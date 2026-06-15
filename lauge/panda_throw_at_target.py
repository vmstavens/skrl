import functools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax
import tyro
from brax import envs
from brax.base import System
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from jax import numpy as jp
from orbax import checkpoint as ocp

import wandb
from lauge.cfg import Cfg

from .learning import create_data_directory
from .mjx_panda_throw_at_target_ladep import PandaThrowAtTarget


class PandaThrowAtTargetArgs(Cfg):
    """
    Configuration arguments for training and environment setup.
    """

    algo: Callable = ppo.train
    env_name: str = Path(__file__).stem
    num_evals: int = 20
    # TODO: make sure 150 is the correct episode length, if updated also update frames in the video
    episode_length: int = 500
    action_repeat: int = 1
    unroll_length: int = 10
    num_updates_per_batch: int = 8
    discounting: float = 0.97
    learning_rate: float = 0.001
    entropy_cost: float = 0.02
    seed: int = 0
    # TODO: test a short training section with the loaded keyframe to make sure it starts in the correct position holding the cube.

    num_timesteps: int = 50_000_000
    normalize_observations: bool = True
    num_minibatches: int = 32
    num_envs: int = 2048
    batch_size: int = 512
    reward_scaling: int = 0.01
    # TODO: remove ChatGPT from all scripts if turned in
    normalize_advantage = True  # new ChatGPT thing


args = tyro.cli(PandaThrowAtTargetArgs, description=__doc__)

session = wandb.init(
    # set the wandb project where this run will be logged
    project=args.env_name,
    # track hyperparameters and run metadata
    config=args,
)

# save path init
session_path = create_data_directory(
    environment_name=args.env_name, session_name=session.name
)

# register enfironment
envs.register_environment(args.env_name, PandaThrowAtTarget)
env: PandaThrowAtTarget = envs.get_environment(args.env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


def checkpoint_callback(current_step, make_policy, params):
    """
    A user-defined callback function that can be used for saving
    policy checkpoints
    """
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = session_path / "checkpoints" / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


def evaluation_callback(num_steps, metrics: dict):
    """
    A user-defined callback function for reporting/plotting metrics
    """
    times.append(datetime.now())
    wandb.log(metrics)


# TODO: maybe test without domain_randomization to make training easier, but less robust
def domain_randomize(sys: System, rng: jp.ndarray):
    """Randomizes the mjx.Model."""

    @jax.vmap
    def rand(rng: jp.ndarray):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = (
            jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1])
            + sys.actuator_gainprm[:, 0]
        )
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    sys = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )

    return sys, in_axes


# define the size of my network (different than vims, same as notebook)
make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(32, 32, 32, 32),
    policy_obs_key="state",
    value_hidden_layer_sizes=(256, 256, 256, 256, 256),
    value_obs_key="state",
)

train_fn = functools.partial(
    args.algo,
    num_timesteps=args.num_timesteps,
    num_evals=args.num_evals,
    reward_scaling=args.reward_scaling,
    episode_length=args.episode_length,
    normalize_observations=args.normalize_observations,
    action_repeat=args.action_repeat,
    unroll_length=args.unroll_length,
    num_minibatches=args.num_minibatches,
    num_updates_per_batch=args.num_updates_per_batch,
    discounting=args.discounting,
    learning_rate=args.learning_rate,
    entropy_cost=args.entropy_cost,
    num_envs=args.num_envs,
    batch_size=args.batch_size,
    network_factory=make_networks_factory,
    # randomization_fn=domain_randomize,
    policy_params_fn=checkpoint_callback,
    seed=args.seed,
)

times = [datetime.now()]

# train the algorithm
make_inference_fn, params, metrics = train_fn(
    environment=env, progress_fn=evaluation_callback
)

# add the training time and JIT time to metrics
metrics["time to jit"] = times[1].second - times[0].second
metrics["time to train"] = times[-1].second - times[1].second

# upload all metrics along with times
wandb.log(metrics)

# save model
model.save_params(session_path / "model", params)

# generate a short video demonstrating the capabilities of the policy
demo_video_frames = env.get_demo_video(
    make_inference_fn=make_inference_fn, render_every=1, n_steps=500, params=params
)

# send demo video to wandb
wandb.log({f"{session.name} demo": wandb.Video(demo_video_frames, fps=1.0 / env.dt)})
