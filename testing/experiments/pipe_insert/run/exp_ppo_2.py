from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import imageio
import jax
import numpy as np
import tyro
from brax.io import model as brax_model
from brax.training import acting

# from brax.training.agents import ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from mujoco_playground._src import wrapper
from orbax import checkpoint as ocp

import wandb
from testing.envs.pipe_insert_2 import PipeInsert2


@dataclass
class TrainArgs:
    """CLI configuration for running PPO on the pipe_insert environment."""

    env_name: str = "pipe_insert_2"
    num_timesteps: int = 1_000_000
    num_evals: int = 100
    episode_length: int = 2000
    action_repeat: int = 1
    unroll_length: int = 10
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    discounting: float = 0.99
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    num_envs: int = 256
    batch_size: int = 8192
    seed: int = 0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3

    # logging / checkpointing
    use_wandb: bool = True
    project: str = "pipe_insert_2"
    run_name: str | None = None
    checkpoint_dir: Path = Path("runs/pipe_insert_2")
    checkpoint_every: int = 250_000
    rollout_video: Path | None = None
    rollout_length: int | None = None
    rollout_fps: int = 30


def _maybe_init_wandb(args: TrainArgs) -> wandb.sdk.wandb_run.Run | None:
    if not args.use_wandb:
        return None
    return wandb.init(project=args.project, name=args.run_name, config=asdict(args))


def _log_progress(
    step: int, metrics: Dict[str, Any], run: wandb.sdk.wandb_run.Run | None
) -> None:
    if run is not None:
        wandb.log(metrics, step=step)


def _save_checkpoint(path: Path, params: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(path.as_posix(), params, save_args=save_args, force=True)


def _checkpoint_callback(
    step: int,
    make_inference_fn: Callable[[Any], Any],
    params: Any,
    *,
    args: TrainArgs,
    run: wandb.sdk.wandb_run.Run | None,
) -> None:
    """Periodic checkpoint hook matching the Brax PPO callback signature."""
    del make_inference_fn  # unused but kept for signature compatibility
    # if args.checkpoint_every <= 0 or step % args.checkpoint_every != 0:
    #     return
    # ckpt_path = args.checkpoint_dir / "checkpoints" / f"step_{step}"
    # _save_checkpoint(ckpt_path, params)
    if run is not None:
        wandb.log({"checkpoint_step": step}, step=step)


def _make_eval_env(args: TrainArgs):
    """Create a single-env instance for rendering rollouts."""
    env = PipeInsert2()
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
    )
    # Force batch_size=1 on the internal VmapWrapper so we can unbatch cleanly.
    try:
        env.env.env.batch_size = 1
    except AttributeError:
        pass
    return env


def rollout_history(
    make_policy_fn: Callable[[Any], Callable[[Any, Any], Any]],
    params: Any,
    *,
    args: TrainArgs,
    video_path: Path,
    num_steps: int,
    fps: int,
) -> None:
    """Run a deterministic rollout and write an MP4 to ``video_path``."""
    policy = make_policy_fn(params)
    rng = jax.random.PRNGKey(args.seed)
    env = _make_eval_env(args)

    def _unbatch(state: Any) -> Any:
        return jax.tree_util.tree_map(lambda x: x[0], state)

    rng, reset_key = jax.random.split(rng)
    state = env.reset(reset_key)
    trajectory = [_unbatch(state)]

    for _ in range(num_steps):
        rng, step_key = jax.random.split(rng)
        next_state, _ = acting.actor_step(env, state, policy, step_key)
        trajectory.append(_unbatch(next_state))
        state = next_state

    frames = env.unwrapped.render(trajectory)
    frames = [np.asarray(frame) for frame in frames]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=fps)


def main() -> None:
    args = tyro.cli(TrainArgs, description="Train PPO on the pipe_insert MJX task.")

    run = _maybe_init_wandb(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build the MJX environment; Brax PPO will wrap it via wrap_env_fn.
    env = PipeInsert2()

    train_fn = functools.partial(
        ppo.train,
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
        seed=args.seed,
        gae_lambda=args.gae_lambda,
        clipping_epsilon=args.clipping_epsilon,
        progress_fn=lambda step, metrics: _log_progress(step, metrics, run),
        policy_params_fn=functools.partial(_checkpoint_callback, args=args, run=run),
        network_factory=ppo_networks.make_ppo_networks,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    # Log final metrics and persist parameters
    if run is not None:
        wandb.log(metrics, step=args.num_timesteps)
    brax_model.save_params(args.checkpoint_dir / "model", params)
    _save_checkpoint(args.checkpoint_dir / "checkpoints" / "final", params)

    rollout_path = (
        args.rollout_video
        if args.rollout_video is not None
        else args.checkpoint_dir / "rollout.mp4"
    )
    rollout_history(
        make_inference_fn,
        params,
        args=args,
        video_path=rollout_path,
        num_steps=args.rollout_length or args.episode_length,
        fps=args.rollout_fps,
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
