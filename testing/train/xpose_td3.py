"""Train PPO on XPose robot gripper environment."""

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import brax
import brax.envs as envs
import gym
import jax
import torch
import torch.nn as nn
from brax.envs.wrappers.training import VmapWrapper as brax_VmapWrapper
from mujoco_playground import registry
from mujoco_playground._src.dm_control_suite import cartpole, pendulum
from mujoco_playground.config import dm_control_suite_params

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer

# from skrl.trainers.torch import SequentialTrainer
# from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.trainers.torch.sequential_2 import (
    SEQUENTIAL_TRAINER_2_DEFAULT_CONFIG,
    SequentialTrainer2,
)
from skrl.utils import set_seed
from testing import wrappers as wrap
from testing.envs.cartpole_brax import InvertedPendulum
from testing.envs.xpose import XPose, default_config
from testing.envs.xpose import XPose as mjx_XPose
from testing.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.ppo_utils import get_ppo_default_models
from testing.td3_utils import get_td3_default_models

# c = dm_control_suite_params("CartpoleBalance")


def setup_environment(env_name: str = "xpose", num_envs: int = 256, debug=False):
    # def setup_environment(env_name: str = "xpose", num_envs: int = 256, debug=False):
    """Set up the XPose environment with proper wrapping."""
    env = envs.create("inverted_pendulum", batch_size=num_envs)
    env = wrap_env(env, wrapper="brax")
    return env


def setup_environment_xpose_mjx_fixed(
    batch_size: int = 256,
    episode_length: int = 1000,
    auto_reset: bool = True,
    action_repeat: int = 1,
):
    """Set up the MJX XPose environment with proper wrapping."""

    # Create base environment
    # env = pendulum.SwingUp()
    # env = cartpole.Balance(swing_up=False, sparse=False)
    env = XPose()

    # from mujoco_playground._src.dm_control_suite import point_mass

    # point_mass

    env = wrap.create(
        env,
        batch_size=batch_size,
        episode_length=episode_length,
        auto_reset=auto_reset,
        action_repeat=action_repeat,
    )
    env = wrap_env(env, wrapper="playground")

    return env


def setup_agent(env, device):
    """Set up PPO agent with improved configuration."""
    # Memory
    memory = RandomMemory(memory_size=200_000, num_envs=env.num_envs, device=device)
    expert_memory = RandomMemory(
        memory_size=5000, num_envs=env.num_envs, device=device, replacement=False
    )

    models_td3 = get_td3_default_models(env)

    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/td3.html#configuration-and-hyperparameters
    cfg = TD3_DEFAULT_CONFIG.copy()
    # cfg["exploration_noise"] = GaussianNoise
    cfg["exploration"]["noise"] = GaussianNoise(0, 0.01, device=device)

    cfg["exploration_noise_kwargs"] = {"mean": 0.0, "std": 0.1, "device": device}
    cfg["exploration_scheduler"] = lambda timestep, timesteps: max(
        1 - timestep / timesteps, 1e-2
    )
    # cfg["smooth_regularization_noise"] = GaussianNoise
    cfg["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)

    # cfg["smooth_regularization_noise"] = GaussianNoise
    cfg["smooth_regularization_noise_kwargs"] = {
        "mean": 0.0,
        "std": 0.2,
        "device": device,
    }
    cfg["smooth_regularization_clip"] = 0.5
    cfg["discount_factor"] = 0.98
    cfg["batch_size"] = 128
    cfg["random_timesteps"] = 128
    cfg["learning_starts"] = 128

    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    cfg["experiment"]["write_interval"] = 50
    cfg["experiment"]["checkpoint_interval"] = 1000
    cfg["experiment"]["directory"] = model_path.as_posix()
    cfg["experiment"]["experiment_name"] = Path(__file__).stem
    cfg["experiment"]["wandb"] = True

    # logging to TensorBoard and write checkpoints (in timesteps)

    # Create agent
    agent = TD3(
        models=models_td3,
        memory=memory,
        # expert_memory=expert_memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    return agent


def main():
    """Main training function."""
    # Set seed for reproducibility
    set_seed(27)

    # Setup environment
    print("Setting up environment...")
    mjx = True
    if mjx:
        env = setup_environment_xpose_mjx_fixed(batch_size=256)
        eval_env = setup_environment_xpose_mjx_fixed(batch_size=1)
    else:
        env = setup_environment(num_envs=8000, debug=True)
        eval_env = setup_environment(num_envs=1, debug=True)
        print(".........................................")
    device = env.device

    # Setup agent
    print("Setting up PPO agent...")
    agent = setup_agent(env, device)
    agent.set_mode("train")

    # Setup trainer with increased timesteps
    print("Setting up trainer...")
    cfg_trainer = SEQUENTIAL_TRAINER_2_DEFAULT_CONFIG.copy()
    # cfg_trainer = SEQUENTIAL_TRAINER_DEFAULT_CONFIG.copy()
    cfg_trainer["timesteps"] = 200_000
    cfg_trainer["eval_frequency"] = 1000
    cfg_trainer["headless"] = True

    trainer = SequentialTrainer2(
        cfg=cfg_trainer, env=env, agents=agent, eval_env=eval_env
    )

    # agent = setup_agent(env, device)

    # model_path = "testing/train/results/models/train_brax_td3/checkpoints/agent_30000.pt"

    # agent.load(model_path)
    # agent.set_mode("eval")

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    # trainer.eval()

    quit()

    num_timesteps = 1000

    state, _ = env.reset()

    # Video setup
    frames = []
    video_filename = "policy_rollout_td3.mp4"

    print("Generating frames...")

    for i in range(num_timesteps):
        actions, _, _ = agent.act(states=state, timestep=i, timesteps=num_timesteps)

        # env: mjx_env.MjxEnv = env._unwrapped

        next_states, rewards, terminated, truncated, infos = env.step(
            actions=actions.detach()
        )

        print(rewards)

        # Get rendered frame
        frame = env.render()
        # print(frame.shape)
        # quit()

        frames.append(frame)

        # Update state
        state = next_states

        # Check if episode ended
        if terminated[0]:
            # print(f"Episode ended at step {i}")
            state, _ = env.reset()

    print(f"Generated {len(frames)} frames")

    # Create video from frames
    if frames:
        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
        fps = 30  # Frames per second

        out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

        # Write all frames to video
        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Video saved as: {video_filename}")
    else:
        print("No frames were generated")


if __name__ == "__main__":
    main()
