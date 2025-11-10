# import isaacgym
# import isaacgymenvs
import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import brax
import brax.envs as envs
import cv2
import gym
import jax
import numpy as np
import torch
import torch.nn as nn
from brax.envs.wrappers.training import VmapWrapper as brax_VmapWrapper
from mujoco_playground import registry
from mujoco_playground._src.dm_control_suite import cartpole, pendulum
from mujoco_playground._src.wrapper import mjx_env
from mujoco_playground.config import dm_control_suite_params

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.envs.wrappers.torch import wrap_env

# from algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG

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

# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.ppo_utils import get_ppo_default_models
from testing.shen.BC import BC, BC_DEFAULT_CONFIG
from testing.td3_utils import get_td3_default_models


def setup_environment(
    batch_size: int = 256,
    episode_length: int = 1000,
    auto_reset: bool = True,
    action_repeat: int = 1,
):
    """Set up the MJX XPose environment with proper wrapping."""

    # Create base environment
    # env = cartpole.Balance(swing_up=False, sparse=False)
    # env = pendulum.SwingUp()

    env = XPose()

    env = wrap.create(
        env,
        batch_size=batch_size,
        episode_length=episode_length,
        auto_reset=auto_reset,
        action_repeat=action_repeat,
    )
    env = wrap_env(env, wrapper="playground")

    return env


env = setup_environment(batch_size=1)

device = env.device


def setup_agent(env, device):
    """Set up BC agent with improved configuration."""

    class BCmodel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net = nn.Sequential(
                nn.Linear(self.num_observations, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, self.num_actions),
            )

        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}

    # logging to TensorBoard and write checkpoints (in timesteps)
    models_BC = {}
    models_BC["policy"] = BCmodel(
        env.observation_space, env.action_space, device, clip_actions=True
    )

    with open("memories/expert_memory.pkl", "rb") as f:
        expert_memory = pickle.load(f)

    cfg_BC = BC_DEFAULT_CONFIG.copy()
    cfg_BC["gradient_steps"] = 5
    cfg_BC["batch_size"] = 256
    cfg_BC["demo_file"] = "./Demos/Cab-expert-bc.csv"
    cfg_BC["exploration"]["noise"] = GaussianNoise(0, 0.0001, device=device)
    cfg_BC["smooth_regularization_clip"] = 0.0001

    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    cfg_BC["experiment"]["write_interval"] = 50
    cfg_BC["experiment"]["checkpoint_interval"] = 100
    cfg_BC["experiment"]["directory"] = model_path.as_posix()
    cfg_BC["experiment"]["experiment_name"] = Path(__file__).stem
    cfg_BC["experiment"]["wandb"] = True

    # Create agent
    agent_BC = BC(
        models=models_BC,
        expert_memory=expert_memory,
        cfg=cfg_BC,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    return agent_BC


agent = setup_agent(env, device)

model_path = "testing/train/results/models/train_bc/checkpoints/best_agent.pt"
# model_path = "testing/train/results/models/train_brax_td3/checkpoints/agent_30000.pt"

agent.load(model_path)
agent.set_mode("eval")

num_timesteps = 1000

state, _ = env.reset()

# Video setup
frames = []
video_filename = "policy_rollout_bc.mp4"

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
    # quit()

    frames.append(frame)

    # Update state
    state = next_states

    # Check if episode ended
    if terminated[0]:
        print(f"Episode ended at step {i}")
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


# from mujoco_playground._src.dm_control_suite import point_mass
