# import isaacgym
# import isaacgymenvs
from pathlib import Path
from typing import Optional

import brax.envs as envs
import cv2
import numpy as np
import torch
import torch.nn as nn
from mujoco_playground import registry
from mujoco_playground._src.dm_control_suite import cartpole, pendulum
from mujoco_playground._src.wrapper import mjx_env

from skrl.envs.torch import wrap_env

# from algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.utils import set_seed
from testing import wrappers as wrap
from testing.envs.xpose import XPose, default_config

# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.ppo_utils import get_ppo_default_models


def setup_environment(
    batch_size: int = 256,
    episode_length: int = 1000,
    auto_reset: bool = True,
    action_repeat: int = 1,
):
    """Set up the MJX XPose environment with proper wrapping."""

    # Create base environment
    # env = cartpole.Balance(swing_up=False, sparse=False)
    env = pendulum.SwingUp()

    # env = mjx_XPose()

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

models_ppo = get_ppo_default_models(env)
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["rollouts"] = 128  # memory_size  ## 16 horizon_length
cfg_ppo["learning_epochs"] = 2  # mini_epochs
cfg_ppo["mini_batches"] = 64  # horizaen_length * numberof_actor / minibathch_size  ## 8
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 3e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 2.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 20 and 200 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 50
cfg_ppo["experiment"]["checkpoint_interval"] = 200

device = env.device

memory = RandomMemory(memory_size=128, num_envs=env.num_envs, device=device)
expert_memory = RandomMemory(
    memory_size=5000, num_envs=env.num_envs, device=device, replacement=False
)
agent = PPO(
    models=models_ppo,
    memory=memory,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

model_path = "testing/train/results/models/train_brax/checkpoints/agent_200000.pt"

agent.load(model_path)
agent.set_mode("eval")

num_timesteps = 1000

state, _ = env.reset()

# Video setup
frames = []
video_filename = "policy_rollout_brax.mp4"

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
