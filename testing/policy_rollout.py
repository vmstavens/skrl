# import isaacgym
# import isaacgymenvs
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from mujoco_playground import registry
from mujoco_playground._src.dm_control_suite import cartpole
from mujoco_playground._src.wrapper import mjx_env

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env

# from algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.wrappers.torch.playground_envs import VmapWrapper
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.utils import set_seed
from testing.env import XPose, default_config

from .ppo_utils import get_ppo_default_models


def setup_environment(env_name: str = "xpose", num_envs: int = 256, debug=False):
    """Set up the XPose environment with proper wrapping."""
    if debug:
        env = cartpole.Balance(swing_up=True, sparse=False)
        env = VmapWrapper(env, num_envs)
    else:
        # Register the environment
        registry.dm_control_suite.register_environment(
            env_name=env_name,
            env_class=XPose,
            cfg_class=default_config,
            # env_name=env_name, env_class=XPose, cfg_class=default_config
        )

        # Load and wrap environment
        env = registry.dm_control_suite.load(env_name)
        env = VmapWrapper(env, num_envs)

    env = wrap_env(env)

    return env


env = setup_environment("xpose", num_envs=1)

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

agent = PPO(
    models=models_ppo,
    memory=memory,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

model_path = "testing/results/models/train/checkpoints/agent_50000.pt"

agent.load(model_path)
agent.set_mode("eval")

num_timesteps = 1000

state, _ = env.reset()

# Video setup
frames = []
video_filename = "policy_rollout.mp4"

print("Generating frames...")

for i in range(num_timesteps):
    actions, _, _ = agent.act(states=state, timestep=i, timesteps=num_timesteps)

    # env: mjx_env.MjxEnv = env._unwrapped

    next_states, rewards, terminated, truncated, infos = env.step(
        actions=actions.detach()
    )

    mocap_pos = env._env.unwrapped.unwrapped._state.data.mocap_pos[0][0]
    agent_pos = env._env.unwrapped.unwrapped._state.data.qpos[0]
    agent_ctrl = env._env.unwrapped.unwrapped._state.data.ctrl[0]
    print(np.linalg.norm(mocap_pos - agent_pos))
    print(np.linalg.norm(mocap_pos - agent_ctrl))
    # quit()

    # Get rendered frame
    # frame = env.render()[0]
    frame = env.render(width=500, height=500)
    # break
    # print(f"Frame {i}: {frame.shape=}")

    # # Convert frame to proper format for video
    # if frame.shape[-1] == 3:  # If RGB, convert to BGR for OpenCV
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # print(f"{frame[0][0]=}")
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
