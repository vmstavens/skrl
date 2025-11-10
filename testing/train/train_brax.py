"""Train PPO on XPose robot gripper environment."""

from pathlib import Path
from typing import Any, Optional

import brax
import brax.envs as envs
import jax
import torch
import torch.nn as nn
from brax.envs.wrappers.training import VmapWrapper as brax_VmapWrapper
from mujoco_playground import registry
from mujoco_playground._src.dm_control_suite import cartpole, pendulum, point_mass
from mujoco_playground.config import dm_control_suite_params

from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

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
    # env = mjx_XPose()
    env = point_mass.PointMass()

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
    memory = RandomMemory(
        memory_size=128, num_envs=env.num_envs, device=device, replacement=True
    )
    expert_memory = RandomMemory(
        memory_size=5000, num_envs=env.num_envs, device=device, replacement=False
    )

    models_ppo = get_ppo_default_models(env)

    # PPO configuration
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()

    # Improved training parameters
    cfg_ppo["rollouts"] = 512
    # cfg_ppo["rollouts"] = 128
    cfg_ppo["learning_epochs"] = 4  # Increased
    cfg_ppo["mini_batches"] = 128
    # cfg_ppo["mini_batches"] = 32
    # cfg_ppo["mini_batches"] = 16  # Reduced for better gradient estimates
    cfg_ppo["discount_factor"] = 0.995
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 0.001
    # cfg_ppo["learning_rate"] = 3e-4
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_ppo["random_timesteps"] = 0
    cfg_ppo["learning_starts"] = 0
    cfg_ppo["grad_norm_clip"] = 0.5  # Less restrictive
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["value_clip"] = 0.2
    cfg_ppo["clip_predicted_values"] = True
    cfg_ppo["entropy_loss_scale"] = 0.01  # Added for exploration
    cfg_ppo["value_loss_scale"] = 1.0  # Reduced
    cfg_ppo["kl_threshold"] = 0
    # Reduced reward scaling
    cfg_ppo["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.5

    # Preprocessors
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": device,
    }
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    cfg_ppo["experiment"]["write_interval"] = 50
    cfg_ppo["experiment"]["checkpoint_interval"] = 1000
    cfg_ppo["experiment"]["directory"] = model_path.as_posix()
    cfg_ppo["experiment"]["experiment_name"] = Path(__file__).stem
    cfg_ppo["experiment"]["wandb"] = True

    # Create agent
    agent = PPO(
        models=models_ppo,
        memory=memory,
        expert_memory=expert_memory,
        cfg=cfg_ppo,
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
        env = setup_environment_xpose_mjx_fixed(batch_size=8000)
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

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    # trainer.eval()


if __name__ == "__main__":
    main()
