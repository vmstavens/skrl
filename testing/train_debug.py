"""Train PPO on XPose robot gripper environment."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from mujoco_playground import registry
from mujoco_playground._src.dm_control_suite import cartpole

from skrl.envs.torch import wrap_env
from skrl.envs.wrappers.torch.playground_envs import VmapWrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.utils import set_seed
from testing.env import XPose, default_config
from testing.ppo import PPO, PPO_DEFAULT_CONFIG

from .envs.cartpole import Balance


def setup_environment(env_name: str = "xpose", num_envs: int = 256, debug=False):
    # def setup_environment(env_name: str = "xpose", num_envs: int = 256, debug=False):
    """Set up the XPose environment with proper wrapping."""
    if debug:
        env = Balance()
        # env = cartpole.Balance(swing_up=False, sparse=False)
        env = VmapWrapper(env, num_envs)
    else:
        # Register the environment
        registry.dm_control_suite.register_environment(
            env_name=env_name,
            env_class=XPose,
            cfg_class=default_config,
        )

        # Load and wrap environment
        env = registry.dm_control_suite.load(env_name)
        env = VmapWrapper(env, num_envs)

    env = wrap_env(env)

    return env


def setup_agent(env, device):
    """Set up PPO agent with improved configuration."""
    # Memory
    memory = RandomMemory(memory_size=128, num_envs=env.num_envs, device=device)
    expert_memory = RandomMemory(
        memory_size=5000, num_envs=env.num_envs, device=device, replacement=False
    )

    from .ppo_utils import get_ppo_default_models

    models_ppo = get_ppo_default_models(env)

    # Models - using separate policy and value networks
    # models_ppo = {}
    # models_ppo["policy"] = Shared(env.observation_space, env.action_space, device)
    # models_ppo["value"] = Shared(
    #     env.observation_space, env.action_space, device
    # )  # same instance: shared model

    # PPO configuration
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()

    # Improved training parameters
    cfg_ppo["rollouts"] = 128
    cfg_ppo["learning_epochs"] = 4  # Increased
    cfg_ppo["mini_batches"] = 16  # Reduced for better gradient estimates
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 3e-4
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
    env = setup_environment(num_envs=256, debug=True)
    device = env.device

    # Setup agent
    print("Setting up PPO agent...")
    agent = setup_agent(env, device)
    agent.set_mode("train")

    # Setup trainer with increased timesteps
    print("Setting up trainer...")
    cfg_trainer = SEQUENTIAL_TRAINER_DEFAULT_CONFIG.copy()
    cfg_trainer["timesteps"] = 50_000
    cfg_trainer["eval_frequency"] = 1000
    cfg_trainer["headless"] = True

    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main()
