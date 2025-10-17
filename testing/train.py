"""Train PPO on XPose robot gripper environment."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from mujoco_playground import registry
from mujoco_playground._src.dm_control_suite import cartpole

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
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


class PPOModel(GaussianMixin, DeterministicMixin, Model):
    """Shared model for PPO policy and value networks."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )
        DeterministicMixin.__init__(self, clip_actions)

        # Shared backbone
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )

        # Policy head
        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # Value head
        self.value_layer = nn.Linear(32, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        features = self.net(inputs["states"])

        if role == "policy":
            return self.mean_layer(features), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(features), {}


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


def setup_agent(env, device):
    """Set up PPO agent with configuration."""
    # Memory
    memory = RandomMemory(memory_size=128, num_envs=env.num_envs, device=device)

    # Models
    models_ppo = {}
    models_ppo["policy"] = PPOModel(env.observation_space, env.action_space, device)
    models_ppo["value"] = PPOModel(env.observation_space, env.action_space, device)

    # PPO configuration
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()

    # Training parameters
    cfg_ppo["rollouts"] = 128
    cfg_ppo["learning_epochs"] = 2
    cfg_ppo["mini_batches"] = 64
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
    cfg_ppo["experiment"]["checkpoint_interval"] = 200
    cfg_ppo["experiment"]["directory"] = model_path.as_posix()
    cfg_ppo["experiment"]["experiment_name"] = Path(__file__).stem
    cfg_ppo["experiment"]["wandb"] = True

    # Create agent
    agent = PPO(
        models=models_ppo,
        memory=memory,
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
    # _env_name = "CartpoleBalance"
    env = setup_environment(num_envs=256, debug=False)
    # env = setup_environment(env_name="xpose", num_envs=256)
    device = env.device

    # Setup agent
    print("Setting up PPO agent...")
    agent = setup_agent(env, device)

    # Setup trainer
    print("Setting up trainer...")
    cfg_trainer = SEQUENTIAL_TRAINER_DEFAULT_CONFIG.copy()
    cfg_trainer["timesteps"] = 50000
    cfg_trainer["eval_frequency"] = 1000
    cfg_trainer["headless"] = True

    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main()
