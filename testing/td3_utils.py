import argparse
import os

# import isaacgym
# import isaacgymenvs
from pathlib import Path

import gym
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

# import the skrl components to build the RL system
from skrl.agents.torch.td3 import TD3
from skrl.envs.loaders.torch import load_isaaclab_env

# from algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import Wrapper, wrap_env
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR, KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import compute_space_size, unflatten_tensorized_space


class DeterministicActor(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        dropout_rate=0.1,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            # nn.Dropout(dropout_rate),
            nn.ELU(),
            nn.Linear(256, 128),
            # nn.Dropout(dropout_rate),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class Critic1(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        dropout_rate=0.5,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256),
            # nn.Dropout(dropout_rate),
            nn.ELU(),
            nn.Linear(256, 128),
            # nn.Dropout(dropout_rate),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(
            torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        ), {}


def get_td3_default_models(env: Wrapper):
    device = env.device
    models = {}
    models["policy"] = DeterministicActor(
        env.observation_space, env.action_space, device, clip_actions=True
    )
    models["target_policy"] = DeterministicActor(
        env.observation_space, env.action_space, device, clip_actions=True
    )
    models["critic_1"] = Critic1(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic1(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic1(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic1(env.observation_space, env.action_space, device)

    return models
