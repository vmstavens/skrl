# import isaacgym
# import isaacgymenvs
from pathlib import Path

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env

# from algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import Wrapper, wrap_env
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR, KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import compute_space_size, unflatten_tensorized_space

# set the seed for reproducibility
set_seed(27)


# Define the shared model (stochastic and deterministic models) for the agent using mixins.
class Shared(GaussianMixin, DeterministicMixin, Model):
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

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            # nn.ELU(),
            # nn.Linear(32, 32),
            nn.ELU(),
        )

        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Sequential(nn.Linear(32, 1))
        #     nn.ELU(),
        #     nn.Linear(256, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 1),
        # )
        # nn.Linear(32, 1)
        # self.value_layer = nn.Linear(32, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return (
                self.mean_layer(self.net(inputs["states"])),
                self.log_std_parameter,
                {},
            )
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}


def get_ppo_default_models(env: Wrapper):
    device = env.device
    models_ppo = {}
    models_ppo["policy"] = Shared(env.observation_space, env.action_space, device)
    models_ppo["value"] = Shared(
        env.observation_space, env.action_space, device
    )  # same instance: shared model

    return models_ppo


def get_ppo_default_config():
    pass
