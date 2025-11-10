# import isaacgym
# import isaacgymenvs

import copy
import itertools

import gym
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.torch import load_isaacgym_env_preview4, wrap_env
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import postprocessing, set_seed

# from algorithms.IBRL_active import IBRL
from testing.shen.IBRL import IBRL, IBRL_DEFAULT_CONFIG

# Set random seed for reproducibility across PyTorch, NumPy, and Isaac Gym
RANDOM_SEED = 10  # 10, 11, 12 for reproducing the DRLR paper results
set_seed(RANDOM_SEED)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================
# define RL agent: TD3 (Deterministic models for actors and critics)
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
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.num_actions),
        )

    def compute(self, inputs, role):
        raw_action = self.net(inputs["states"])
        return torch.tanh(raw_action), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.linear_layer_3 = nn.Linear(256, 1)

    def compute(self, inputs, role):
        x = F.relu(
            self.linear_layer_1(
                torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
            )
        )
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


# Define IL agent, BC agent
class ILActor(DeterministicMixin, Model):
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


# ============================================================================
# Load and wrap the Isaac Gym environment
# ============================================================================
# env = isaacgymenvs.make(
#     seed=10,
#     task="FrankaCabinet",
#     num_envs=10,
#     sim_device="cuda:0",
#     rl_device="cuda:0",
#     graphics_device_id=0,
#     headless=True,
# )


env = wrap_env(env)
device = env.device

# ============================================================================
# # Instantiate replay and expert memory buffer
# ============================================================================
memory = RandomMemory(
    memory_size=350000, num_envs=env.num_envs, device=device, replacement=True
)
expert_memory = RandomMemory(
    memory_size=15000, num_envs=env.num_envs, device=device, replacement=True
)


# ============================================================================
# # Initiate an IL agent in order to load a trained BC agent
# ============================================================================
# Behavior clone (BC) requires 1 model to learn the expert behavior, directly load
models_BC = {}
models_BC["policy"] = ILActor(
    env.observation_space, env.action_space, device, clip_actions=True
)
# Load the full saved state
saved_state = torch.load("./RefAgent/cab_td3_imperfect/checkpoints/agent_60000.pt")
# Extract just the policy weights
policy_state = saved_state["policy"]
models_BC["policy"].load_state_dict(policy_state)
models_BC["policy"].eval()


# ============================================================================
# # Initiate the RL agent: TD3
# ============================================================================
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#spaces-and-models
models_td3 = {}
models_td3["policy"] = DeterministicActor(
    env.observation_space, env.action_space, device, clip_actions=True
)
models_td3["target_policy"] = DeterministicActor(
    env.observation_space, env.action_space, device, clip_actions=True
)
models_td3["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_td3["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_td3["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_td3["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_td3.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Create Emseabling Q networks
ensemble_size = 5
# Create ensemble of critics (each with unique parameters)
critics = []
target_critics = []
for i in range(ensemble_size):
    # Create new critic instance for each position
    critic = Critic(env.observation_space, env.action_space, device)

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    critic.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    target_critic = copy.deepcopy(critic)  # Create target as deep copy

    critics.append(critic)
    target_critics.append(target_critic)

models_td3["critics"] = critics
models_td3["target_critics"] = target_critics


# Configure and instantiate the agent.
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#configuration-and-hyperparameters
cfg_IBRL = IBRL_DEFAULT_CONFIG.copy()
cfg_IBRL["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
# cfg_IBRL["exploration"]["noise"] = None
cfg_IBRL["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
cfg_IBRL["smooth_regularization_clip"] = 0.5
cfg_IBRL["gradient_steps"] = 1
cfg_IBRL["RED-Q_enable"] = True
cfg_IBRL["offline"] = True
cfg_IBRL["batch_size"] = 128
cfg_IBRL["random_timesteps"] = 0
cfg_IBRL["learning_starts"] = 0
cfg_IBRL["learning_rate"] = 3e-4
cfg_IBRL["num_envs"] = env.num_envs
# cfg_IBRL["demo_file"] = "/home/chen/Downloads/new/memories/Cab-expert-bc.csv"
cfg_IBRL["demo_file"] = "./Demos/cab_imperfect.csv"
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_IBRL["experiment"]["write_interval"] = 500
cfg_IBRL["experiment"]["checkpoint_interval"] = 1000

agent_IBRL = IBRL(
    models=models_td3,
    models_il=models_BC,
    memory=memory,
    expert_memory=expert_memory,
    cfg=cfg_IBRL,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


# # Configure and instantiate the RL trainer
# # agent_IBRL.load("./runs/CoL-cab-50000-10/checkpoints/agent_50000.pt")
# cfg = {"timesteps":350000, "headless": False}
# trainer = SequentialTrainer(cfg=cfg,
#                             env=env,
#                             agents=agent_IBRL)
#
# # start training
# trainer.train()

# Evaluate policy
# load checkpoint (agent)
# agent_IBRL.load("./runs/IBRL1/checkpoints/agent_30000.pt")
# agent_IBRL.load("./runs/IBRLe/checkpoints/agent_100000.pt")
agent_IBRL.load("./runs/25-10-09_23-38-59-988669_IBRL/checkpoints/agent_6000.pt")
# Configure and instantiate the RL trainer
cfg = {"timesteps": 4000, "headless": False}
trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent_IBRL, agents_scope=[])
# evaluate the agent
trainer.eval()
