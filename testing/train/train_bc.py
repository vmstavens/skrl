# import isaacgym
# import isaacgymenvs
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mujoco_playground._src.dm_control_suite import cartpole, pendulum

from skrl.envs.torch import load_isaacgym_env_preview4, wrap_env
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import postprocessing, set_seed
from testing import wrappers as wrap
from testing.envs.xpose import XPose
from testing.shen.BC import BC, BC_DEFAULT_CONFIG

from .demon import TransitionDataset
from .demonstration import DemonstrationDataset

# set the seed for reproducibility
set_seed(11)


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


# Define the BC NN architecture (stochastic and deterministic models) for the agents using mixins.
# - DeterministicActor: takes as input the environment's observation/state and returns an action
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


env = setup_environment()

device = env.device

data_path = Path("./data/norm_smooth_data/")
data_files = list(data_path.glob("*.json"))


def reward_fn(state, action, next_state):
    return -np.linalg.norm(state)  # Negative distance as reward


def termination_fn(state, next_state):
    return np.linalg.norm(state) < 0.0001


dataset = TransitionDataset(
    json_paths=data_files, reward_fn=reward_fn, termination_fn=termination_fn
)

# Get all transitions from the dataset
states = []
actions = []
next_states = []
rewards = []
dones = []

for i in range(len(dataset)):
    transition = dataset[i]
    states.append(transition["state"].numpy())
    actions.append(transition["action"].numpy())

    if "next_state" in transition:
        next_states.append(transition["next_state"].numpy())
    else:
        # Handle case where next_state might not be available
        next_states.append(transition["state"].numpy())  # Fallback

    if "reward" in transition:
        rewards.append(transition["reward"].numpy())
    else:
        rewards.append(0.0)  # Default reward

    if "done" in transition:
        dones.append(transition["done"].numpy())
    else:
        dones.append(False)  # Default not done

# Convert to arrays
states = np.array(states)
actions = np.array(actions)
next_states = np.array(next_states)
rewards = np.array(rewards)
dones = np.array(dones)

# Convert to tensors
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)
next_states = torch.tensor(next_states, dtype=torch.float32)
rewards = torch.tensor(rewards, dtype=torch.float32)
terminated = torch.tensor(dones, dtype=torch.float32)

memory_size = len(states)

a_dim = actions.shape[1]  # Get action dimension from data
o_dim = states.shape[1]  # Get observation dimension from data

# Create expert memory
expert_memory = RandomMemory(memory_size=memory_size)
expert_memory.create_tensor(name="states", size=o_dim, dtype=torch.float32)
expert_memory.create_tensor(name="actions", size=a_dim, dtype=torch.float32)
expert_memory.create_tensor(name="next_states", size=o_dim, dtype=torch.float32)
expert_memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
expert_memory.create_tensor(name="terminated", size=1, dtype=torch.float32)

# Add samples to memory
expert_memory.add_samples(
    states=states,
    actions=actions,
    next_states=next_states,
    rewards=rewards.unsqueeze(-1),  # Add dimension for reward size
    terminated=terminated.unsqueeze(-1),  # Add dimension for terminated size
)

with open("memories/expert_memory.pkl", "wb") as f:
    pickle.dump(expert_memory, f)

# instantiate the agent's models, Behavior clone (BC) requires 1 model to learn the expert behavior
models_BC = {}
models_BC["policy"] = BCmodel(
    env.observation_space, env.action_space, device, clip_actions=True
)
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

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_BC.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

agent_BC = BC(
    models=models_BC,
    expert_memory=expert_memory,
    cfg=cfg_BC,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# Configure and instantiate the RL trainer
cfg = {"timesteps": 2000, "headless": True}
trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent_BC)

# start training
trainer.train()

# # Evaluate policy
# # load checkpoint (agent)
# agent_BC.load("./runs/BC-CAB-256128/checkpoints/agent_5000.pt")
# # Configure and instantiate the RL trainer
# cfg = {"timesteps": 3000, "headless": False}
# trainer = SequentialTrainer(cfg=cfg,
#                             env=env,
#                             agents=agent_BC,
#                             agents_scope=[])
# # evaluate the agent
# trainer.eval()
