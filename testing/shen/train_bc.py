# import isaacgym
# import isaacgymenvs
import json
import numpy as np
import torch
import torch.nn as nn

from skrl.envs.torch import load_isaacgym_env_preview4, wrap_env
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import postprocessing, set_seed
from testing.shen.BC import BC, BC_DEFAULT_CONFIG

# set the seed for reproducibility
set_seed(11)


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


# Load and wrap the Isaac Gym environment

import brax.envs as envs
from brax.envs.wrappers.training import VmapWrapper

env = envs.create("inverted_pendulum")
env = VmapWrapper(env, 10)

env = wrap_env(env)
device = env.device

# Instantiate a RandomMemory (without replacement) as shared experience replay memory
expert_memory = RandomMemory(
    memory_size=20000, num_envs=env.num_envs, device=device, replacement=False
)

expert_memory.create_tensor("states", env.observation_space.shape[0])
expert_memory.create_tensor("next_states", env.observation_space.shape[0])
expert_memory.create_tensor("actions", env.action_space.shape[0])
expert_memory.create_tensor("rewards", 1)
expert_memory.create_tensor("terminated", 1)

from glob import glob

file_names = glob("data/*.json")

expert_data = {
    "states": [],
    "next_states": [],
    "actions": [],
    "rewards": [],
    "terminated": []
}

for fn in file_names:
    with open(fn, "r") as f:
        datai = json.load(f)
    expert_data["states"].extend(datai["states"])
    expert_data["next_states"].extend(datai["next_states"])
    expert_data["actions"].extend(datai["actions"])
    expert_data["rewards"].extend(datai["rewards"])
    expert_data["terminated"].extend(datai["terminated"])

print(file_names)
print(np.array(expert_data["states"]).shape)
print(np.array(expert_data["next_states"]).shape)
print(np.array(expert_data["actions"]).shape)
print(np.array(expert_data["rewards"]).shape)
print(np.array(expert_data["terminated"]).shape)
quit()

# # instantiate the agent's models, Behavior clone (BC) requires 1 model to learn the expert behavior
models_BC = {}
models_BC["policy"] = BCmodel(
    env.observation_space, env.action_space, device, clip_actions=True
)
cfg_BC = BC_DEFAULT_CONFIG.copy()
cfg_BC["gradient_steps"] = 5
cfg_BC["batch_size"] = 256
# cfg_BC["demo_file"] = "./Demos/Cab-expert-bc.csv"
cfg_BC["exploration"]["noise"] = GaussianNoise(0, 0.0001, device=device)
cfg_BC["smooth_regularization_clip"] = 0.0001

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
