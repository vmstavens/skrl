"""Train diffusion policy on pipe_insert data and generate rollouts."""

# import isaacgym
# import isaacgymenvs
import collections
import glob
import json
import os
import pickle
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# import the skrl components to build the RL system
from datasets.pushert import normalize_data, unnormalize_data
from skrl import logger
from skrl.envs.torch import load_isaacgym_env_preview4, wrap_env
from skrl.envs.wrappers.torch import Wrapper
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# from algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
# Import the skrl components to build the RL system
# Import the skrl components to build the RL system
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed

# from skrl.trainers.torch import SequentialTrainer
# from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from testing import wrappers as wrap
from testing.envs.pipe_insert import PipeInsert
from testing.envs.pipe_insert_2 import PipeInsert2
from testing.envs.pipe_insert_3 import PipeInsert3
from testing.envs.xpose import XPose
from testing.experiments.trainer.sequential_trainer_plus import SequentialTrainerPlus
from testing.experiments.trainer.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
)

# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.shen.BC import BC, BC_DEFAULT_CONFIG
from testing.shen.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    DiffusionPolicy,
)
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

# from algorithms.IBRL_active import IBRL
# from testing.shen.ibrl import IBRL, IBRL_DEFAULT_CONFIG
# from testing.shen.IBRL import IBRL, IBRL_DEFAULT_CONFIG
from testing.shen.drlr import DRLR_DEFAULT_CONFIG
from testing.shen.ibrl_rl import IBRL_RL_DEFAULT_CONFIG
from testing.shen.ibrl_sac_o_o2_v2 import IBRL_SAC_DEFAULT_CONFIG
from testing.shen.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.train.demon import TransitionDataset


class PipeInsertDataset(Dataset):
    """Create horizon-aligned state/action sequences from JSON demos."""

    def __init__(
        self,
        data_dir: Path,
        pred_horizon: int,
        obs_horizon: int,
        pct: float = 1.0,
        subsample_every: int = 1,
        states_label: str = "states",
        actions_label: str = "actions",
    ):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.subsample_every = max(1, int(subsample_every))

        self.state_sequences: List[np.ndarray] = []
        self.action_sequences: List[np.ndarray] = []

        all_states: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []

        data_files = data_dir.glob("*.json")
        data_files = list(data_files)
        data_files = data_files[: int(len(data_files) * pct)]

        def _subsample(
            states: np.ndarray, actions: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if self.subsample_every <= 1:
                return states, actions
            step = self.subsample_every
            # Subsample states
            states_sub = states[::step]
            # Accumulate actions between sampled states
            accum_actions = []
            for i in range(0, len(actions), step):
                accum_actions.append(actions[i : i + step].sum(axis=0))
            actions_sub = np.stack(accum_actions, axis=0)
            min_len = min(len(states_sub), len(actions_sub))
            return states_sub[:min_len], actions_sub[:min_len]

        if len(data_files) == 0:
            raise ValueError((f"No files found in {data_dir}"))

        for path in sorted(data_files):
            # for path in sorted(data_dir.glob("*.json")):
            with open(path, "r") as f:
                data: dict = json.load(f)

            if states_label not in data.keys():
                raise ValueError(
                    f"Key '{states_label}' not in {path} (keys: {data.keys()}) "
                )

            states = np.asarray(data[states_label], dtype=np.float32)
            # states = np.asarray(data["states"], dtype=np.float32)
            actions = np.asarray(data[actions_label], dtype=np.float32)

            if len(states) == 0:
                raise ValueError(f"Obs states is empty with label {states_label=}")
            if len(actions) == 0:
                raise ValueError(f"Obs actions is empty with label {actions_label=}")

            # actions = np.asarray(data["actions"], dtype=np.float32)
            states, actions = _subsample(states, actions)
            if len(states) == 0 or len(actions) == 0:
                continue
            if len(states) < pred_horizon or len(actions) < pred_horizon:
                continue
            for start in range(0, len(states) - pred_horizon + 1):
                s_seq = states[start : start + pred_horizon]
                a_seq = actions[start : start + pred_horizon]
                self.state_sequences.append(s_seq)
                self.action_sequences.append(a_seq)
            all_states.append(states)
            all_actions.append(actions)

        with open("testing/experiments/pipe_insert/tmp/test.json", "w") as f:
            json.dump(
                {"states": states.tolist(), "actions": actions.tolist()}, f, indent=4
            )

        # Stats for normalization
        stacked_states = np.concatenate(all_states, axis=0)
        stacked_actions = np.concatenate(all_actions, axis=0)
        self.stats = {
            "obs": {
                "min": stacked_states.min(axis=0),
                "max": stacked_states.max(axis=0),
            },
            "action": {
                "min": stacked_actions.min(axis=0),
                "max": stacked_actions.max(axis=0),
            },
        }

        if not self.state_sequences or not self.action_sequences:
            raise ValueError(f"No usable sequences were found in {data_dir}")

        self.obs_dim = self.state_sequences[0].shape[-1]
        self.action_dim = self.action_sequences[0].shape[-1]

    def __len__(self) -> int:
        return len(self.state_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self.state_sequences[idx]
        actions = self.action_sequences[idx]
        norm_states = normalize_data(states, self.stats["obs"])
        norm_actions = normalize_data(actions, self.stats["action"])
        obs_seq = norm_states[: self.obs_horizon]
        act_seq = norm_actions
        return torch.from_numpy(obs_seq), torch.from_numpy(act_seq)


_SEED = 10


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
            nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU()
        )

        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(32, 1)

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


class PPODeterministicActor(DeterministicMixin, Model):
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
            nn.Linear(self.num_observations, 64),
            nn.Dropout(dropout_rate),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.Dropout(dropout_rate),
            nn.ELU(),
            nn.Linear(32, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}



# define models (stochastic and deterministic models) using mixins
# class SAC_StochasticActor(GaussianMixin, Model):
#     def __init__(
#         self,
#         observation_space,
#         action_space,
#         device,
#         clip_actions=False,
#         clip_log_std=True,
#         min_log_std=-20,
#         max_log_std=2,
#         reduction="sum",
#     ):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(
#             self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
#         )

#         self.linear_layer_1 = nn.Linear(self.num_observations, 256)
#         self.linear_layer_2 = nn.Linear(256, 256)
#         self.action_layer = nn.Linear(256, self.num_actions)

#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
#         action_low = torch.as_tensor(
#             self.action_space.low, device=self.device, dtype=torch.float32
#         )
#         action_high = torch.as_tensor(
#             self.action_space.high, device=self.device, dtype=torch.float32
#         )
#         self.register_buffer("action_scale", (action_high - action_low) / 2.0)
#         self.register_buffer("action_bias", (action_high + action_low) / 2.0)
#         self._log_prob_epsilon = 1e-6

#     def compute(self, inputs, role):
#         x = F.relu(self.linear_layer_1(inputs["states"]))
#         x = F.relu(self.linear_layer_2(x))
#         return self.action_layer(x), self.log_std_parameter, {}

#     def act(self, inputs, role=""):
#         mean_actions, log_std, outputs = self.compute(inputs, role)

#         if self._g_clip_log_std:
#             log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

#         self._g_log_std = log_std
#         self._g_num_samples = mean_actions.shape[0]

#         distribution = torch.distributions.Normal(mean_actions, log_std.exp())
#         self._g_distribution = distribution

#         taken_actions = inputs.get("taken_actions")
#         if taken_actions is None:
#             raw_actions = distribution.rsample()
#             squashed_actions = torch.tanh(raw_actions)
#             actions = squashed_actions * self.action_scale + self.action_bias
#         else:
#             actions = taken_actions
#             squashed_actions = (actions - self.action_bias) / self.action_scale.clamp_min(
#                 self._log_prob_epsilon
#             )
#             squashed_actions = torch.clamp(
#                 squashed_actions,
#                 -1.0 + self._log_prob_epsilon,
#                 1.0 - self._log_prob_epsilon,
#             )
#             raw_actions = torch.atanh(squashed_actions)

#         log_prob = distribution.log_prob(raw_actions)
#         log_prob -= torch.log(
#             self.action_scale.clamp_min(self._log_prob_epsilon)
#             * (1 - squashed_actions.pow(2)).clamp_min(self._log_prob_epsilon)
#         )
#         log_prob = log_prob.sum(dim=-1, keepdim=True)

#         outputs["mean_actions"] = (
#             torch.tanh(mean_actions) * self.action_scale + self.action_bias
#         )
#         outputs["raw_mean_actions"] = mean_actions
#         return actions, log_prob, outputs


class SAC_Critic(DeterministicMixin, Model):
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


class SAC_DeterministicActor(DeterministicMixin, Model):
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


class SAC_Critic1(DeterministicMixin, Model):
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


def setup_environment(
    batch_size: int = 100,
    episode_length: int = 1000,
    auto_reset: bool = True,
    action_repeat: int = 1,
    version: int = 2,
    # warp: bool = False,
    impl: str = "warp",
    skrl: bool = False,
):
    """Set up the MJX XPose environment with proper wrapping."""

    # Create base environment
    # env = cartpole.Balance(swing_up=False, sparse=False)
    # env = pendulum.SwingUp()

    versions = [1, 2, 3]

    if version not in versions:
        raise ValueError(f"Invalid version {version} expected {versions}")

    _env_versions = [PipeInsert, PipeInsert2, PipeInsert3]

    if version == 2:
        env = _env_versions[version - 1](impl=impl)
    elif version == 3:
        from testing.envs.pipe_insert_3 import default_config

        dc = default_config()
        dc.episode_length = episode_length
        env = _env_versions[version - 1](impl=impl, config=dc)
    else:
        env = _env_versions[version - 1]()

    # if version == 1:
    #     env = PipeInsert()
    # if version == 2:
    #     env = PipeInsert2()

    # if warp:
    #     from testing.wrappers_warp import WarpWrapper

    #     env = wrap.create(
    #         env,
    #         batch_size=batch_size,
    #         episode_length=episode_length,
    #         auto_reset=auto_reset,
    #         action_repeat=action_repeat,
    #     )
    #     env = wrap_env(env, wrapper="playground")
    #     env = WarpWrapper(env)

    # else:
    # env = wrap.create(
    #     env,
    #     batch_size=batch_size,
    #     episode_length=episode_length,
    #     auto_reset=auto_reset,
    #     action_repeat=action_repeat,
    # )

    if skrl:
        env = wrap_env(env, wrapper="playground_skrl")
    else:
        env = wrap.create(
            env,
            batch_size=batch_size,
            episode_length=episode_length,
            auto_reset=auto_reset,
            action_repeat=action_repeat,
        )
        env = wrap_env(env, wrapper="playground")

    return env


def get_expert_memory_2(
    expert_data_dir: str,
    states_label: str = "states",
    actions_label: str = "actions",
    rewards_label: str = "rewards",
    next_states_label: str = "next_states",
    dones_label: str = "dones",
) -> RandomMemory:
    data_path = Path(expert_data_dir)
    data_files = list(data_path.glob("*.json"))

    dataset = TransitionDataset(
        json_paths=data_files,
        state_key=states_label,
        action_key=actions_label,
        reward_key=rewards_label,
        next_state_key=next_states_label,
        done_key=dones_label,
    )

    transitions = dataset.transitions
    states = np.stack([t["state"] for t in transitions], axis=0)
    actions = np.stack([t["action"] for t in transitions], axis=0)

    if dataset.has_next_states:
        next_states = np.stack(
            [t.get("next_state", t["state"]) for t in transitions], axis=0
        )
    else:
        next_states = states.copy()

    if dataset.has_rewards:
        rewards = np.asarray(
            [t.get("reward", 0.0) for t in transitions], dtype=np.float32
        )
    else:
        rewards = np.zeros(len(transitions), dtype=np.float32)

    if dataset.has_dones:
        dones = np.asarray([t.get("done", False) for t in transitions], dtype=bool)
    else:
        dones = np.zeros(len(transitions), dtype=bool)

    states = torch.from_numpy(states).float()
    actions = torch.from_numpy(actions).float()
    next_states = torch.from_numpy(next_states).float()
    rewards = torch.from_numpy(rewards).float()
    terminated = torch.from_numpy(dones).bool()

    memory_size = len(states)

    a_dim = actions.shape[1]
    o_dim = states.shape[1]

    expert_memory = RandomMemory(memory_size=memory_size)
    expert_memory.create_tensor(name="states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="actions", size=a_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="next_states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
    expert_memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

    expert_memory.add_samples(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards.unsqueeze(-1),
        terminated=terminated.unsqueeze(-1),
    )
    return expert_memory


def _get_expert_memory2(expert_data_dir: str) -> RandomMemory:
    data_path = Path(expert_data_dir)  # data/norm_smooth_states_and_actions
    data_files = list(data_path.glob("*.json"))

    dataset = TransitionDataset(json_paths=data_files)

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
    terminated = torch.tensor(dones, dtype=torch.bool)

    memory_size = len(states)

    a_dim = actions.shape[1]  # Get action dimension from data
    o_dim = states.shape[1]  # Get observation dimension from data

    # Create expert memory
    expert_memory = RandomMemory(memory_size=memory_size)
    expert_memory.create_tensor(name="states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="actions", size=a_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="next_states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
    expert_memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

    # Add samples to memory
    expert_memory.add_samples(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards.unsqueeze(-1),  # Add dimension for reward size
        terminated=terminated.unsqueeze(-1),  # Add dimension for terminated size
    )
    return expert_memory


def get_expert_memory(expert_data_dir: str = "data/norm_data") -> RandomMemory:
    # raise NotImplementedError("Yo not implemented yet")
    # data_path = Path("data/raw")
    # data_path = Path("data/norm_smooth_states_and_actions")
    # data_path = Path("data/norm_data")
    data_path = Path(
        expert_data_dir
        # "./data/norm_smooth_data_test_2/"
    )  # data/norm_smooth_states_and_actions
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
    return expert_memory


def get_memory(
    env,
    tensor_names: list[str] = [
        "states",
        "actions",
        "rewards",
        "next_states",
        "terminated",
    ],
    capacity: int = 350_000,
) -> RandomMemory:
    memory = RandomMemory(
        memory_size=capacity, num_envs=env.num_envs, device=env.device, replacement=True
    )
    memory.create_tensor(
        name=tensor_names[0], size=env.observation_space, dtype=torch.float32
    )
    memory.create_tensor(
        name=tensor_names[1], size=env.action_space, dtype=torch.float32
    )
    memory.create_tensor(name=tensor_names[2], size=1, dtype=torch.float32)
    memory.create_tensor(
        name=tensor_names[3], size=env.observation_space, dtype=torch.float32
    )
    memory.create_tensor(name=tensor_names[4], size=1, dtype=torch.bool)
    return memory


def get_ppo_memory(
    env,
    tensor_names: list[str] = [
        "states",
        "actions",
        "rewards",
        "truncated",
        "log_prob",
        "values",
        "returns",
        "advantages",
        "terminated",
    ],
    capacity: int = 35_000,
) -> RandomMemory:
    memory = RandomMemory(
        memory_size=capacity, num_envs=env.num_envs, device=env.device, replacement=True
    )
    # memory.create_tensor(
    #     name=tensor_name[0], size=env.observation_space, dtype=torch.float32
    # )
    # memory.create_tensor(
    #     name=tensor_name[1], size=env.action_space, dtype=torch.float32
    # )
    # memory.create_tensor(name=tensor_name[2], size=1, dtype=torch.float32)
    # memory.create_tensor(
    #     name=tensor_name[3], size=env.observation_space, dtype=torch.float32
    # )
    # memory.create_tensor(name=tensor_name[4], size=1, dtype=torch.bool)
    # memory.create_tensor(name=tensor_name[5], size=1, dtype=torch.float)
    # memory.create_tensor(name=tensor_name[6], size=1, dtype=torch.bool)

    memory.create_tensor(
        name=tensor_names[0], size=env.observation_space, dtype=torch.float32
    )
    memory.create_tensor(
        name=tensor_names[1], size=env.action_space, dtype=torch.float32
    )
    memory.create_tensor(name=tensor_names[2], size=1, dtype=torch.float32)
    memory.create_tensor(name=tensor_names[3], size=1, dtype=torch.bool)
    memory.create_tensor(name=tensor_names[4], size=1, dtype=torch.float32)
    memory.create_tensor(name=tensor_names[5], size=1, dtype=torch.float32)
    memory.create_tensor(name=tensor_names[6], size=1, dtype=torch.float32)
    memory.create_tensor(name=tensor_names[7], size=1, dtype=torch.float32)
    memory.create_tensor(name=tensor_names[8], size=1, dtype=torch.bool)

    return memory


def generate_expert_memory() -> None:
    raise NotImplementedError("yo not implemented yet")
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

    file_name = "memories/expert_memory.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(expert_memory, f)

    print(f"Saved memory to {file_name}")


def get_td3_models(env) -> dict:
    device = env.device
    models_td3 = {}
    models_td3["policy"] = DeterministicActor(
        env.observation_space, env.action_space, device, clip_actions=True
    )
    models_td3["target_policy"] = DeterministicActor(
        env.observation_space, env.action_space, device, clip_actions=True
    )
    models_td3["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models_td3["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models_td3["target_critic_1"] = Critic(
        env.observation_space, env.action_space, device
    )
    models_td3["target_critic_2"] = Critic(
        env.observation_space, env.action_space, device
    )

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_td3.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    return models_td3


def get_ppo_models(env) -> dict:
    device = env.device
    models_ppo = {}
    models_ppo["policy"] = Shared(env.observation_space, env.action_space, device)
    models_ppo["value"] = Shared(
        env.observation_space, env.action_space, device
    )  # same instance: shared model
    return models_ppo


def get_dp_models(env: Wrapper, config: dict) -> tuple[dict, EMAModel]:
    dp_models = {}
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    device = env.device
    dp_models["model"] = ConditionalUnet1D(a_dim, config).to(device)
    # dp_models["model"] = ConditionalUnet1D(input_dim, dp_config).to(device)
    ema = EMAModel(dp_models["model"].parameters(), power=config["ema_power"])
    dp_models["ema_model"] = ConditionalUnet1D(a_dim, config).to(device)
    return dp_models, ema


def get_sac_models(env: Wrapper) -> dict:
    models_SAC = {}
    models_SAC["policy"] = SAC_StochasticActor(
        env.observation_space, env.action_space, env.device, clip_actions=True, clip_log_std=True
    )
    models_SAC["critic_1"] = Critic(env.observation_space, env.action_space, env.device)
    models_SAC["critic_2"] = Critic(env.observation_space, env.action_space, env.device)
    models_SAC["target_critic_1"] = Critic(
        env.observation_space, env.action_space, env.device
    )
    models_SAC["target_critic_2"] = Critic(
        env.observation_space, env.action_space, env.device
    )

    # initialize models' parameters (weights and biases)
    for model in models_SAC.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    return models_SAC

def get_dp_config(exp_name: str, env: Wrapper, wandb: bool = False) -> dict:
    dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG.copy()
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)
    dp_config["experiment"]["directory"] = model_path.as_posix()
    dp_config["experiment"]["experiment_name"] = exp_name
    dp_config["experiment"]["wandb"] = wandb
    # trainer_config["write_interval"] = 1
    # trainer_config["checkpoint_interval"] = 10
    # trainer_config["batch_size"] = 256 // 4
    # trainer_config["epochs"] = 100
    # trainer_config["shuffle"] = False
    dp_config["eval_frequency"] = 10
    dp_config["batch_size"] = 32

    dp_config["obs_dim"] = env.observation_space.shape[0]
    dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]

    dp_config["pred_horizon"] = 16
    dp_config["action_horizon"] = 1

    return dp_config


def get_ppo_config(exp_name: str, env, wandb: bool = False) -> dict:
    device = env.device
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 128  # memory_size  ## 16 horizon_length
    cfg_ppo["learning_epochs"] = 2  # mini_epochs
    cfg_ppo["mini_batches"] = (
        64  # horizaen_length * numberof_actor / minibathch_size  ## 8
    )
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
    cfg_ppo["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": device,
    }
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints each 20 and 200 timesteps respectively
    cfg_ppo["experiment"]["write_interval"] = 50
    cfg_ppo["experiment"]["checkpoint_interval"] = 200
    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    # cfg_IBRL["experiment"]["write_interval"] = 50
    # cfg_IBRL["experiment"]["checkpoint_interval"] = 100
    cfg_ppo["experiment"]["directory"] = model_path.as_posix()
    cfg_ppo["experiment"]["experiment_name"] = exp_name
    cfg_ppo["experiment"]["wandb"] = wandb
    return cfg_ppo


def get_ibrl_config(exp_name: str, env) -> dict:
    device = env.device
    cfg_IBRL = IBRL_RL_DEFAULT_CONFIG.copy()
    cfg_IBRL["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
    # cfg_IBRL["exploration"]["noise"] = None
    cfg_IBRL["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
    cfg_IBRL["smooth_regularization_clip"] = 0.5
    cfg_IBRL["gradient_steps"] = 1
    cfg_IBRL["RED-Q_enable"] = False
    # cfg_IBRL["RED-Q_enable"] = True
    cfg_IBRL["offline"] = False
    # cfg_IBRL["offline"] = True
    cfg_IBRL["batch_size"] = 128
    cfg_IBRL["random_timesteps"] = 0
    cfg_IBRL["learning_starts"] = 0
    cfg_IBRL["learning_rate"] = 3e-4
    cfg_IBRL["num_envs"] = env.num_envs
    # cfg_IBRL["demo_file"] = "/home/chen/Downloads/new/memories/Cab-expert-bc.csv"
    cfg_IBRL["demo_file"] = "./Demos/cab_imperfect.csv"
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    cfg_IBRL["experiment"]["write_interval"] = 500
    cfg_IBRL["experiment"]["checkpoint_interval"] = 1
    # cfg_IBRL["experiment"]["checkpoint_interval"] = 1000

    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    # cfg_IBRL["experiment"]["write_interval"] = 50
    # cfg_IBRL["experiment"]["checkpoint_interval"] = 100
    cfg_IBRL["experiment"]["directory"] = model_path.as_posix()
    cfg_IBRL["experiment"]["experiment_name"] = exp_name
    cfg_IBRL["experiment"]["wandb"] = True
    return cfg_IBRL

# def _get_ibrl_sac_dp_config():
#     test = {
#   "actor": {
#     "value": "both"
#   },
#   "polyak": {
#     "value": 0.005
#   },

#   "headless": {
#     "value": true
#   },
#   "num_envs": {
#     "value": 100
#   },
#   "timesteps": {
#     "value": 200000
#   },
#   "batch_size": {
#     "value": 128
#   },
#   "exploration": {
#     "value": {
#       "noise": "<skrl.resources.noises.torch.gaussian.GaussianNoise object at 0x734fd4fae1e0>",
#       "timesteps": null,
#       "final_scale": 0.001,
#       "initial_scale": 1
#     }
#   },
#   "il_ctrl_scale": {
#     "value": 1
#   },
#   "learn_entropy": {
#     "value": true # should be true
#   },
#   "learning_rate": {
#     "value": 0.0003
#   },
#   "rl_ctrl_scale": {
#     "value": 1
#   },
#   "decision_block": {
#     "value": false
#   },
#   "grad_norm_clip": {
#     "value": 0
#   },
#   "gradient_steps": {
#     "value": 1
#   },
#   "rewards_shaper": {
#     "value": null
#   },
#   "target_entropy": {
#     "value": null
#   },
#   "discount_factor": {
#     "value": 0.99
#   },
#   "learning_starts": {
#     "value": 0
#   },
#   "mixed_precision": {
#     "value": false
#   },
#   "target_critic_1": {
#     "value": {
#       "linear_layer_1": "Linear(in_features=9, out_features=256, bias=True)",
#       "linear_layer_2": "Linear(in_features=256, out_features=256, bias=True)",
#       "linear_layer_3": "Linear(in_features=256, out_features=1, bias=True)"
#     }
#   },
#   "target_critic_2": {
#     "value": {
#       "linear_layer_1": "Linear(in_features=9, out_features=256, bias=True)",
#       "linear_layer_2": "Linear(in_features=256, out_features=256, bias=True)",
#       "linear_layer_3": "Linear(in_features=256, out_features=1, bias=True)"
#     }
#   },
#   "environment_info": {
#     "value": "episode"
#   },
#   "log_rollout_exit": {
#     "value": false
#   },
#   "log_rollout_path": {
#     "value": "testing/experiments/pipe_insert/tmp/success_ibrl_mjx.json"
#   },
#   "random_timesteps": {
#     "value": 0
#   },
#   "soft_update_beta": {
#     "value": 0.2
#   },
#   "warmup_timesteps": {
#     "value": 10000
#   },
#   "log_rollout_steps": {
#     "value": 10000
#   },
#   "rollout_video_dir": {
#     "value": "/home/vims/git/skrl/testing/experiments/pipe_insert/V2/.runs/train_ibrl_both_actor_both__warmup_timesteps_10000__soft_update_beta_0.2_20260313_09_28_05/media"
#   },
#   "critic_subset_size": {
#     "value": 2
#   },
#   "policy_subset_size": {
#     "value": 2
#   },
#   "state_preprocessor": {
#     "value": null
#   },
#   "actor_learning_rate": {
#     "value": 0.0001 # obs
#   },
#   "disable_progressbar": {
#     "value": false
#   },
#   "critic_learning_rate": {
#     "value": 0.0001 # obs
#   },
#   "rollout_video_prefix": {
#     "value": "train_rollout"
#   },
#   "entropy_learning_rate": {
#     "value": 0.0003
#   },
#   "initial_entropy_value": {
#     "value": 0.01
#   },
#   "stochastic_evaluation": {
#     "value": false
#   },
#   "learning_rate_scheduler": {
#     "value": null
#   },
#   "rollout_video_env_index": {
#     "value": 0
#   },
#   "rollout_video_num_steps": {
#     "value": 10000
#   },
#   "close_environment_at_exit": {
#     "value": true
#   },
#   "smooth_regularization_clip": {
#     "value": 0.5
#   },
#   "smooth_regularization_noise": {
#     "value": "<skrl.resources.noises.torch.gaussian.GaussianNoise object at 0x734fd3b2bfe0>"
#   },
#   "rollout_video_every_episodes": {
#     "value": 0
#   }
# }

def get_ibrl_sac_dp_config(exp_name, env, wandb):
    cfg = IBRL_SAC_DEFAULT_CONFIG.copy()

    cfg["exploration"] = {}
    cfg["exploration"]["noise"] = GaussianNoise(0, 0.1, device=env.device)
    # cfg_IBRL["exploration"]["noise"] = None

    cfg["actor_learning_rate"] = 1e-4
    cfg["critic_learning_rate"] = 1e-4

    cfg["discount_factor"] = 0.99
    cfg["batch_size"] = 128
    cfg["random_timesteps"] = 0  # Add some random exploration at the start
    cfg["learning_starts"] = 0   # Start learning after some experience
    cfg["learn_entropy"] = True
    # cfg["learn_entropy"] = True
    cfg["grad_norm_clip"] = 1.0     # Add gradient clipping for stability
    cfg["learning_rate"] = 3e-4     # Standard SAC learning rate
    cfg["initial_entropy_value"] = 0.01     # Entropy learning rate
    cfg["entropy_learning_rate"] = 3e-4     # Entropy learning rate
    cfg["num_envs"] = env.num_envs
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 1000

    # Experiment configuration
    model_path = Path(__file__).parent / f".runs/{exp_name}/models"
    model_path.mkdir(parents=True, exist_ok=True)

    # project name
    cfg["experiment"]["wandb_kwargs"].setdefault("project", "pipe-insert-v2")

    print(f"creating {model_path}")

    cfg["experiment"]["directory"] = model_path.as_posix()
    cfg["experiment"]["experiment_name"] = exp_name
    cfg["experiment"]["wandb"] = wandb
    cfg["experiment"].setdefault("wandb_kwargs", {})
    cfg["experiment"]["wandb_kwargs"].setdefault("resume", "never")

    return cfg

def get_bc_models(env) -> dict:
    device = env.device
    models_BC = {}
    models_BC["policy"] = BCmodel(
        env.observation_space, env.action_space, device, clip_actions=True
    )
    for model in models_BC.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    return models_BC


def get_drlr_config(exp_name: str, env, wandb: bool = True) -> dict:
    device = env.device
    
    # DRLR_DEFAULT_CONFIG = {
    #     "gradient_steps": 1,            # gradient steps
    #     "batch_size": 64,               # training batch size

    #     "discount_factor": 0.99,        # discount factor (gamma)
    #     "polyak": 0.005,                # soft update hyperparameter (tau)

    #     "actor_learning_rate": 3e-4,    # actor learning rate
    #     "critic_learning_rate": 3e-4,   # critic learning rate
    #     "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    #     "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    #     "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    #     "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    #     "random_timesteps": 0,          # random exploration steps
    #     "learning_starts": 0,           # learning starts after this many steps

    #     "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    #     "learn_entropy": True,          # learn entropy
    #     "entropy_learning_rate": 3e-4,  # entropy learning rate
    #     "initial_entropy_value": 0.01,     # initial entropy value
    #     "target_entropy": None,         # target entropy

    #     # RED-Q specific parameters
    #     "RED-Q_enable": True,  # use REDQ?
    #     "ensemble_size": 5,  # number of critics in ensemble (N)
    #     "critic_subset_size": 2,  # number of critics to sample for target computation (M)
    #     "policy_subset_size": 2,  # number of critics to sample for policy updates

    #     "offline": False,
    #     "BC": False,
    #     "demo_file": "",
    #     "num_envs": 1,

    #     "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    #     "mixed_precision": False,       # enable automatic mixed precision for higher performance

    #     "experiment": {
    #         "directory": "",            # experiment's parent directory
    #         "experiment_name": "",      # experiment name
    #         "write_interval": "auto",   # TensorBoard writing interval (timesteps)

    #         "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
    #         "store_separately": False,          # whether to store checkpoints separately

    #         "wandb": False,             # whether to use Weights & Biases
    #         "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    #     }
    # }

    cfg_IBRL = DRLR_DEFAULT_CONFIG.copy()

    cfg_IBRL["discount_factor"] = 0.99
    cfg_IBRL["batch_size"] = 128
    cfg_IBRL["random_timesteps"] = 0  # Add some random exploration at the start
    cfg_IBRL["learning_starts"] = 0   # Start learning after some experience
    cfg_IBRL["learn_entropy"] = True
    cfg_IBRL["grad_norm_clip"] = 1.0     # Add gradient clipping for stability
    cfg_IBRL["learning_rate"] = 3e-4     # Standard SAC learning rate
    cfg_IBRL["initial_entropy_value"] = 0.1     # Entropy learning rate
    cfg_IBRL["RED-Q_enable"] = False     #enable RED-Q
    cfg_IBRL["offline"] = False       # not important here
    cfg_IBRL["num_envs"] = env.num_envs
    cfg_IBRL["demo_file"] = "./Demos/cab_imperfect.csv"
    cfg_IBRL["experiment"]["write_interval"] = 100
    cfg_IBRL["experiment"]["checkpoint_interval"] = 1000




    # cfg_IBRL["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
    # # cfg_IBRL["exploration"]["noise"] = None
    # cfg_IBRL["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
    # cfg_IBRL["smooth_regularization_clip"] = 0.5
    cfg_IBRL["gradient_steps"] = 1
    cfg_IBRL["RED-Q_enable"] = False
    # cfg_IBRL["RED-Q_enable"] = True
    cfg_IBRL["offline"] = False
    # cfg_IBRL["offline"] = True

    cfg_IBRL["decision_block"] = False

    # cfg_IBRL["batch_size"] = 128
    # cfg_IBRL["random_timesteps"] = 0
    # cfg_IBRL["learning_starts"] = 0
    # cfg_IBRL["learning_rate"] = 3e-4
    # cfg_IBRL["num_envs"] = env.num_envs
    # cfg_IBRL["demo_file"] = "/home/chen/Downloads/new/memories/Cab-expert-bc.csv"
    # cfg_IBRL["demo_file"] = "./Demos/cab_imperfect.csv"
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    # cfg_IBRL["experiment"]["write_interval"] = 100
    # cfg_IBRL["experiment"]["write_interval"] = 1
    # cfg_IBRL["experiment"]["write_interval"] = 500
    # cfg_IBRL["experiment"]["checkpoint_interval"] = 1000

    # Experiment configuration
    model_path = Path(__file__).parent / f".runs/{exp_name}/models"
    model_path.mkdir(parents=True, exist_ok=True)

    # project name
    cfg_IBRL["experiment"]["wandb_kwargs"].setdefault("project", "pipe-insert-v2")

    print(f"creating {model_path}")

    cfg_IBRL["experiment"]["directory"] = model_path.as_posix()
    cfg_IBRL["experiment"]["experiment_name"] = exp_name
    cfg_IBRL["experiment"]["wandb"] = wandb
    cfg_IBRL["experiment"].setdefault("wandb_kwargs", {})
    cfg_IBRL["experiment"]["wandb_kwargs"].setdefault("resume", "never")
    return cfg_IBRL


def get_td3_config(exp_name: str, env, wandb: bool = False) -> dict:
    device = env.device
    cfg_IBRL = DRLR_DEFAULT_CONFIG.copy()
    cfg_IBRL["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
    # cfg_IBRL["exploration"]["noise"] = None
    cfg_IBRL["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
    cfg_IBRL["smooth_regularization_clip"] = 0.5
    cfg_IBRL["gradient_steps"] = 1
    cfg_IBRL["RED-Q_enable"] = False
    # cfg_IBRL["RED-Q_enable"] = True
    cfg_IBRL["offline"] = False
    # cfg_IBRL["offline"] = True
    cfg_IBRL["batch_size"] = 128
    cfg_IBRL["random_timesteps"] = 0
    cfg_IBRL["learning_starts"] = 0
    cfg_IBRL["learning_rate"] = 3e-4
    cfg_IBRL["num_envs"] = env.num_envs
    # cfg_IBRL["demo_file"] = "/home/chen/Downloads/new/memories/Cab-expert-bc.csv"
    cfg_IBRL["demo_file"] = "./Demos/cab_imperfect.csv"
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    cfg_IBRL["experiment"]["write_interval"] = 100
    # cfg_IBRL["experiment"]["write_interval"] = 500
    cfg_IBRL["experiment"]["checkpoint_interval"] = 1000

    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    cfg_IBRL["experiment"]["directory"] = model_path.as_posix()
    cfg_IBRL["experiment"]["experiment_name"] = exp_name
    cfg_IBRL["experiment"]["wandb"] = wandb
    return cfg_IBRL


def get_bc_config(exp_name: str, env, wandb: bool = True) -> dict:
    device = env.device
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
    cfg_BC["experiment"]["experiment_name"] = exp_name
    cfg_BC["experiment"]["wandb"] = wandb
    return cfg_BC


def get_trainer(
    env, agent, timesteps: int = 350_000, trainer_cfg: Optional[dict] = None
):
    cfg = {"timesteps": timesteps, "headless": True}
    if trainer_cfg:
        cfg.update(trainer_cfg)
    trainer = SequentialTrainerPlus(cfg=cfg, env=env, agents=agent)
    return trainer


def exp_set_seed(seed: Optional[int] = None):
    if seed is not None:
        set_seed(seed)
    else:
        set_seed(_SEED)


def rollout_markov(
    file_name: str,
    env,
    agent,
    num_timesteps: int = 1000,
    end_on_terminate: bool = False,
):
    agent.set_mode("eval")
    state, _ = env.reset()
    frames = []

    data = {"states": [], "actions": []}

    print("Performing rollout")
    for i in tqdm(range(num_timesteps)):
        actions, _, _ = agent.act(states=state, timestep=i, timesteps=num_timesteps)
        data["actions"].append(actions.tolist())
        data["states"].append(state.tolist())
        next_states, rewards, terminated, truncated, infos = env.step(
            actions=actions.detach()
        )
        frame = env.render()
        frames.append(frame)
        state = next_states
        if terminated[0]:
            if end_on_terminate:
                break
            state, _ = env.reset()

    with open("data/tmp/data.json", "w") as f:
        json.dump(data, f, indent=4)

    # Create video from frames
    if frames:
        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Use pathlib for proper path handling
        output_path = Path(file_name)

        # Create temp file in the same directory with a proper name
        temp_file_name = output_path.parent / f"temp_{output_path.name}"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing {len(frames)} frames to temporary file: {temp_file_name}")

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30

        out = cv2.VideoWriter(str(temp_file_name), fourcc, fps, (width, height))

        # Write all frames to video
        for frame in frames:
            # Ensure frame is in correct format (BGR for OpenCV)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to BGR if needed
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        print(f"Temporary video saved: {temp_file_name}")

        # Check if temp file was created successfully
        if not temp_file_name.exists():
            print(f"Error: Temporary file was not created: {temp_file_name}")
            return

        # Convert to H.264 using FFmpeg
        print("Converting to H.264 format...")
        try:
            # Run FFmpeg conversion
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # -y to overwrite output file
                    "-i",
                    str(temp_file_name),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            print(f"Successfully converted to H.264: {output_path}")

            # Remove temporary file
            temp_file_name.unlink()
            print(f"Removed temporary file: {temp_file_name}")

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e}")
            print(f"FFmpeg stderr: {e.stderr}")

            # Check if temp file exists and has content
            if temp_file_name.exists():
                file_size = temp_file_name.stat().st_size
                print(f"Temporary file exists with size: {file_size} bytes")

                # Try to use the temp file directly
                try:
                    temp_file_name.rename(output_path)
                    print(f"Renamed temporary file to: {output_path}")
                except Exception as rename_error:
                    print(f"Failed to rename temporary file: {rename_error}")
            else:
                print("Temporary file does not exist")

        except FileNotFoundError:
            print("FFmpeg not found. Please install ffmpeg.")
            # Try to rename the temp file
            try:
                temp_file_name.rename(output_path)
                print(f"Renamed temporary file to: {output_path}")
            except Exception as rename_error:
                print(f"Failed to rename temporary file: {rename_error}")

    else:
        print("No frames were generated")


def rollout_history(
    file_name: str,
    env,
    agent,
    obs_horizon: int,
    num_timesteps: int = 1000,
    end_on_terminate: bool = False,
):
    agent.set_mode("eval")
    state, _ = env.reset()
    frames = []

    # Initialize deque for storing observation sequences
    state_sequence = deque(maxlen=obs_horizon)

    # print("-------------------- in rollout -------------------------")
    # print(f"{state.shape=}")
    # print(f"{obs_horizon=}")

    # Initialize the sequence with the first state (repeated to fill the sequence)
    for _ in range(obs_horizon):
        state_sequence.append(state)
        # state_sequence.append(state.detach().cpu())

    data = {
        "states": [],
        "actions": [],
        "state_sequences": [],  # Optional: store the sequences for debugging
    }

    print("Performing rollout with sequence processing")
    for i in tqdm(range(num_timesteps)):
        # Convert the state sequence to a numpy array for the agent
        state_sequence_array = torch.stack(list(state_sequence)).permute(1, 0, 2)
        # state_sequence_array = torch.stack(list(state_sequence)).to(agent.device)

        # states

        # print(f"{state_sequence_array.shape=}")

        # Agent now receives a sequence of states and should return a sequence of actions
        actions, _, _ = agent.act(
            states=state_sequence_array, timestep=i, timesteps=num_timesteps
        )
        # print(f"{actions.shape=}")

        # Store data
        data["actions"].append(actions.tolist())
        data["states"].append(state.tolist())
        data["state_sequences"].append(state_sequence_array.tolist())  # Optional

        # Execute the action (you might need to handle single action vs sequence)
        # If agent returns a sequence, you might want to use the last action or all of them
        if isinstance(actions, np.ndarray) and actions.ndim > 1:
            # If we have a sequence of actions, use the most recent one
            current_action = actions[-1] if len(actions) > 0 else actions[0]
        else:
            current_action = actions

        # print(f"{current_action.shape=}")

        one_action = current_action[:, 0, :]

        next_states, rewards, terminated, truncated, infos = env.step(
            actions=one_action.detach() if hasattr(one_action, "detach") else one_action
        )
        # print(f"{next_states.shape=}")
        # print(f"{rewards.shape=}")
        # print(f"{terminated.shape=}")

        # Render and collect frame
        frame = env.render()
        frames.append(frame)

        # Update the state sequence with the new state
        state_sequence.append(next_states)
        # state_sequence.append(next_states.copy())
        state = next_states

        if terminated[0]:
            if end_on_terminate:
                break
            # Reset environment and state sequence
            state, _ = env.reset()
            state_sequence.clear()
            for _ in range(obs_horizon):
                state_sequence.append(state)
                # state_sequence.append(state.copy())

    # with open("data/tmp/data.json", "w") as f:
    #     json.dump(data, f, indent=4)

    print(f"{len(frames)=} ---- VIDEO ----------------------")

    # Create video from frames (your existing video code remains the same)
    if frames:
        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Use pathlib for proper path handling
        output_path = Path(file_name)

        # Create temp file in the same directory with a proper name
        temp_file_name = output_path.parent / f"temp_{output_path.name}"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing {len(frames)} frames to temporary file: {temp_file_name}")

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30

        out = cv2.VideoWriter(str(temp_file_name), fourcc, fps, (width, height))

        # Write all frames to video
        for frame in frames:
            # Ensure frame is in correct format (BGR for OpenCV)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to BGR if needed
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        print(f"Temporary video saved: {temp_file_name}")

        # Check if temp file was created successfully
        if not temp_file_name.exists():
            print(f"Error: Temporary file was not created: {temp_file_name}")
            return

        # Convert to H.264 using FFmpeg
        print("Converting to H.264 format...")
        try:
            # Run FFmpeg conversion
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # -y to overwrite output file
                    "-i",
                    str(temp_file_name),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            print(f"Successfully converted to H.264: {output_path}")

            # Remove temporary file
            temp_file_name.unlink()
            print(f"Removed temporary file: {temp_file_name}")

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e}")
            print(f"FFmpeg stderr: {e.stderr}")

            # Check if temp file exists and has content
            if temp_file_name.exists():
                file_size = temp_file_name.stat().st_size
                print(f"Temporary file exists with size: {file_size} bytes")

                # Try to use the temp file directly
                try:
                    temp_file_name.rename(output_path)
                    print(f"Renamed temporary file to: {output_path}")
                except Exception as rename_error:
                    print(f"Failed to rename temporary file: {rename_error}")
            else:
                print("Temporary file does not exist")

        except FileNotFoundError:
            print("FFmpeg not found. Please install ffmpeg.")
            # Try to rename the temp file
            try:
                temp_file_name.rename(output_path)
                print(f"Renamed temporary file to: {output_path}")
            except Exception as rename_error:
                print(f"Failed to rename temporary file: {rename_error}")

    else:
        print("No frames were generated")


def rollout_general(
    env,
    agent,
    config: dict,
    stats: dict = None,
    max_steps: int = 200,
    seed: int = None,
    normalize_obs: bool = False,
    normalize_act: bool = False,
    render: bool = True,
) -> Tuple[float, List[np.ndarray], dict]:
    """
    General rollout function that works with both trajectory-based and single-step policies.

    Args:
        env: The environment
        agent: The policy agent (can be diffusion or standard policy)
        config: Configuration dictionary containing:
            - obs_horizon: For trajectory-based policies (default=1 for single-step)
            - action_horizon: For trajectory-based policies (default=1 for single-step)
        stats: Statistics for normalization (optional)
        max_steps: Maximum number of steps
        seed: Random seed for environment
        normalize_obs: Whether to normalize observations
        normalize_act: Whether to unnormalize actions
        render: Whether to render frames

    Returns:
        max_reward: Maximum reward during rollout
        imgs: List of rendered frames (if render=True)
        rollout_data: Dictionary containing states, actions, rewards, etc.
    """
    # Set agent to evaluation mode
    if hasattr(agent, "set_mode"):
        agent.set_mode("eval")

    # Extract config values with defaults
    obs_horizon = config.get("obs_horizon", 1)
    action_horizon = config.get("action_horizon", 1)

    # Set seed if provided
    if seed is not None:
        if hasattr(env, "seed"):
            env.seed(seed)
        elif hasattr(env, "_np_random_seed"):
            env._np_random_seed = seed

    # Initialize environment
    obs, info = env.reset()

    # Initialize data storage
    rollout_data = {
        "states": [obs],
        # "states": [obs.copy()],
        "actions": [],
        "rewards": [],
        "dones": [],
        "infos": [info],
    }

    # Initialize for trajectory-based policies
    if obs_horizon > 1:
        obs_deque = collections.deque([obs.cpu()] * obs_horizon, maxlen=obs_horizon)
    else:
        current_obs = obs

    # Initialize rendering
    imgs = []
    if render:
        try:
            imgs.append(env.render())
        except:
            render = False

    # Main rollout loop
    done = False
    step_idx = 0
    total_reward = 0

    while not done and step_idx < max_steps:
        # Prepare observation based on policy type
        if obs_horizon > 1:
            # Trajectory-based policy (e.g., diffusion)
            obs_seq = np.stack(obs_deque)

            # Normalize if required
            if normalize_obs and stats is not None and "obs" in stats:
                obs_input = normalize_data(obs_seq, stats["obs"])
            else:
                obs_input = obs_seq

            # Convert to tensor if needed
            if hasattr(agent, "device"):
                obs_tensor = (
                    torch.from_numpy(obs_input)
                    .to(agent.device, dtype=torch.float32)
                    .unsqueeze(0)
                )
            else:
                obs_tensor = obs_input

        else:
            # Single-step policy
            if normalize_obs and stats is not None and "obs" in stats:
                obs_input = normalize_data(obs.reshape(1, -1), stats["obs"])
            else:
                obs_input = obs.reshape(1, -1)

            if hasattr(agent, "device"):
                obs_tensor = torch.from_numpy(obs_input).to(
                    agent.device, dtype=torch.float32
                )
            else:
                obs_tensor = obs_input

        # Get action from agent
        with torch.no_grad():
            # Try different agent interfaces
            if hasattr(agent, "predict"):
                # Diffusion policy style
                action_pred = agent.predict(obs_tensor)

                if obs_horizon > 1:
                    # Take only the action_horizon steps
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[0, start:end, :].detach().cpu().numpy()
                else:
                    action = action_pred.detach().cpu().numpy()

            elif hasattr(agent, "act"):
                # skrl style
                result = agent.act(
                    states=obs_tensor, timestep=step_idx, timesteps=max_steps
                )

                if isinstance(result, tuple):
                    # (actions, log_probs, entropies) format
                    actions = result[0]
                else:
                    actions = result

                # Handle different action formats
                if isinstance(actions, torch.Tensor):
                    action = actions.detach().cpu().numpy()
                else:
                    action = actions

                # Handle sequence outputs
                if action.ndim > 2:
                    action = action.squeeze(0)
                if action.ndim > 1 and action.shape[0] > 1:
                    # Take the first action if sequence
                    action = action[0]
            else:
                raise ValueError("Agent must have either 'predict' or 'act' method")

        # Unnormalize action if required
        if normalize_act and stats is not None and "action" in stats:
            action = unnormalize_data(action, stats["action"])

        # Ensure action has correct shape
        if action.ndim > 1 and action.shape[0] > 1:
            # Multiple actions (action horizon)
            actions_to_execute = action
        else:
            # Single action
            actions_to_execute = [action.squeeze()]

        # Execute actions
        for i in range(len(actions_to_execute)):
            if done or step_idx >= max_steps:
                break

            # Step environment
            act = actions_to_execute[i]
            act = torch.from_numpy(act)
            # print(f"{type(act)=}")
            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            # Store data
            rollout_data["actions"].append(act)
            rollout_data["rewards"].append(reward.cpu().numpy())
            rollout_data["dones"].append(done)
            rollout_data["infos"].append(info)
            rollout_data["states"].append(next_obs)

            total_reward += reward

            # Update observation
            if obs_horizon > 1:
                obs_deque.append(next_obs.cpu().numpy())
                current_obs_seq = np.stack(obs_deque)
            else:
                current_obs = next_obs

            # Render
            if render:
                try:
                    imgs.append(env.render())
                except:
                    pass

            step_idx += 1

    # Clean up
    env.close()

    # Calculate metrics
    max_reward = max(rollout_data["rewards"]) if rollout_data["rewards"] else 0.0
    mean_reward = np.mean(rollout_data["rewards"]) if rollout_data["rewards"] else 0.0

    # Add summary to rollout data
    rollout_data["summary"] = {
        "total_steps": step_idx,
        "total_reward": total_reward,
        "max_reward": max_reward,
        "mean_reward": mean_reward,
        "episode_length": len(rollout_data["rewards"]),
    }

    return max_reward, imgs, rollout_data


def create_data(
    input_dir: str, output_dir: str = "data/test", state_filter: bool = False
):
    def low_pass_filter(data, window_size=5):
        """Apply simple moving average filter"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode="same")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files in the data directory
    path = Path(input_dir)
    data_files = glob.glob((path / "*.json").as_posix())

    # Define colors for different files
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_files)))

    for i, file_path in enumerate(data_files):
        with open(file_path) as f:
            data = json.load(f)

        # Original data
        states_original = np.array(data["states"])
        actions_original = np.array(data["actions"])

        s_x_orig = states_original[:, 0]
        s_y_orig = states_original[:, 1]
        s_z_orig = states_original[:, 2]
        a_x_orig = actions_original[:, 0]
        a_y_orig = actions_original[:, 1]
        a_z_orig = actions_original[:, 2]

        # Apply low-pass filter to create filtered data
        window_size = 100

        if state_filter:
            s_x_filt = low_pass_filter(s_x_orig, window_size)
            s_y_filt = low_pass_filter(s_y_orig, window_size)
            s_z_filt = low_pass_filter(s_z_orig, window_size)
        else:
            s_x_filt = s_x_orig
            s_y_filt = s_y_orig
            s_z_filt = s_z_orig

        a_x_filt = low_pass_filter(a_x_orig, window_size)
        a_y_filt = low_pass_filter(a_y_orig, window_size)
        a_z_filt = low_pass_filter(a_z_orig, window_size)

        # Create new actions array with filtered data
        actions_filtered = np.column_stack((a_x_filt, a_y_filt, a_z_filt))
        states_filtered = np.column_stack((s_x_filt, s_y_filt, s_z_filt))

        # Create new data dictionary with filtered actions
        data_filtered = data.copy()
        data_filtered["actions"] = actions_filtered.tolist()
        data_filtered["states"] = states_filtered.tolist()

        # Generate output filename
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        # Save filtered data
        with open(output_path, "w") as f:
            json.dump(data_filtered, f, indent=2)

        print(f"Saved data to {output_dir}")

        # Get filename for legend (without extension)
        label = os.path.splitext(filename)[0]

        # Create comparison figure
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f"Data Comparison: {label}", fontsize=16, fontweight="bold")

        # Plot states comparison
        time_original = np.arange(len(states_original))
        time_filtered = np.arange(len(states_filtered))

        # State X
        axes[0, 0].plot(
            time_original, s_x_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[0, 0].plot(time_filtered, s_x_filt, "r-", label="Filtered", linewidth=2)
        axes[0, 0].set_ylabel("State X")
        axes[0, 0].set_title("State X - Before vs After Filtering")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # State Y
        axes[1, 0].plot(
            time_original, s_y_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[1, 0].plot(time_filtered, s_y_filt, "r-", label="Filtered", linewidth=2)
        axes[1, 0].set_ylabel("State Y")
        axes[1, 0].set_title("State Y - Before vs After Filtering")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # State Z
        axes[2, 0].plot(
            time_original, s_z_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[2, 0].plot(time_filtered, s_z_filt, "r-", label="Filtered", linewidth=2)
        axes[2, 0].set_ylabel("State Z")
        axes[2, 0].set_xlabel("Time Steps")
        axes[2, 0].set_title("State Z - Before vs After Filtering")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Plot actions comparison
        # Action X
        axes[0, 1].plot(
            time_original, a_x_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[0, 1].plot(time_filtered, a_x_filt, "r-", label="Filtered", linewidth=2)
        axes[0, 1].set_ylabel("Action X")
        axes[0, 1].set_title("Action X - Before vs After Filtering")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Action Y
        axes[1, 1].plot(
            time_original, a_y_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[1, 1].plot(time_filtered, a_y_filt, "r-", label="Filtered", linewidth=2)
        axes[1, 1].set_ylabel("Action Y")
        axes[1, 1].set_title("Action Y - Before vs After Filtering")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Action Z
        axes[2, 1].plot(
            time_original, a_z_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[2, 1].plot(time_filtered, a_z_filt, "r-", label="Filtered", linewidth=2)
        axes[2, 1].set_ylabel("Action Z")
        axes[2, 1].set_xlabel("Time Steps")
        axes[2, 1].set_title("Action Z - Before vs After Filtering")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        figure_filename = f"{label}_comparison.png"
        figure_path = os.path.join(output_dir, figure_filename)
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison figure to {figure_path}")

        # Close the figure to free memory
        plt.close(fig)

    print("Processing completed!")
