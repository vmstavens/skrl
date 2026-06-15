"""Train diffusion policy on pipe_insert data and generate rollouts."""

# import isaacgym
# import isaacgymenvs
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# import the skrl components to build the RL system
from datasets.pushert import normalize_data
from skrl.envs.torch import wrap_env
from skrl.envs.wrappers.torch import Wrapper
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# from algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
# Import the skrl components to build the RL system
# Import the skrl components to build the RL system
from skrl.utils import set_seed

# from skrl.trainers.torch import SequentialTrainer
# from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from testing import wrappers as wrap
from testing.envs.pipe_insert import PipeInsert
from testing.envs.pipe_insert_2 import PipeInsert2
from testing.envs.pipe_insert_3 import PipeInsert3
from testing.experiments.trainer.sequential_trainer_plus import SequentialTrainerPlus

# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.shen.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
)
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

# from algorithms.IBRL_active import IBRL
# from testing.shen.ibrl import IBRL, IBRL_DEFAULT_CONFIG
# from testing.shen.IBRL import IBRL, IBRL_DEFAULT_CONFIG
from testing.shen.ibrl_sac_o_o2_v2 import IBRL_SAC_DEFAULT_CONFIG
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


# class Critic(DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False):
#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions)

#         self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 256)
#         self.linear_layer_2 = nn.Linear(256, 256)
#         self.linear_layer_3 = nn.Linear(256, 1)

#     def compute(self, inputs, role):
#         x = F.relu(
#             self.linear_layer_1(
#                 torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
#             )
#         )
#         x = F.relu(self.linear_layer_2(x))
#         return self.linear_layer_3(x), {}


class StochasticActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        # min_log_std=-5,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

        self.linear_layer_1 = nn.Linear(self.num_observations, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.action_layer = nn.Linear(256, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))

        return torch.tanh(self.action_layer(x)), self.log_std_parameter, {}


# class StochasticActor(GaussianMixin, Model):
#     def __init__(
#         self,
#         observation_space,
#         action_space,
#         device,
#         clip_actions=False,
#         clip_log_std=True,
#         min_log_std=-5,
#         max_log_std=2,
#         reduction="sum",
#     ):
#         Model.__init__(
#             self,
#             observation_space=observation_space,
#             action_space=action_space,
#             device=device,
#         )
#         GaussianMixin.__init__(
#             self,
#             clip_actions=clip_actions,
#             clip_log_std=clip_log_std,
#             min_log_std=min_log_std,
#             max_log_std=max_log_std,
#             reduction=reduction,
#         )

#         self.net = nn.Sequential(
#             nn.Linear(self.num_observations, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.num_actions),
#             # nn.Tanh(),
#         )
#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

#     def compute(self, inputs, role):
#         return self.net(inputs["states"]), self.log_std_parameter, {}

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================
# define RL agent: SAC (stochastic models for actor, and deterministic models for critic) using mixins
# class StochasticActor(GaussianMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False,
#                  clip_log_std=True, min_log_std=-5, max_log_std=2, reduction="sum"):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#         self.linear_layer_1 = nn.Linear(self.num_observations, 256)
#         self.linear_layer_2 = nn.Linear(256, 256)
#         self.action_layer = nn.Linear(256, self.num_actions)

#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

#     def compute(self, inputs, role):
#         x = F.relu(self.linear_layer_1(inputs["states"]))
#         x = F.relu(self.linear_layer_2(x))
#         return torch.tanh(self.action_layer(x)), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
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
    models_SAC["policy"] = StochasticActor(
        env.observation_space,
        env.action_space,
        env.device,
        clip_actions=True,
        clip_log_std=True,
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


def get_ibrl_sac_dp_config(exp_name, env, wandb):
    cfg = IBRL_SAC_DEFAULT_CONFIG.copy()

    cfg["actor_learning_rate"] = 3e-4
    cfg["critic_learning_rate"] = 3e-4

    # maybe use?
    # cfg["state_preprocessor"] = RunningStandardScaler

    cfg["discount_factor"] = 0.99
    cfg["batch_size"] = 256  # from ibrl paper
    cfg["random_timesteps"] = 0  # Add some random exploration at the start
    cfg["learning_starts"] = 0  # Start learning after some experience
    cfg["learn_entropy"] = True
    # cfg["learn_entropy"] = False
    # cfg["learn_entropy"] = True
    # cfg["learn_entropy"] = True
    # cfg["learn_entropy"] = True
    # cfg["learn_entropy"] = False
    # cfg["learn_entropy"] = True
    cfg["grad_norm_clip"] = 0.1  # Add gradient clipping for stability
    cfg["learning_rate"] = 3e-4  # Standard SAC learning rate
    cfg["initial_entropy_value"] = 0.01  # Entropy learning rate
    # cfg["initial_entropy_value"] = 0.001  # Entropy learning rate
    # cfg["initial_entropy_value"] = 0.1  # Entropy learning rate
    # cfg["initial_entropy_value"] = 0.01  # Entropy learning rate
    # cfg["initial_entropy_value"] = 0.01  # Entropy learning rate
    # cfg["initial_entropy_value"] = 0.01  # Entropy learning rate
    # cfg["initial_entropy_value"] = 0.1  # Entropy learning rate
    # cfg["initial_entropy_value"] = 0.01  # Entropy learning rate
    cfg["entropy_learning_rate"] = 3e-4  # Entropy learning rate
    # cfg["entropy_learning_rate"] = 3e-4  # Entropy learning rate
    # cfg["entropy_learning_rate"] = 1e-4     # Entropy learning rate
    # cfg["entropy_learning_rate"] = 3e-4     # Entropy learning rate
    cfg["num_envs"] = env.num_envs
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 1000

    # Experiment configuration
    model_path = Path(__file__).parent / f".runs/{exp_name}/models"
    model_path.mkdir(parents=True, exist_ok=True)

    # project name
    cfg["experiment"]["wandb_kwargs"].setdefault("project", "pipe-insert-v3")

    print(f"creating {model_path}")

    cfg["experiment"]["directory"] = model_path.as_posix()
    cfg["experiment"]["experiment_name"] = exp_name
    cfg["experiment"]["wandb"] = wandb
    cfg["experiment"].setdefault("wandb_kwargs", {})
    cfg["experiment"]["wandb_kwargs"].setdefault("resume", "never")

    return cfg


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
