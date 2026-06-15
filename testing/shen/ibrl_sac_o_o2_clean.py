import copy
import itertools
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

# from skrl import config, logger
# from algorithms.IBRLbase import Agent
from skrl.memories.torch import Memory, RandomMemory
from skrl.models.torch import Model
from skrl.utils import postprocessing

# from skrl.agents.torch import Agent
from testing.shen.base_ibrl_agent import Agent
from utils.performance import timer

logging.basicConfig(level=logging.WARNING)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.WARNING)

# fmt: off
# [start-config-dict-torch]
IBRL_SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "decision_block": False,        # if the decision block from https://arxiv.org/abs/2509.04069 should be used

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 3e-4,    # actor learning rate
    "critic_learning_rate": 3e-4,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 3e-4,  # entropy learning rate
    "initial_entropy_value": 0.01,     # initial entropy value
    "target_entropy": None,         # target entropy

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },

    # added
    "soft_update_beta": 10,
}
# [end-config-dict-torch]
# fmt: on

# # fmt: off
# # [start-config-dict-torch]
# IBRL_SAC_DEFAULT_CONFIG = {
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
#     },

#     # added
#     "soft_update_beta": 10,
# }
# # [end-config-dict-torch]
# # fmt: on


# IBRL(SAC)
class IBRL(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        models_il: Dict[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]],
        expert_memory: Optional[Union[Memory, Tuple[Memory]]],
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Soft Actor-Critic (SAC)

        https://arxiv.org/abs/1801.01290

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(IBRL_SAC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            models_il=models_il,
            memory=memory,
            expert_memory=expert_memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )
        # IL model
        self.IL_policy = self.models_il.get("policy", None)

        # models
        self.policy = self.models.get("policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        self.RED_Q = self.cfg["RED-Q_enable"]
        if self.RED_Q:
            # RED-Q: ensemble of critics
            self.critics = self.models.get("critics", [])
            self.target_critics = self.models.get("target_critics", [])
        else:
            self.critic_1 = self.models.get("critic_1", None)
            self.critic_2 = self.models.get("critic_2", None)
            self.critics = [self.critic_1, self.critic_2]
            self.target_critic_1 = self.models.get("target_critic_1", None)
            self.target_critic_2 = self.models.get("target_critic_2", None)
            self.target_critics = [self.target_critic_1, self.target_critic_2]

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2
        self.demo_mean = 0
        self.demo_cov = 0

        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # configuration

        self._decision_block: bool = self.cfg.get("decision_block", False)

        self._gradient_steps: int = self.cfg["gradient_steps"]
        self._batch_size: int = self.cfg["batch_size"]

        self._discount_factor: float = self.cfg["discount_factor"]
        self._polyak: float = self.cfg["polyak"]

        self._actor_learning_rate: float = self.cfg["actor_learning_rate"]
        self._critic_learning_rate: float = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]  # 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]
        # RED-Q parameters
        self._ensemble_size = self.cfg["ensemble_size"]
        self._critic_subset_size = self.cfg["critic_subset_size"]
        self._policy_subset_size = self.cfg["policy_subset_size"]
        self._offline = self.cfg["offline"]
        self._BC = self.cfg["BC"]
        self._num_envs = self.cfg["num_envs"]

        # self._demo_file = self.cfg["demo_file"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(
                device=self._device_type, enabled=self._mixed_precision
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(
                        np.float32
                    )
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam(
                [self.log_entropy_coefficient], lr=self._entropy_learning_rate
            )

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if (
            self.policy is not None
            and self.critic_1 is not None
            and self.critic_2 is not None
        ):
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self._actor_learning_rate
            )
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                lr=self._critic_learning_rate,
            )
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(
                **self.cfg["state_preprocessor_kwargs"]
            )
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        self.expert_memory = expert_memory

        # TODO: Added for dp compatibility
        self._obs_horizon = self.models_il["policy"].cfg["obs_horizon"]

        self._obs_queues = [
            deque(maxlen=self._obs_horizon) for _ in range(self._num_envs)
        ]

        self._soft_update_beta = self.cfg["soft_update_beta"]

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(
                name="states", size=self.observation_space, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="next_states", size=self.observation_space, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="actions", size=self.action_space, dtype=torch.float32
            )
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self._tensors_names = [
                "states",
                "actions",
                "rewards",
                "next_states",
                "terminated",
            ]

            if not self._offline:
                # Calculate how many samples we can add (min of available space or total expert data)
                total_samples = self.expert_memory.get_tensor_by_name("states").shape[0]
                memory_capacity = self.memory.memory_size
                available_space = memory_capacity - len(self.memory)
                samples_to_add = min(total_samples, available_space)
                if samples_to_add > 0:
                    logger.warning(
                        f"Be aware that len(expert_memory) > len(memory) (i.e. {len(self.expert_memory)} > {len(self.memory)}) the loaded expert data into memory will therefore wrap."
                    )
                    # Add samples in env-sized chunks to satisfy memory's expected shape
                    start_idx = 0
                    while start_idx < samples_to_add:
                        end_idx = min(start_idx + memory_capacity, samples_to_add)

                        for offset in range(start_idx, end_idx, self.memory.num_envs):
                            batch = {}
                            batch_end = min(offset + self.memory.num_envs, end_idx)
                            for tn in self._tensors_names:
                                tensor = self.expert_memory.get_tensor_by_name(
                                    tn
                                ).squeeze(1)[offset:batch_end, :]
                                logger.debug(f"{tn}: {tensor.shape=}")
                                batch[tn] = tensor

                            self.memory.add_samples(**batch)
                        start_idx = end_idx
                # _result = {}
                # for tn in self._tensors_names:
                #     tensor = self.expert_memory.get_tensor_by_name(tn).squeeze(1)[
                #         :100, :
                #     ]
                #     logger.warning(
                #         "More expert data is loaded than there is space in memory provided"
                #     )
                #     logger.info(f"{tensor.shape=}")
                #     _result[tn] = tensor
                # self.memory.add_samples(**_result)
                # exp_memory = postprocessing.MemoryFileIterator(self._demo_file)
                # for k, data0 in exp_memory:
                #     # self.expert_memory.add_samples(d)
                #     keys = list(data0.keys())
                #     N = len(data0[keys[0]])
                #     for i in range(0, N):
                #         self.memory.add_samples(
                #             states=torch.Tensor(np.array(data0[keys[3]][i])),
                #             actions=torch.Tensor(np.array(data0[keys[0]][i])),
                #             rewards=torch.Tensor(np.array(data0[keys[2]][i])),
                #             next_states=torch.Tensor(np.array(data0[keys[1]][i])),
                #             terminated=torch.Tensor(np.array(data0[keys[4]][i])),
                #         )
            # logger.info(
            #     f"loaded expert data into memory successfully, loaded {len(_result.values()[0])}"
            # )

            # create tensors in memory

            # if isinstance(self.expert_memory, (Memory, RandomMemory)):
            #     if len(self.expert_memory) == 0:
            #         self.expert_memory.create_tensor(
            #             name="states", size=self.observation_space, dtype=torch.float32
            #         )
            #         self.expert_memory.create_tensor(
            #             name="next_states",
            #             size=self.observation_space,
            #             dtype=torch.float32,
            #         )
            #         self.expert_memory.create_tensor(
            #             name="actions", size=self.action_space, dtype=torch.float32
            #         )
            #         self.expert_memory.create_tensor(
            #             name="rewards", size=1, dtype=torch.float32
            #         )
            #         self.expert_memory.create_tensor(
            #             name="terminated", size=1, dtype=torch.bool
            #         )

            #         # for i in range(N):
            #         #     print(f"{exp_data['observation'].shape=}")
            #         #     self.expert_memory.add_samples(
            #         #         states=exp_data["observation"][i],
            #         #         actions=exp_data["action"][i],
            #         #         rewards=exp_data["reward"][i],
            #         #         next_states=exp_data["next_observation"][i],
            #         #         terminated=exp_data["done"][i],
            #         #     )

            #         print(f"Loaded {N} expert samples successfully")

            expert_states_r = self.expert_memory.sample(
                names=["states"],
                batch_size=self._batch_size,
            )[0][0]

            self.demo_mean = torch.mean(expert_states_r, dim=0)
            cov = torch.cov(expert_states_r.T)
            self.demo_cov = torch.inverse(cov)

    def _select_act(
        self,
        rl_obs: torch.Tensor,
        il_obs: torch.Tensor,
        exp_obs: torch.Tensor,
        soft: bool,
        target: bool,
    ):
        logger.debug("select act")
        logger.debug(f"\t{rl_obs.shape=}")  # (num_envs, o_dim)
        logger.debug(f"\t{il_obs.shape=}")  # (num_envs, o_horizon, o_dim)
        logger.debug(f"\t{exp_obs.shape=}")  # (batch_size, o_dim)

        with timer("\tfind rl action", logger.debug):
            if target:
                # target policy smoothing
                rl_actions, next_log_prob, _ = self.policy.act(
                    {"states": rl_obs}, role="policy"
                )
            else:
                # sample stochastic actions
                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision
                ):
                    rl_actions, _, outputs = self.policy.act(
                        {"states": self._state_preprocessor(rl_obs)}, role="policy"
                    )

        logger.debug(f"\t{rl_actions.shape=}")  # here i want (batch_size, a_dim)

        with timer("\tfind il action", logger.debug):
            # Get IL actions
            self.IL_policy.eval()
            il_actions, _, _ = self.IL_policy.act(
                # il_actions, _, _ = self.IL_policy.act(
                {"states": self._state_preprocessor(il_obs)},
                role="policy",
            )

        # il_actions, _, _ = self.IL_policy.act(
        #     # il_actions, _, _ = self.IL_policy.act(
        #     {"states": self._state_preprocessor(exp_obs)},
        #     role="policy",
        # )
        # TODO: extract the one action from the trajectory

        logger.debug(f"\t{rl_actions.shape=}")
        logger.debug(f"\t{il_actions.shape=}")
        # TODO: Here we slice to get a actions instance instead of a traj
        il_actions = il_actions[
            :, 0
        ]  # (batch_size, a_horizon, a_dim) → (batch_size, a_dim)

        # Change this line - instead of concatenating, stack the actions
        # rl_bc_actions = torch.cat([rl_actions, il_actions], dim=1)
        rl_bc_actions = torch.stack([rl_actions, il_actions], dim=1)

        logger.debug(f"{rl_bc_actions.shape=}")
        batch_size, _, num_action = rl_bc_actions.size()

        logger.debug(f"{batch_size=}")
        logger.debug(f"{num_action=}")

        logger.debug(f"{rl_actions[0]=}")
        logger.debug(f"{il_actions[0]=}")
        logger.debug(f"{exp_obs[0]=}")

        # rl action (10, 6) (num_envs, a_dim)
        # il action (1, 16, 3) (x1, x2, x3) (batch, pred_horizon, a_dim)
        logger.debug(f"\t{rl_actions.shape=}")
        logger.debug(f"\t{il_actions.shape=}")
        logger.debug(f"\t{exp_obs.shape=}")
        logger.debug(f"\t{rl_bc_actions.shape=}")
        logger.debug(f"\t{rl_bc_actions.size()=}")

        # Stack actions and get batch dimensions
        # rl_bc_actions = torch.stack([rl_actions, il_actions], dim=1)
        # TODO: This is only used without the decision block introduced by
        # https://arxiv.org/pdf/2509.04069
        # batch_size, num_action, _ = (
        #     rl_bc_actions.size()
        # )  # get dimensions values, bsize:batch size

        logger.debug(f"\t{exp_obs.shape=}")
        logger.debug(f"\t{il_actions.shape=}")

        # Compute min Q-values for both policies
        target_q_il = self._compute_min_q_values(exp_obs, il_actions)

        logger.debug(f"\t{rl_obs.shape=}")
        logger.debug(f"\t\t{rl_obs[0]=}")
        logger.debug(f"\t{rl_actions.shape=}")
        logger.debug(f"\t\t{rl_actions[0]=}")

        target_q_rl = self._compute_min_q_values(rl_obs, rl_actions)

        logger.debug(f"\t{target_q_il.shape=}")
        logger.debug(f"\t{target_q_rl.shape=}")
        logger.debug(f"\t{target_q_il[0]=}")
        logger.debug(f"\t{target_q_rl[0]=}")

        logger.debug(f"\t{target_q_il[0]=}")
        logger.debug(f"\t{target_q_rl[0]=}")

        # TODO: Why is this not used? → probably new decision block from
        # https://arxiv.org/pdf/2509.04069
        # Stack Q-values
        # target_q_values = torch.stack([target_q_rl, target_q_il], dim=1).view(
        #     batch_size, num_action
        # )

        if self._decision_block:
            # TODO: Be aware that we here are parsing the expert state and expert state 2
            # instead of the obs from the interaction
            if torch.mean(target_q_il) > torch.mean(target_q_rl):
                # IL wins: reuse the single-step IL action to keep shape (num_envs, a_dim)
                actions = il_actions
                # No log-prob available for IL policy in this branch
                if target:
                    next_log_prob = None
            else:
                actions = rl_actions
        else:
            # Stack Q-values
            target_q_values = torch.hstack([target_q_rl, target_q_il])
            # target_q_values = torch.stack([target_q_rl, target_q_il], dim=1).view(
            #     batch_size, num_action
            # )
            # (batch_size, 2)
            logger.debug(f"\t{target_q_values.shape=}")
            logger.debug(f"\t\t{target_q_values=}")

            # here target_q_values are [batchsize, 2], since there are one for each rl and il
            # here we have the q-values for all actions i.e. if we have an action space A ∈ R²
            # then a_rl ∈ R² and a_il ∈ R², i.e. a ∈ R⁴ as a = [a_rl, a_il].
            # the q values are thus also in q ∈ R⁴, meaning we can pick specific elements from
            # rl or il that each produce a greater q value.

            # Select actions based on strategy
            if soft:
                # Boltzmann exploration
                # convert q values to a probability distribution using softmax
                probs = F.softmax(target_q_values * self._soft_update_beta, dim=1)
                logger.debug(f"\t{probs.shape=}")
                logger.debug(f"\t\t{probs=}")
                # her we sample from the probability distribution, such that this is a
                # list of length num_actions with either a 0 or a 1. If it is a 0, i means we want
                # the rl action if it is 1 we want the il action.
                # action_indices = probs.multinomial(1)  # Keep as column vector for proper indexing
                # actions = rl_bc_actions[torch.arange(batch_size), action_indices.squeeze(1)]

                action_indices = probs.multinomial(1).squeeze(1)  # Shape: [10]
                actions = rl_bc_actions[
                    torch.arange(batch_size), action_indices
                ]  # Shape: [10, 2]

                # action_indices = probs.multinomial(1).squeeze(1)
                logger.debug(f"{action_indices.shape=}")
                logger.debug(f"\t{action_indices}")
                logger.debug(f"{rl_bc_actions.shape=}")
                # we here index over the entire batch (using torch.arange(batch_size)) in the rows and
                # use the action indecies for either rl or il in the columns
                #

                actions = rl_bc_actions[torch.arange(batch_size), action_indices]
                logger.debug(f"{actions.shape=}")
                # actions = rl_actions * (1 - action_indices) + il_actions * action_indices
            else:
                # Greedy selection
                # action_indices = target_q_values.argmax(dim=1)
                # actions = rl_bc_actions[torch.arange(batch_size), action_indices]
                # action_indices = target_q_values.argmax(dim=1, keepdim=True)  # Keep dimensions
                # actions = rl_bc_actions[torch.arange(batch_size), action_indices.squeeze(1)]
                action_indices = target_q_values.argmax(dim=1)  # Shape: [10]
                actions = rl_bc_actions[
                    torch.arange(batch_size), action_indices
                ]  # Shape: [10, 2]

                # actions = rl_actions
                # actions = il_actions

            # return actions, _, _

            # actions = rl_actions

        logger.debug(f"\t{actions[0]=} {actions.shape=}")

        if not target:
            self.track_data(
                "Q-network / select_rl_Q (mean)", torch.mean(target_q_rl).item()
            )
            self.track_data(
                "Q-network / select_il_Q (mean)", torch.mean(target_q_il).item()
            )
            return actions, _, _
        else:
            return actions, next_log_prob, _

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """

        # logger.debug("add states to queue (ensure that there is always 2 states)")
        # logger.debug("act")
        # logger.debug(f"\t{states.shape=}")
        # logger.debug(f"\t\t{states[0]=}")

        for i, queue in enumerate(self._obs_queues):
            while len(queue) < self._obs_horizon:
                queue.append(states[i])

        for i, s in enumerate(states):
            self._obs_queues[i].append(s)
        # self._obs_queue.append(states)

        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act(
                {"states": self._state_preprocessor(states)}, role="policy"
            )
        logger.debug(f"\t{self._state_preprocessor(states).shape=} ")

        # sample from expert buffer
        (
            expert_states_r,
            expert_actions_r,
            expert_rewards_r,
            expert_next_states_r,
            expert_dones_r,
        ) = self.expert_memory.sample(
            names=self._tensors_names,
            # TODO: Obs, we are sampling num envs instead of batch_size in order to follow the shape
            # of the observations coming from the environment.
            batch_size=self.cfg["num_envs"],
            # batch_size=self.cfg["batch_size"],
        )[0]

        logger.debug(f"\t{self.cfg['num_envs']=}")
        logger.debug(f"\t{expert_states_r.shape=}")

        diff = states - self.demo_mean
        left = torch.matmul(diff, self.demo_cov)
        dist_sq = (left * diff).sum(dim=1)
        M_dist = torch.sqrt(dist_sq)

        # TODO: So here we transform the list of deques of states into a properly batched data
        # sample for DP to accept
        # here we have that il_states are (n_envs, obs_horizon, o_dim)
        # il_states = torch.stack(list(self._obs_queues), axis=1)
        deque_tensors = [torch.stack(list(q), dim=0) for q in self._obs_queues]
        il_states = torch.stack(deque_tensors, dim=0)

        # TODO: Here we are deciding which action to take, the observations parsed are
        # logger.debug(f"\t{states.shape=}")
        # logger.debug(f"\t\t{states[0]=}")
        # logger.info(f"\t{il_states.shape=}")
        # logger.debug(f"\t\t{il_states[0]=}")
        # logger.debug(f"\t{expert_states_r.shape=}")
        # logger.debug(f"\t\t{expert_states_r[0]=}")
        # quit()

        # select actions
        with timer("select action", logger.debug):
            actions, _, _ = self._select_act(
                rl_obs=states,
                il_obs=il_states,
                exp_obs=expert_states_r,
                soft=True,
                target=False,
            )
        # actions, _, _ = self._select_act(states, states, soft=True, target=False)

        logger.debug(f"Return actions {actions.shape=}")

        self.track_data("Loss / states BC loss", torch.mean(M_dist).item())

        return actions, None, None

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        with timer("record_transition", logger.debug):
            super().record_transition(
                states,
                actions,
                rewards,
                next_states,
                terminated,
                truncated,
                infos,
                timestep,
                timesteps,
            )

            if timestep < self._random_timesteps + self._learning_starts - 1:
                self.expert_memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )
                for expert_memory in self.secondary_memories:
                    expert_memory.add_samples(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        terminated=terminated,
                        truncated=truncated,
                    )

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(
        self, states: torch.Tensor, timestep: int, timesteps: int
    ) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # save the state for diffusion policy observation horizon
        self._states = states

    def post_interaction(
        self, next_states: torch.BoolTensor, timestep: int, timesteps: int
    ) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # save the next state for diffusion policy observation horizon
        self._next_states = next_states

        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

        # for i, done in enumerate(terminated):
        #     if done:
        #         self._obs_queues[i].clear()

        # if timestep >= self._learning_starts:
        #     self.set_mode("train")
        #     with timer("update", logger_fn=logger.debug):
        #         self._update(timestep, timesteps)
        #     self.set_mode("eval")

        # # write tracking data and checkpoints
        # super().post_interaction(timestep, timesteps)

    def _compute_min_q_values(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Helper to compute target Q-values using both critics"""
        # RED-Q: compute target values using ensemble

        logger.debug("5 _compute_min_q_values")
        logger.debug(f"\t{states.shape=}")
        logger.debug(f"\t{actions.shape=}")
        logger.debug(f"\t{self.RED_Q=}")
        logger.debug(f"\t{len(self.target_critics)=}")

        # Compute target values
        if self.RED_Q:
            # RED-Q: randomly sample subset of critics to compute target Q value
            random_critic_indices = torch.randperm(len(self.critics))[
                : self._critic_subset_size
            ]
            target_q_values_list = []
            for idx in random_critic_indices:
                target_q_val, _, _ = self.target_critics[idx].act(
                    {"states": states, "taken_actions": actions},
                    role=f"target_critic_{idx}",
                )
                target_q_values_list.append(target_q_val)
        else:
            target_q_values_list = []
            for idx in [0, 1]:
                if len(states.shape) == 1:
                    states = states.unsqueeze(0)
                if len(actions.shape) == 1:
                    actions = actions.unsqueeze(0)
                # states = states.squeeze()
                # target_q_val, _, _ = self.target_critics.act(
                target_q_val, _, _ = self.target_critics[idx].act(
                    {"states": states, "taken_actions": actions},
                    role=f"target_critic_{idx + 1}",
                )
                logger.debug(f"\t\t{states.shape=}")
                logger.debug(f"\t\t{actions.shape=}")
                logger.debug(f"\t\t{target_q_val.shape=}")
                target_q_values_list.append(target_q_val)

        target_q_values = torch.hstack(target_q_values_list)  # (num_envs, 2)
        # target_q_values = torch.stack(target_q_values_list, axis=1)
        logger.debug(f"{target_q_values.shape=}")

        target_q_value = torch.min(target_q_values, dim=1).values.unsqueeze(
            -1
        )  # (num_envs, 1) here the 1 is an
        # index, if we should use IL or RL

        # target_q_value = torch.min(target_q_values, dim=0)[
        #     0
        # ]
        logger.debug(f"\t{target_q_value.shape=}")
        # (num_envs, 2 * a_dim)
        return target_q_value

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        logger.debug("_update")

        # gradient steps
        for gradient_step in range(self._gradient_steps):
            if self._offline:
                raise ValueError(f"obs: {self._offline=} must be False...")
                # here we only get expert memory
                (
                    sampled_states,
                    sampled_actions,
                    sampled_rewards,
                    sampled_next_states,
                    sampled_dones,
                ) = self.expert_memory.sample(
                    names=self._tensors_names, batch_size=self._batch_size
                )[0]

            else:
                # here we mix expert memory and sampled memory

                # sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
                #     self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]
                logger.debug(f"\t{len(self.memory)=}")
                logger.debug(f"\t{int(self._batch_size * 1)=}")
                (
                    sampled_states_r,
                    sampled_actions_r,
                    sampled_rewards_r,
                    sampled_next_states_r,
                    sampled_dones_r,
                ) = self.memory.sample(
                    names=self._tensors_names, batch_size=int(self._batch_size * 1)
                )[0]

                # (
                #     expert_states_r,
                #     expert_actions_r,
                #     expert_rewards_r,
                #     expert_next_states_r,
                #     expert_dones_r,
                # ) = self.expert_memory.sample(
                #     names=self._tensors_names,
                #     # batch_size=int(self._batch_size * 1),
                #     # names=self._tensors_names,
                #     batch_size=int(self._batch_size * 0),
                # )[0]
                # logger.info(f"\t{expert_states_r.shape=}")
                (
                    expert_states,
                    expert_actions,
                    expert_rewards,
                    expert_next_states,
                    expert_dones,
                ) = self.expert_memory.sample(
                    names=self._tensors_names, batch_size=int(self._batch_size)
                )[0]

                logger.debug(f"\t{expert_next_states.shape=}")
                logger.debug(f"\t\t{expert_next_states[0]=}")

                logger.debug(f"\t{sampled_states_r.shape=}")
                logger.debug(f"\t\t{sampled_states_r[0]=}")

                # logger.debug(f"\t{expert_states_r.shape=}")
                # logger.debug(f"\t\t{expert_states_r[0]=}")

                logger.debug(f"\t{sampled_actions_r.shape=}")
                logger.debug(f"\t\t{sampled_actions_r[0]=}")

                # logger.debug(f"\t{expert_actions_r.shape=}")
                # logger.debug(f"\t\t{expert_actions_r[0]=}")

                sampled_states = sampled_states_r
                sampled_actions = sampled_actions_r
                # sampled_states = torch.cat((sampled_states_r, expert_states_r))
                # sampled_actions = torch.cat((sampled_actions_r, expert_actions_r))

                logger.debug(f"\t{sampled_states.shape=}")
                logger.debug(f"\t\t{sampled_states[0]=}")
                logger.debug(f"\t{sampled_actions.shape=}")
                logger.debug(f"\t\t{sampled_actions[0]=}")

                sampled_next_states = sampled_next_states_r
                # sampled_next_states = torch.cat(
                #     (sampled_next_states_r, expert_next_states_r)
                # )

                logger.debug(f"\t{sampled_next_states.shape=}")
                logger.debug(f"\t\t{sampled_next_states[0]=}")

                logger.debug(f"\t{expert_states.shape=}")
                logger.debug(f"\t\t{expert_states[0]=}")

                sampled_rewards = sampled_rewards_r
                # sampled_rewards = torch.cat((sampled_rewards_r, expert_rewards_r))
                sampled_dones = sampled_dones_r
                # sampled_dones = torch.cat((sampled_dones_r, expert_dones_r))

                logger.debug(f"\t{sampled_rewards.shape=}")
                logger.debug(f"\t\t{sampled_rewards[0]=}")

                logger.debug(f"\t{sampled_dones.shape=}")
                logger.debug(f"\t\t{sampled_dones[0]=}")

            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(
                    sampled_next_states, train=True
                )

                # compute target values
                with torch.no_grad():
                    # next_actions, next_log_prob, _ = self.policy.act({"states": sampled_next_states}, role="policy")
                    # TODO: here we add o2 as well
                    # TODO: here we attempt to build the DP state queue. We here choose the expert memory
                    # due to the Bootstrapping error as described in https://arxiv.org/pdf/2509.04069

                    il_states = torch.stack([expert_states, expert_next_states], axis=1)
                    logger.debug(f"\t{il_states.shape=}")
                    logger.debug(f"\t\t{il_states[0]=}")

                    # TODO: Here we need to figure out if we should use the expert or the samples
                    # obs for DP
                    # actions, next_log_prob, _
                    next_actions, next_log_prob, _ = self._select_act(
                        rl_obs=sampled_next_states,
                        il_obs=il_states,
                        exp_obs=expert_next_states,
                        soft=True,
                        target=True,
                    )
                    logger.debug(f"{next_log_prob=}")
                    # next_actions, next_log_prob, _ = self._select_act(
                    #     sampled_next_states, expert_next_states, soft=True, target=True
                    # )
                    # next_actions, next_log_prob, _ = self._select_act(sampled_next_states, sampled_next_states, soft=True,
                    #                                       target=True)

                    target_q1_values, _, _ = self.target_critic_1.act(
                        {"states": sampled_next_states, "taken_actions": next_actions},
                        role="target_critic_1",
                    )
                    target_q2_values, _, _ = self.target_critic_2.act(
                        {"states": sampled_next_states, "taken_actions": next_actions},
                        role="target_critic_2",
                    )
                    if next_log_prob is None:
                        # Deterministic policy branch: no log-prob contribution
                        next_log_prob = torch.zeros_like(target_q1_values)
                    target_q_values = (
                        torch.min(target_q1_values, target_q2_values)
                        - self._entropy_coefficient * next_log_prob
                    )
                    target_values = (
                        sampled_rewards
                        + self._discount_factor
                        * (sampled_dones).logical_not()
                        * target_q_values
                    )
                    # self.track_data(
                    #     "Discount Reward ",
                    #     self._discount_factor
                    #     * (sampled_dones).logical_not()
                    #     * target_q_values,
                    # )

                # compute critic loss
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": sampled_actions},
                    role="critic_1",
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": sampled_actions},
                    role="critic_2",
                )

                critic_loss = (
                    F.mse_loss(critic_1_values, target_values)
                    + F.mse_loss(critic_2_values, target_values)
                ) / 2

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            # if config.torch.is_distributed:
            #     self.critic_1.reduce_parameters()
            #     self.critic_2.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.critic_1.parameters(), self.critic_2.parameters()
                    ),
                    self._grad_norm_clip,
                )

            self.scaler.step(self.critic_optimizer)

            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                # compute policy (actor) loss
                actions, log_prob, _ = self.policy.act(
                    {"states": sampled_states}, role="policy"
                )
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": actions},
                    role="critic_1",
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": actions},
                    role="critic_2",
                )

                # print(f"{actions=}")
                # print(f"{sampled_actions=}")

                bc_loss = F.mse_loss(actions, sampled_actions)
                # print(f"{bc_loss=}")

                policy_loss = (
                    self._entropy_coefficient
                    * (
                        log_prob
                        if log_prob is not None
                        else torch.zeros_like(critic_1_values)
                    )
                    - torch.min(critic_1_values, critic_2_values)
                ).mean()

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

            # if config.torch.is_distributed:
            #     self.policy.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            # entropy learning
            if self._learn_entropy:
                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision
                ):
                    safe_log_prob = (
                        log_prob
                        if log_prob is not None
                        else torch.zeros_like(critic_1_values)
                    )
                    # compute entropy loss
                    entropy_loss = -(
                        self.log_entropy_coefficient
                        * (safe_log_prob + self._target_entropy).detach()
                    ).mean()

                # optimization step (entropy)
                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)

                # compute entropy coefficient
                self._entropy_coefficient = torch.exp(
                    self.log_entropy_coefficient.detach()
                )

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())
                # print(f"{bc_loss.item()=}")
                # if torch.isnan(bc_loss):
                #     print(f"{bc_loss.item()=}")

                self.track_data("Loss / BC loss", bc_loss.item())

                self.track_data(
                    "Q-network / Q1 (max)", torch.max(critic_1_values).item()
                )
                self.track_data(
                    "Q-network / Q1 (min)", torch.min(critic_1_values).item()
                )
                self.track_data(
                    "Q-network / Q1 (mean)", torch.mean(critic_1_values).item()
                )

                self.track_data(
                    "Q-network / Q2 (max)", torch.max(critic_2_values).item()
                )
                self.track_data(
                    "Q-network / Q2 (min)", torch.min(critic_2_values).item()
                )
                self.track_data(
                    "Q-network / Q2 (mean)", torch.mean(critic_2_values).item()
                )

                self.track_data(
                    "Target / Target (max)", torch.max(target_values).item()
                )
                self.track_data(
                    "Target / Target (min)", torch.min(target_values).item()
                )
                self.track_data(
                    "Target / Target (mean)", torch.mean(target_values).item()
                )
                self.track_data(
                    "Target / sampled_rewards (mean)",
                    torch.mean(sampled_rewards).item(),
                )

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data(
                        "Coefficient / Entropy coefficient",
                        self._entropy_coefficient.item(),
                    )

                if self._learning_rate_scheduler:
                    self.track_data(
                        "Learning / Policy learning rate",
                        self.policy_scheduler.get_last_lr()[0],
                    )
                    self.track_data(
                        "Learning / Critic learning rate",
                        self.critic_scheduler.get_last_lr()[0],
                    )
