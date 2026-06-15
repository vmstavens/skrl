import copy
import itertools
import json
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

from skrl import config, logger

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

    "warmup_timesteps": 10_000,    # number of timesteps to conclude before mode is switched from il to both

    "il_ctrl_scale": 1.0,           # since il needs traj, sends to overshoot, this is to prevent that 
    "rl_ctrl_scale": 1.0,           # since il needs traj, sends to overshoot, this is to prevent that 

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
    # "entropy_learning_rate": 1e-4,  # entropy learning rate
    "entropy_learning_rate": 3e-4,  # entropy learning rate
    # "entropy_learning_rate": 3e-3,  # entropy learning rate
    # "initial_entropy_value": 0.2,     # initial entropy value
    "initial_entropy_value": 0.2,   # initial entropy value
    # "initial_entropy_value": 0.1,     # initial entropy value
    # "initial_entropy_value": 0.01,     # initial entropy value
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
    "soft_update_beta": 1,         # use small for good IL (small ~0.2) original IBRL used 10
    # "soft_update_beta": 0.2,         # use small for good IL (small ~0.2) original IBRL used 10
    "actor": "both",                # should either be "rl", "il" or "both"
}
# [end-config-dict-torch]
# fmt: on


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
        """Imitation Bootstrapped Reinforcement Learning (IBRL) - SAC Based

        https://arxiv.org/abs/2311.02198

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param models_il: Imitation learning models used by the agent
        :type models_il: dictionary of skrl.models.torch.Model
        :param memory: Memory to store the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memories.torch.Memory, list of skrl.memories.torch.Memory or None
        :param expert_memory: Expert demonstration memory buffer
        :type expert_memory: skrl.memories.torch.Memory, list of skrl.memories.torch.Memory or None
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

        # memories
        self.expert_memory = expert_memory
        self.memory = memory

        self._tensors_names = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
        ]

        # Ensure that memory and expert_memory have the correct tensors
        assert set(self.memory.get_tensor_names()).issubset(self._tensors_names), (
            f"Memory error: memory should have the tensor names {self._tensors_names}, but got {self.memory.get_tensor_names()}"
        )
        assert set(self.expert_memory.get_tensor_names()).issubset(
            self._tensors_names
        ), (
            f"Memory error: expert_memory should have the tensor names {self._tensors_names}, but got {self.expert_memory.get_tensor_names()}"
        )

        # IL model
        self.IL_policy = self.models_il["policy"]

        # models
        self.policy = self.models["policy"]

        self.critic_1 = self.models["critic_1"]
        self.critic_2 = self.models["critic_2"]
        self.critics = [self.critic_1, self.critic_2]

        self.target_critic_1 = self.models["target_critic_1"]
        self.target_critic_2 = self.models["target_critic_2"]
        self.target_critics = [self.target_critic_1, self.target_critic_2]

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        self.expert_mean = 0
        self.expert_cov = 0

        # OBS I will try :----------------------------------------------------------:
        # broadcast models' parameters in distributed runs

        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)
        # OBS I will try - mid:----------------------------------------------------------:

        # freeze target networks with respect to optimizers (update via .update_parameters())
        # self.target_critic_1.freeze_parameters(True)
        # self.target_critic_2.freeze_parameters(True)

        # # update target networks (hard update)
        # self.target_critic_1.update_parameters(self.critic_1, polyak=1)
        # self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # OBS I will try - end:----------------------------------------------------------:

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
        self._random_timesteps: int = self.cfg["random_timesteps"]
        self._learning_starts: int = self.cfg["learning_starts"]  # 0
        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._entropy_learning_rate: float = self.cfg["entropy_learning_rate"]
        self._learn_entropy: bool = self.cfg["learn_entropy"]
        self._entropy_coefficient: float = self.cfg["initial_entropy_value"]
        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._mixed_precision: bool = self.cfg["mixed_precision"]
        self._soft_update_beta = self.cfg["soft_update_beta"]
        self._actor: str = self.cfg["actor"]

        self._actors = ["rl", "il", "both"]

        self._il_ctrl_scale = self.cfg["il_ctrl_scale"]
        self._rl_ctrl_scale = self.cfg["rl_ctrl_scale"]
        self._warmup_timesteps = self.cfg["warmup_timesteps"]

        assert self._actor in self._actors, (
            f"In config: 'actor' should be one of {self._actors} but got {self._actor}"
        )

        # used to keep track of observations for diffusion policy
        self._states: torch.Tensor = None
        self._prev_states: torch.Tensor = None

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

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)

        # print(f"{self._entropy_coefficient=}")
        # print(f"{self._entropy_learning_rate=}")
        # print(f"{self._learn_entropy=}")
        # print(f"{self._target_entropy=}")
        # print(f"{self.log_entropy_coefficient=}")
        # quit()

        if False:
            # if False:
            # How many timesteps exist in expert and how many timesteps fit in target memory
            expert_T = self.expert_memory.get_tensor_by_name("states").shape[0]
            target_T = self.memory.memory_size

            # how many timesteps we can still insert (skrl memories measure length in timesteps)
            available_T = target_T - len(self.memory)
            steps_to_add = min(expert_T, available_T)

            if steps_to_add <= 0:
                return

            if expert_T > available_T:
                logger.warning(
                    f"Expert memory has more data than available space in memory "
                    f"({expert_T} > {available_T}). Extra data will be skipped."
                )

            # Add one timestep at a time: each call gets shape (num_envs, dim)
            for t in range(steps_to_add):
                batch = {}
                for tn in self._tensors_names:
                    x = self.expert_memory.get_tensor_by_name(tn)

                    # Handle possible extra singleton dim from some skrl memories: (T, 1, N, D) -> (T, N, D)
                    if x.ndim == 4 and x.shape[1] == 1:
                        x = x.squeeze(1)

                    # Now x should be (T, N, D) or (T, N) for scalars (rare). Make it (N, D) at timestep t.
                    step = x[t]

                    # Ensure 2D for scalars: (N,) -> (N, 1)
                    if step.ndim == 1:
                        step = step.unsqueeze(-1)

                    batch[tn] = step

                self.memory.add_samples(**batch)

        expert_states = self.expert_memory.sample(
            names=["states"], batch_size=self._batch_size
        )[0][0]

        # compute the expert mean and covariance of states based on a sampled batch
        # expert_states: torch.Tensor = expert_batch["states"]
        self.expert_mean = torch.mean(expert_states, dim=0)
        cov = torch.cov(expert_states.T)
        self.expert_cov = torch.inverse(cov)

    def _select_act(
        self,
        rl_obs: torch.Tensor,
        il_obs: torch.Tensor,
        exp_obs: torch.Tensor,
        soft: bool,
        target: bool,
        timestep: int,
    ):
        """Select an action by comparing RL and IL policies and their Q-values.

        :param rl_obs: Observations for the RL policy
        :type rl_obs: torch.Tensor
        :param il_obs: Observations for the IL policy
        :type il_obs: torch.Tensor
        :param exp_obs: Expert observations used for IL Q evaluation
        :type exp_obs: torch.Tensor
        :param soft: Whether to sample actions with a softmax strategy
        :type soft: bool
        :param target: Whether to use target policy/Q networks
        :type target: bool
        :return: Selected actions, log-probabilities (if available), and extra outputs
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]
        """

        if target:
            # target policy smoothing
            rl_actions, next_log_prob, _ = self.policy.act(
                {"states": rl_obs}, role="policy"
            )
            # print("rl next_log_prob", next_log_prob)
        else:
            # sample stochastic actions
            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                rl_actions, _, outputs = self.policy.act(
                    {"states": self._state_preprocessor(rl_obs)}, role="policy"
                )

        # Get IL actions
        # a_{il} ← µ ( s_{t} )
        il_actions, _, _ = self.IL_policy.act(
            {"states": self._state_preprocessor(il_obs)},
            role="policy",
        )

        # act out the 0'th index
        action_index = 0
        il_actions = il_actions[:, action_index, :]

        # scale il_actions
        il_actions = il_actions * self._il_ctrl_scale
        rl_actions = rl_actions * self._rl_ctrl_scale

        # Change this line - instead of concatenating, stack the actions
        # this leaves (   )
        rl_il_actions = torch.stack([rl_actions, il_actions], dim=1)

        batch_size, _, num_action = rl_il_actions.size()

        # rl action (10, 6) (num_envs, a_dim)
        # il action (1, 16, 3) (x1, x2, x3) (batch, pred_horizon, a_dim)

        # Stack actions and get batch dimensions
        # rl_bc_actions = torch.stack([rl_actions, il_actions], dim=1)
        # TODO: This is only used without the decision block introduced by
        # https://arxiv.org/pdf/2509.04069
        # batch_size, num_action, _ = (
        #     rl_bc_actions.size()
        # )  # get dimensions values, bsize:batch size

        # Compute min Q-values for both policies
        if self._decision_block:
            # Q_il ← Q_phi' ( s_{t} , µ_{psi} ( s'_{t} ) )
            target_q_il = self._compute_min_q_values(exp_obs, il_actions)
        else:
            # Q_il ← Q_phi' ( s_{t} , µ_{psi} ( s_{t} ) )
            # since il with diffusion policy needs (num_envs, pred_horizon, o_dim)
            # then il_obs has one too many dimension, and we correct with

            # Here the structure is [envs, o_horizon, o_dim], since we want the newest observation and
            # il_obs has the structure [prev_obs, obs]
            il_obs = il_obs[:, -1, :]
            target_q_il = self._compute_min_q_values(il_obs, il_actions)

        # Q_{rl} ← Q_phi' ( s_{t} , π_{theta} ( s_{t} ) )
        target_q_rl = self._compute_min_q_values(rl_obs, rl_actions)

        # TODO: Why is this not used? → probably new decision block from
        # https://arxiv.org/pdf/2509.04069
        # Stack Q-values
        # target_q_values = torch.stack([target_q_rl, target_q_il], dim=1).view(
        #     batch_size, num_action
        # )

        if self._decision_block:
            # see: https://gitlab.sdu.dk/cshen/drlr/-/blob/main/algorithms/DRLR.py?ref_type=heads
            # TODO: Be aware that we here are parsing the expert state and expert state 2
            # instead of the obs from the interaction
            if torch.mean(target_q_il) > torch.mean(target_q_rl):
                # IL wins: reuse the single-step IL action to keep shape (num_envs, a_dim)
                actions = il_actions
            else:
                actions = rl_actions
        else:
            # Stack Q-values
            target_q_values = torch.hstack([target_q_rl, target_q_il])

            # here target_q_values are [batchsize, 2], since there are one for each rl and il
            # here we have the q-values for all actions i.e. if we have an action space A ∈ R²
            # then a_rl ∈ R² and a_il ∈ R², i.e. a ∈ R⁴ as a = [a_rl, a_il].
            # the q values are thus also in q ∈ R⁴, meaning we can pick specific elements from
            # rl or il that each produce a greater q value.

            # Select actions based on strategy
            #  see: https://gitlab.sdu.dk/cshen/drlr/-/blob/main/algorithms/IBRL.py?ref_type=heads
            if soft:
                # Boltzmann exploration
                # convert q values to a probability distribution using softmax
                probs = F.softmax(target_q_values * self._soft_update_beta, dim=1)
                # her we sample from the probability distribution, such that this is a
                # list of length num_actions with either a 0 or a 1. If it is a 0, i means we want
                # the rl action if it is 1 we want the il action.
                # action_indices = probs.multinomial(1)  # Keep as column vector for proper indexing
                # actions = rl_bc_actions[torch.arange(batch_size), action_indices.squeeze(1)]

                # OBS: should this be action_indices = probs.multinomial(1)?
                action_indices = probs.multinomial(1)  # Shape: [num_envs]
                # action_indices = probs.multinomial(1).squeeze(1)  # Shape: [num_envs]

                # how many percent of the action taken is il
                pct_of_il = action_indices.sum().item() / len(action_indices)

                # OBS Better logging needed
                if self._actor == "rl":
                    il_ratio = 0.0
                elif self._actor == "il":
                    il_ratio = 1.0
                elif self._actor == "both":
                    il_ratio = 1.0 if timestep < self._warmup_timesteps else pct_of_il
                else:
                    raise ValueError(
                        f"[ERROR]: {self._actor=}, has to be ('rl', 'il' or 'both')"
                    )
                self.track_data("Which / il_selection_ratio", il_ratio)

                # we here index over the entire batch (using torch.arange(batch_size)) in the rows and
                # use the action indecies for either rl or il in the columns

                # OBS: holy shit this might be it!
                actions = (
                    rl_actions * (1 - action_indices) + il_actions * action_indices
                )
            else:
                # Greedy selection
                action_indices = target_q_values.argmax(dim=1)  # Shape: [num_envs]
                actions = rl_il_actions[torch.arange(batch_size), action_indices]

        # here "actions" is the product of both agents

        actor_id = 0

        if self._actor == "rl":
            actions = rl_actions
            actor_id = 1
        elif self._actor == "il":
            actions = il_actions
            actor_id = 2
        elif self._actor == "both":
            actor_id = 3
            if timestep < self._warmup_timesteps:
                actor_id = 4
                actions = il_actions
        else:
            raise ValueError("Wrong actor")

        self.track_data("Which / Actor", actor_id)

        if not target:
            if self._decision_block:
                self.track_data(
                    "Q-network / select_rl_Q (mean)", torch.mean(target_q_rl).item()
                )
                self.track_data(
                    "Q-network / select_il_Q (mean)", torch.mean(target_q_il).item()
                )
            else:
                self.track_data(
                    "Q-network / select_rl_Q (max)", torch.max(target_q_rl).item()
                )
                self.track_data(
                    "Q-network / select_il_Q (max)", torch.max(target_q_il).item()
                )
            return actions, _, _
        else:
            return actions, next_log_prob, _

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process environment states and return an action tuple.

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions, log-probabilities, and extra outputs (unused)
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]
        """
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act(
                {"states": self._state_preprocessor(states)}, role="policy"
            )

        # sample from expert buffer
        (
            expert_states,
            expert_actions,
            expert_rewards,
            expert_next_states,
            expert_dones,
        ) = self.expert_memory.sample(
            names=self._tensors_names,
            # OBS: Obs, we are sampling num envs instead of batch_size in order to follow the shape
            # of the observations coming from the environment.
            batch_size=self.cfg["num_envs"],
        )[0]

        # compute states BC loss to track state-OOD behavior
        diff = states - self.expert_mean
        left = torch.matmul(diff, self.expert_cov)
        dist_sq = (left * diff).sum(dim=1)
        M_dist = torch.sqrt(dist_sq)
        self.track_data("Loss / states BC loss", torch.mean(M_dist).item())

        # here states (num_envs, o_dim) and next_states (num_envs, o_dim)
        # we need il_states (num_envs, pred_horizon, o_dim)
        # therefore axis=1

        il_states = torch.stack([self._prev_states, self._states], axis=1)

        # Here, when self._decision_block = False, then expert_states is never used!
        actions, _, _ = self._select_act(
            rl_obs=states,
            il_obs=il_states,
            exp_obs=expert_states,
            soft=True,
            target=False,
            timestep=timestep,
        )

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
        """Record an environment transition in memory and expert buffer.

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
        """Callback called before the interaction with the environment.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # save the state for diffusion policy observation horizon
        self._states = states
        if self._prev_states is None:
            self._prev_states = self._states
        else:
            if self._prev_states.shape[0] != self._states.shape[0]:
                self._prev_states = self._states

    def post_interaction(
        self, next_states: torch.Tensor, timestep: int, timesteps: int
    ) -> None:
        """Callback called after the interaction with the environment.

        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.BoolTensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # if timestep >= self._warmup_timesteps:
        #     self._actor = "both"

        # save the next state for diffusion policy observation horizon
        self._prev_states = next_states

        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _compute_min_q_values(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample min Q-values from both target critics.

        :param states: Batch of states
        :type states: torch.Tensor
        :param actions: Batch of actions
        :type actions: torch.Tensor
        :return: Min Q-values across critics
        :rtype: torch.Tensor
        """

        target_q_values_list = []

        for idx in [0, 1]:
            if len(states.shape) == 1:
                states = states.unsqueeze(0)
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)
            target_q_val, _, _ = self.target_critics[idx].act(
                {"states": states, "taken_actions": actions},
                role=f"target_critic_{idx + 1}",
            )
            target_q_values_list.append(target_q_val)

        target_q_values = torch.hstack(target_q_values_list)  # (num_envs, 2)
        # target_q_values = torch.stack(target_q_values_list, axis=1)

        target_q_value = torch.min(target_q_values, dim=1).values.unsqueeze(
            -1
        )  # (num_envs, 1) here the 1 is an
        # index, if we should use IL or RL

        # target_q_value = torch.min(target_q_values, dim=0)[
        #     0
        # ]
        # (num_envs, 2 * a_dim)
        return target_q_value

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # gradient steps
        for gradient_step in range(self._gradient_steps):
            # here we mix expert memory and sampled memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_dones,
            ) = self.memory.sample(
                names=self._tensors_names, batch_size=self._batch_size
            )[0]

            (
                expert_states,
                expert_actions,
                expert_rewards,
                expert_next_states,
                expert_dones,
            ) = self.expert_memory.sample(
                names=self._tensors_names, batch_size=self._batch_size
            )[0]

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

                    il_next_states = torch.stack(
                        [sampled_states, sampled_next_states], axis=1
                    )
                    il_expert_states = torch.stack(
                        [expert_states, expert_next_states], axis=1
                    )

                    # TODO: Here we need to figure out if we should use the expert or the samples
                    # obs for DP
                    # OBS: soft = true, is that okay? in ibrl it uses soft false
                    next_actions, next_log_prob, _ = self._select_act(
                        rl_obs=sampled_next_states,
                        il_obs=il_next_states,
                        exp_obs=il_expert_states,
                        soft=True,
                        target=True,
                        timestep=timestep,
                    )

                    target_q1_values, _, _ = self.target_critic_1.act(
                        {"states": sampled_next_states, "taken_actions": next_actions},
                        role="target_critic_1",
                    )
                    target_q2_values, _, _ = self.target_critic_2.act(
                        {"states": sampled_next_states, "taken_actions": next_actions},
                        role="target_critic_2",
                    )
                    # if next_log_prob is None:
                    #     # Deterministic policy branch: no log-prob contribution
                    #     next_log_prob = torch.zeros_like(target_q1_values)
                    # print(
                    #     # target_q1_values,
                    #     # target_q2_values,
                    #     # self._entropy_coefficient,
                    #     next_log_prob,
                    # )
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
                    # OBS now has added this
                    # discout_q_values = (
                    #     self._discount_factor
                    #     * (sampled_dones).logical_not()
                    #     * target_q_values
                    # ).mean()

                # compute critic loss
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": sampled_actions},
                    role="critic_1",
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": sampled_actions},
                    role="critic_2",
                )

                # OBS sum, not average
                critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(
                    critic_2_values, target_values
                )
                # critic_loss = (
                #     F.mse_loss(critic_1_values, target_values)
                #     + F.mse_loss(critic_2_values, target_values)
                # ) / 2

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

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

                bc_loss = F.mse_loss(actions, sampled_actions)

                policy_loss = (
                    self._entropy_coefficient * log_prob
                    - torch.min(critic_1_values, critic_2_values)
                ).mean()

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            self.track_data("Loss / Target Entropy", self._target_entropy)
            self.track_data("Loss / Log Prob", log_prob.mean().item())
            self.track_data("Loss / Log Entropy Coeff", self.log_entropy_coefficient.item())


            # entropy learning
            if self._learn_entropy:
                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision
                ):
                    # safe_log_prob = (
                    #     log_prob
                    #     if log_prob is not None
                    #     else torch.zeros_like(critic_1_values)
                    # )
                    # compute entropy loss

                    # print(f"{self._target_entropy=}")
                    # print(f"{log_prob.mean()=}")
                    # print(f"{self.log_entropy_coefficient=}")


                    entropy_loss = -(
                        self.log_entropy_coefficient
                        * (log_prob + self._target_entropy).detach()
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
                    if hasattr(self.policy, "get_log_std"):
                        policy_log_std = self.policy.get_log_std()
                        self.track_data("Policy / Log std (max)", torch.max(policy_log_std).item())
                        self.track_data("Policy / Log std (min)", torch.min(policy_log_std).item())
                        self.track_data("Policy / Log std (mean)", torch.mean(policy_log_std).item())


                if self._learning_rate_scheduler:
                    self.track_data(
                        "Learning / Policy learning rate",
                        self.policy_scheduler.get_last_lr()[0],
                    )
                    self.track_data(
                        "Learning / Critic learning rate",
                        self.critic_scheduler.get_last_lr()[0],
                    )
