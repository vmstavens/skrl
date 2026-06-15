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
DRLR_SAC_DEFAULT_CONFIG = {
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

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

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
    "soft_update_beta": 0.2,         # use small for good IL (small ~0.2) original IBRL used 10
    # "soft_update_beta": 1,         # This is the inverse temperature 
    # "soft_update_beta": 10,         # use small for good IL (small ~0.2) original IBRL used 10
    # "soft_update_beta": 0.2,         # use small for good IL (small ~0.2) original IBRL used 10
    "actor": "both",                # should either be "rl", "il" or "both"
}
# [end-config-dict-torch]
# fmt: on


class DRLR(Agent):
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
        _cfg = copy.deepcopy(DRLR_SAC_DEFAULT_CONFIG)
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

        self.expert_mean_states = 0
        self.expert_cov_states = 0

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
        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]
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
        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.clip_actions_min = torch.tensor(
                self.action_space.low, device=self.device, dtype=torch.float32
            )
            self.clip_actions_max = torch.tensor(
                self.action_space.high, device=self.device, dtype=torch.float32
            )
        else:
            self.clip_actions_min = None
            self.clip_actions_max = None

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
        expert_rewards = self.expert_memory.sample(
            names=["rewards"], batch_size=self._batch_size
        )[0][0]

        # compute the expert mean and covariance of states based on a sampled batch
        self.expert_mean_states  = torch.mean(expert_states, dim=0)
        self.expert_mean_rewards = torch.mean(expert_rewards, dim=0)
        cov = torch.cov(expert_states.T)
        self.expert_cov_states = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0], device= cov.device))

        # ---------------------
        # expert_returns = self.expert_memory.sample(names=["returns"],
        #                                             batch_size=N)[0][0]

        # self.demo_mean = torch.mean(expert_returns, dim=0)

        # expert_states_r = self.expert_memory.sample(names=["observations"],
        #                                             batch_size=N-500)[0][0]
        # self.demo_mean0 = torch.mean(expert_states_r, dim=0)
        # cov = torch.cov(expert_states_r.T)
        # self.demo_cov = torch.inverse(cov + 1e-6 * torch.eye(cov.shape[0], device= cov.device))


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
            {"states": self._state_preprocessor(exp_obs)},
            role="policy",
        )

        # act out the 0'th index
        il_actions = il_actions[:, 0, :]

        # scale il_actions
        il_actions = il_actions * self._il_ctrl_scale
        rl_actions = rl_actions * self._rl_ctrl_scale

        if len(exp_obs.shape) == 3:
            exp_obs = exp_obs[:, -1, :]

        target_q_il = self._compute_min_q_values(exp_obs, il_actions)
        target_q_rl = self._compute_min_q_values(rl_obs, rl_actions)

        if torch.mean(target_q_il) > torch.mean(target_q_rl):
            # IL wins: reuse the single-step IL action to keep shape (num_envs, a_dim)
            il_actions, _, _ = self.IL_policy.act(
                {"states": self._state_preprocessor(il_obs)}, role="policy"
            )
            actions = il_actions
            self.track_data("Which / Actor", 1)
        else:
            self.track_data("Which / Actor", 0)
            actions = rl_actions

        if len(actions.shape) == 3:
            actions = actions[:,0,:]
    
        if not target:
            self.track_data("Q-network / select_rl_Q (mean)", torch.mean(target_q_rl).item())
            self.track_data("Q-network / select_il_Q (mean)", torch.mean(target_q_il).item())
            self.track_data("Q-network / debug (mean)", self.expert_mean_rewards.item())
            il_selected = (torch.mean(target_q_il) > torch.mean(target_q_rl)).float()
            self.track_data("Online / IL selection probability", il_selected.item())
            return actions, _, outputs
        else:
            il_selected = (torch.mean(target_q_il) > torch.mean(target_q_rl)).float()
            # -------
            self.track_data("Bootstrap / select_rl_Q (mean)", torch.mean(target_q_rl).item())
            self.track_data("Bootstrap / select_il_Q (mean)", torch.mean(target_q_il).item())
            self.track_data("Bootstrap / IL selection probability", il_selected.item())
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

        if timestep < self._warmup_timesteps:
            il_states = torch.stack([self._prev_states, self._states], axis=1)
            il_actions, _, _ = self.IL_policy.act(
                {"states": self._state_preprocessor(il_states)}, role="policy"
            )
            return il_actions[:,0,:], None, None


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
        diff = states - self.expert_mean_states
        left = torch.matmul(diff, self.expert_cov_states)
        dist_sq = (left * diff).sum(dim=1)
        M_dist = torch.sqrt(dist_sq)

        self.track_data("Loss / states BC loss", torch.mean(M_dist).item())

        il_states = torch.stack([self._prev_states, self._states], axis=1)
        il_expert_states = torch.stack([expert_states, expert_next_states], axis=1)
 
        # Here, when self._decision_block = False, then expert_states is never used!

        actions, _, output = self._select_act(
            rl_obs=states,
            il_obs=il_states,
            exp_obs=il_expert_states,
            soft=True,
            target=False,
            timestep=timestep,
        )

        return actions, None, output

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

        target_q_value = torch.min(target_q_values, dim=1).values.unsqueeze(-1)
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

                with torch.no_grad():
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

                self.track_data(
                    "Loss / self._entropy_coefficient * log_prob",
                    (self._entropy_coefficient * log_prob).mean().item(),
                )
                self.track_data(
                    "Loss / torch.min(critic_1_values, critic_2_values)",
                    (torch.min(critic_1_values, critic_2_values)).mean().item(),
                )

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            if self._learn_entropy:
                self.track_data("Loss / Target Entropy", self._target_entropy)
                self.track_data("Loss / Log Prob", log_prob.mean().item())
                self.track_data(
                    "Loss / Log Entropy Coeff", self.log_entropy_coefficient.item()
                )

            # entropy learning
            if self._learn_entropy:
                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision
                ):
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
                self.track_data(
                    "Target / Target Q (max)", torch.max(target_q_values).item()
                )
                self.track_data(
                    "Target / Target Q (min)", torch.min(target_q_values).item()
                )
                self.track_data(
                    "Target / Target Q (mean)", torch.mean(target_q_values).item()
                )
                self.track_data(
                    "Target / Next log prob (max)", torch.max(next_log_prob).item()
                )
                self.track_data(
                    "Target / Next log prob (min)", torch.min(next_log_prob).item()
                )
                self.track_data(
                    "Target / Next log prob (mean)", torch.mean(next_log_prob).item()
                )

                self.track_data("Loss / actions BC loss", bc_loss.item())

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
                        self.track_data(
                            "Policy / Log std (max)", torch.max(policy_log_std).item()
                        )
                        self.track_data(
                            "Policy / Log std (min)", torch.min(policy_log_std).item()
                        )
                        self.track_data(
                            "Policy / Log std (mean)", torch.mean(policy_log_std).item()
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
