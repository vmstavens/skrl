import copy
import itertools
from typing import Any, Dict, Optional, Tuple, Union

import gym
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.ma.core import argmax

from performance import timer
from skrl import logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import postprocessing

from .IBRLbase import Agent

IBRL_DEFAULT_CONFIG = {
    "gradient_steps": 1,  # gradient steps
    "batch_size": 256,  # training batch size
    "discount_factor": 0.99,  # discount factor (gamma)
    "polyak": 0.005,  # soft update hyperparameter (tau)
    "soft_update_beta": 1,
    "actor_learning_rate": 1e-4,  # actor learning rate, Paper:1e-4
    "critic_learning_rate": 1e-4,  # critic learning rate, Paper:1e-4
    "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "random_timesteps": 0,  # random exploration steps
    "learning_starts": 0,  # learning starts after this many steps
    "grad_norm_clip": 0,  # clipping coefficient for the norm of the gradients
    "exploration": {
        "noise": None,  # exploration noise
        "initial_scale": 1.0,  # initial scale for the noise
        "final_scale": 1e-3,  # final scale for the noise
        "timesteps": None,  # timesteps for the noise decay
    },
    "policy_delay": 2,  # policy delay update with respect to critic update
    "smooth_regularization_noise": None,  # smooth noise for regularization
    "smooth_regularization_clip": 0.1,  # clip for smooth regularization
    # RED-Q specific parameters
    "RED-Q_enable": True,  # use REDQ?
    "ensemble_size": 5,  # number of critics in ensemble (N)
    "critic_subset_size": 2,  # number of critics to sample for target computation (M)
    "policy_subset_size": 2,  # number of critics to sample for policy updates
    "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "offline": False,
    "demo_file": "",
    "num_envs": 1,
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": 500,  # TensorBoard writing interval (timesteps)
        "checkpoint_interval": 1000,  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}


class IBRL(Agent):
    def __init__(
        self,
        models: Dict[str, Model],
        models_il: Dict[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        expert_memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[
            Union[int, Tuple[int], gym.Space, gymnasium.Space]
        ] = None,
        action_space: Optional[
            Union[int, Tuple[int], gym.Space, gymnasium.Space]
        ] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """IBRL(https://arxiv.org/abs/2311.02198)
        :param models_il: NN used for Imitation learning in IBRL paper, It is MLPs.
        :type models: dictionary of skrl.models.torch.Model
        :param models: NNs used by the RL agent, in IBRL paper, they are all MLPs.
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(IBRL_DEFAULT_CONFIG)
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

        self.init()
        # IL model
        self.IL_policy = self.models_il.get("policy", None)
        # models
        self.policy = self.models.get("policy", None)
        self.target_policy = self.models.get("target_policy", None)
        # self.RED_Q = self.cfg["RED-Q_enable"]
        # if self.RED_Q:
        #     # RED-Q: Random ensemble of critics
        #     self.critics = self.models.get("critics", [])
        #     self.target_critics = self.models.get("target_critics", [])
        # else:
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.critics = [self.critic_1, self.critic_2]
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)
        self.target_critics = [self.target_critic_1, self.target_critic_2]

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        for i, critic in enumerate(self.critics):
            self.checkpoint_modules[f"critic_{i}"] = critic
        for i, target_critic in enumerate(self.target_critics):
            self.checkpoint_modules[f"target_critic_{i}"] = target_critic

        if self.target_policy is not None and len(self.target_critics) > 0:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            for target_critic in self.target_critics:
                target_critic.freeze_parameters(True)

            # update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            for i, target_critic in enumerate(self.target_critics):
                target_critic.update_parameters(self.critics[i], polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._soft_update_beta = self.cfg["soft_update_beta"]
        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._policy_delay = self.cfg["policy_delay"]
        self._critic_update_counter = 0

        self._smooth_regularization_noise = self.cfg["smooth_regularization_noise"]
        self._smooth_regularization_clip = self.cfg["smooth_regularization_clip"]

        # RED-Q parameters
        self._ensemble_size = self.cfg["ensemble_size"]
        self._critic_subset_size = self.cfg["critic_subset_size"]
        self._policy_subset_size = self.cfg["policy_subset_size"]

        self._offline = self.cfg["offline"]
        self._num_envs = self.cfg["num_envs"]
        self._demo_file = self.cfg["demo_file"]
        self._rewards_shaper = self.cfg["rewards_shaper"]

        # set up optimizers and learning rate schedulers
        if self.policy is not None and len(self.critics) > 0:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self._actor_learning_rate
            )
            # self.IL_policy_optimizer = torch.optim.Adam(self.IL_policy.parameters(), lr=self._actor_learning_rate)

            # Create optimizer for all critics in ensemble
            all_critic_params = []
            for critic in self.critics:
                all_critic_params.extend(critic.parameters())
            self.critic_optimizer = torch.optim.Adam(
                all_critic_params, lr=self._critic_learning_rate
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

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent"""
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
            # if not self._offline:
            #     exp_memory = postprocessing.MemoryFileIterator(self._demo_file)
            #     for k, data0 in exp_memory:
            #         # self.expert_memory.add_samples(d)
            #         keys = list(data0.keys())
            #         N = len(data0[keys[0]])
            #         for i in range(0, N):
            #             self.memory.add_samples(
            #                 states=torch.Tensor(np.array(data0[keys[3]][i])),
            #                 actions=torch.Tensor(np.array(data0[keys[0]][i])),
            #                 rewards=torch.Tensor(np.array(data0[keys[2]][i])),
            #                 next_states=torch.Tensor(np.array(data0[keys[1]][i])),
            #                 terminated=torch.Tensor(np.array(data0[keys[4]][i])),
            #             )
            #     print("load memory successfully")

            # create tensors in memory
            # # self.expert_memory
            # if self.expert_memory is not None:
            #     self.expert_memory.create_tensor(
            #         name="states", size=self.observation_space, dtype=torch.float32
            #     )
            #     self.expert_memory.create_tensor(
            #         name="next_states", size=self.observation_space, dtype=torch.float32
            #     )
            #     self.expert_memory.create_tensor(
            #         name="actions", size=self.action_space, dtype=torch.float32
            #     )
            #     self.expert_memory.create_tensor(
            #         name="rewards", size=1, dtype=torch.float32
            #     )
            #     self.expert_memory.create_tensor(
            #         name="terminated", size=1, dtype=torch.bool
            #     )
            #     self._tensors_names = [
            #         "states",
            #         "actions",
            #         "rewards",
            #         "next_states",
            #         "terminated",
            #     ]
            #     expert_memory = postprocessing.MemoryFileIterator(self._demo_file)

            #     for j, data in expert_memory:
            #         print(f"{j=}")
            #         # self.expert_memory.add_samples(d)
            #         keys = list(data.keys())
            #         N = len(data[keys[0]])
            #         for i in range(0, N):
            #             self.expert_memory.add_samples(
            #                 states=torch.Tensor(np.array(data[keys[3]][i])),
            #                 actions=torch.Tensor(np.array(data[keys[0]][i])),
            #                 rewards=torch.Tensor(np.array(data[keys[2]][i])),
            #                 next_states=torch.Tensor(np.array(data[keys[1]][i])),
            #                 terminated=torch.Tensor(np.array(data[keys[4]][i])),
            #             )
            # self.expert_memory.load("/home/chen/Downloads/memories/2000_new.pt")
            # self.expert_memory.save("/home/chen/Downloads/memories", "csv")
            print("load expert memory successfully")

        # clip noise bounds
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(
                self.action_space.low, device=self.device
            )
            self.clip_actions_max = torch.tensor(
                self.action_space.high, device=self.device
            )

        # backward compatibility: torch < 1.9 clamp method does not support tensors
        self._backward_compatibility = tuple(
            map(int, (torch.__version__.split(".")[:2]))
        ) < (1, 9)

    def _select_act(
        self, obs: torch.Tensor, exp_obs: torch.Tensor, soft: bool, target: bool
    ):
        with timer("select act"):
            if target:
                # target policy smoothing
                rl_actions, _, _ = self.target_policy.act(
                    {"states": obs}, role="target_policy"
                )

                noises = torch.clamp(
                    self._smooth_regularization_noise.sample(rl_actions.shape),
                    min=-self._smooth_regularization_clip,
                    max=self._smooth_regularization_clip,
                )
                rl_actions.add_(noises)
                if self._backward_compatibility:
                    rl_actions = torch.max(
                        torch.min(rl_actions, self.clip_actions_max),
                        self.clip_actions_min,
                    )
                else:
                    rl_actions.clamp_(
                        min=self.clip_actions_min, max=self.clip_actions_max
                    )

            else:
                rl_actions, _, outputs = self.policy.act(
                    {"states": self._state_preprocessor(obs)}, role="policy"
                )

            # Get IL actions
            self.IL_policy.eval()
            # TODO: OBS ibrl also needs this
            il_actions, _, _ = self.IL_policy.act(
                {"states": self._state_preprocessor(obs)}, role="policy"
            )  ## ibrl
            # exp_il_actions, _, _ = self.IL_policy.act(
            #     {"states": self._state_preprocessor(exp_obs)}, role="policy"
            # )  ## drlr

            il_actions, _, _ = self.IL_policy.act(
                {"states": self._state_preprocessor(obs)}, role="policy"
            )

            # Stack actions and get batch dimensions
            rl_bc_actions = torch.stack([rl_actions, il_actions], dim=1)
            batch_size, num_action, _ = (
                rl_bc_actions.size()
            )  # get dimensions values, bsize:batch size

            # Compute min Q-values for both policies
            target_q_rl = self._compute_min_q_values(obs, rl_actions)
            target_q_il = self._compute_min_q_values(obs, il_actions)
            # target_q_il = self._compute_min_q_values(exp_obs, exp_il_actions)

            # Stack Q-values
            target_q_values = torch.stack([target_q_rl, target_q_il], dim=1).view(
                batch_size, num_action
            )

            # Select actions based on strategy
            if soft:
                # Boltzmann exploration
                # probs = F.softmax(target_q_values * self._soft_update_beta, dim=1)
                probs = torch.nn.functional.softmax(
                    target_q_values * self._soft_update_beta, dim=1
                )
                action_indices = probs.multinomial(1)
                actions = (
                    rl_actions * (1 - action_indices) + il_actions * action_indices
                )
            else:
                # Greedy selection
                action_indices = target_q_values.argmax(dim=1)
                actions = rl_bc_actions[torch.arange(batch_size), action_indices]

            # TODO: Obs here we overwrite actions to be IL for testing rewards performance
            # actions = il_actions

            if not target:
                self.track_data(
                    "Q-network / select_rl_Q (mean)", torch.mean(target_q_rl).item()
                )
                self.track_data(
                    "Q-network / select_il_Q (mean)", torch.mean(target_q_il).item()
                )
                return actions, _, outputs
            else:
                return actions, _, _

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Compute RL action a_RL = pi(s_t)+ rand, and compute IL action a_IL = miu(s_t)

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        with timer("act"):
            # sample random actions, warming up
            if timestep < self._random_timesteps:
                return self.policy.random_act(
                    {"states": self._state_preprocessor(states)}, role="policy"
                )

            # for pretrain stage, still need exploration noises
            if (
                self._random_timesteps
                < timestep
                < self._learning_starts + self._random_timesteps
            ):
                actions, _, outputs = self.policy.act(
                    {"states": self._state_preprocessor(states)}, role="policy"
                )
                # add exloration noise
                if self._exploration_noise is not None:
                    # sample noises
                    noises = self._exploration_noise.sample(actions.shape)

                    # define exploration timesteps
                    scale = self._exploration_final_scale
                    if self._exploration_timesteps is None:
                        self._exploration_timesteps = timesteps

                    # apply exploration noise
                    if timestep <= self._exploration_timesteps:
                        scale = (1 - timestep / self._exploration_timesteps) * (
                            self._exploration_initial_scale
                            - self._exploration_final_scale
                        ) + self._exploration_final_scale
                        noises.mul_(scale)

                        # modify actions
                        actions.add_(noises)
                        if self._backward_compatibility:
                            actions = torch.max(
                                torch.min(actions, self.clip_actions_max),
                                self.clip_actions_min,
                            )
                        else:
                            actions.clamp_(
                                min=self.clip_actions_min, max=self.clip_actions_max
                            )
                return actions, _, outputs

            (
                expert_states_r,
                expert_actions_r,
                expert_rewards_r,
                expert_next_states_r,
                expert_dones_r,
            ) = self.expert_memory.sample(
                names=self._tensors_names, batch_size=self._num_envs
            )[0]

            # select actions
            # actions, _, outputs = self._select_act(
            #     states, expert_states_r, soft=False, target=False
            # )  ## drlr act_select

            actions, _, outputs = self._select_act(
                states, states, soft=False, target=False
            )  ## original ibrl

            # actions, _, _ = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")

            # add exloration noise
            if self._exploration_noise is not None:
                # sample noises
                noises = self._exploration_noise.sample(actions.shape)

                # define exploration timesteps
                scale = self._exploration_final_scale
                if self._exploration_timesteps is None:
                    self._exploration_timesteps = timesteps

                # apply exploration noise
                if timestep <= self._exploration_timesteps:
                    scale = (1 - timestep / self._exploration_timesteps) * (
                        self._exploration_initial_scale - self._exploration_final_scale
                    ) + self._exploration_final_scale
                    noises.mul_(scale)

                    # modify actions
                    actions.add_(noises)
                    if self._backward_compatibility:
                        actions = torch.max(
                            torch.min(actions, self.clip_actions_max),
                            self.clip_actions_min,
                        )
                    else:
                        actions.clamp_(
                            min=self.clip_actions_min, max=self.clip_actions_max
                        )

                    # record noises
                    self.track_data(
                        "Exploration / Exploration noise (max)",
                        torch.max(noises).item(),
                    )
                    self.track_data(
                        "Exploration / Exploration noise (min)",
                        torch.min(noises).item(),
                    )
                    self.track_data(
                        "Exploration / Exploration noise (mean)",
                        torch.mean(noises).item(),
                    )

                else:
                    # record noises
                    self.track_data("Exploration / Exploration noise (max)", 0)
                    self.track_data("Exploration / Exploration noise (min)", 0)
                    self.track_data("Exploration / Exploration noise (mean)", 0)

            return actions, _, outputs

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

        with timer("record transition"):
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

            # if timestep > self._random_timesteps + self._learning_starts - 1:
            if self.memory is not None:
                # reward shaping
                if self._rewards_shaper is not None:
                    rewards = self._rewards_shaper(rewards, timestep, timesteps)

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
        # if timestep == timesteps-1:
        #     self.memory.save("/home/chen/Downloads/new/eval", "csv")
        # print("Recording memory from record_transition...")
        # self.memory.save("/home/chen/Downloads", "csv")

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _compute_min_q_values(self, states, actions):
        """Helper to compute target Q-values using both critics"""
        # RED-Q: compute target values using ensemble

        # # Compute target values
        # if self.RED_Q:
        #     # RED-Q: randomly sample subset of critics to compute target Q value
        #     random_critic_indices = torch.randperm(len(self.critics))[
        #         : self._critic_subset_size
        #     ]
        #     target_q_values_list = []
        #     for idx in random_critic_indices:
        #         target_q_val, _, _ = self.target_critics[idx].act(
        #             {"states": states, "taken_actions": actions},
        #             role=f"target_critic_{idx}",
        #         )
        #         target_q_values_list.append(target_q_val)
        # else:
        target_q_values_list = []
        for idx in [0, 1]:
            target_q_val, _, _ = self.target_critics[idx].act(
                {"states": states, "taken_actions": actions},
                role=f"target_critic_{idx + 1}",
            )
            target_q_values_list.append(target_q_val)
        target_q_values = torch.stack(target_q_values_list, dim=0)
        target_q_value = torch.min(target_q_values, dim=0)[0]
        return target_q_value

    def _update(
        self,
        timestep: int,
        timesteps: int,
        pre_train=False,
        soft=False,
        dynamics_bc_loss=True,
        experience_buffer_ratio=1,
        expert_buffer_ratio=1,
    ) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        with timer("update"):
            # gradient steps
            for gradient_step in range(self._gradient_steps):
                # warm up
                if timestep < self._random_timesteps:
                    pass
                if self._offline:
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
                    # sample a batch from memory
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
                        expert_states_r,
                        expert_actions_r,
                        expert_rewards_r,
                        expert_next_states_r,
                        expert_dones_r,
                    ) = self.expert_memory.sample(
                        names=self._tensors_names,
                        batch_size=int(self._batch_size * expert_buffer_ratio),
                    )[0]

                with torch.no_grad():
                    sampled_states = self._state_preprocessor(
                        sampled_states, train=True
                    )
                    sampled_next_states = self._state_preprocessor(
                        sampled_next_states, train=True
                    )

                    with torch.no_grad():
                        if self._offline:
                            # target policy smoothing
                            next_actions, _, _ = self.target_policy.act(
                                {"states": sampled_next_states}, role="target_policy"
                            )
                            noises = torch.clamp(
                                self._smooth_regularization_noise.sample(
                                    next_actions.shape
                                ),
                                min=-self._smooth_regularization_clip,
                                max=self._smooth_regularization_clip,
                            )
                            next_actions.add_(noises)
                        else:
                            # select next actions
                            next_actions, _, _ = self._select_act(
                                sampled_next_states,
                                expert_states_r,
                                soft=False,
                                target=True,
                            )  # IBRL action selection module

                        # Compute final target values
                        target_qvalues = self._compute_min_q_values(
                            sampled_next_states, next_actions
                        )
                        target_values = (
                            sampled_rewards
                            + self._discount_factor
                            * sampled_dones.logical_not()
                            * target_qvalues
                        )
                        # Plot discount value for debugging
                        discount_values = (
                            self._discount_factor
                            * sampled_dones.logical_not()
                            * target_qvalues
                        )

                # compute critic loss for ensemble
                critic_values, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": sampled_actions},
                    role="critic_1",
                )

                critic_loss = F.mse_loss(critic_values, target_values)

                # optimization step (critic)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                if self._grad_norm_clip > 0:
                    # Clip gradients for all critics in ensemble
                    all_critic_params = []
                    for critic in self.critics:
                        all_critic_params.extend(critic.parameters())
                    nn.utils.clip_grad_norm_(all_critic_params, self._grad_norm_clip)
                self.critic_optimizer.step()

                # delayed update
                self._critic_update_counter += 1
                if not self._critic_update_counter % self._policy_delay:
                    actions, _, _ = self.policy.act(
                        {"states": sampled_states}, role="policy"
                    )
                    il_actions, _, _ = self.IL_policy.act(
                        {"states": sampled_states}, role="policy"
                    )

                    # randomly sample subset of critics for policy update
                    policy_critic_indices = torch.randperm(len(self.critics))[
                        : self._policy_subset_size
                    ]
                    policy_critic_values_list = []
                    il_policy_critic_values_list = []
                    for idx in policy_critic_indices:
                        critic_values, _, _ = self.critics[idx].act(
                            {"states": sampled_states, "taken_actions": actions},
                            role=f"critic_{idx}",
                        )
                        il_critic_values, _, _ = self.critics[idx].act(
                            {"states": sampled_states, "taken_actions": il_actions},
                            role=f"critic_{idx}",
                        )
                        policy_critic_values_list.append(critic_values)
                        il_policy_critic_values_list.append(il_critic_values)

                    avg_critic_values = torch.stack(
                        policy_critic_values_list, dim=0
                    ).mean(dim=0)
                    actor_loss = -avg_critic_values.mean()

                    # L2 regularization
                    lambda_l2 = 0
                    l2_penalty = sum(
                        torch.sum(param**2) for param in self.policy.parameters()
                    )
                    loss_l2 = lambda_l2 * l2_penalty

                    if self._offline:
                        bc_loss = F.mse_loss(actions, sampled_actions)
                        policy_loss = actor_loss + 1 * bc_loss
                    else:
                        actions_bc, _, _ = self.policy.act(
                            {"states": sampled_states}, role="policy"
                        )
                        bc_loss = F.mse_loss(
                            actions_bc, sampled_actions
                        )  # To measure OOD actions
                        policy_loss = actor_loss

                    # optimization step (policy)
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self._grad_norm_clip
                        )
                    self.policy_optimizer.step()

                # update target networks
                for i, target_critic in enumerate(self.target_critics):
                    target_critic.update_parameters(
                        self.critics[i], polyak=self._polyak
                    )
                self.target_policy.update_parameters(self.policy, polyak=self._polyak)

                # update learning rate
                if self._learning_rate_scheduler:
                    self.policy_scheduler.step()
                    self.critic_scheduler.step()

                # record data
                if not self._critic_update_counter % self._policy_delay:
                    self.track_data("Loss / Policy loss", policy_loss.item())
                    self.track_data("Loss / BC loss", bc_loss.item())
                    self.track_data("Loss / Critic loss", critic_loss.item())

                    self.track_data(
                        "Q-network / Q (max)", torch.max(avg_critic_values).item()
                    )
                    self.track_data(
                        "Q-network / Q (min)", torch.min(avg_critic_values).item()
                    )
                    self.track_data(
                        "Q-network / Q (mean)", torch.mean(avg_critic_values).item()
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
                    "Target / discount (mean)", torch.mean(discount_values).item()
                )
                self.track_data(
                    "Target / sampled_rewards (mean)",
                    torch.mean(sampled_rewards).item(),
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
