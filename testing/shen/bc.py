import copy
import itertools
from typing import Any, Dict, Optional, Tuple, Union

import gym
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import postprocessing

from .base import Agent

BC_DEFAULT_CONFIG = {
    "gradient_steps": 1,  # gradient steps
    "batch_size": 256,  # training batch size
    "discount_factor": 0.99,  # discount factor (gamma)
    "polyak": 0.005,  # soft update hyperparameter (tau)
    "actor_learning_rate": 1e-3,  # actor learning rate
    "learning_rate_scheduler": None,  # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},  # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})
    "state_preprocessor": None,  # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},  # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "random_timesteps": 0,  # random exploration steps
    "learning_starts": 0,  # learning starts after this many steps
    "grad_norm_clip": 0,  # clipping coefficient for the norm of the gradients
    "exploration": {
        "noise": None,  # exploration noise
        "initial_scale": 0.5,  # initial scale for the noise
        "final_scale": 1e-3,  # final scale for the noise
        "timesteps": None,  # timesteps for the noise decay
    },
    "smooth_regularization_noise": None,  # smooth noise for regularization
    "smooth_regularization_clip": 0.1,  # clip for smooth regularization
    "demo_file": "",
    "rewards_shaper": None,  # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": 50,  # TensorBoard writing interval (timesteps)
        "checkpoint_interval": 100,  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}


class BC(Agent):
    def __init__(
        self,
        models: Dict[str, Model],
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
        """Behavior Clone (BC)

        :param models: Models used by the agent
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
        _cfg = copy.deepcopy(BC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            expert_memory=expert_memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        self._tensors_names = self.expert_memory.get_tensor_names()

        # models
        self.policy = self.models.get("policy", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]
        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._smooth_regularization_noise = self.cfg["smooth_regularization_noise"]
        self._smooth_regularization_clip = self.cfg["smooth_regularization_clip"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._demo_file = self.cfg["demo_file"]

        # set up optimizers and learning rate schedulers
        if self.policy is not None:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self._actor_learning_rate
            )
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer

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
        # # self.expert_memory
        # if self.expert_memory is not None:
        #     self.expert_memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
        #     self.expert_memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
        #     self.expert_memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
        #     self.expert_memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
        #     self.expert_memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

        #     self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated"]
        #     expert_memory = postprocessing.MemoryFileIterator(self._demo_file)
        #     for j, data in expert_memory:
        #         keys = list(data.keys())
        #         N = len(data[keys[0]])
        #         for i in range(0, N):
        #             self.expert_memory.add_samples(states=torch.Tensor(np.array(data[keys[3]][i])),
        #                                            actions=torch.Tensor(np.array(data[keys[0]][i])),
        #                                            rewards=torch.Tensor(np.array(data[keys[2]][i])),
        #                                            next_states=torch.Tensor(np.array(data[keys[1]][i])),
        #                                            terminated=torch.Tensor(np.array(data[keys[4]][i])))
        # self.expert_memory.load("/home/chen/Downloads/memories/2000_new.pt")
        # self.expert_memory.save("/home/chen/Downloads/memories", "csv")
        print("load memory successfully")

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
        # sample deterministic actions
        actions, _, _ = self.policy.act(
            {"states": self._state_preprocessor(states)}, role="policy"
        )

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

                # record noises
                self.track_data(
                    "Exploration / Exploration noise (max)", torch.max(noises).item()
                )
                self.track_data(
                    "Exploration / Exploration noise (min)", torch.min(noises).item()
                )
                self.track_data(
                    "Exploration / Exploration noise (mean)", torch.mean(noises).item()
                )

            else:
                # record noises
                self.track_data("Exploration / Exploration noise (max)", 0)
                self.track_data("Exploration / Exploration noise (min)", 0)
                self.track_data("Exploration / Exploration noise (mean)", 0)

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

        # #Todo add flag for generating BC policies
        # # storage transition in memory
        # self.expert_memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
        #                             terminated=terminated, truncated=truncated)
        # # self.eval_memory.save("/home/chen/Downloads/new", "pt")
        if timestep == timesteps - 1:
            self.expert_memory.save("./Demos", "csv")
        # pass

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
        self.set_mode("train")
        self._update(timestep, timesteps)
        self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # Sample expert buffer
        (
            sampled_states,
            sampled_actions,
            sampled_rewards,
            sampled_next_states,
            sampled_dones,
        ) = self.expert_memory.sample(
            names=self._tensors_names, batch_size=self._batch_size
        )[0]

        # print(sampled_states)

        # gradient steps
        for gradient_step in range(self._gradient_steps):
            # print('sampled_states', sampled_states)
            sampled_states = self._state_preprocessor(
                sampled_states, train=not gradient_step
            )
            # print('sampled_states', sampled_states)

            # sample noises
            noises = self._exploration_noise.sample(sampled_states.shape)
            sampled_states.add_(noises)

            # Compute actor loss
            actions, _, _ = self.policy.act({"states": sampled_states}, role="policy")
            bc_loss = F.mse_loss(actions, sampled_actions)
            # print('actions', actions)
            # print('sampled_actions', sampled_actions)

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            bc_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
            self.policy_optimizer.step()

            # update target networks
            self.policy.update_parameters(self.policy, polyak=self._polyak)

            # update learning rate.
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()

            self.track_data("Loss / Policy loss", bc_loss.item())

            if self._learning_rate_scheduler:
                self.track_data(
                    "Learning / Policy learning rate",
                    self.policy_scheduler.get_last_lr()[0],
                )
