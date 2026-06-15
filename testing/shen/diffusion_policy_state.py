import logging
import os
from collections import deque
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import randn_tensor

# from utils.dpy.dpy.diffusion_policy_state import DiffusionPolicyState
# from utils.dpy.models import state as mod
# from utils.dpy.models.state import EMAModel
# from utils.dpy.utils import cfg
# from utils.model import ModuleWrapper
from skrl.agents.torch.base import Agent, Model

from .dp_models import ConditionalUnet1D, EMAModel, ModuleWrapper

logging.basicConfig(level=logging.INFO)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.WARNING)
# Prevent propagation to root logger to avoid duplicate handling
logger.propagate = False

DIFFUSION_POLICY_STATE_DEFAULT_CONFIG = {
    # Model architecture
    # "input_dim": 2,
    # "global_cond_dim": 10,  # obs_horizon * obs_dim
    "diffusion_step_embed_dim": 256,
    "down_dims": [256, 512, 1024],
    "kernel_size": 5,
    "n_groups": 8,
    # Training
    "pred_horizon": 16,
    "obs_horizon": 2,
    "action_horizon": 8,
    "num_diffusion_iters": 100,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "ema_power": 0.75,
    # "num_workers": 0,
    "num_workers": 1,
    "num_epochs": 100,
    "max_steps": 200,
    # "max_steps": 200,
    "eval_frequency": 10,
    # Scheduler
    "beta_schedule": "squaredcos_cap_v2",
    "clip_sample": True,
    "prediction_type": "epsilon",
    "lr_scheduler_cfg": {"num_warmup_steps": 500, "num_training_steps": 10_000},
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


# class DiffusionPolicy(Agent):
#     def __init__(
#         self,
#         models: dict[str, Model],
#         ema: EMAModel,
#         device: str = "cuda",
#         observation_space=None,
#         dataloader=None,
#         action_space=None,
#         memory=None,
#         config=None,
#     ):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.config = {**DIFFUSION_POLICY_STATE_DEFAULT_CONFIG, **(config or {})}

#         # Validate config
#         # if not cfg.is_valid(DIFFUSION_POLICY_STATE_DEFAULT_CONFIG, self.config):
#         #     raise ValueError("Invalid config for DiffusionPolicy")

#         # Load config values
#         self._num_diffusion_iters: int = self.config["num_diffusion_iters"]
#         self._beta_schedule: str = self.config["beta_schedule"]
#         self._clip_sample: bool = self.config["clip_sample"]
#         self._prediction_type: str = self.config["prediction_type"]
#         self._ema_power: float = self.config["ema_power"]
#         self._learning_rate: float = self.config["learning_rate"]
#         self._weight_decay: float = self.config["weight_decay"]
#         self._num_warmup_steps: int = self.config["lr_scheduler_cfg"][
#             "num_warmup_steps"
#         ]
#         self._num_training_steps: int = self.config["lr_scheduler_cfg"][
#             "num_training_steps"
#         ]
#         self._obs_horizon: int = self.config["obs_horizon"]
#         self._pred_horizon: int = self.config["pred_horizon"]

#         # self._num_envs = self.config["num_envs"]

#         # Save models
#         self.models = {k: ModuleWrapper(v).to(device) for k, v in models.items()}
#         self.model = self.models["model"]
#         self.ema_model = self.models["ema_model"]
#         self.ema = ema

#         super().__init__(
#             self.models, memory, observation_space, action_space, device, config
#         )

#         # Noise scheduler
#         self.noise_scheduler = DDPMScheduler(
#             num_train_timesteps=self._num_diffusion_iters,
#             beta_schedule=self._beta_schedule,
#             clip_sample=self._clip_sample,
#             prediction_type=self._prediction_type,
#         )

#         # Optimizer + LR scheduler
#         self.optimizer = None
#         self.lr_scheduler = None
#         self.configure_optimizers(
#             dataloader=dataloader, num_epochs=self.config["num_epochs"]
#         )

#         # Register for checkpointing
#         self.checkpoint_modules = {
#             "policy": self.model,
#             "ema_model": self.ema_model,
#             "optimizer": self.optimizer,
#             "lr_scheduler": self.lr_scheduler,
#         }

#         self.is_trained = False
#         self.set_mode("train")

#         self.obs_queues = deque(maxlen=self._obs_horizon)
#         # self.obs_queues = [
#         #     deque(maxlen=self._obs_horizon) for i in range(self._num_envs)
#         # ]

#     def init(self, trainer_cfg=None):
#         self.set_mode("train")
#         super().init(trainer_cfg)

#     def configure_optimizers(self, dataloader=None, num_epochs=None):
#         if self.optimizer is None:
#             self.optimizer = torch.optim.AdamW(
#                 self.model.parameters(),
#                 lr=self._learning_rate,
#                 weight_decay=self._weight_decay,
#             )
#         if self.lr_scheduler is None:
#             num_training_steps = (
#                 len(dataloader) * num_epochs
#                 if dataloader and num_epochs
#                 else self._num_training_steps
#             )
#             self.lr_scheduler = get_scheduler(
#                 name="cosine",
#                 optimizer=self.optimizer,
#                 num_warmup_steps=self._num_warmup_steps,
#                 num_training_steps=num_training_steps,
#             )

#     def prepare_observation_condition(self, obs: torch.Tensor) -> torch.Tensor:
#         return obs[:, : self._obs_horizon, :].flatten(start_dim=1)

#     def _update(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
#         """Perform one supervised training step"""
#         # self.model.train()

#         if len(states.shape) == 2:
#             states = states.unsqueeze(0)
#         if len(actions.shape) == 2:
#             actions = actions.unsqueeze(0)

#         logger.info(f"{states.shape=}")

#         # Conditioning
#         obs_cond = self.prepare_observation_condition(states)

#         # Sample noise
#         noise = torch.randn_like(actions)
#         timesteps = torch.randint(
#             0, self._num_diffusion_iters, (actions.shape[0],), device=self.device
#         ).long()

#         # Add noise to actions
#         noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

#         inputs = {
#             "noisy_actions": noisy_actions,
#             "timestep": timesteps,
#             "global_cond": obs_cond,
#         }
#         logger.info(f"\t{noisy_actions.shape=}")
#         logger.info(f"\t{obs_cond.shape=}")
#         for k, v in inputs.items():
#             logger.info(k, v.shape)

#         # Predict noise
#         noise_pred, _, _ = self.model.act(inputs=inputs)

#         out, _, _ = self.model.act(inputs)

#         # Loss
#         # print(f"{type(noise_pred)=} | {type(noise)=}")
#         loss = nn.functional.mse_loss(noise_pred, noise)

#         if self.model.training:  # only train if in training mode
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             self.lr_scheduler.step()
#             self.ema.step(self.model.parameters())

#         # Optimize
#         # self.optimizer.zero_grad()
#         # loss.backward()
#         # self.optimizer.step()

#         # # EMA update
#         # self.ema.step(self.model.parameters())

#         return loss.detach()

#     def act(
#         self,
#         states: Union[torch.Tensor, dict],
#         timestep: int = 0,
#         timesteps: int = 0,
#         role: str = "policy",
#         num_inference_steps=None,
#     ):
#         """Generate actions using DDIM sampling"""

#         if isinstance(states, dict):
#             states = states["states"]

#         # if len(states.shape) == 2:
#         #     states = states.unsqueeze(0)

#         # (batch, obs_horizon, state)
#         # print(f"{states.shape=}")

#         # self.obs_queues.append(states)

#         # fill up obs buffer
#         # while len(self.obs_queues) < self.obs_queues.maxlen:
#         #     self.obs_queues.append(states)

#         # states = torch.tensor(self.obs_queues)
#         # states = torch.stack(list(self.obs_queues))

#         logger.info(f"{states.shape=}")
#         self.ema_model.eval()
#         num_inference_steps = num_inference_steps or self._num_diffusion_iters

#         obs_cond = self.prepare_observation_condition(states)
#         logger.info(f"{obs_cond.shape=}")
#         #  i want a obs cond that is (batch, hor, o_dim)

#         shape = (
#             states.shape[0],
#             self._pred_horizon,
#             self.model._unwrapped_module._input_dim,
#         )
#         logger.info(f"{shape=}")
#         noisy_actions = randn_tensor(shape, device=self.device)

#         self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

#         logger.info(f"{noisy_actions.shape=}")
#         logger.info(f"{obs_cond.shape=}")
#         for t in self.noise_scheduler.timesteps:
#             inputs = {
#                 "noisy_actions": noisy_actions,
#                 "timestep": t,
#                 "global_cond": obs_cond,
#             }
#             # quit()

#             noise_pred, _, _ = self.ema_model.act(inputs=inputs)
#             # print(f"{noise_pred.shape=}")
#             # noise_pred = self.ema_model.act(noisy_actions, t, global_cond=obs_cond)
#             noisy_actions = self.noise_scheduler.step(
#                 noise_pred, t, noisy_actions
#             ).prev_sample

#         return noisy_actions, None, {}

#     def _predict_action(self, o: torch.Tensor) -> torch.Tensor:
#         pass

#     def eval(self) -> None:
#         self.ema.copy_to(self.ema_model.parameters())
#         self.model.eval()
#         self.ema_model.eval()

#     def train(self) -> None:
#         self.model.train()
#         self.ema_model.train()

#     def to(self, device: str = "cuda"):
#         for _, v in self.models.items():
#             v.to(device)


class DiffusionPolicy(Agent):
    def __init__(
        self,
        a_dim: int,
        o_dim: int,
        models: dict[str, Model],
        ema: EMAModel,
        device: str = "cuda",
        observation_space=None,
        dataloader=None,
        action_space=None,
        memory=None,
        config=None,
        stats: Optional[dict[str, Any]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {**DIFFUSION_POLICY_STATE_DEFAULT_CONFIG, **(config or {})}
        self.stats = stats

        self._a_dim: int = a_dim
        self._o_dim: int = o_dim

        # Load config values
        self._num_diffusion_iters: int = self.config["num_diffusion_iters"]
        self._beta_schedule: str = self.config["beta_schedule"]
        self._clip_sample: bool = self.config["clip_sample"]
        self._prediction_type: str = self.config["prediction_type"]
        self._ema_power: float = self.config["ema_power"]
        self._learning_rate: float = self.config["learning_rate"]
        self._weight_decay: float = self.config["weight_decay"]
        self._num_warmup_steps: int = self.config["lr_scheduler_cfg"][
            "num_warmup_steps"
        ]
        self._num_training_steps: int = self.config["lr_scheduler_cfg"][
            "num_training_steps"
        ]
        self._obs_horizon: int = self.config["obs_horizon"]
        self._pred_horizon: int = self.config["pred_horizon"]
        self._act_horizon: int = self.config["action_horizon"]

        # Save models
        self.models = {k: ModuleWrapper(v) for k, v in models.items()}
        # self.models = {k: ModuleWrapper(v).to(device) for k, v in models.items()}
        self.model = self.models["model"]
        self.ema_model = self.models["ema_model"]
        self.ema = ema

        super().__init__(
            self.models, memory, observation_space, action_space, device, config
        )

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self._num_diffusion_iters,
            beta_schedule=self._beta_schedule,
            clip_sample=self._clip_sample,
            prediction_type=self._prediction_type,
        )

        # Optimizer + LR scheduler
        self.optimizer = None
        self.lr_scheduler = None
        self.configure_optimizers(
            dataloader=dataloader, num_epochs=self.config["num_epochs"]
        )

        # Register for checkpointing
        self.checkpoint_modules = {
            "policy": self.model,
            "ema_model": self.ema_model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }

        self.is_trained = False
        self.set_mode("train")

    def init(self, trainer_cfg=None):
        self.set_mode("train")
        super().init(trainer_cfg)

    def configure_optimizers(self, dataloader=None, num_epochs=None):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay,
            )
        if self.lr_scheduler is None:
            num_training_steps = (
                len(dataloader) * num_epochs
                if dataloader and num_epochs
                else self._num_training_steps
            )
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=self._num_warmup_steps,
                num_training_steps=num_training_steps,
            )

    def prepare_observation_condition(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, : self._obs_horizon, :].flatten(start_dim=1)

    def _update(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Perform one supervised training step"""
        # self.model.train()

        if len(states.shape) == 2:
            states = states.unsqueeze(1)
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(1)

        # Conditioning
        obs_cond = self.prepare_observation_condition(states)
        # print(f"{obs_cond.shape=}")

        # Sample noise
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self._num_diffusion_iters, (actions.shape[0],), device=self.device
        ).long()

        # Add noise to actions
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        inputs = {
            "actions": noisy_actions,
            "timestep": timesteps,
            "global_cond": obs_cond,
        }

        noise_pred, _, _ = self.model.act(inputs=inputs)

        out, _, _ = self.model.act(inputs)

        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        if self.model.training:  # only train if in training mode
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.ema.step(self.model.parameters())

        # Optimize
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # # EMA update
        # self.ema.step(self.model.parameters())

        return loss.detach()

    def act(
        self,
        states: Union[torch.Tensor, dict],
        timestep: int = 0,
        timesteps: int = 0,
        role: str = "policy",
        num_inference_steps=None,
        normalize_obs: Optional[bool] = None,
        unnormalize_act: Optional[bool] = None,
    ) -> tuple[torch.Tensor, None, dict]:
        """Generate actions using DDIM sampling

        expected dimensions [num_envs, obs_horizon, o_dim]

        """

        if isinstance(states, dict):
            states = states["states"]

        if not torch.is_tensor(states):
            states = torch.as_tensor(states, device=self.device, dtype=torch.float32)

        if self.stats is not None:
            do_normalize = normalize_obs if normalize_obs is not None else True
        else:
            do_normalize = False

        if do_normalize:
            stats = self.stats["obs"]
            min_val = torch.as_tensor(
                stats["min"], device=states.device, dtype=states.dtype
            )
            max_val = torch.as_tensor(
                stats["max"], device=states.device, dtype=states.dtype
            )
            range_val = max_val - min_val
            range_val = torch.where(
                range_val == 0, torch.ones_like(range_val), range_val
            )
            states = 2.0 * (states - min_val) / range_val - 1.0

        self.ema_model.eval()
        num_inference_steps = num_inference_steps or self._num_diffusion_iters
        # print(f"\t{states.shape=}")

        # print(f"{states.shape=} this should be 2,1,6")
        obs_cond = self.prepare_observation_condition(states)
        # print(f"{obs_cond.shape=} this should be 2,1,12")
        # logger.info(f"\t{obs_cond.shape=}")
        # print(f"{obs_cond.shape=}")
        #  i want a obs cond that is (batch, hor, o_dim)

        shape = (
            states.shape[0],
            self._pred_horizon,
            self.model._unwrapped_module._a_dim,
        )
        # print(f"{shape=}")
        # logger.info(f"{shape=}")
        noisy_actions = randn_tensor(shape, device=self.device)
        # print(f"{noisy_actions.shape=}")

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        for t in self.noise_scheduler.timesteps:
            inputs = {
                "actions": noisy_actions,
                "timestep": t,
                "global_cond": obs_cond,
            }

            noise_pred, _, _ = self.ema_model.act(inputs=inputs)
            # print(f"{noise_pred.shape=}")
            # print(f"{noise_pred.shape=}")
            # noise_pred = self.ema_model.act(noisy_actions, t, global_cond=obs_cond)
            noisy_actions = self.noise_scheduler.step(
                noise_pred, t, noisy_actions
            ).prev_sample
            # print(f"\t{noisy_actions.shape=}")

        if self.stats is not None:
            do_unnormalize = unnormalize_act if unnormalize_act is not None else True
        else:
            do_unnormalize = False

        if do_unnormalize:
            stats = self.stats["action"]
            min_val = torch.as_tensor(
                stats["min"], device=noisy_actions.device, dtype=noisy_actions.dtype
            )
            max_val = torch.as_tensor(
                stats["max"], device=noisy_actions.device, dtype=noisy_actions.dtype
            )
            range_val = max_val - min_val
            range_val = torch.where(
                range_val == 0, torch.ones_like(range_val), range_val
            )
            noisy_actions = 0.5 * (noisy_actions + 1.0) * range_val + min_val

        # print(f"{noisy_actions.shape=}")
        # print(f"{noisy_actions[:,:self._act_horizon,:]=}")
        # print(f"{noisy_actions[:,:self._act_horizon,:].shape=}")
        return noisy_actions, None, {}

    def _predict_action(self, o: torch.Tensor) -> torch.Tensor:
        pass

    def set_mode(self, mode: str):
        # print(f"in set_mode of DP with mode '{mode}'")

        if mode == "eval":
            # print("setting to 'eval'")
            self.eval()
        elif mode == "train":
            self.train()
        else:
            raise ValueError(
                f"Wrong Mode: choose either 'train' or 'eval', but got '{mode}'"
            )

    def eval(self) -> None:
        # print(f"In DP eval() {self.stats=}")

        # self.ema.copy_to(self.ema_model.parameters())
        self.model.eval()
        self.ema_model.eval()

    def train(self) -> None:
        # self.ema.copy_to(self.ema_model.parameters())
        self.model.train()
        self.ema_model.train()

    def to(self, device: str = "cuda"):
        for _, v in self.models.items():
            v.to(device)

    def save(self, path: str):
        """Save model weights and configuration."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer
            else None,
            "scheduler_state_dict": self.lr_scheduler.state_dict()
            if self.lr_scheduler
            else None,
            "config": self.config,
            "is_trained": self.is_trained,
            "a_dim": self._a_dim,
            "o_dim": self._o_dim,
            "stats": self.stats,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, a_dim: int = None, o_dim: int = None, device: str = None):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        a_dim = a_dim if a_dim is not None else checkpoint.get("a_dim")
        o_dim = o_dim if o_dim is not None else checkpoint.get("o_dim")
        if a_dim is None or o_dim is None:
            raise ValueError("Both action and observation dims must be provided.")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dp_models = {}
        dp_models["model"] = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=config)
        ema = EMAModel(dp_models["model"].parameters(), power=config["ema_power"])
        dp_models["ema_model"] = ConditionalUnet1D(
            a_dim=a_dim, o_dim=o_dim, config=config
        )

        policy = cls(
            a_dim=a_dim,
            o_dim=o_dim,
            models=dp_models,
            ema=ema,
            device=device,
            config=config,
            stats=checkpoint.get("stats"),
        )
        policy.model.load_state_dict(checkpoint["model_state_dict"])
        policy.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if checkpoint["optimizer_state_dict"]:
            policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"]:
            policy.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        policy.is_trained = checkpoint["is_trained"]

        return policy
