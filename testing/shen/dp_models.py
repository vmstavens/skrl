import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import randn_tensor

# env import
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from skrl.models.torch import DeterministicMixin, Model

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
    "num_workers": 1,
    "num_epochs": 100,
    "max_steps": 200,
    "eval_frequency": 10,
    # Scheduler
    "beta_schedule": "squaredcos_cap_v2",
    "clip_sample": True,
    "prediction_type": "epsilon",
    "lr_scheduler_cfg": {"num_warmup_steps": 500, "num_training_steps": 10_000},
}


class ModuleWrapper(DeterministicMixin, Model):
    def __init__(
        self,
        module: nn.Module,
        device: Optional[str] = None,
        observation_space=None,
        action_space=None,
        clip_actions=False,
    ):
        if device is None:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # init base classes
        Model.__init__(self, observation_space, action_space, self.device)
        DeterministicMixin.__init__(self, clip_actions)

        self._unwrapped_module = module
        # self._unwrapped_module = module.to(self.device)

    def compute(self, inputs: dict, role):
        # pick states and actions if available
        return self._unwrapped_module.forward(**inputs), {}


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# class ConditionalUnet1D(nn.Module):
#     def __init__(self, a_dim: int, o_dim: int, config: dict):
#         super().__init__()
#         self.config = config

#         self._a_dim: int = a_dim
#         self._o_dim: int = o_dim
#         self._down_dims: list[int] = config["down_dims"]
#         self._diffusion_step_embed_dim: int = config["diffusion_step_embed_dim"]
#         self._global_cond_dim: int = self._o_dim * self.config["obs_horizon"]
#         self._kernel_size: int = config["kernel_size"]
#         self._n_groups: int = config["n_groups"]

#         all_dims = [self._a_dim] + list(self._down_dims)
#         start_dim = self._down_dims[0]

#         dsed = self._diffusion_step_embed_dim
#         diffusion_step_encoder = nn.Sequential(
#             SinusoidalPosEmb(dsed),
#             nn.Linear(dsed, dsed * 4),
#             nn.Mish(),
#             nn.Linear(dsed * 4, dsed),
#         )
#         cond_dim = dsed + self._global_cond_dim

#         in_out = list(zip(all_dims[:-1], all_dims[1:]))
#         mid_dim = all_dims[-1]

#         self.mid_modules = nn.ModuleList(
#             [
#                 ConditionalResidualBlock1D(
#                     mid_dim,
#                     mid_dim,
#                     cond_dim=cond_dim,
#                     kernel_size=self._kernel_size,
#                     n_groups=self._n_groups,
#                 ),
#                 ConditionalResidualBlock1D(
#                     mid_dim,
#                     mid_dim,
#                     cond_dim=cond_dim,
#                     kernel_size=self._kernel_size,
#                     n_groups=self._n_groups,
#                 ),
#             ]
#         )

#         down_modules = nn.ModuleList([])
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (len(in_out) - 1)
#             down_modules.append(
#                 nn.ModuleList(
#                     [
#                         ConditionalResidualBlock1D(
#                             dim_in,
#                             dim_out,
#                             cond_dim=cond_dim,
#                             kernel_size=self._kernel_size,
#                             n_groups=self._n_groups,
#                         ),
#                         ConditionalResidualBlock1D(
#                             dim_out,
#                             dim_out,
#                             cond_dim=cond_dim,
#                             kernel_size=self._kernel_size,
#                             n_groups=self._n_groups,
#                         ),
#                         Downsample1d(dim_out) if not is_last else nn.Identity(),
#                     ]
#                 )
#             )

#         up_modules = nn.ModuleList([])
#         # NOTE: leaving your structure intact; we will enforce temporal alignment via match_T
#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (len(in_out) - 1)
#             up_modules.append(
#                 nn.ModuleList(
#                     [
#                         ConditionalResidualBlock1D(
#                             dim_out * 2,
#                             dim_in,
#                             cond_dim=cond_dim,
#                             kernel_size=self._kernel_size,
#                             n_groups=self._n_groups,
#                         ),
#                         ConditionalResidualBlock1D(
#                             dim_in,
#                             dim_in,
#                             cond_dim=cond_dim,
#                             kernel_size=self._kernel_size,
#                             n_groups=self._n_groups,
#                         ),
#                         Upsample1d(dim_in) if not is_last else nn.Identity(),
#                     ]
#                 )
#             )

#         final_conv = nn.Sequential(
#             Conv1dBlock(start_dim, start_dim, kernel_size=self._kernel_size),
#             nn.Conv1d(start_dim, self._a_dim, 1),
#         )

#         self.diffusion_step_encoder = diffusion_step_encoder
#         self.down_modules = down_modules
#         self.up_modules = up_modules
#         self.final_conv = final_conv

#     @staticmethod
#     def _match_T(x: torch.Tensor, T: int) -> torch.Tensor:
#         """
#         Ensure x has temporal length T along the last dimension (Conv1d format: [B, C, T]).
#         Crops if too long, right-pads with zeros if too short.
#         """
#         Tx = x.shape[-1]
#         if Tx == T:
#             return x
#         if Tx > T:
#             return x[..., :T]
#         # pad right side
#         return nn.functional.pad(x, (0, T - Tx))

#     def forward(
#         self,
#         actions: torch.Tensor,
#         timestep: Union[torch.Tensor, float, int],
#         global_cond: Optional[torch.Tensor] = None,
#     ):
#         # actions: [B, T, A] -> Conv1d expects [B, A, T]
#         actions = actions.moveaxis(-1, -2)
#         T_in = actions.shape[-1]

#         if not torch.is_tensor(timestep):
#             timestep = torch.tensor([timestep], dtype=torch.long, device=actions.device)
#         elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
#             timestep = timestep[None].to(actions.device)

#         timestep = timestep.expand(actions.shape[0])
#         global_feature = self.diffusion_step_encoder(timestep)

#         if global_cond is not None:
#             if not global_cond.is_cuda:
#                 global_cond = global_cond.to(global_feature.device)
#             global_feature = torch.cat([global_feature, global_cond], axis=-1)

#         x = actions
#         h = []

#         # Down path
#         for resnet, resnet2, downsample in self.down_modules:
#             x = resnet(x, global_feature)
#             x = resnet2(x, global_feature)
#             h.append(x)
#             x = downsample(x)

#         # Mid
#         for mid_module in self.mid_modules:
#             x = mid_module(x, global_feature)

#         # Up path
#         for resnet, resnet2, upsample in self.up_modules:
#             skip = h.pop()

#             # Critical: ConvTranspose1d can produce off-by-1 length.
#             # Enforce x to match skip length before concatenation.
#             x = self._match_T(x, skip.shape[-1])

#             x = torch.cat((x, skip), dim=1)
#             x = resnet(x, global_feature)
#             x = resnet2(x, global_feature)

#             x = upsample(x)

#         # Final projection
#         x = self.final_conv(x)

#         # Critical: enforce final temporal length to match input action horizon exactly
#         x = self._match_T(x, T_in)

#         # back to [B, T, A]
#         x = x.moveaxis(-1, -2)
#         return x


class ConditionalUnet1D(nn.Module):
    def __init__(self, a_dim: int, o_dim: int, config: dict):
        super().__init__()
        self.config = config

        self._a_dim: int = a_dim
        self._o_dim: int = o_dim
        self._down_dims: list[int] = config["down_dims"]
        self._diffusion_step_embed_dim: int = config["diffusion_step_embed_dim"]
        self._global_cond_dim: int = self._o_dim * self.config["obs_horizon"]
        # print(f"{self._a_dim=}")
        # print(f"{self._o_dim=}")
        # print(f"{self._global_cond_dim=}")
        # print(f"{self.config["obs_horizon"]=}")
        # self._global_cond_dim: int = config["global_cond_dim"]
        self._kernel_size: int = config["kernel_size"]
        self._n_groups: int = config["n_groups"]

        all_dims = [self._a_dim] + list(self._down_dims)
        start_dim = self._down_dims[0]

        dsed = self._diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + self._global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=self._kernel_size,
                    n_groups=self._n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=self._kernel_size,
                    n_groups=self._n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=self._kernel_size,
                            n_groups=self._n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=self._kernel_size,
                            n_groups=self._n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=self._kernel_size,
                            n_groups=self._n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=self._kernel_size,
                            n_groups=self._n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=self._kernel_size),
            nn.Conv1d(start_dim, self._a_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

    def forward(
        self,
        actions: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=Optional[torch.Tensor],
    ):
        actions = actions.moveaxis(-1, -2)
        # print(f"{actions.shape=}")

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=actions.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(actions.device)

        timestep = timestep.expand(actions.shape[0])
        global_feature = self.diffusion_step_encoder(timestep)
        # print(f"{global_feature.shape=}")

        if global_cond is not None:
            # print(
            #     f"in global_cond is not None, we got {global_feature.shape=} {global_cond.shape=}"
            # )
            if not global_cond.is_cuda:
                global_cond = global_cond.to(global_feature.device)
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # print(f"{global_feature.shape=}")
        # print(f"{global_cond.shape=}")
        x = actions
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            # print(f"{x.shape=}")
            # print(f"{global_feature.shape=}")
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)
        return x


class EMAModel:
    """Exponential Moving Average model wrapper."""

    def __init__(self, parameters: list[torch.Tensor], power: float = 0.75):
        self.parameters = list(parameters)
        self.power = power
        self.shadow_params = [p.clone().detach() for p in self.parameters]

    def step(self, parameters: list[torch.Tensor]):
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.data = s_param.data * self.power + param.data * (1 - self.power)

    def copy_to(self, parameters: list[torch.Tensor]):
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)
