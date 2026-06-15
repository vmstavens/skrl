from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import torch

from skrl.envs.wrappers.torch.base import Wrapper as skrl_Wrapper
from skrl.utils.spaces.torch import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)


class WarpWrapper(skrl_Wrapper):
    """Torch wrapper for mujoco_warp-based vector environments."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._env = env
        self._unwrapped = env

    @property
    def observation_space(self) -> gym.Space:
        return convert_gym_space(
            self._unwrapped.observation_space, squeeze_batch_dimension=True
        )

    @property
    def action_space(self) -> gym.Space:
        return convert_gym_space(
            self._unwrapped.action_space, squeeze_batch_dimension=True
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        observation = self._env.reset()
        observation = torch.as_tensor(observation, device=self.device)
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation)
        )
        return observation, {}

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        actions = unflatten_tensorized_space(self.action_space, actions)
        if torch.is_tensor(actions):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)

        observation, reward, terminated, truncated, info = self._env.step(actions_np)

        observation = torch.as_tensor(observation, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        terminated = torch.as_tensor(terminated, device=self.device)
        truncated = torch.as_tensor(truncated, device=self.device)

        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation)
        )

        reward = reward.view(-1, 1)
        terminated = terminated.bool().view(-1, 1)
        truncated = truncated.bool().view(-1, 1)

        return observation, reward, terminated, truncated, info

    def render(self, *args, **kwargs) -> Any:
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)
        return None

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()
