import contextlib
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple

import cv2
import gym
import gymnasium
import jax
import jax.numpy as jp
import mujoco as mj
import mujoco.mjx as mjx
import mujoco_playground._src.mjx_env as mjx_env
import numpy as np
import torch

# from brax.io.torch import jax_to_torch, torch_to_jax
# NOTE: The following line will emit a warning and raise ImportError if `torch`
# isn't available.
from flax import struct
from gym import spaces
from gym.vector import utils
from mujoco_playground._src.mjx_env import MjxEnv

# Assuming these imports from mujoco_playground
from mujoco_playground._src.mjx_env import State as MjxState
from mujoco_playground._src.wrapper_torch import _jax_to_torch, _torch_to_jax

from skrl import logger
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)


class VmapWrapper:
    """Vectorizes MjxEnv using vmap for parallel environments."""

    def __init__(self, env: Any, batch_size: int):
        self._env = env
        self.batch_size = batch_size

        # Vmap the environment functions
        self.reset = jax.vmap(env.reset)
        self.step = jax.vmap(env.step)

        # Copy necessary attributes
        self.dt = env.dt
        self.mj_model = env.mj_model
        self.mjx_model = env.mjx_model
        self.action_size = env.action_size
        self.observation_size = getattr(env, "observation_size", None)
        self._gym_disable_underscore_compat = True

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value


class AutoResetWrapper:
    """Automatically resets MJX envs that are done."""

    def __init__(self, env: Any):
        self._env = env

        # Copy attributes
        self.dt = env.dt
        self.mj_model = env.mj_model
        self.mjx_model = env.mjx_model
        self.action_size = env.action_size
        self.batch_size = getattr(env, "batch_size", 1)
        self._gym_disable_underscore_compat = True

    def reset(self, rng: jax.Array) -> Any:
        state = self._env.reset(rng)
        # Store initial state for auto-reset
        state.info["first_state"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: Any, action: jax.Array) -> Any:
        state = self._env.step(state, action)

        # Auto-reset: where done is True, use initial state/obs
        def where_done(x, y):
            done = state.done
            if done.shape and len(done.shape) > 0:
                # Add dimensions for broadcasting
                done = jp.reshape(
                    done, done.shape + (1,) * (len(x.shape) - len(done.shape))
                )
            return jp.where(done, x, y)

        # Reset states where done
        data = jax.tree_map(where_done, state.info["first_state"], state.data)
        obs = jax.tree_map(where_done, state.info["first_obs"], state.obs)

        # Keep reward and done as is (reward should be 0 for reset states)
        reward = jp.where(state.done, 0.0, state.reward)

        return state.replace(data=data, obs=obs, reward=reward)


class TorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(self, env: gym.Env, device: Optional[torch.device] = None):
        super().__init__(env)
        self.device = device if device is not None else torch.device("cpu")

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            obs = self._jax_to_torch(obs).to(self.device)
            return obs, info
        else:
            obs = self._jax_to_torch(result).to(self.device)
            return obs

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = self._torch_to_jax(action)

        result = self.env.step(action)
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            obs = self._jax_to_torch(obs).to(self.device)
            reward = self._jax_to_torch(reward).to(self.device).float()
            terminated = self._jax_to_torch(terminated).to(self.device).bool()
            truncated = self._jax_to_torch(truncated).to(self.device).bool()
            return obs, reward, terminated, truncated, info
        else:  # Gym format
            obs, reward, done, info = result
            obs = self._jax_to_torch(obs).to(self.device)
            reward = self._jax_to_torch(reward).to(self.device).float()
            done = self._jax_to_torch(done).to(self.device).bool()
            return obs, reward, done, info

    def _jax_to_torch(self, x):
        """Convert JAX array to PyTorch tensor."""
        if hasattr(x, "__array__"):
            return torch.from_numpy(np.asarray(x))
        return x

    def _torch_to_jax(self, x):
        """Convert PyTorch tensor to JAX array."""
        if isinstance(x, torch.Tensor):
            return np.asarray(x.cpu().numpy())
        return x


class GymWrapper(gym.Env):
    """A wrapper that converts Brax Env to one that follows Gym API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: MjxEnv, seed: int = 0, backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        self.seed(seed)
        self.backend = backend
        self._state = None

        obs = np.inf * np.ones(self._env.observation_size, dtype="float32")
        self.observation_space = spaces.Box(-obs, obs, dtype="float32")

        action = jax.tree.map(np.array, self._env.mjx_model.actuator_ctrlrange)
        self.action_space = spaces.Box(action[:, 0], action[:, 1], dtype="float32")

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        # We return device arrays for pytorch users.
        return obs

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        # We return device arrays for pytorch users.
        return obs, reward, done, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self):
        return super().render()


class VectorGymWrapper(gym.vector.VectorEnv):
    """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: MjxEnv, seed: int = 0, backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        if not hasattr(self._env, "batch_size"):
            raise ValueError("underlying env must be batched")

        self.num_envs = self._env.batch_size
        self.seed(seed)
        self.backend = backend
        self._state = None
        self._renderer = None

        obs = np.inf * np.ones(self._env.observation_size, dtype="float32")
        obs_space = spaces.Box(-obs, obs, dtype="float32")
        self.observation_space = utils.batch_space(obs_space, self.num_envs)

        action = jax.tree.map(np.array, self._env.mj_model.actuator_ctrlrange)
        action_space = spaces.Box(action[:, 0], action[:, 1], dtype="float32")
        self.action_space = utils.batch_space(action_space, self.num_envs)

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        return obs

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        return obs, reward, done, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode="human", width=256, height=256, cam_name: str = "lookatcart"):
        if self._renderer is None:
            self._renderer = mj.Renderer(self._env.mj_model, width=width, height=height)
            if self._env.mj_model.ncam == 0:
                logger.warning(
                    "You seem not to have defined a camera in your simulation, this i strongly recommended. Defaulting to the abstract camera."
                )
                self._cam_id = -1
            else:
                self._cam_id = self._env.mj_model.cam(cam_name).id
        if mode == "rgb_array":
            state: MjxState = self._state
            data = mjx.get_data(self._env.mj_model, state.data)[0]
            self._renderer.update_scene(data=data, camera=self._cam_id)
            image = np.zeros(shape=(width, height, 3))
            image = self._renderer.render()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        else:
            return super().render(mode=mode)


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: Any, device: Optional[torch.device] = None) -> None:
        """MJX Playground environment wrapper for skrl.

        :param env: The environment to wrap
        :type env: Any supported MJX environment
        :param device: PyTorch device, defaults to None
        :type device: Optional[torch.device]
        """
        super().__init__(env)

        # Check if environment is already wrapped with VmapWrapper
        if hasattr(env, "batch_size") and env.batch_size > 1:
            # Environment is already batched
            gym_env = self._create_gym_env(env)
        else:
            # Apply auto-reset and create gym env
            auto_reset_env = AutoResetWrapper(env)
            gym_env = self._create_gym_env(auto_reset_env)

        # Convert to PyTorch
        self._env = TorchWrapper(gym_env, device=device)
        self._unwrapped = (
            self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        )

    def _create_gym_env(self, env: Any) -> gym.Env:
        """Create a Gym environment from MJX environment."""
        return GymWrapper(env)

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        return convert_gym_space(
            self._unwrapped.observation_space, squeeze_batch_dimension=True
        )

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        return convert_gym_space(
            self._unwrapped.action_space, squeeze_batch_dimension=True
        )

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment"""
        observation, reward, terminated, truncated, info = self._env.step(
            unflatten_tensorized_space(self.action_space, actions)
        )
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation)
        )
        return (
            observation,
            reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            info,
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment"""
        result = self._env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation = result
            info = {}

        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation)
        )
        return observation, info

    def render(self, *args, **kwargs) -> Any:
        """Render the environment"""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        if hasattr(self._env, "close"):
            self._env.close()
