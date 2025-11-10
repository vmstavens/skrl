import contextlib
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple

import cv2
import gym
import gymnasium
import jax
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
from jax import numpy as jp

# Assuming these imports from mujoco_playground
from mujoco_playground._src.mjx_env import MjxEnv
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


class TorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(self, env: gym.Env, device: Optional[torch.device] = None):
        """Creates a gym Env to one that outputs PyTorch tensors."""
        super().__init__(env)
        self.device = device

    def reset(self):
        obs = super().reset()
        return _jax_to_torch(obs)

    def step(self, action):
        # print(f".venv/lib/python3.12/site-packages/brax/envs/wrappers/torch.py {action=}")
        # print(f".venv/lib/python3.12/site-packages/brax/envs/wrappers/torch.py {type(action)=}")
        action = _torch_to_jax(action)
        # print(f"{action[0]=}")
        obs, reward, done, info = super().step(action)
        obs = _jax_to_torch(obs)
        # print(f"{obs[0]=}")
        reward = _jax_to_torch(reward)
        # print(f"{reward[0]=}")
        done = _jax_to_torch(done)
        # print(f"{done[0]=}")
        # print(len(done))
        # quit()
        # print("--------------------------------------")
        # info = _jax_to_torch(info)
        return obs, reward, done, info


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

        action = jax.tree.map(np.array, self._env.mj_model.actuator_ctrlrange)
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

    def render(self, mode="human", width=256, height=256):
        if mode == "rgb_array":
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError("must call reset or step before rendering")
            return image.render_array(
                sys, state.pipeline_state, width=width, height=height
            )
        else:
            return super().render(mode=mode)  # just raise an exception


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


class VmapWrapper(Wrapper):
    """Vectorizes MjxEnv."""

    def __init__(self, env: MjxEnv, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> MjxState:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self._env.reset)(rng)

    def step(self, state: MjxState, action: jax.Array) -> MjxState:
        return jax.vmap(self._env.step)(state, action)


class BraxAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self._env.reset(rng)
        state.info["first_state"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self._env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, state.info["first_state"], state.data)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(data=data, obs=obs)


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: MjxEnv) -> None:
        """Brax environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Brax environment
        """
        super().__init__(env)

        # TODO: See mujoco_playground _src wrappers

        # from mujoco_playground._src.wrapper import wrap_for_brax_training

        # env = wrap_for_brax_training(env)

        env = BraxAutoResetWrapper(env)
        env = VectorGymWrapper(env)
        env = TorchWrapper(env, device=self.device)

        self.env = env

        self._env = env
        self._unwrapped = env.unwrapped

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
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        observation, reward, terminated, info = self._env.step(
            unflatten_tensorized_space(self.action_space, actions)
        )
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation)
        )
        truncated = torch.zeros_like(terminated)
        return (
            observation,
            reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            info,
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        observation = self._env.reset()
        observation = flatten_tensorized_space(
            tensorize_space(self.observation_space, observation)
        )
        return observation, {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        frame = self._env.render(mode="rgb_array", **kwargs)

        # render the frame using OpenCV
        # try:
        #     import cv2

        #     cv2.imshow("env", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     cv2.waitKey(1)
        # except ImportError as e:
        #     logger.warning(
        #         f"Unable to import opencv-python: {e}. Frame will not be rendered."
        #     )
        return frame

    def close(self) -> None:
        """Close the environment"""
        # self._env.close() raises AttributeError: 'VectorGymWrapper' object has no attribute 'closed'
        pass
