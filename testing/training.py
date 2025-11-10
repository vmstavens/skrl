import abc
from typing import (
    Any,
    Callable,
    ClassVar,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import gym
import gymnasium
import jax
import jax.numpy as jp
import mujoco
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
import torch

# NOTE: The following line will emit a warning and raise ImportError if `torch`
# isn't available.
from brax.io.torch import Device, jax_to_torch, torch_to_jax
from gym import spaces
from gym.vector import utils
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import MjxEnv, State

from skrl import logger

# from skrl.envs.wrappers.torch.base import Wrapper
# from skrl.envs.wrappers.torch.base import Wrapper as skrl_Wrapper
from skrl.utils.spaces.torch import (
    convert_gym_space,
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)

# from skrl.utils.spaces.jax import (
#     convert_gym_space,
#     flatten_tensorized_space,
#     tensorize_space,
#     unflatten_tensorized_space,
# )

Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]


class Wrapper(mjx_env.MjxEnv):
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Any):  # pylint: disable=super-init-not-called
        self.env = env

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return self.env.reset(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.env.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self.env.mjx_model

    @property
    def xml_path(self) -> str:
        return self.env.xml_path

    def render(
        self,
        trajectory: List[mjx_env.State],
        height: int = 240,
        width: int = 320,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
    ) -> Sequence[np.ndarray]:
        return self.env.render(
            trajectory, height, width, camera, scene_option, modify_scene_fns
        )


class TorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(self, env: gym.Env, device: Optional[Device] = None):
        """Creates a gym Env to one that outputs PyTorch tensors."""
        super().__init__(env)
        self.device = device

    def reset(self):
        print(self.env.reset)
        obs = self.env.reset()  # super().reset()
        return jax_to_torch(obs, device=self.device)

    def step(self, action):
        action = torch_to_jax(action)
        obs, reward, done, info = super().step(action)
        obs = jax_to_torch(obs, device=self.device)
        reward = jax_to_torch(reward, device=self.device)
        done = jax_to_torch(done, device=self.device)
        info = jax_to_torch(info, device=self.device)
        return obs, reward, done, info


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

        action = jax.tree.map(np.array, self._env.mjx_model.actuator_ctrlrange)
        # action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
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

    # def render(self, mode="human", width=256, height=256):
    #     if mode == "rgb_array":
    #         sys, state = self._env.sys, self._state
    #         if state is None:
    #             raise RuntimeError("must call reset or step before rendering")
    #         state_list = [state.take(i).pipeline_state for i in range(self.num_envs)]
    #         return np.stack(
    #             image.render_array(sys, state_list, width=width, height=height)
    #         )
    #     else:
    #         return super().render(mode=mode)  # just raise an exception

    def render(self, mode="human", width=256, height=256, cam_name: str = None):
        if self._renderer is None:
            self._renderer = mj.Renderer(self._env.mj_model, width=width, height=height)
            if self._env.mj_model.ncam == 0:
                logger.warning(
                    "You seem not to have defined a camera in your simulation, this i strongly recommended. Defaulting to the abstract camera."
                )
                self._cam_id = -1
            else:
                if cam_name is None:
                    self._cam_id = 0
                else:
                    self._cam_id = self._env.mj_model.cam(cam_name).id
        if mode == "rgb_array":
            state: State = self._state
            data = mjx.get_data(self._env.mj_model, state.data)[0]
            self._renderer.update_scene(data=data, camera=self._cam_id)
            image = np.zeros(shape=(width, height, 3))
            image = self._renderer.render()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        else:
            return super().render(mode=mode)


class Env(abc.ABC):
    """Interface for driving training and inference."""

    @abc.abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

    @property
    @abc.abstractmethod
    def observation_size(self) -> ObservationSize:
        """The size of the observation vector returned in step and reset."""

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """The size of the action vector expected by step."""

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""

    @property
    def unwrapped(self) -> "Env":
        return self


class PlaygroundWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Brax environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Brax environment
        """
        super().__init__(env)

        env = VectorGymWrapper(env)
        env = TorchWrapper(env, device=self.device)

        self._env = env
        # self._unwrapped = env.unwrapped
        self._unwrapped = env  # stop unwrapping, just keep reference

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
        # observation is already a torch.Tensor
        # observation = flatten_tensorized_space(observation)
        # return observation, {}
        # flatten only if needed
        # if isinstance(observation, dict):
        #     observation = flatten_tensorized_space(
        #         tensorize_space(self.observation_space, observation)
        #     )
        # return observation, {}

        # observation = flatten_tensorized_space(tensorize_space(self.observation_space, observation))
        # return observation, {}

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


class BraxAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_state"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, state.info["first_state"], state.data)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(data=data, obs=obs)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: MjxEnv, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["steps"] = jp.zeros(rng.shape[:-1])
        state.info["truncation"] = jp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info["episode_done"] = jp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics["sum_reward"] = jp.zeros(rng.shape[:-1])
        episode_metrics["length"] = jp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
        state.info["episode_metrics"] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info["steps"] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info["steps"] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info["episode_done"]
        state.info["episode_metrics"]["sum_reward"] += jp.sum(rewards, axis=0)
        state.info["episode_metrics"]["sum_reward"] *= 1 - prev_done
        state.info["episode_metrics"]["length"] += self.action_repeat
        state.info["episode_metrics"]["length"] *= 1 - prev_done
        for metric_name in state.metrics.keys():
            if metric_name != "reward":
                state.info["episode_metrics"][metric_name] += state.metrics[metric_name]
                state.info["episode_metrics"][metric_name] *= 1 - prev_done
        state.info["episode_done"] = done
        return state.replace(done=done)


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state: State = self.env.reset(rng)
        state.info["first_pipeline_state"] = state
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state: State = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info["first_pipeline_state"], state
        )
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class VmapWrapper(Wrapper):
    """Vectorizes a Brax environment over batch dimension using jax.vmap."""

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array):
        """Reset all batched environments."""
        # Determine if rng is a single key or already batched
        if rng.ndim == 1:  # shape (2,)
            if self.batch_size is None:
                raise ValueError(
                    "VmapWrapper requires batch_size when rng is unbatched."
                )
            rngs = jax.random.split(rng, self.batch_size)
        elif rng.ndim == 2 and (rng.shape[-1] == 2):
            # Already batched RNGs
            rngs = rng
            if self.batch_size is not None and rngs.shape[0] != self.batch_size:
                raise ValueError(
                    f"RNG batch size mismatch: got {rngs.shape[0]}, expected {self.batch_size}"
                )
        else:
            raise ValueError(
                f"Unexpected RNG shape {rng.shape}, expected (2,) or (B, 2)"
            )

        # Vectorized reset
        return jax.vmap(self.env.reset)(rngs)

    def step(self, state, action):
        """Step all batched environments."""
        return jax.vmap(self.env.step)(state, action)

    def close(self):
        """Release references to avoid GC recursion at teardown."""
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None


def create(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Env:
    """Creates an environment from the registry.

    Args:
    env_name: environment name string
    episode_length: length of episode
    action_repeat: how many repeated actions to take per environment step
    auto_reset: whether to auto reset the environment after an episode is done
    batch_size: the number of environments to batch together
    **kwargs: keyword argments that get passed to the Env class constructor

    Returns:
    env: an environment
    """

    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = VmapWrapper(env, batch_size=batch_size)  # pytype: disable=wrong-arg-types
    env = BraxAutoResetWrapper(env)
    return env
    # TODO: see brax wrapper for nice structure
