# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cartpole environment matching Brax inverted pendulum."""

from typing import Any, Dict, Optional, Union
import warnings

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cartpole.xml"


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=512,
      render_width=64,
      render_height=64,
      enabled_geom_groups=[0, 1, 2],
      use_rasterizer=False,
      history=3,
  )


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      vision_config=default_vision_config(),
  )


def _rgba_to_grayscale(rgba: jax.Array) -> jax.Array:
  """
  Intensity-weigh the colors.
  This expects the input to have the channels in the last dim.
  """
  r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray


class Balance(mjx_env.MjxEnv):
    """Cartpole environment matching Brax inverted pendulum."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)
        self._vision = self._config.vision

        self._xml_path = _XML_PATH.as_posix()
        self._model_assets = common.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            _XML_PATH.read_text(), self._model_assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()
        self._timeout = 1000  # Episode length matching Brax

        if self._vision:
            try:
                # pylint: disable=import-outside-toplevel
                from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
            except ImportError:
                warnings.warn("Madrona MJX not installed. Cannot use vision with.")
                return
            self.renderer = BatchRenderer(
                m=self._mjx_model,
                gpu_id=self._config.vision_config.gpu_id,
                num_worlds=self._config.vision_config.render_batch_size,
                batch_render_view_width=self._config.vision_config.render_width,
                batch_render_view_height=self._config.vision_config.render_height,
                enabled_geom_groups=np.asarray(
                    self._config.vision_config.enabled_geom_groups
                ),
                enabled_cameras=np.asarray([
                    0,
                ]),
                add_cam_debug_geo=False,
                use_rasterizer=self._config.vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )

    def _post_init(self) -> None:
        slider_jid = self._mj_model.joint("slider").id
        self._slider_qposadr = self._mj_model.jnt_qposadr[slider_jid]
        hinge_1_jid = self._mj_model.joint("hinge_1").id
        self._hinge_1_qposadr = self._mj_model.jnt_qposadr[hinge_1_jid]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state matching Brax."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # Initialize qpos with small random noise around zero, matching Brax
        qpos = jax.random.uniform(
            rng1, (self.mjx_model.nq,), minval=-0.01, maxval=0.01
        )
        
        # Initialize qvel with small random noise around zero, matching Brax  
        qvel = jax.random.uniform(
            rng2, (self.mjx_model.nv,), minval=-0.01, maxval=0.01
        )

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

        metrics = {}
        info = {"rng": rng, "step_count": 0}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        
        if self._vision:
            render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
            info.update({"render_token": render_token})
            obs = _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0
            obs_history = jp.tile(obs, (self._config.vision_config.history, 1, 1))
            info.update({"obs_history": obs_history})
            obs = {"pixels/view_0": obs_history.transpose(1, 2, 0)}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Run one timestep of the environment's dynamics matching Brax."""
        # Scale action from [-1,1] to actuator limits, matching Brax
        action_min = self.mjx_model.actuator_ctrlrange[:, 0]
        action_max = self.mjx_model.actuator_ctrlrange[:, 1]
        scaled_action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        data = mjx_env.step(self.mjx_model, state.data, scaled_action, self.n_substeps)
        
        # Get observation
        obs = self._get_obs(data, state.info)
        
        # Brax-style reward: +1 for each timestep the pole is upright
        reward = 1.0
        
        # Brax-style termination: pole angle > 0.2 radians
        # Note: In MJX, the pole angle is stored differently than in Brax
        # We need to extract the actual angle from the quaternion/rotation matrix
        pole_angle = self._get_pole_angle(data)
        done = jp.where(jp.abs(pole_angle) > 0.2, 1.0, 0.0)
        
        # Also check for timeout (1000 steps like Brax)
        current_step = state.info.get("step_count", 0) + 1
        timeout = current_step >= self._timeout
        done = jp.where(timeout, 1.0, done)
        
        # Update metrics and info
        metrics = state.metrics.copy()
        info = state.info.copy()
        info["step_count"] = current_step
        info["timeout"] = timeout
        info["pole_angle"] = pole_angle

        if self._vision:
            _, rgb, _ = self.renderer.render(state.info["render_token"], data)
            obs_history = state.info["obs_history"]
            obs_history = jp.roll(obs_history, 1, axis=0)
            obs_history = obs_history.at[0].set(
                _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0
            )
            info["obs_history"] = obs_history
            obs = {"pixels/view_0": obs_history.transpose(1, 2, 0)}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _get_pole_angle(self, data: mjx.Data) -> jax.Array:
        """Extract pole angle from MJX data to match Brax observation."""
        # The pole is body 2 in the cartpole model
        # We can get the angle from the quaternion or rotation matrix
        # For a simple hinge joint, the angle is in qpos[1]
        return data.qpos[1]

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        """Observe cartpole body position and velocities matching Brax."""
        del info  # Unused
        
        # Brax observation: [cart_position, pole_angle, cart_velocity, pole_angular_velocity]
        cart_position = data.qpos[0]  # slider joint position
        pole_angle = data.qpos[1]     # hinge joint angle
        cart_velocity = data.qvel[0]  # slider joint velocity  
        pole_angular_velocity = data.qvel[1]  # hinge joint angular velocity
        
        return jp.array([cart_position, pole_angle, cart_velocity, pole_angular_velocity])

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 1  # Matching Brax - single continuous action

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model