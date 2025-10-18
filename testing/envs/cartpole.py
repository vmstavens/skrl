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
"""Cartpole environment."""

import warnings
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cartpole.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
    )




class Balance(mjx_env.MjxEnv):
    """Cartpole environment with balance task."""

    _CART_RANGE = (-0.25, 0.25)
    _ANGLE_COSINE_RANGE = (0.995, 1)

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)

        self._xml_path = _XML_PATH.as_posix()
        self._model_assets = common.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            _XML_PATH.read_text(), self._model_assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

        # if the angles goes outside of 20 degrees, reset and punish
        self._dropped_threshold = 0.2


    def _post_init(self) -> None:
        slider_jid = self._mj_model.joint("slider").id
        self._slider_qposadr = self._mj_model.jnt_qposadr[slider_jid]
        hinge_1_jid = self._mj_model.joint("hinge_1").id
        self._hinge_1_qposadr = self._mj_model.jnt_qposadr[hinge_1_jid]

    def _reset_balance(self, rng: jax.Array) -> jax.Array:
        rng1, rng2 = jax.random.split(rng, 2)

        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[self._slider_qposadr].set(
            jax.random.uniform(rng1, (), minval=-0.1, maxval=0.1)
        )
        qpos = qpos.at[1:].set(
            jax.random.uniform(
                rng2, (self.mjx_model.nq - 1,), minval=-0.034, maxval=0.034
            )
        )

        return qpos

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._reset_balance(rng)

        rng, rng1 = jax.random.split(rng, 2)
        qvel = 0.01 * jax.random.normal(rng1, (self.mjx_model.nv,))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)


        def _print_debug(qpos, qvel):
            print(f"We are resetting :")
            print(f"  qpos: {qpos}")
            print(f"  qvel: {qvel}")
            print("---------------------")

        # jax.debug.callback(
        #     _print_debug,
        #     qpos,
        #     qvel
        # )


        metrics = {
            "reward/upright": jp.zeros(()),
            "reward/centered": jp.zeros(()),
            "reward/small_control": jp.zeros(()),
            "reward/small_velocity": jp.zeros(()),
            "reward/cart_in_bounds": jp.zeros(()),
            "reward/angle_in_bounds": jp.zeros(()),
        }
        info = {"rng": rng, "step_count": jp.array([0])}

        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name

        obs = self._get_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        reward = self._get_reward(data, action, state.info, state.metrics)

        obs = self._get_obs(data, state.info)

        # Get current step count and increment
        current_step = state.info.get("step_count", 0) + 1
        timeout = current_step >= self._config.episode_length

        hinge_pos = data.qpos[self._hinge_1_qposadr]
        dropped = jp.abs(hinge_pos) > self._dropped_threshold

        # reward = reward if not dropped else reward - 10
        reward = jp.where(dropped, reward - 10, reward)

        # Check for termination conditions
        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = nan_condition | timeout | dropped
        done = done.astype(float)

        # Update info - reset step_count if episode is done
        
        info = state.info.copy()
        info["step_count"] = jp.where(done, 0, current_step)  # Reset to 0 if done
        info["timeout"] = timeout

        # Debug callback
        def _print_debug(nan_condition, timeout, dropped, done, current_step, new_step_count, qpos_hinge, qpos_slide):
            print(f"Step {current_step} -> {new_step_count}:")
            print(f"  nan_condition: {nan_condition}")
            print(f"  timeout: {timeout}")
            print(f"  dropped: {dropped}")
            print(f"  hinge: {qpos_hinge}")
            print(f"  slide: {qpos_slide}")
            print(f"  done: {done}")
            if done:
                print(f"  done: {done} ===============")
                # exit()
            print("---")

        # jax.debug.callback(
        #     _print_debug,
        #     nan_condition,
        #     timeout,
        #     dropped,
        #     done,
        #     current_step,
        #     info["step_count"],
        #     qpos_hinge, qpos_slide
        # )

        return mjx_env.State(data, obs, reward, done, state.metrics, info)


    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.
        cart_position = data.qpos[self._slider_qposadr]
        pole_angle_cos = data.xmat[2:, 2, 2]  # zz.
        pole_angle_sin = data.xmat[2:, 0, 2]  # xz.
        return jp.concatenate(
            [
                cart_position.reshape(1),
                pole_angle_cos,
                pole_angle_sin,
                data.qvel,
            ]
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        del info  # Unused.
        pole_angle_cos = data.xmat[2, 2, 2]
        upright = (pole_angle_cos + 1) / 2
        metrics["reward/upright"] = upright

        cart_position = data.qpos[self._slider_qposadr]
        centered = reward.tolerance(cart_position, margin=2)
        centered = (1 + centered) / 2
        metrics["reward/centered"] = centered

        small_control = reward.tolerance(
            action[0], margin=1, value_at_margin=0, sigmoid="quadratic"
        )
        small_control = (4 + small_control) / 5
        metrics["reward/small_control"] = small_control

        angular_vel = data.qvel[1:]
        small_velocity = reward.tolerance(angular_vel, margin=5).min()
        small_velocity = (1 + small_velocity) / 2
        metrics["reward/small_velocity"] = small_velocity

        return upright * small_control * small_velocity * centered

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
