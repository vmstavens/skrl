import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import mujoco as mj
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
    )


@dataclass
class GoalState:
    qpos: jp.ndarray = field(default_factory=lambda: jp.array([0, 0, 0]))
    qvel: jp.ndarray = field(default_factory=lambda: jp.array([0, 0, 0]))
    ctrl: jp.ndarray = field(default_factory=lambda: jp.array([0, 0, 0]))


class XPose(mjx_env.MjxEnv):
    """Simple 3D position control environment."""

    def init(self) -> mj.MjModel:
        # Create a simple XML string instead of using file
        # <option timestep="0.01" iterations="1" ls_iterations="4">
        #     <flag contact="disable" eulerdamp="disable"/>
        # </option>
        #
        xml_string = """
        <mujoco>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <quality shadowsize="4096"/>
            <map stiffness="700" shadowscale="0.5" fogstart="1" fogend="150" zfar="40" haze="1"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="1 1 1" width="512" height="512"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
            width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>

            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="10 10" texuniform="true"/>
        </asset>


        <option iterations="1" ls_iterations="5" timestep="0.004" integrator="implicitfast">
            <flag eulerdamp="disable"/>
        </option>

            <worldbody>
                <geom name="target" type="sphere" size="0.01" contype="0" conaffinity="0" rgba="0 0.5 0 0.3"/>
                <camera name="lookat" mode="targetbody" target="agent" pos="0.2 0.2 0.2"/>
                <body name="agent" gravcomp="1">
                    <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
                    <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
                    <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
                    <geom name="agent" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0" rgba="0.5 0 0 0.3"/>
                </body>
            </worldbody>
            <actuator>
                <position name="x" joint="x" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
                <position name="y" joint="y" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
                <position name="z" joint="z" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
            </actuator>
        </mujoco>
        """
        # xml_string = """
        # <mujoco>

        # <option iterations="1" ls_iterations="5" timestep="0.004" integrator="implicitfast">
        #     <flag eulerdamp="disable"/>
        # </option>

        #     <worldbody>
        #         <camera name="lookat" mode="targetbody" target="agent" pos="0 -2 2"/>

        #         <body name="agent" gravcomp="1">
        #             <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
        #             <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
        #             <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
        #             <geom name="agent" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0"/>
        #         </body>
        #     </worldbody>
        #     <actuator>
        #         <position name="x" joint="x" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
        #         <position name="y" joint="y" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
        #         <position name="z" joint="z" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
        #     </actuator>
        # </mujoco>
        # """

        scene = mj.MjSpec.from_string(xml_string)
        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)

        self._mj_model = self.init()
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self) -> None:
        # Store joint addresses
        x_jid = self._mj_model.joint("x").id
        self._x_qposadr = self._mj_model.jnt_qposadr[x_jid]
        y_jid = self._mj_model.joint("y").id
        self._y_qposadr = self._mj_model.jnt_qposadr[y_jid]
        z_jid = self._mj_model.joint("z").id
        self._z_qposadr = self._mj_model.jnt_qposadr[z_jid]

        # Store actuator IDs
        self._x_ctrl_id = self._mj_model.actuator("x").id
        self._y_ctrl_id = self._mj_model.actuator("y").id
        self._z_ctrl_id = self._mj_model.actuator("z").id

        # Get joint limits for randomization
        self._qpos_min = jp.array(
            [
                self._mj_model.jnt_range[self._x_qposadr, 0],
                self._mj_model.jnt_range[self._y_qposadr, 0],
                self._mj_model.jnt_range[self._z_qposadr, 0],
            ]
        )
        self._qpos_max = jp.array(
            [
                self._mj_model.jnt_range[self._x_qposadr, 1],
                self._mj_model.jnt_range[self._y_qposadr, 1],
                self._mj_model.jnt_range[self._z_qposadr, 1],
            ]
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Split RNG for position and velocity
        rng1, rng2 = jax.random.split(rng, 2)

        # Initialize with random position
        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[self._x_qposadr].set(
            jax.random.uniform(rng1, minval=-0.5, maxval=0.5)
        )
        qpos = qpos.at[self._y_qposadr].set(
            jax.random.uniform(rng1, minval=-0.5, maxval=0.5)
        )
        qpos = qpos.at[self._z_qposadr].set(
            jax.random.uniform(rng1, minval=-0.5, maxval=0.5)
        )

        # Small random velocity
        qvel = 0.01 * jax.random.normal(rng2, (self.mjx_model.nv,))

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

        metrics = {}
        info = {"rng": rng}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)

        # Simple reward: negative distance to origin
        current_pos = jp.array(
            [
                data.qpos[self._x_qposadr],
                data.qpos[self._y_qposadr],
                data.qpos[self._z_qposadr],
            ]
        )
        # jax.debug.print(f"{current_pos=}")
        dist_to_goal = jp.linalg.norm(current_pos)
        reward = -dist_to_goal

        obs = self._get_obs(data, state.info)

        # Check for termination
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused
        # Return current 3D position
        return jp.array(
            [
                data.qpos[self._x_qposadr],
                data.qpos[self._y_qposadr],
                data.qpos[self._z_qposadr],
            ]
        )

    @property
    def observation_size(self) -> int:
        return 3

    @property
    def action_size(self) -> int:
        return 3

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
