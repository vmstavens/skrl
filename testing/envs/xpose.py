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


class XPose(mjx_env.MjxEnv):
    """Simple 3D position control environment."""

    def init(self) -> mj.MjModel:
        # Create a simple XML string instead of using file
        # <option timestep="0.01" iterations="1" ls_iterations="4">
        #     <flag contact="disable" eulerdamp="disable"/>
        # </option>
        #
        #         <body name="target" mocap="true">
        #     <geom name="target" type="sphere" size="0.005" contype="0" conaffinity="0" rgba="0 0.5 0 0.3"/>
        # </body>
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

                <body name="target">
                    <geom name="target" type="sphere" size="0.005" contype="0" conaffinity="0" rgba="0 0.5 0 0.3"/>
                </body>
                <camera name="lookat" mode="targetbody" target="target" pos="0.4 0.4 0.4"/>
                <body name="agent" gravcomp="1">
                    <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
                    <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
                    <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
                    <geom name="agent" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0" rgba="0.5 0 0 0.3"/>
                </body>
            </worldbody>
            <actuator>
                <position name="x" joint="x" ctrlrange="-1 1" kp="10" kv="10" ctrllimited="true"/>
                <position name="y" joint="y" ctrlrange="-1 1" kp="10" kv="10" ctrllimited="true"/>
                <position name="z" joint="z" ctrlrange="-1 1" kp="10" kv="10" ctrllimited="true"/>
            </actuator>
        </mujoco>
        """

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
        self._slide_jid = [
            self._mj_model.joint("x").id,
            self._mj_model.joint("y").id,
            self._mj_model.joint("z").id,
        ]
        self._slide_qposadr = [
            self._mj_model.jnt_qposadr[self._slide_jid[0]],
            self._mj_model.jnt_qposadr[self._slide_jid[1]],
            self._mj_model.jnt_qposadr[self._slide_jid[2]],
        ]
        self._slide_cid = [
            self._mj_model.actuator("x").id,
            self._mj_model.actuator("y").id,
            self._mj_model.actuator("z").id,
        ]

        # Get joint limits for randomization
        self._qpos_min = jp.array(
            [
                self._mj_model.jnt_range[self._slide_qposadr[0], 0],
                self._mj_model.jnt_range[self._slide_qposadr[1], 0],
                self._mj_model.jnt_range[self._slide_qposadr[2], 0],
            ]
        )
        self._qpos_max = jp.array(
            [
                self._mj_model.jnt_range[self._slide_qposadr[0], 1],
                self._mj_model.jnt_range[self._slide_qposadr[1], 1],
                self._mj_model.jnt_range[self._slide_qposadr[2], 1],
            ]
        )

        # Store mocap body ID for target
        self._target_body_id = self._mj_model.body("target").id
        self._target_mocap_id = 0  # Since we only have one mocap body

        # Target position bounds (within reasonable range)
        self._target_bounds = jp.array([[-0.8, -0.8, 0.1], [0.8, 0.8, 0.8]])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Split RNG for agent position, velocity, and target position
        rng1, rng2, rng3 = jax.random.split(rng, 3)

        # Initialize agent with random position
        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[self._slide_qposadr[0]].set(
            jax.random.uniform(rng1, minval=-0.5, maxval=0.5)
        )
        qpos = qpos.at[self._slide_qposadr[1]].set(
            jax.random.uniform(rng1, minval=-0.5, maxval=0.5)
        )
        qpos = qpos.at[self._slide_qposadr[2]].set(
            jax.random.uniform(rng1, minval=0.1, maxval=0.5)  # Start above ground
        )

        # Small random velocity
        qvel = 0.01 * jax.random.normal(rng2, (self.mjx_model.nv,))

        # Initialize data
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

        metrics = {}
        info = {}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # a = action
        a = state.data.ctrl + action
        data = mjx_env.step(self.mjx_model, state.data, a, self.n_substeps)

        # Get current agent and target positions
        agent_pos = jp.array(
            [
                data.qpos[self._slide_qposadr[0]],
                data.qpos[self._slide_qposadr[1]],
                data.qpos[self._slide_qposadr[2]],
            ]
        )

        # Reward: negative distance to target + bonus for being close
        dist_to_target = jp.linalg.norm(agent_pos)
        reward = -dist_to_target

        # Add small bonus for being very close to target
        # close_bonus = dist_to_target < 0.02
        # reward += close_bonus

        obs = self._get_obs(data, state.info)

        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        success = jp.all(agent_pos < 0.01)

        # Combine all termination conditions
        done = nan_condition | success
        # done = nan_condition | out_of_bounds
        done = done.astype(float)

        info = state.info.copy()

        return mjx_env.State(data, obs, reward, done, state.metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        # Return agent position and target position (6D observation)
        agent_pos = jp.array(
            [
                data.qpos[self._slide_qposadr[0]],
                data.qpos[self._slide_qposadr[1]],
                data.qpos[self._slide_qposadr[2]],
            ]
        )

        # Return both absolute positions and relative position
        return agent_pos

    @property
    def observation_size(self) -> int:
        return 3  # agent_pos(3) + target_pos(3) + relative_pos(3)

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
