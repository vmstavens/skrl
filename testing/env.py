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
                <body name="target" mocap="true">
                    <geom name="target" type="sphere" size="0.005" contype="0" conaffinity="0" rgba="0 0.5 0 0.3"/>
                </body>
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
        self._timeout = 500  # timesteps
        self._post_init()

    def _post_init(self) -> None:
        # Store joint addresses
        # x_jid = self._mj_model.joint("x").id
        # self._x_qposadr = self._mj_model.jnt_qposadr[x_jid]
        # y_jid = self._mj_model.joint("y").id
        # self._y_qposadr = self._mj_model.jnt_qposadr[y_jid]
        # z_jid = self._mj_model.joint("z").id
        # self._z_qposadr = self._mj_model.jnt_qposadr[z_jid]

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

        # Store actuator IDs
        # self._x_ctrl_id = self._mj_model.actuator("x").id
        # self._y_ctrl_id = self._mj_model.actuator("y").id
        # self._z_ctrl_id = self._mj_model.actuator("z").id

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

        # Set random target position
        target_pos = jax.random.uniform(
            rng3,
            shape=(3,),
            minval=self._target_bounds[0],
            maxval=self._target_bounds[1],
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._target_mocap_id].set(target_pos)
        )

        metrics = {}
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "agent_pos": jp.array(
                [
                    qpos[self._slide_qposadr[0]],
                    qpos[self._slide_qposadr[1]],
                    qpos[self._slide_qposadr[2]],
                ]
            ),
        }

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
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
        target_pos = data.mocap_pos[self._target_mocap_id]

        # Reward: negative distance to target + bonus for being close
        dist_to_target = jp.linalg.norm(agent_pos - target_pos)
        ctrl_cost = jp.linalg.norm(action)
        reward = -dist_to_target

        # Add small bonus for being very close to target
        close_bonus = 1.0 * (dist_to_target < 0.05)
        reward += close_bonus

        obs = self._get_obs(data, state.info)

        # Track step count in info and calculate timeout in timesteps
        current_step = state.info.get("step_count", 0) + 1
        # FIX: Ensure both are JAX arrays before comparison
        # current_step_jax = jp.array(current_step, dtype=jp.float32)
        # timeout_jax = jp.array(self._timeout, dtype=jp.float32)
        # timeout = current_step_jax >= timeout_jax
        timeout = current_step >= self._timeout

        # Check for termination (out of bounds, NaN, or timeout)
        out_of_bounds = jp.any(agent_pos < self._target_bounds[0] + 0.1) | jp.any(
            agent_pos > self._target_bounds[1] + 0.1
        )
        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        # Combine all termination conditions
        done = nan_condition | out_of_bounds | timeout
        done = done.astype(float)

        # def _print_debug(nan_cond, out_bounds, timeout_val, done_val, step_count):
        #     print(f"Step {step_count}:")
        #     print(f"  nan_condition: {nan_cond}")
        #     print(f"  out_of_bounds: {out_bounds}")
        #     print(f"  timeout: {timeout_val}")
        #     print(f"  done: {done_val}")
        #     print("---")

        # jax.debug.callback(
        #     _print_debug, nan_condition, out_of_bounds, timeout, done, current_step
        # )

        # # Debug print - use jax.debug.print with proper formatting
        # jax.debug.print(
        #     "nan_condition: {}, out_of_bounds: {}, timeout: {}, done: {}",
        #     nan_condition,
        #     out_of_bounds,
        #     timeout,
        #     done,
        # )

        # Or print individual values:
        # jax.debug.print("Step: {}, Timeout: {}", current_step, timeout)
        # jax.debug.print("Done flag: {}", done)
        # Update info with current positions and step info
        info = state.info.copy()
        info.update(
            {
                "agent_pos": agent_pos,
                "target_pos": target_pos,
                "distance": dist_to_target,
                "ctrl_cost": ctrl_cost,
                "step_count": current_step,  # Track current step
                "timeout": timeout,  # Add timeout info
            }
        )

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
        target_pos = data.mocap_pos[self._target_mocap_id]

        # Relative position to target
        relative_pos = agent_pos - target_pos

        # Return both absolute positions and relative position
        return jp.concatenate([agent_pos, target_pos, relative_pos])

    @property
    def observation_size(self) -> int:
        return 9  # agent_pos(3) + target_pos(3) + relative_pos(3)

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
