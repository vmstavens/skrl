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
        xml_string = """
        <mujoco>

        <option iterations="1" ls_iterations="5" timestep="0.004" integrator="implicitfast">
            <flag eulerdamp="disable"/>
        </option>

            <worldbody>
                <body name="agent" gravcomp="1">
                    <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
                    <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
                    <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
                    <geom name="agent" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0"/>
                </body>
            </worldbody>
            <actuator>
                <position name="x" joint="x" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
                <position name="y" joint="y" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
                <position name="z" joint="z" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
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
        # print(self.mjx_model, state.data, action, self.n_substeps)
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


# class XPose(mjx_env.MjxEnv):
#     """Robot gripper environment with mocap control."""

#     def __init__(
#         self,
#         config: config_dict.ConfigDict = default_config(),
#         config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
#     ):
#         # Don't pass batch_size to parent - create config without it
#         if "batch_size" in config:
#             del config["batch_size"]

#         super().__init__(config, config_overrides=config_overrides)

#         # Store batch_size separately for your use
#         self.batch_size = config.get("batch_size", 1)

#         self._goal_state = GoalState()
#         self._mj_model = self.init()
#         self._mj_model.opt.timestep = self.sim_dt
#         self._mjx_model = mjx.put_model(self._mj_model)
#         self._post_init()
#         self._ctrl_scale = 0.05

#     def init(self) -> mj.MjModel:
#         _empty_scene = """
#     <mujoco model="cart-pole">
#     <compiler angle="radian"/>
#     <option gravity="0 0 -9.82" integrator="implicitfast" cone="elliptic">
#         <flag contact="disable" eulerdamp="disable"/>
#     </option>
#     <asset>
#         <texture type="2d" colorspace="auto" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
#         <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
#     </asset>
#     <worldbody>
#         <geom name="floor" size="4 4 0.2" pos="0 0 -0.05" type="plane" material="groundplane"/>
#         <body name="agent">
#         <joint name="x" pos="0 0 0" axis="1 0 0" type="slide" range="-1 1"/>
#         <joint name="y" pos="0 0 0" axis="0 1 0" type="slide" range="-1 1"/>
#         <joint name="z" pos="0 0 0" axis="0 0 1" type="slide" range="-1 1"/>
#         <geom name="agent" size="0.01 0.01 0.01" type="box" contype="0" conaffinity="0"/>
#         </body>
#     </worldbody>
#     <actuator>
#         <general name="x" joint="x" ctrlrange="-1 1" biastype="affine" gainprm="1000" biasprm="0 -1000 -1000"/>
#         <general name="y" joint="y" ctrlrange="-1 1" biastype="affine" gainprm="1000" biasprm="0 -1000 -1000"/>
#         <general name="z" joint="z" ctrlrange="-1 1" biastype="affine" gainprm="1000" biasprm="0 -1000 -1000"/>
#     </actuator>
#     </mujoco>
#     """
#         scene = mj.MjSpec.from_string(_empty_scene)
#         m = scene.compile()
#         return m

#     def _post_init(self) -> None:
#         # Store individual joint addresses like the cartpole example
#         pass
#         self._x_qposadr = int(self._mj_model.jnt_qposadr[self._mj_model.joint("x").id])
#         self._y_qposadr = int(self._mj_model.jnt_qposadr[self._mj_model.joint("y").id])
#         self._z_qposadr = int(self._mj_model.jnt_qposadr[self._mj_model.joint("z").id])

#         # Store actuator IDs
#         self._x_ctrl_id = self._mj_model.actuator("x").id
#         self._y_ctrl_id = self._mj_model.actuator("y").id
#         self._z_ctrl_id = self._mj_model.actuator("z").id

#         # Get joint limits
#         self._qpos_min = jp.array(
#             [
#                 self._mj_model.jnt_range[self._x_qposadr, 0],
#                 self._mj_model.jnt_range[self._y_qposadr, 0],
#                 self._mj_model.jnt_range[self._z_qposadr, 0],
#             ]
#         )
#         self._qpos_max = jp.array(
#             [
#                 self._mj_model.jnt_range[self._x_qposadr, 1],
#                 self._mj_model.jnt_range[self._y_qposadr, 1],
#                 self._mj_model.jnt_range[self._z_qposadr, 1],
#             ]
#         )

#     def reset(self, rng: jax.Array) -> mjx_env.State:
#         # Check if we're being traced for shape inference
#         if hasattr(rng, "aval"):
#             # Return minimal state for shape inference
#             data = mjx.make_data(self.mjx_model)
#             obs = jp.zeros(3)
#             reward, done = jp.zeros(2)
#             metrics = {}
#             info = {"rng": rng}
#             return mjx_env.State(data, obs, reward, done, metrics, info)

#         # Normal reset logic (same as cartpole)
#         rng1, rng2, rng3 = jax.random.split(rng, 3)

#         qpos = jp.zeros(self.mjx_model.nq)
#         qpos = qpos.at[self._x_qposadr].set(
#             jax.random.uniform(rng1, minval=self._qpos_min[0], maxval=self._qpos_max[0])
#         )
#         qpos = qpos.at[self._y_qposadr].set(
#             jax.random.uniform(rng2, minval=self._qpos_min[1], maxval=self._qpos_max[1])
#         )
#         qpos = qpos.at[self._z_qposadr].set(
#             jax.random.uniform(rng3, minval=self._qpos_min[2], maxval=self._qpos_max[2])
#         )

#         qvel = 0.01 * jax.random.normal(rng, (self.mjx_model.nv,))

#         data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

#         metrics = {}
#         info = {"rng": rng}

#         reward, done = jp.zeros(2)
#         obs = self._get_obs(data, info)
#         return mjx_env.State(data, obs, reward, done, metrics, info)

#     # def reset(self, rng: jax.Array) -> mjx_env.State:
#     #     rng1, rng2, rng3 = jax.random.split(rng, 3)

#     #     # # Start with zeros like cartpole example
#     #     # qpos = jp.zeros(self.mjx_model.nq)
#     #     # qvel = jp.zeros(self.mjx_model.nv)
#     #     # ctrl = jp.zeros(self.mjx_model.nu)

#     #     # # Set random positions for each joint individually
#     #     # qpos = qpos.at[self._x_qposadr].set(
#     #     #     jax.random.uniform(rng1, minval=self._qpos_min[0], maxval=self._qpos_max[0])
#     #     # )
#     #     # qpos = qpos.at[self._y_qposadr].set(
#     #     #     jax.random.uniform(rng2, minval=self._qpos_min[1], maxval=self._qpos_max[1])
#     #     # )
#     #     # qpos = qpos.at[self._z_qposadr].set(
#     #     #     jax.random.uniform(rng3, minval=self._qpos_min[2], maxval=self._qpos_max[2])
#     #     # )

#     #     # jax.debug.callback(self.print_debug, "\tqpos", qpos)

#     #     # # Set control to match position (position control)
#     #     # ctrl = ctrl.at[self._x_ctrl_id].set(qpos[self._x_qposadr])
#     #     # ctrl = ctrl.at[self._y_ctrl_id].set(qpos[self._y_qposadr])
#     #     # ctrl = ctrl.at[self._z_ctrl_id].set(qpos[self._z_qposadr])

#     #     # print(f"\t{type(qpos)=} {qpos=} {qpos[0]=}")
#     #     # print(f"\t{type(ctrl)=} {ctrl=} {ctrl[0]=}")
#     #     # print(f"\t{type(qvel)=} {qvel=} {qvel[0]=}")
#     #     # # print(self._mjx_model)
#     #     qvel = jp.zeros(self.mjx_model.nv)
#     #     qpos = jp.zeros(self.mjx_model.nq)
#     #     ctrl = jp.zeros(self.mjx_model.nu)

#     #     # Try this instead of mjx_env.init
#     #     def safe_init(model, qpos=None, qvel=None, ctrl=None):
#     #         """Initialize data without going through the problematic path"""
#     #         data = mjx.make_data(model)
#     #         if qpos is not None:
#     #             data = data.replace(qpos=qpos)
#     #         if qvel is not None:
#     #             data = data.replace(qvel=qvel)
#     #         if ctrl is not None:
#     #             data = data.replace(ctrl=ctrl)
#     #         return data

#     #     qvel = jp.zeros(self.mjx_model.nv)
#     #     qpos = jp.zeros(self.mjx_model.nq)

#     #     # Use safe_init instead
#     #     data = safe_init(self.mjx_model, qpos=qpos, qvel=qvel)
#     #     # data = mjx_env.init(model=self.mjx_model)
#     #     # data = mjx_env.init(model=self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

#     #     metrics = {}
#     #     info = {"rng": rng}

#     #     reward, done = jp.zeros(2)
#     #     obs = self._get_obs(data, info)
#     #     return mjx_env.State(data, obs, reward, done, metrics, info)

#     # Debug with callback
#     def print_debug(self, name, value):
#         print(f"{name} = {value}")
#         return None

#     def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
#         data = state.data

#         jax.debug.callback(self.print_debug, "data.qpos", data.qpos)

#         # Debug shapes before the problematic operation
#         current_x_pos = data.qpos[self._x_qposadr]
#         jax.debug.callback(self.print_debug, "current_x_pos", current_x_pos)

#         # Scale action
#         jax.debug.callback(self.print_debug, "action", action)
#         action = action * self._ctrl_scale
#         jax.debug.callback(self.print_debug, "action * self._ctrl_scale", action)

#         new_ctrl = jp.zeros(self.mjx_model.nu, dtype=jp.float32)
#         # Scale action
#         action = action * self._ctrl_scale

#         # Update control based on action (position control)
#         current_x_pos = data.qpos[self._x_qposadr]
#         current_y_pos = data.qpos[self._y_qposadr]
#         current_z_pos = data.qpos[self._z_qposadr]

#         new_ctrl = jp.zeros(self.mjx_model.nu)
#         # new_ctrl = new_ctrl.at[self._x_ctrl_id].set(current_x_pos + action[0])
#         # new_ctrl = new_ctrl.at[self._y_ctrl_id].set(current_y_pos + action[1])
#         # new_ctrl = new_ctrl.at[self._z_ctrl_id].set(current_z_pos + action[2])

#         target_x = current_x_pos + action[0]
#         target_y = current_y_pos + action[1]
#         target_z = current_z_pos + action[2]

#         # Check if targets are within joint limits
#         target_x = jp.clip(target_x, self._qpos_min[0], self._qpos_max[0])
#         target_y = jp.clip(target_y, self._qpos_min[1], self._qpos_max[1])
#         target_z = jp.clip(target_z, self._qpos_min[2], self._qpos_max[2])

#         new_ctrl = new_ctrl.at[self._x_ctrl_id].set(target_x)
#         new_ctrl = new_ctrl.at[self._y_ctrl_id].set(target_y)
#         new_ctrl = new_ctrl.at[self._z_ctrl_id].set(target_z)

#         # Step simulation with new control
#         jax.debug.callback(self.print_debug, "new_ctrl.shape", new_ctrl.shape)
#         jax.debug.callback(self.print_debug, "new_ctrl", new_ctrl)
#         jax.debug.callback(self.print_debug, "self.n_substeps", self.n_substeps)

#         print(data)
#         print(new_ctrl)
#         print(self.n_substeps)

#         # Usage in your step method:
#         stepped_data = mjx_env.step(
#             model=self.mjx_model,
#             data=data,
#             action=jp.zeros(self.mjx_model.nu),
#             # model=self.mjx_model, data=data, action=new_ctrl, n_substeps=self.n_substeps
#         )

#         jax.debug.callback(self.print_debug, "stepped_data.qpos", stepped_data.qpos)

#         # Simple reward: negative distance to origin
#         current_pos = jp.array(
#             [
#                 stepped_data.qpos[self._x_qposadr],
#                 stepped_data.qpos[self._y_qposadr],
#                 stepped_data.qpos[self._z_qposadr],
#             ]
#         )
#         dist_to_goal = jp.linalg.norm(current_pos)
#         reward = -dist_to_goal

#         dist_to_goal = jp.linalg.norm(current_pos)
#         reward = -dist_to_goal

#         obs = self._get_obs(stepped_data, state.info)

#         # Check for termination
#         done = jp.isnan(stepped_data.qpos).any() | jp.isnan(stepped_data.qvel).any()

#         jax.debug.callback(self.print_debug, "done", done)
#         done = done.astype(float)

#         return mjx_env.State(stepped_data, obs, reward, done, state.metrics, state.info)

#     def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
#         del info  # Unused
#         # Return current position
#         return data.qpos

#     @property
#     def action_size(self) -> int:
#         return 3

#     @property
#     def observation_size(self) -> int:
#         return 3

#     @property
#     def mj_model(self) -> mujoco.MjModel:
#         return self._mj_model

#     @property
#     def mjx_model(self) -> mjx.Model:
#         return self._mjx_model

#     @property
#     def xml_path(self) -> str:
#         return "empty_scene"
