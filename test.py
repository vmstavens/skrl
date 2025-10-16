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
"""Robot gripper environment with mocap control."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import mujoco as mj
import numpy as np
import torch
from brax.envs.wrappers import training as brax_training
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import acrobot, common
from robot_descriptions import robotiq_2f85_mj_description

from skrl.envs.torch import wrap_env


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
        batch_size=64,
    )


# @dataclass
# class InitState:
#     qpos: jp.ndarray = jp.array([-3.28598e-17,-7.48728e-18,0.699964,-5.42405e-07,1,-7.05886e-20,3.18162e-06,0.00293229,0.000156209,0.00395541,-0.00494461,0.00293229,0.000156209,0.00395634,-0.00494645])
#     qvel: jp.ndarray =jp.array([3.54424e-16,7.52221e-17,-8.00541e-14,-1.90932e-15,-8.01069e-15,-1.13917e-18,6.11965e-08,-3.16056e-11,6.08475e-08,-5.74649e-08,6.11965e-08,-3.16026e-11,6.08474e-08,-5.74639e-08])
#     ctrl: jp.ndarray = jp.array([0])
#     mpos: jp.ndarray = jp.array([0,0,0.7])
#     mquat: jp.ndarray = jp.array([6.12323e-17,1,0,0])


@dataclass
class InitState:
    qpos: jp.ndarray = field(
        default_factory=lambda: jp.array(
            [
                -3.28598e-17,
                -7.48728e-18,
                0.699964,
                -5.42405e-07,
                1,
                -7.05886e-20,
                3.18162e-06,
                0.00293229,
                0.000156209,
                0.00395541,
                -0.00494461,
                0.00293229,
                0.000156209,
                0.00395634,
                -0.00494645,
            ]
        )
    )
    qvel: jp.ndarray = field(
        default_factory=lambda: jp.array(
            [
                3.54424e-16,
                7.52221e-17,
                -8.00541e-14,
                -1.90932e-15,
                -8.01069e-15,
                -1.13917e-18,
                6.11965e-08,
                -3.16056e-11,
                6.08475e-08,
                -5.74649e-08,
                6.11965e-08,
                -3.16026e-11,
                6.08474e-08,
                -5.74639e-08,
            ]
        )
    )
    ctrl: jp.ndarray = field(default_factory=lambda: jp.array([0]))
    mpos: jp.ndarray = field(default_factory=lambda: jp.array([0, 0, 0.7]))
    mquat: jp.ndarray = field(default_factory=lambda: jp.array([6.12323e-17, 1, 0, 0]))


class XPose(mjx_env.MjxEnv):
    """Robot gripper environment with mocap control."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        self.batch_size = config["batch_size"]
        super().__init__(config, config_overrides=config_overrides)

        self._init_state = InitState()

        self._mj_model = self.init()
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def init(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent.parent
        # scene path

        _empty_scene = """
<mujoco model="empty scene">

    <compiler angle="radian" autolimits="true" />
    <option timestep="0.002"
        integrator="implicitfast"
        solver="Newton"
        gravity="0 0 -9.82"
        cone="elliptic"
        sdf_iterations="5"
        sdf_initpoints="30"
        noslip_iterations="2"
    >
        <!-- impratio="100" -->
        <!-- mjMAXCONPAIR="10" -->
        <flag multiccd="enable" nativeccd="enable" />
        <!-- <flag nativeccd="enable" /> -->
    </option>

    <custom>
        <numeric data="15" name="max_contact_points" />
        <numeric data="15" name="max_geom_pairs" />
    </custom>

    <extension>
        <plugin plugin="mujoco.sensor.touch_grid" />
        <!-- <plugin plugin="mujoco.elasticity.solid" /> -->
        <!-- <plugin plugin="mujoco.elasticity.shell" /> -->
    </extension>

    <statistic center="0.3 0 0.3" extent="0.8" meansize="0.08" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="120" elevation="-20" offwidth="2000" offheight="2000" />
        <!-- <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080" /> -->

    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
            height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
            reflectance="0.2" />
    </asset>
    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.5" type="plane" material="groundplane"
            solimp="0.0 0.0 0.0 0.0 1" />
        <!-- <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" /> -->

    </worldbody>
</mujoco>
"""

        scene = mj.MjSpec.from_string(_empty_scene)

        gripper = mj.MjSpec().from_file(robotiq_2f85_mj_description.MJCF_PATH)

        # add mocap to scene
        b_mocap = scene.worldbody.add_body(
            name="mocap", mocap=True, pos=[0, 0, 0.7], euler=[np.pi, 0, 0]
        )
        b_mocap.add_geom(
            name="mocap",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
        )

        gripper.worldbody.first_body().add_site(name="tcp", pos=[0, 0, 0.15])

        joint = (
            scene.worldbody.add_frame(
                name="cable",
                pos=[0, 0, 0.7],
                euler=np.array([np.pi, 0, 0]),
            )
            .attach_body(gripper.worldbody.first_body())
            .add_freejoint(name="free")
        )

        scene.add_equality(
            name="mocap",
            type=mj.mjtEq.mjEQ_WELD,
            data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            active=True,
            name1="mocap",
            name2="base_mount",
            objtype=mj.mjtObj.mjOBJ_BODY,
            solref=[0.0000001, 1],
        )

        m = scene.compile()
        return m

    def _post_init(self) -> None:
        free_jid = self._mj_model.joint("free").id
        self._free_qposaddr = self._mj_model.jnt_qposadr[free_jid + 3]
        self._space_max = jp.array([1, 1, 1])
        self._space_min = jp.array([-1, -1, 0.3])

        # Get mocap body ID
        self._mocap_id = self._mj_model.body("mocap").id
        # Fixed orientation (from init state)
        self._fixed_mocap_quat = self._init_state.mquat.copy()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Generate random position for mocap
        random_position = jax.random.uniform(
            rng, shape=(3,), minval=self._space_min, maxval=self._space_max
        )

        # Create initial qpos and qvel from init state
        qpos = self._init_state.qpos.copy()
        qvel = self._init_state.qvel.copy()

        # Set initial mocap position with fixed orientation
        mocap_pos = jp.array(
            [random_position[0], random_position[1], random_position[2]]
        )

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

        # Set mocap position (fixed orientation)
        data = data.replace(
            mocap_pos=mocap_pos.reshape(1, 3),
            mocap_quat=self._fixed_mocap_quat.reshape(1, 4),
        )

        self._reset_randomize(data, rng)

        metrics = {}
        info = {"rng": rng}

        reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name

        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = state.data

        print(f"{action=}")
        print(f"{data.mocap_pos[0]=}")

        # Update only mocap position (keep orientation fixed)
        new_mocap_pos = data.mocap_pos[0] + action
        new_mocap_pos = jp.clip(new_mocap_pos, self._space_min, self._space_max)

        # Replace only the mocap_pos, keep mocap_quat the same
        new_data = data.replace(
            mocap_pos=new_mocap_pos.reshape(1, 3),
            mocap_quat=self._fixed_mocap_quat.reshape(1, 4),  # Keep fixed orientation
        )

        # Step the simulation with zero control inputs (mocap drives everything via equality constraint)
        stepped_data = mjx_env.step(
            self.mjx_model, new_data, jp.zeros(self.mjx_model.nu), self.n_substeps
        )

        reward = self._get_reward(stepped_data, action, state.info, state.metrics)
        obs = self._get_obs(stepped_data, state.info)

        # Check for termination conditions
        done = jp.isnan(stepped_data.qpos).any() | jp.isnan(stepped_data.qvel).any()
        done = done.astype(float)

        return mjx_env.State(stepped_data, obs, reward, done, state.metrics, state.info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info  # Unused.

        # Get gripper TCP position from site
        tcp_site_id = self._mj_model.site("tcp").id
        tcp_pos = data.site_xpos[tcp_site_id]

        # Get mocap position
        mocap_pos = data.mocap_pos[0]

        # Get gripper joint positions (finger states)
        gripper_joint_pos = data.qpos[7:]  # Adjust indices based on your robot

        # Get gripper joint velocities
        gripper_joint_vel = data.qvel[6:]  # Adjust indices based on your robot

        return jp.concatenate(
            [
                tcp_pos,  # 3D TCP position
                mocap_pos,  # 3D mocap target position
                gripper_joint_pos,  # Gripper finger positions
                gripper_joint_vel,  # Gripper finger velocities
            ]
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        del info, metrics  # Unused for now

        # Get TCP and mocap positions
        tcp_site_id = self._mj_model.site("tcp").id
        tcp_pos = data.site_xpos[tcp_site_id]
        mocap_pos = data.mocap_pos[0]

        # Reward for reaching target (negative distance)
        distance = jp.linalg.norm(tcp_pos - mocap_pos)
        reach_reward = -distance

        # Small penalty for large actions to encourage smooth control
        action_penalty = -0.01 * jp.linalg.norm(action)

        total_reward = reach_reward + action_penalty
        return total_reward

    def _reset_randomize(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        # Add randomization if needed
        return data

    @property
    def action_size(self) -> int:
        return 3  # 3D position control for mocap

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        raise ValueError("obs...")


if __name__ == "__main__":
    env = XPose()
    env_name = "xpose"

    from mujoco_playground import registry

    registry.dm_control_suite.register_environment(
        env_name=env_name, env_class=XPose, cfg_class=default_config
    )
    env = registry.dm_control_suite.load(env_name)

    num_envs = 10

    # env = VmapWrapper(env, num_envs)  # pytype: disable=wrong-arg-types
    # env = brax_training.VmapWrapper(env, num_envs)  # pytype: disable=wrong-arg-types
    # wrap(env)
    from brax.envs import create
    from mujoco_playground import wrapper

    from skrl.envs.wrappers.torch.playground_envs import VmapWrapper

    env = VmapWrapper(env, num_envs)

    env = wrap_env(env)

    state = env.reset()

    a = env.action_space.sample()

    print(f"{a=}")

    for _ in range(100):
        next_states, rewards, terminated, truncated, infos = env.step(
            torch.ones(num_envs)
        )
        print(f"{rewards=}")
        # print(f"{terminated=}")
        # print(f"{next_states=}")
        # print(f"{state=}")
    print("done")
