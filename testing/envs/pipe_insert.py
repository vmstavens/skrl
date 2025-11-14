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

from testing.mj import ObjType
from utils.mjx import get_pose


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
    )


def mjs_cable(
    model_name: str = "cable",
    prefix: str = "cable:",
    curve: str = "0 s 0",
    count: str = "10 1 1",
    twist: float = 60000.0,
    bend: float = 10000000.0,
    vmax: float = 0,
    size: str = 1,
    segment_size: float = 0.002,
    mass: float = 0.00035,
    rgba: list = "0.2 0.2 0.2 1",
    initial: str = "free",
) -> mj.MjSpec:
    _XML = f"""
<mujoco model="{model_name}">
    <extension>
        <plugin plugin="mujoco.elasticity.cable"/>
    </extension>

    <worldbody>
    
            <composite prefix="{prefix}" type="cable" curve="{curve}" count="{count}" size="{size}" initial="{initial}">
                <plugin plugin="mujoco.elasticity.cable">
                    <config key="twist" value="{twist}" />
                    <config key="bend" value="{bend}" />
                    <config key="vmax" value="{vmax}" />
                </plugin>
                <joint kind="main" damping="1e-2" armature="0.001" />
                <geom type="capsule"
                    size="{segment_size}"
                    rgba="{rgba}"
                    mass="{mass}"
                    friction="0.3 0.3 0.3"
                    condim="4"
                    solref="0.00001 1 "
                />
            </composite>
    </worldbody>

</mujoco>
    """

    return mj.MjSpec.from_string(_XML)


def pipe(length: float = 0.1) -> mj.MjSpec:
    _XML = f"""
    <mujoco>
        <worldbody>
            <body euler="0 0 0" pos="0 0 0">
                <replicate sep="hole:" count="30" euler="0 0 20">
                    <geom type="box" solref="0.000000001 1" pos="0 -0.018 0" size=".004 .001 {length / 2}" friction="0.2 0.2 0.2" />
                </replicate>
            </body>
        </worldbody>
    </mujoco>
    """
    return mj.MjSpec().from_string(_XML)


class CableInsert(mjx_env.MjxEnv):
    """Simple 3D position control environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)

        self._mj_model = self._init()
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)

        self._post_init()

    def _init(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        pip = pipe()

        pip.worldbody.first_body().add_site(
            name="target_1", pos=[0, 0, -0.05], group=1, rgba=[1, 0, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_2", pos=[0, 0, 0], group=1, rgba=[1, 1, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_3", pos=[0, 0, 0.05], group=1, rgba=[0, 1, 0, 1]
        )

        cable = mjs_cable(count="40 1 1")

        cable.bodies[1].add_site(name="keypoint_3", group=1, rgba=[0, 1, 0, 1])
        cable.bodies[3].add_site(name="keypoint_2", group=1, rgba=[1, 1, 0, 1])
        cable.bodies[5].add_site(name="keypoint_1", group=1, rgba=[1, 0, 0, 1])

        scene.worldbody.add_camera(
            name="cam",
            pos=[0.721, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
        )

        # <camera pos="0.721 0.234 0.156" xyaxes="-0.037 0.999 0.000 -0.001 -0.000 1.000"/>

        gripper = scene.worldbody.add_body(
            name="gripper", pos=[0, 0.4, 0.1], euler=[0, 0, 1.57]
        )
        gripper.add_geom(
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
            # rgba=[0, 0, 0, 0],
        )
        gripper.add_joint(name="y", type=mj.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        gripper.add_joint(name="z", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
        gripper.add_joint(
            name="theta",
            type=mj.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
            range=[-np.pi / 3, np.pi / 3],
        )
        scene.add_actuator(target="y", trntype=mj.mjtTrn.mjTRN_JOINT).set_to_position(
            kp=100, kv=100
        )
        scene.add_actuator(target="z", trntype=mj.mjtTrn.mjTRN_JOINT).set_to_position(
            kp=100, kv=100
        )
        scene.add_actuator(
            target="theta",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=[-np.pi / 3, np.pi / 3],
        ).set_to_position(kp=300, kv=100)

        scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 0]).attach_body(
            pip.worldbody.first_body()
        )
        scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
            cable.worldbody.first_body()
        )

        b = None
        b0 = cable.worldbody.first_body()
        for i in range(10):
            b = b0.first_body()
            b0 = b

        scene.add_equality(
            name="weld",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_BODY,
            name1="gripper",
            name2=b.name,
            # name2=f"{left.modelname}/link_left_wrist_x",
            data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            solref=[0.000000001, 1],
        )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _post_init(self) -> None:
        # Store joint addresses
        self._slide_jid = [
            self._mj_model.joint("y").id,
            self._mj_model.joint("z").id,
            self._mj_model.joint("theta").id,
        ]
        self._slide_qposadr = [
            self._mj_model.jnt_qposadr[self._slide_jid[0]],
            self._mj_model.jnt_qposadr[self._slide_jid[1]],
            self._mj_model.jnt_qposadr[self._slide_jid[2]],
        ]
        self._slide_cid = [
            self._mj_model.actuator("y").id,
            self._mj_model.actuator("z").id,
            self._mj_model.actuator("theta").id,
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

        self.target_ids = [self._mj_model.site(f"target_{i + 1}").id for i in range(3)]
        self.keypoint_ids = [
            self._mj_model.site(f"keypoint_{i + 1}").id for i in range(3)
        ]

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

        T_w_target_1 = get_pose(
            self._mjx_model, data, self.target_ids[0], obj_type=ObjType.SITE
        )
        T_w_target_2 = get_pose(
            self._mjx_model, data, self.target_ids[1], obj_type=ObjType.SITE
        )
        T_w_target_3 = get_pose(
            self._mjx_model, data, self.target_ids[2], obj_type=ObjType.SITE
        )

        T_w_keypoint_1 = get_pose(
            self._mjx_model, data, self.keypoint_ids[0], obj_type=ObjType.SITE
        )
        T_w_keypoint_2 = get_pose(
            self._mjx_model, data, self.keypoint_ids[1], obj_type=ObjType.SITE
        )
        T_w_keypoint_3 = get_pose(
            self._mjx_model, data, self.keypoint_ids[2], obj_type=ObjType.SITE
        )

        dist_to_target_1 = jp.linalg.norm(
            T_w_target_1.translation() - T_w_keypoint_1.translation()
        )
        dist_to_target_2 = jp.linalg.norm(
            T_w_target_2.translation() - T_w_keypoint_2.translation()
        )
        dist_to_target_3 = jp.linalg.norm(
            T_w_target_3.translation() - T_w_keypoint_3.translation()
        )

        dist_to_target = dist_to_target_1 + dist_to_target_2 + dist_to_target_3
        reward = -dist_to_target

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
