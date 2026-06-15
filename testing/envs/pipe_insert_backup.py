import time
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
from mujoco import glfw, mjx
from mujoco_playground._src import mjx_env

from utils.mjx import ObjType, get_pose, is_stable


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
                    solref="0.001 3"
                />
            </composite>
    </worldbody>

</mujoco>
    """

    return mj.MjSpec.from_string(_XML)


def pipe(length: float = 0.1, n_elements: int = 30) -> mj.MjSpec:
    # <geom type="box" rgba="1 0 0 1" solref="0.000000001 1" pos="0 -0.018 0" size=".004 .001 {length / 2}" friction="0.2 0.2 0.2" />
    _XML = f"""
    <mujoco>
        <worldbody>
            <body >
                <replicate sep="hole:" count="{n_elements}" euler="0 0 20">
                    <geom type="box" rgba="1 0 0 1" solref="0.00001 2" pos="0 -0.018 0" size=".004 .001 {length / 2}" friction="0.2 0.2 0.2" />
                </replicate>
            </body>
        </worldbody>
    </mujoco>
    """
    return mj.MjSpec().from_string(_XML)


def pipe2(length: float = 0.1) -> mj.MjSpec:
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


def empty() -> mj.MjSpec:
    # <flag multiccd="enable" nativeccd="enable" />
    _XML = """
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
            ls_iterations="10"
        >
            <!-- impratio="100" -->
            <!-- mjMAXCONPAIR="10" -->
                <flag eulerdamp="disable" />
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
    return mj.MjSpec().from_string(_XML)


class PipeInsert(mjx_env.MjxEnv):
    """Simple 3D position control environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)

        self._mj_model = self._init2()
        self.ctrl_scale: float = 0.01
        # self._mj_model = self._init()
        # self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)

        self._post_init()

    def _init(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = mj.MjSpec().from_file((_HERE / "envs/assets/empty.xml").as_posix())

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

        cable.bodies[1].add_site(
            name="keypoint_3", group=1, rgba=[0, 1, 0, 1], size=[0.01, 0.01, 0.01]
        )
        cable.bodies[3].add_site(
            name="keypoint_2", group=1, rgba=[1, 1, 0, 1], size=[0.01, 0.01, 0.01]
        )
        cable.bodies[5].add_site(
            name="keypoint_1", group=1, rgba=[1, 0, 0, 1], size=[0.01, 0.01, 0.01]
        )

        scene.worldbody.add_camera(
            name="cam",
            pos=[0.721, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
        )

        # <camera pos="0.721 0.234 0.156" xyaxes="-0.037 0.999 0.000 -0.001 -0.000 1.000"/>

        gripper = scene.worldbody.add_body(
            name="gripper", pos=[0, 0.45, 0.1], euler=[0, 0, 1.57]
        )
        gripper.add_geom(
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
            # rgba=[0, 0, 0, 0],
        )
        gripper.add_joint(
            name="x", type=mj.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.2, 0.2]
        )
        gripper.add_joint(
            name="y", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-0.2, 0.2]
        )
        gripper.add_joint(
            name="z", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.2, 0.2]
        )
        gripper.add_joint(
            name="rx",
            type=mj.mjtJoint.mjJNT_HINGE,
            axis=[1, 0, 0],
            range=[-np.pi / 3, np.pi / 3],
        )
        gripper.add_joint(
            name="ry",
            type=mj.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
            range=[-np.pi / 3, np.pi / 3],
        )
        gripper.add_joint(
            name="rz",
            type=mj.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            range=[-np.pi / 3, np.pi / 3],
        )
        scene.add_actuator(
            name="x", target="x", trntype=mj.mjtTrn.mjTRN_JOINT, ctrlrange=[-0.2, 0.2]
        ).set_to_position(kp=10, kv=100)
        scene.add_actuator(
            name="y", target="y", trntype=mj.mjtTrn.mjTRN_JOINT, ctrlrange=[-0.2, 0.2]
        ).set_to_position(kp=10, kv=100)
        scene.add_actuator(
            name="z", target="z", trntype=mj.mjtTrn.mjTRN_JOINT, ctrlrange=[-0.2, 0.2]
        ).set_to_position(kp=10, kv=100)
        scene.add_actuator(
            name="rx",
            target="rx",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=[-np.pi / 3, np.pi / 3],
        ).set_to_position(kp=10, kv=1000)
        scene.add_actuator(
            name="ry",
            target="ry",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=[-np.pi / 3, np.pi / 3],
        ).set_to_position(kp=10, kv=1000)
        scene.add_actuator(
            name="rz",
            target="rz",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=[-np.pi / 3, np.pi / 3],
        ).set_to_position(kp=10, kv=1000)

        scene.worldbody.add_frame(pos=[0, 0, 0.2], euler=[1.57, 0, 0]).attach_body(
            # scene.worldbody.add_frame(pos=[0, 0, 0.2], euler=[0, 1.57, 0.157]).attach_body(
            # scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 0]).attach_body(
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
            # b.add_site()

        scene.add_equality(
            name="weld",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_BODY,
            name1="gripper",
            name2=b.name,
            data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            solref=[0.000001, 2],
            # solref=[0.000001, 2],
        )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _init2(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        pip = pipe2()

        pip.worldbody.first_body().add_site(
            name="target_1", pos=[0, 0.015, -0.05], group=1, rgba=[1, 0, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_2", pos=[0, 0.015, 0], group=1, rgba=[1, 1, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_3", pos=[0, 0.015, 0.05], group=1, rgba=[0, 1, 0, 1]
        )

        cable = mjs_cable(count="40 1 1", twist=60000.0 * 2, bend=10000000.0 * 2)

        cable.bodies[1].add_site(name="keypoint_3", group=1, rgba=[0, 1, 0, 1])
        cable.bodies[3].add_site(name="keypoint_2", group=1, rgba=[1, 1, 0, 1])
        cable.bodies[5].add_site(name="keypoint_1", group=1, rgba=[1, 0, 0, 1])

        scene.worldbody.add_camera(
            name="cam",
            pos=[0.721, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
        )

        # <camera pos="0.721 0.234 0.156" xyaxes="-0.037 0.999 0.000 -0.001 -0.000 1.000"/>

        SLIDE_RANGE = [-0.3, 0.3]

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
        gripper.add_joint(
            name="x", type=mj.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=SLIDE_RANGE
        )
        gripper.add_joint(
            name="y", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=SLIDE_RANGE
        )
        gripper.add_joint(
            name="z", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=SLIDE_RANGE
        )
        # gripper.add_joint(
        #     name="rx",
        #     type=mj.mjtJoint.mjJNT_HINGE,
        #     axis=[1, 0, 0],
        #     range=[-np.pi / 3, np.pi / 3],
        # )
        # gripper.add_joint(
        #     name="ry",
        #     type=mj.mjtJoint.mjJNT_HINGE,
        #     axis=[0, 1, 0],
        #     range=[-np.pi / 3, np.pi / 3],
        # )
        # gripper.add_joint(
        #     name="rz",
        #     type=mj.mjtJoint.mjJNT_HINGE,
        #     axis=[0, 0, 1],
        #     range=[-np.pi / 3, np.pi / 3],
        # )
        scene.add_actuator(
            name="x", target="x", trntype=mj.mjtTrn.mjTRN_JOINT, ctrlrange=SLIDE_RANGE
        ).set_to_position(kp=100, kv=100)
        scene.add_actuator(
            name="y", target="y", trntype=mj.mjtTrn.mjTRN_JOINT, ctrlrange=SLIDE_RANGE
        ).set_to_position(kp=100, kv=100)
        scene.add_actuator(
            name="z", target="z", trntype=mj.mjtTrn.mjTRN_JOINT, ctrlrange=SLIDE_RANGE
        ).set_to_position(kp=100, kv=100)
        # scene.add_actuator(
        #     name="rx",
        #     target="rx",
        #     trntype=mj.mjtTrn.mjTRN_JOINT,
        #     ctrlrange=[-np.pi / 3, np.pi / 3],
        # ).set_to_position(kp=100, kv=1000)
        # scene.add_actuator(
        #     name="ry",
        #     target="ry",
        #     trntype=mj.mjtTrn.mjTRN_JOINT,
        #     ctrlrange=[-np.pi / 3, np.pi / 3],
        # ).set_to_position(kp=100, kv=1000)
        # scene.add_actuator(
        #     name="rz",
        #     target="rz",
        #     trntype=mj.mjtTrn.mjTRN_JOINT,
        #     ctrlrange=[-np.pi / 3, np.pi / 3],
        # ).set_to_position(kp=100, kv=1000)

        scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 3.14]).attach_body(
            # scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 0]).attach_body(
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
            data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            solref=[0.000001, 2],
        )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

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

        self.target_ids = [self._mj_model.site(f"target_{i + 1}").id for i in range(3)]
        self.keypoint_ids = [
            self._mj_model.site(f"keypoint_{i + 1}").id for i in range(3)
        ]
        self.QPOS = np.load("testing/experiments/pipe_insert/constants/qpos.npy")
        self.QVEL = np.load("testing/experiments/pipe_insert/constants/qvel.npy")
        self.CTRL = np.load("testing/experiments/pipe_insert/constants/ctrl.npy")

        self.QPOS = jp.array(self.QPOS)
        self.QVEL = jp.array(self.QVEL)
        self.CTRL = jp.array(self.CTRL)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Split RNG for agent position, velocity, and target position)

        # Initialize data
        data = mjx_env.init(
            self.mjx_model, qpos=self.QPOS, qvel=self.QVEL, ctrl=self.CTRL
        )
        metrics = {}
        info = {}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # a = action

        _data = state.data
        a = state.data.ctrl + action
        # a = state.data.ctrl + self.ctrl_scale * action
        data = mjx_env.step(self.mjx_model, state.data, a, self.n_substeps)

        # Reward: negative distance to target + bonus for being close
        reward = self._get_reward(data)

        obs = self._get_obs(_data, data, state.info)

        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        # Combine all termination conditions
        done = nan_condition

        # Use lax.cond instead of if statement
        def quit_simulation(_):
            # jax.debug.print("Quitting due to instability")
            # You can't actually call quit() in JAX, so we need a JAX-compatible alternative
            # Option A: Mark as done and let the outer loop handle it
            return state.replace(data=data, info=state.info)

        def continue_simulation(_):
            return state.replace(data=data, info=state.info)

        # Instead of quit(), we'll mark the episode as done
        state = jax.lax.cond(
            ~is_stable(data),  # Note: ~ instead of "not" for JAX arrays
            quit_simulation,
            continue_simulation,
            operand=None,
        )

        # Check if all conditions are True and conditionally raise error
        # jax.lax.cond(jp.all(nan_condition), raise_error, no_op, operand=None)
        # done = nan_condition | out_of_bounds
        done = done.astype(float)

        info = state.info.copy()

        return mjx_env.State(data, obs, reward, done, state.metrics, info)

    def _get_reward(self, data: mjx.Data) -> float:
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

        return reward

    def _get_site_pos(self, data: mjx.Data) -> jax.Array:
        # relative distances to targets
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

        o = jp.array([dist_to_target_1, dist_to_target_2, dist_to_target_3])

        # Return both absolute positions and relative position
        return o

    def _get_obs(
        self, _data: mjx.Data, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        # Return agent position and target position (6D observation)
        s0 = self._get_site_pos(_data)
        s1 = self._get_site_pos(data)
        dt = self.mj_model.opt.timestep
        v1 = (s1 - s0) / dt

        o = jp.concatenate([s1, v1])
        return o

    @property
    def observation_size(self) -> int:
        """Compute observation size by calling _get_obs with dummy data"""
        # Create dummy data for size computation
        dummy_data = mjx.make_data(self.mjx_model)
        dummy_info = {}

        # Get observation and check its shape
        obs = self._get_obs(dummy_data, dummy_data, dummy_info)
        # print(f"{obs=}")
        # print(f"{obs.shape[0]=}")
        return obs.shape[0]  # Get the last dimension (feature size)
        # return 6  # Get the last dimension (feature size)

    @property
    def action_size(self) -> int:
        return self.mj_model.nu

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


import glfw
import mujoco.viewer

if __name__ == "__main__":
    env = PipeInsert()
    model = env.mj_model
    data = mj.MjData(model)

    def cb(key: int) -> None:
        if key is glfw.KEY_SPACE:
            qpos = data.qpos
            qvel = data.qvel
            ctrl = data.ctrl
            np.save("testing/experiments/pipe_insert/constants/qpos.npy", qpos)
            np.save("testing/experiments/pipe_insert/constants/qvel.npy", qvel)
            np.save("testing/experiments/pipe_insert/constants/ctrl.npy", ctrl)

    with mujoco.viewer.launch_passive(
        model=model, data=data, key_callback=cb
    ) as viewer:
        # set gui camera to the specified in the model
        viewer.cam.azimuth = model.vis.global_.azimuth
        viewer.cam.elevation = model.vis.global_.elevation
        viewer.cam.lookat = model.stat.center
        viewer.cam.distance = model.stat.extent

        _scene = viewer.user_scn

        while viewer.is_running():
            step_start = time.time()

            # step simulation one time step
            mj.mj_step(model, data)

            # input()

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
