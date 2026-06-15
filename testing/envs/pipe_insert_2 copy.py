import pickle
import time
import warnings
import xml.etree.ElementTree as ET
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
        ctrl_dt=0.002,
        sim_dt=0.002,
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
                <!-- <joint kind="main" damping="1e-2" armature="0.001" /> -->
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


def pipe(length: float = 0.1) -> mj.MjSpec:
    _XML = f"""
    <mujoco>
        <worldbody>
            <body euler="0 0 0" pos="0 0 0">
                <replicate sep="hole:" count="40" euler="0 0 20">
                    <geom type="box" solref="0.0000000001 1" pos="0 -0.03 0" size=".008 .001 {length / 2}" friction="0.2 0.2 0.2" />
                </replicate>
            </body>
        </worldbody>
    </mujoco>
    """
    return mj.MjSpec().from_string(_XML)


def empty() -> mj.MjSpec:
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
            <!-- <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" /> -->

        </worldbody>

    </mujoco>
    """
    return mj.MjSpec().from_string(_XML)


def _parse_float_list(value: Optional[str]) -> list[float]:
    if not value:
        return []
    return [float(item) for item in value.strip().split()]


def _load_keyframe_xml(path: Path, key_name: str = "bent") -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    key_elem = root.find(f".//keyframe/key[@name='{key_name}']")
    if key_elem is None:
        key_elem = root.find(".//keyframe/key")
    if key_elem is None:
        raise ValueError(f"No <key> found in keyframe XML: {path}")

    time_value = float(key_elem.attrib.get("time", "0"))
    qpos = _parse_float_list(key_elem.attrib.get("qpos"))
    qvel = _parse_float_list(key_elem.attrib.get("qvel"))
    ctrl = _parse_float_list(key_elem.attrib.get("ctrl"))
    act = _parse_float_list(key_elem.attrib.get("act"))
    mpos = _parse_float_list(key_elem.attrib.get("mpos"))
    mquat = _parse_float_list(key_elem.attrib.get("mquat"))

    key_kwargs: dict[str, Any] = {
        "name": key_elem.attrib.get("name", key_name),
        "time": time_value,
    }
    if qpos:
        key_kwargs["qpos"] = qpos
    if qvel:
        key_kwargs["qvel"] = qvel
    if ctrl:
        key_kwargs["ctrl"] = ctrl
    if act:
        key_kwargs["act"] = act
    if mpos:
        key_kwargs["mpos"] = mpos
    if mquat:
        key_kwargs["mquat"] = mquat

    return key_kwargs


class PipeInsert2(mjx_env.MjxEnv):
    """Simple 3D position control environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)

        self._slide_limits = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (-0.3, 0.3),
        }

        self.keys = {
            "qpos": "testing/experiments/pipe_insert/constants/v3/qpos.npy",
            "qvel": "testing/experiments/pipe_insert/constants/v3/qvel.npy",
            "ctrl": "testing/experiments/pipe_insert/constants/v3/ctrl.npy",
        }

        self._mj_model = self._init()
        self.ctrl_scale: float = 0.01
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model)

        self._post_init()

    def _init(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        keyframe_path = _HERE / "envs/tmp/keyframe.xml"
        if keyframe_path.exists():
            key_kwargs = _load_keyframe_xml(keyframe_path, key_name="bent")
            scene.add_key(**key_kwargs)

        pip = pipe()

        pip.worldbody.first_body().add_site(
            name="target_1", pos=[0, 0.025, -0.05], group=1, rgba=[1, 0, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_2", pos=[0, 0.025, 0], group=1, rgba=[1, 1, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_3", pos=[0, 0.025, 0.05], group=1, rgba=[0, 1, 0, 1]
        )
        # pip.worldbody.first_body().add_site(
        #     name="target_1", pos=[0, 0.015, -0.05], group=1, rgba=[1, 0, 0, 1]
        # )
        # pip.worldbody.first_body().add_site(
        #     name="target_2", pos=[0, 0.015, 0], group=1, rgba=[1, 1, 0, 1]
        # )
        # pip.worldbody.first_body().add_site(
        #     name="target_3", pos=[0, 0.015, 0.05], group=1, rgba=[0, 1, 0, 1]
        # )

        twist = 60000.0 * 2
        bend = 10000000.0 * 2
        # twist = 60000.0 * 2
        # bend = 10000000.0 * 2

        cable = mjs_cable(count="40 1 1", twist=twist, bend=bend)

        print(cable.to_xml())
        # quit()

        # cable = mjs_cable(count="40 1 1", twist=60000.0 * 2, bend=10000000.0 * 2)

        cable.bodies[1].add_site(name="keypoint_3", group=1, rgba=[0, 1, 0, 1])
        cable.bodies[3].add_site(name="keypoint_2", group=1, rgba=[1, 1, 0, 1])
        cable.bodies[5].add_site(name="keypoint_1", group=1, rgba=[1, 0, 0, 1])

        # print(cable.to_file("/home/vims/git/skrl/testing/envs/tmp/cable_mj.xml"))

        scene.worldbody.add_camera(
            name="cam",
            pos=[1.2, 0.234, 0.156],
            # pos=[0.721, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
        )

        _b = scene.worldbody.add_body(name="am i falling?!", pos=[0, 0, 0])
        _g = _b.add_geom(
            name="am i falling?!",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            contype=0,
            conaffinity=0,
        )
        # _b.add_freejoint()

        # <camera pos="0.721 0.234 0.156" xyaxes="-0.037 0.999 0.000 -0.001 -0.000 1.000"/>

        gripper = scene.worldbody.add_body(
            # name="gripper",
            # pos=[0, 0.4, 0.1],
            # euler=[0, 0, 0],
            name="gripper",
            pos=[0, 0.4, 0.1],
            euler=[0, 0, 0],
            # euler=[0, 0, 1.57],
        )
        gripper.add_geom(
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
            # rgba=[0, 0, 0, 0],
        )
        gripper.add_joint(
            name="x",
            type=mj.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            range=self._slide_limits["x"],
        )
        gripper.add_joint(
            name="y",
            type=mj.mjtJoint.mjJNT_SLIDE,
            axis=[0, 1, 0],
            range=self._slide_limits["y"],
        )
        gripper.add_joint(
            name="z",
            type=mj.mjtJoint.mjJNT_SLIDE,
            axis=[0, 0, 1],
            range=self._slide_limits["z"],
        )
        # after creating `gripper`
        gripper.add_site(
            name="grip_cable_site",
            pos=[0.0, 0.0, 0.0],  # choose where on the gripper
            euler=[0.0, 0.0, 1.5707963],  # rotate attachment frame as needed
            group=1,
            rgba=[1, 0, 1, 1],
        )

        scene.add_actuator(
            name="x",
            target="x",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=self._slide_limits["x"],
        ).set_to_position(kp=100, kv=100)
        scene.add_actuator(
            name="y",
            target="y",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=self._slide_limits["y"],
        ).set_to_position(kp=100, kv=100)
        scene.add_actuator(
            name="z",
            target="z",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=self._slide_limits["z"],
        ).set_to_position(kp=100, kv=100)

        scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 3.14]).attach_body(
            # scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 0]).attach_body(
            pip.worldbody.first_body()
        )
        scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
            # scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
            cable.worldbody.first_body()
        )

        b = None
        b0 = cable.worldbody.first_body()
        for i in range(10):
            b = b0.first_body()
            b0 = b

        b.add_site(
            name="cable_weld_site",
            pos=[0.0, 0.0, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 0.0],
            group=1,
            rgba=[0, 1, 1, 1],
        )

        scene.add_equality(
            name="weld",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="grip_cable_site",
            name2="cable_weld_site",
            # omit data => MuJoCo uses the current relative pose as the target
            solref=[0.000001, 2],
        )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _post_init(self) -> None:
        self.target_ids = [self._mj_model.site(f"target_{i + 1}").id for i in range(3)]
        self.keypoint_ids = [
            self._mj_model.site(f"keypoint_{i + 1}").id for i in range(3)
        ]
        self._qpos0 = jp.array(np.load(self.keys["qpos"]))
        self._qvel0 = jp.array(np.load(self.keys["qvel"]))
        self._ctrl0 = jp.array(np.load(self.keys["ctrl"]))
        self._key_id: Optional[int] = None
        try:
            self._key_id = self._mj_model.key("bent").id
        except Exception:
            self._key_id = None

        if self._key_id is not None:
            self._qpos0 = jp.array(self._mj_model.key_qpos[self._key_id])
            self._qvel0 = jp.array(self._mj_model.key_qvel[self._key_id])
            self._ctrl0 = jp.array(self._mj_model.key_ctrl[self._key_id])

        self.qpos0_gripper = self._qpos0[:3].copy()
        self.qpos0_cable = self._qpos0[3:6].copy()
        self.ctrl0 = self._ctrl0[:3].copy()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Split RNG
        rng, rng_delta = jax.random.split(rng)

        qpos = self._qpos0
        qvel = self._qvel0
        ctrl = self._ctrl0
        act = jp.zeros(self.mj_model.na)
        if self._key_id is not None:
            qpos = jp.array(self._mj_model.key_qpos[self._key_id])
            qvel = jp.array(self._mj_model.key_qvel[self._key_id])
            ctrl = jp.array(self._mj_model.key_ctrl[self._key_id])
            if self._mj_model.na:
                act = jp.array(self._mj_model.key_act[self._key_id])

        # Match the manual initialization from __main__.
        # delta = jax.random.uniform(rng_delta, shape=(3,), minval=-0.3, maxval=0.3)
        delta = jp.zeros(3)
        qpos = qpos.at[:3].set(self.qpos0_gripper + delta)
        qpos = qpos.at[3:6].set(self.qpos0_cable + delta)
        ctrl = ctrl.at[:3].set(self.ctrl0 + delta)

        # Zero velocities if no keyframe is present
        if self._key_id is None:
            qvel = jp.zeros(self.mj_model.nv)
        qacc = jp.zeros(self.mj_model.nv)
        qfrc_applied = jp.zeros(self.mj_model.nv)
        xfrc_applied = jp.zeros((self.mj_model.nbody, 6))

        # Initialize MuJoCo-X data
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl, act=act)

        print(f"{self.mjx_model.nu=}")
        print(f"{self.mjx_model.nv=}")
        print(f"{self.mjx_model.na=}")
        print(f"{self.mjx_model.nq=}")
        print(f"{self.mjx_model.nbody=}")
        print(f"{self.mjx_model.ngeom=}")
        print(f"{self.mjx_model.njnt=}")

        data = data.replace(qvel=qvel)
        data = data.replace(qacc=qacc)
        data = data.replace(qfrc_applied=qfrc_applied)
        data = data.replace(xfrc_applied=xfrc_applied)

        metrics = {}
        info = {}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.zeros(3)
        a = state.data.ctrl + action

        # a = state.data.ctrl + self.ctrl_scale * action
        data = mjx_env.step(self.mjx_model, state.data, a, self.n_substeps)

        # Reward: negative distance to target + bonus for being close
        dist_to_target = self._get_distance(data)
        reward = -dist_to_target

        obs = self._get_obs(data, state.info)

        # This is to check for stability. If the simulation has turned unstable these will be None
        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        threshold = 0.5

        # Create bool (remember this is not branching so this is legal)
        # if we are within 2 cm of the target, then we are successful
        success = dist_to_target <= threshold
        # success = dist_to_target <= 0.02

        def _print_reward(_):
            part1 = dist_to_target
            part2 = jp.asarray(threshold)
            part3 = part1 <= part2

            # part4 = self.mjx_model.dof_armature[self._id]
            # part5 = self.mjx_model.dof_damping[self._id]

            # jax.debug.print("part4 = {part4}", part4=part4)
            # jax.debug.print("part5 = {part5}", part5=part5)

            # jax.debug.print("success success: {success}", success=success)
            # jax.debug.print("\tsuccess reward: {reward}", reward=reward)
            # jax.debug.print("\tsuccess part1: {part1}", part1=part1)
            # jax.debug.print("\tsuccess part2: {part2}", part2=part2)
            # jax.debug.print("\tsuccess part3: {part3}", part3=part3)
            return None

        # jax.lax.cond(success, _print_reward, lambda _: None, operand=None)

        # Combine all termination conditions
        done = jp.logical_or(nan_condition, success)

        # Use lax.cond instead of if statement
        # def quit_simulation(_):
        #     # jax.debug.print("Quitting due to instability")
        #     # You can't actually call quit() in JAX, so we need a JAX-compatible alternative
        #     # Option A: Mark as done and let the outer loop handle it
        #     return state.replace(data=data, info=state.info)

        # def continue_simulation(_):
        #     return state.replace(data=data, info=state.info)

        # # Instead of quit(), we'll mark the episode as done
        # state = jax.lax.cond(
        #     ~is_stable(data),  # Note: ~ instead of "not" for JAX arrays
        #     quit_simulation,
        #     continue_simulation,
        #     operand=None,
        # )

        # Check if all conditions are True and conditionally raise error
        # jax.lax.cond(jp.all(nan_condition), raise_error, no_op, operand=None)
        # done = nan_condition | out_of_bounds
        done = done.astype(float)

        info = state.info.copy()

        return mjx_env.State(data, obs, reward, done, state.metrics, info)

    def _get_distance(self, data: mjx.Data) -> jax.Array:
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

        return dist_to_target

    def _get_reward(self, data: mjx.Data) -> float:
        return -self._get_distance(data)

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

    def _from_to(self, data: mjx.Data, _from: int, _to: int) -> jax.Array:
        T_w_from = get_pose(self._mjx_model, data, _from, obj_type=ObjType.SITE)
        T_w_to = get_pose(self._mjx_model, data, _to, obj_type=ObjType.SITE)
        return T_w_to.translation() - T_w_from.translation()

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        # Return agent position and target position (6D observation)
        # s0 = self._get_site_pos(_data)
        # s1 = self._get_site_pos(data)
        # dt = self.mj_model.opt.timestep
        # v1 = (s1 - s0) / dt

        # o = jp.concatenate([s1, v1])

        # T_w_target_3.translation() - T_w_keypoint_3.translation()

        r1 = self._from_to(data, self.keypoint_ids[0], self.target_ids[0])
        r2 = self._from_to(data, self.keypoint_ids[1], self.target_ids[1])
        r3 = self._from_to(data, self.keypoint_ids[2], self.target_ids[2])

        o = jp.concatenate([r1, r2, r3]).flatten()

        return o

    @property
    def observation_size(self) -> int:
        """Compute observation size by calling _get_obs with dummy data"""
        # Create dummy data for size computation
        dummy_data = mjx.make_data(self.mjx_model)
        dummy_info = {}

        # Get observation and check its shape
        obs = self._get_obs(dummy_data, dummy_info)
        # print(f"{obs=}")
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
    env = PipeInsert2()
    model = env.mj_model
    data = mj.MjData(model)

    # rng = jax.random.PRNGKey(0)
    # state = env.reset(rng)

    # how does it work wrt. get mjx data vs. mj data
    #   make sure to use mj_model for mj_data
    #   make sure to use mjx_model for mjx_data
    # data = mjx.get_data(env.mj_model, state.data)

    # data.qpos[:] = np.load(env.keys["qpos"])
    # data.qvel[:] = np.load(env.keys["qvel"])
    # data.ctrl[:] = np.load(env.keys["ctrl"])
    # delta = np.random.uniform(-0.3, 0.3, 3)
    # data.qpos[:3] = np.asarray(env.qpos0_gripper) + delta
    # data.qpos[3:6] = np.asarray(env.qpos0_cable) + delta
    # data.ctrl[:3] = np.asarray(env.ctrl0) + delta
    # data.qvel[:] = 0.0
    # mj.mj_forward(model, data)
    # print(f"{data=}")
    # print(f"{data.qvel[:10]=}")
    # print(f"{data.qacc[:10]=}")
    # quit()

    joint_ids = [model.joint(name).id for name in ["x", "y", "z"]]
    ctrl_ids = [model.actuator(name).id for name in ["x", "y", "z"]]

    def _load_base_state() -> None:
        if env._key_id is not None:
            data.qpos[:] = np.asarray(model.key_qpos[env._key_id])
            data.qvel[:] = np.asarray(model.key_qvel[env._key_id])
            data.ctrl[:] = np.asarray(model.key_ctrl[env._key_id])
        else:
            data.qpos[:] = np.load(env.keys["qpos"])
            data.qvel[:] = np.load(env.keys["qvel"])
            data.ctrl[:] = np.load(env.keys["ctrl"])

    def cb(key: int) -> None:
        if key is glfw.KEY_SPACE:
            mj.mj_setKeyframe(model, data, 0)
            # qpos = data.qpos
            # qvel = data.qvel
            # ctrl = data.ctrl
            # np.save("testing/experiments/pipe_insert/constants/v3/qpos.npy", qpos)
            # np.save("testing/experiments/pipe_insert/constants/v3/qvel.npy", qvel)
            # np.save("testing/experiments/pipe_insert/constants/v3/ctrl.npy", ctrl)
            print("saved")

        elif key is glfw.KEY_PERIOD:
            delta = np.random.uniform(-0.3, 0.3, 3)
            _load_base_state()
            data.qpos[:3] = np.asarray(env.qpos0_gripper) + delta
            data.qpos[3:6] = np.asarray(env.qpos0_cable) + delta
            data.ctrl[:3] = np.asarray(env.ctrl0) + delta
            data.qvel[:] = 0.0
            mj.mj_forward(model, data)

    # delta = np.random.uniform(-0.3, 0.3, 3)
    # delta = np.zeros(3)
    # data.qpos[:] = np.load(env.keys["qpos"])
    # data.qvel[:] = np.load(env.keys["qvel"])
    # data.ctrl[:] = np.load(env.keys["ctrl"])
    # data.qpos[joint_ids] = np.asarray(env.qpos0_gripper) + delta
    # # data.qpos[:3] = np.asarray(env.qpos0_gripper) + delta
    # data.qpos[3:6] = np.asarray(env.qpos0_cable) + delta
    # data.ctrl[:3] = np.asarray(env.qpos0_cable) + delta

    # print(np.asarray(env.ctrl0) + delta)
    # print(np.asarray(env.qpos0_gripper) + delta)
    # quit()

    delta = np.random.uniform(-0.3, 0.3, 3)
    _load_base_state()
    # data.qpos[:3] = np.asarray(env.qpos0_gripper) + delta
    # data.qpos[3:6] = np.asarray(env.qpos0_cable) + delta
    # data.ctrl[:3] = np.asarray(env.ctrl0) + delta
    # data.qvel[:] = 0.0
    # mj.mj_forward(model, data)
    with mujoco.viewer.launch_passive(
        model=model, data=data, key_callback=cb
    ) as viewer:
        # set gui camera to the specified in the model
        viewer.cam.azimuth = model.vis.global_.azimuth
        viewer.cam.elevation = model.vis.global_.elevation
        viewer.cam.lookat = model.stat.center
        viewer.cam.distance = model.stat.extent

        # mj.mj_forward(model, data)

        _scene = viewer.user_scn

        # input("go?")
        while viewer.is_running():
            step_start = time.time()

            # step simulation one time step
            mj.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
