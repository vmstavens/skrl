import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
from robot_descriptions import robotiq_2f85_mj_description, ur10e_mj_description

from lauge.dominik.cable import _parse_float_list


def mjx_cable(
    *,
    model_name: str = "cable",
    prefix: str = "cable:",
    curve: str = "0 s 0",
    count: Union[str, int] = "10 1 1",
    twist: float = 60000.0,
    bend: float = 10000000.0,
    vmax: float = 0,
    size: Union[str, float, int] = 1,
    segment_size: float = 0.002,
    mass: float = 0.00035,
    rgba: Union[str, list[float]] = "0.2 0.2 0.2 1",
    initial: str = "free",
    solref: Optional[list] = None,
) -> mj.MjSpec:
    del model_name, curve, vmax
    base_pos = [0.0, 0.0, 0.0]
    damping = 1e-2
    armature = 0.001
    friction = [0.7, 0.7, 0.7]
    if solref is None:
        solref = [0.000001, 1.0]
    # solref = [0.001, 3.0]
    condim = 4
    geom_type = "capsule"
    add_freejoint = initial == "free"
    if isinstance(count, str):
        count_tokens = count.strip().split()
        n_segments = int(count_tokens[0]) if count_tokens else 0
    else:
        n_segments = int(count)
    size_value = float(size.split()[0]) if isinstance(size, str) else float(size)
    segment_length = size_value / max(n_segments, 1)
    radius = segment_size
    rgba = _parse_float_list(rgba) if isinstance(rgba, str) else rgba
    if len(rgba) != 4:
        rgba = [0.2, 0.2, 0.2, 1.0]
    name_prefix = prefix[:-1] if prefix.endswith(":") else prefix
    geom_type_map = {
        "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mj.mjtGeom.mjGEOM_CAPSULE,
        "sphere": mj.mjtGeom.mjGEOM_SPHERE,
        "box": mj.mjtGeom.mjGEOM_BOX,
    }
    if geom_type not in geom_type_map:
        raise ValueError(f"Unsupported geom_type: {geom_type}")

    if n_segments <= 0:
        raise ValueError("n_segments must be > 0")

    # Cable runs along +Y in the local frame, matching curve="0 s 0".
    geom_euler = [-np.pi / 2, 0.0, 0.0]

    def section_properties(
        geom_type: str, geom_size: list[float]
    ) -> tuple[float, float, float]:
        # Match plugin/cable.cc section property computations using geom_size.
        if geom_type in ("cylinder", "capsule"):
            r = geom_size[0]
            j = np.pi * r**4 / 2.0
            iy = iz = np.pi * r**4 / 4.0
            return j, iy, iz
        if geom_type == "box":
            h = geom_size[1]
            w = geom_size[2]
            a = max(h, w)
            b = min(h, w)
            j = a * b**3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - (b**4) / (a**4) / 12.0))
            iy = (2.0 * w) ** 3 * (2.0 * h) / 12.0
            iz = (2.0 * h) ** 3 * (2.0 * w) / 12.0
            return j, iy, iz
        return 0.0, 0.0, 0.0

    spec = mj.MjSpec()
    root = spec.worldbody.add_body(name=f"{name_prefix}:root", pos=base_pos)
    if add_freejoint:
        root.add_joint(name=f"{name_prefix}:free", type=mj.mjtJoint.mjJNT_FREE)

    first_joint_idx = 1 if n_segments > 1 else None
    last_joint_idx = n_segments - 1 if n_segments > 1 else None

    first_body_idx = 0
    last_body_idx = n_segments - 1

    parent = root
    for i in range(n_segments):
        if i == 0:
            body = parent.add_body(name=f"{name_prefix}:Bfirst")
        else:
            if i == last_body_idx:
                body_name = f"{name_prefix}:Blast"
            else:
                body_name = f"{name_prefix}:B{i}"
            body = parent.add_body(
                name=body_name,
                pos=[0.0, segment_length, 0.0],
            )
            joint_pos = [0.0, -segment_length / 2.0, 0.0]

        if geom_type in ("cylinder", "capsule"):
            geom_size = [radius, segment_length / 2.0, 0]
        elif geom_type == "sphere":
            geom_size = [radius, 0.0, 0.0]
        else:
            geom_size = [radius, segment_length / 2.0, radius]

        # Match plugin stiffness behavior: k = (J*G)/L for twist, (Iy*E)/L & (Iz*E)/L for bend.
        if i > 0:
            j, iy, iz = section_properties(geom_type, geom_size)
            length = max(segment_length, 1e-9)
            k_twist = (j * twist) / length
            k_bend_y = (iy * bend) / length
            k_bend_z = (iz * bend) / length

            # Ball joint uses a single stiffness; approximate from bend/twist contributions.
            k_ball = (k_bend_y + k_bend_z + k_twist) / 3.0
            if i == first_joint_idx:
                joint_name = f"{name_prefix}:Jfirst"
            elif i == last_joint_idx:
                joint_name = f"{name_prefix}:Jlast"
            else:
                joint_name = f"{name_prefix}:J{i}"
            body.add_joint(
                name=joint_name,
                type=mj.mjtJoint.mjJNT_BALL,
                pos=joint_pos,
                damping=damping,
                armature=armature,
                stiffness=k_ball,
            )

        geom = body.add_geom(
            name=(
                f"{name_prefix}:Gfirst"
                if i == first_body_idx
                else f"{name_prefix}:Glast"
                if i == last_body_idx
                else f"{name_prefix}:G{i}"
            ),
            type=geom_type_map[geom_type],
            size=geom_size,
            euler=geom_euler,
            rgba=rgba,
            mass=mass,
            friction=friction,
            condim=condim,
            solref=solref,
        )
        if geom_type in ("cylinder", "capsule"):
            geom.fromto = [
                0.0,
                -segment_length / 2.0,
                0.0,
                0.0,
                segment_length / 2.0,
                0.0,
            ]

        parent = body

    return spec


def empty(floor: bool = False) -> mj.MjSpec:

    # integrator="RK4"
    _XML = f"""
        <mujoco model="empty scene">

        <compiler angle="radian" autolimits="true" />
        <option 
            timestep="0.0001" 
            integrator="RK4" 
            solver="Newton" 
            gravity="0 0 -9.82" 
            cone="elliptic" 
            noslip_iterations="3" 
        >
            <flag multiccd="enable" />
        </option>

        <statistic center="0.3 0 0.3" extent="0.8" meansize="0.08" />

        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
            <rgba haze="0.15 0.25 0.35 1" />
            <global azimuth="120" elevation="-20" offwidth="2000" offheight="2000" />
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
            {'<geom name="floor" size="0 0 0.5" type="plane" material="groundplane" />' if floor else ""}
        </worldbody>

    </mujoco>
    """
    return mj.MjSpec.from_string(_XML)


def _parse_str_list(value: Optional[Union[str, list[float], list[str]]]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return " ".join(str(x) for x in value)


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
    solref: Optional[Union[str, list[str], list[float]]] = None,
) -> mj.MjSpec:
    solref_str = _parse_str_list(solref) if solref is not None else "0.001 3"

    # <joint kind="main" damping="1e-2" armature="0.001" />
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
                    solref="{solref_str}"
                />
            </composite>
    </worldbody>

</mujoco>
    """

    return mj.MjSpec.from_string(_XML)


def _mjx_cable(
    *,
    model_name: str = "cable",
    prefix: str = "cable:",
    curve: str = "0 s 0",
    count: Union[str, int] = "10 1 1",
    twist: float = 60000.0,
    bend: float = 10000000.0,
    vmax: float = 0,
    size: Union[str, float, int] = 1,
    segment_size: float = 0.002,
    mass: float = 0.00035,
    rgba: Union[str, list[float]] = "0.2 0.2 0.2 1",
    initial: str = "free",
    match_composite: bool = False,
) -> mj.MjSpec:
    del model_name, curve, vmax
    base_pos = [0.0, 0.0, 0.0]
    damping = 1e-2
    armature = 0.001
    friction = [0.3, 0.3, 0.3]
    solref = [0.001, 3.0]
    condim = 4
    geom_type = "capsule"
    add_freejoint = initial == "free"
    if isinstance(count, str):
        count_tokens = count.strip().split()
        n_segments = int(count_tokens[0]) if count_tokens else 0
    else:
        n_segments = int(count)
    size_value = float(size.split()[0]) if isinstance(size, str) else float(size)
    segment_length = size_value / max(n_segments, 1)
    radius = segment_size
    # Match composite cable: capsule cylinder length equals segment spacing,
    # so hemispherical ends overlap with neighbors by 2*radius.
    capsule_half = segment_length / 2.0
    rgba = _parse_float_list(rgba) if isinstance(rgba, str) else rgba
    if len(rgba) != 4:
        rgba = [0.2, 0.2, 0.2, 1.0]
    name_prefix = prefix[:-1] if prefix.endswith(":") else prefix
    geom_type_map = {
        "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mj.mjtGeom.mjGEOM_CAPSULE,
        "sphere": mj.mjtGeom.mjGEOM_SPHERE,
        "box": mj.mjtGeom.mjGEOM_BOX,
    }
    if geom_type not in geom_type_map:
        raise ValueError(f"Unsupported geom_type: {geom_type}")

    if n_segments <= 0:
        raise ValueError("n_segments must be > 0")

    # Cable runs along +Y in the local frame, matching curve="0 s 0".
    geom_euler = [-np.pi / 2, 0.0, 0.0]

    def section_properties(
        geom_type: str, geom_size: list[float]
    ) -> tuple[float, float, float]:
        # Match plugin/cable.cc section property computations using geom_size.
        if geom_type in ("cylinder", "capsule"):
            r = geom_size[0]
            j = np.pi * r**4 / 2.0
            iy = iz = np.pi * r**4 / 4.0
            return j, iy, iz
        if geom_type == "box":
            h = geom_size[1]
            w = geom_size[2]
            a = max(h, w)
            b = min(h, w)
            j = a * b**3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - (b**4) / (a**4) / 12.0))
            iy = (2.0 * w) ** 3 * (2.0 * h) / 12.0
            iz = (2.0 * h) ** 3 * (2.0 * w) / 12.0
            return j, iy, iz
        return 0.0, 0.0, 0.0

    spec = mj.MjSpec()
    root = spec.worldbody.add_body(name=f"{name_prefix}:root", pos=base_pos)
    if add_freejoint:
        root.add_joint(name=f"{name_prefix}:free", type=mj.mjtJoint.mjJNT_FREE)

    first_joint_idx = 0
    last_joint_idx = n_segments - 1 if n_segments > 1 else None

    first_body_idx = 0
    last_body_idx = n_segments - 1

    parent = root
    for i in range(n_segments):
        if i == 0:
            body = parent.add_body(
                name=f"{name_prefix}:Bfirst",
                pos=[0.0, segment_length / 2.0, 0.0],
            )
        else:
            if i == last_body_idx:
                body_name = f"{name_prefix}:Blast"
            else:
                body_name = f"{name_prefix}:B{i}"
            body = parent.add_body(
                name=body_name,
                pos=[0.0, segment_length, 0.0],
            )
        joint_pos = [0.0, -segment_length / 2.0, 0.0]

        if geom_type in ("cylinder", "capsule"):
            geom_size = [radius, capsule_half, 0]
        elif geom_type == "sphere":
            geom_size = [radius, 0.0, 0.0]
        else:
            geom_size = [radius, segment_length / 2.0, radius]

        # Match plugin stiffness behavior: k = (J*G)/L for twist, (Iy*E)/L & (Iz*E)/L for bend.
        j, iy, iz = section_properties(geom_type, geom_size)
        length = max(segment_length, 1e-9)
        k_twist = (j * twist) / length
        k_bend_y = (iy * bend) / length
        k_bend_z = (iz * bend) / length

        # Ball joint uses a single stiffness; approximate from bend/twist contributions.
        k_ball = (k_bend_y + k_bend_z + k_twist) / 3.0
        if match_composite:
            k_ball = 0.0
        if i == first_joint_idx:
            joint_name = f"{name_prefix}:Jfirst"
        elif i == last_joint_idx:
            joint_name = f"{name_prefix}:Jlast"
        else:
            joint_name = f"{name_prefix}:J{i}"
        body.add_joint(
            name=joint_name,
            type=mj.mjtJoint.mjJNT_BALL,
            pos=joint_pos,
            damping=damping,
            armature=armature,
            stiffness=k_ball,
        )

        geom = body.add_geom(
            name=(
                f"{name_prefix}:Gfirst"
                if i == first_body_idx
                else f"{name_prefix}:Glast"
                if i == last_body_idx
                else f"{name_prefix}:G{i}"
            ),
            type=geom_type_map[geom_type],
            size=geom_size,
            euler=geom_euler,
            rgba=rgba,
            mass=mass,
            friction=friction,
            condim=condim,
            solref=solref,
        )
        if geom_type in ("cylinder", "capsule"):
            geom.fromto = [
                0.0,
                -capsule_half,
                0.0,
                0.0,
                capsule_half,
                0.0,
            ]

        parent = body

    return spec


def init() -> mj.MjModel:
    scene = empty()

    _f = scene.worldbody.add_frame(
        name="cable", pos=[-0.92, -0.17, 0.05], euler=[0, 0, 1.57]
    )
    cable = mjx_cable(model_name="cable", initial="free", count="20 1 1", size=1)
    cable = mjs_cable(
        model_name="cable",
        initial="free",
        bend=10_000,
        twist=10_000,
        count="20 1 1",
        size=0.5,
        segment_size=0.005,
        # solref=[0.000001, 2],
    )
    _f.attach_body(cable.worldbody.first_body(), prefix="cable")

    arm = mj.MjSpec.from_file(ur10e_mj_description.MJCF_PATH)
    gripper = mj.MjSpec.from_file("assets/robotiq_hande/hande.xml")
    gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)
    _f = scene.worldbody.add_frame(name="robot")

    s_as: mj.MjsSite = arm.site("attachment_site")

    s_as.attach_body(gripper.worldbody.first_body(), prefix="gripper/")

    # for g in arm.worldbody.find_all("geom"):
    #     g: mj.MjsGeom = g
    #     g.solref = [0.001, 3]

    _f.attach_body(arm.worldbody.first_body(), prefix="arm/")

    # b_box = scene.worldbody.add_body(name="box", pos=[-0.92, -0.17, 0.05])
    # b_box.add_freejoint()
    # b_box.add_geom(
    #     name="box",
    #     size=[0.02, 0.02, 0.02],
    #     type=mj.mjtGeom.mjGEOM_BOX,
    #     solref=[0.0001, 1],
    # )

    # scene.compile()
    # scene.to_file("testing/envs/assets/cable_arm_gripper.xml")

    # scene.from_file("testing/envs/assets/cable_arm_gripper.xml")

    # g_0 = scene.worldbody.first_geom()

    # geoms = scene.worldbody.find_all("geom")
    # _g = None
    # for g in geoms:
    #     if "pad" in g.name:
    #         g.solref = [0.001, 3]

    # print(g_0.name)
    # print(g_0.solref)
    # # g_0.solref = [0.0000001, 3]
    # print(g_0.solref)
    # # quit()

    return scene.compile()


def _build_model() -> mj.MjModel:

    scene = empty(floor=True)
    # 1) goal
    b_goal = scene.worldbody.add_body(name="goal", mocap=True, pos=[0.1, 0.1, 0.1])
    b_goal.add_geom(
        name="goal",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
        rgba=[1, 0, 0, 1],
    )
    s_goal = b_goal.add_site(name="goal")

    # 2) mocap
    b_mocap = scene.worldbody.add_body(
        name="mocap", mocap=True, pos=[-0.1, -0.1, 0.5], euler=[3.14, 0, 0]
    )
    b_mocap.add_geom(
        name="mocap",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
    )
    s_mocap = b_mocap.add_site(name="mocap")

    # 3) gripper (welded to mocap site)
    gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)
    b_gripper = gripper.worldbody.first_body()
    b_gripper.add_freejoint()
    b_gripper.add_site(name="gripper")
    f_gripper = scene.worldbody.add_frame(
        name="gripper", pos=[-0.1, -0.1, 0.5], euler=[3.14, 0, 0]
    )
    f_gripper.attach_body(b_gripper, prefix="gripper/")

    b_item = scene.worldbody.add_body(name="item", pos=[0, 0, 0.02])
    b_item.add_geom(
        name="item",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        rgba=[0, 1, 0, 1],
    )
    b_item.add_site(name="item")
    b_item.add_freejoint()

    scene.add_equality(
        name="weld",
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_SITE,
        name1="gripper/gripper",
        name2="mocap",
        # omit data => MuJoCo uses the current relative pose as the target
        solref=[0.000001, 2],
    )

    return scene.compile()


m = _build_model()

d = mj.MjData(m)


def cb(key: int) -> None:
    if key is glfw.KEY_SPACE:
        global d
        d.actuator("arm/gripper/fingers_actuator").ctrl = 255


jnts = [
    "arm/shoulder_pan_joint",
    "arm/shoulder_lift_joint",
    "arm/elbow_joint",
    "arm/wrist_1_joint",
    "arm/wrist_2_joint",
    "arm/wrist_3_joint",
]

acts = [
    "arm/shoulder_pan",
    "arm/shoulder_lift",
    "arm/elbow",
    "arm/wrist_1",
    "arm/wrist_2",
    "arm/wrist_3",
]

ref = np.array([0, -0.942, 1.64, -2.32, -1.57, 0])
# ref = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

with mujoco.viewer.launch_passive(model=m, data=d, key_callback=cb) as viewer:
    # for i, jn in enumerate(jnts):
    #     d.joint(jn).qpos = ref[i]
    # for i, jn in enumerate(acts):
    #     d.actuator(jn).ctrl = ref[i]
    while viewer.is_running():
        step_start = time.time()

        # step simulation one time step
        mj.mj_step(m, d)

        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
