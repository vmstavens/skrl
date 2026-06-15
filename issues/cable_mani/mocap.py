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


def empty() -> mj.MjSpec:

    # integrator="RK4"
    _XML = """
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
            <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" />
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
    )
    _f.attach_body(cable.worldbody.first_body(), prefix="cable")

    gripper = mj.MjSpec.from_file("assets/robotiq_hande/hande.xml")
    gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)
    _f = scene.worldbody.add_frame(name="gripper")

    b_mocap = scene.worldbody.add_body(name="mocap", mocap=True)
    b_mocap.add_geom(
        name="mocap",
        size=[0.02, 0.02, 0.02],
        type=mj.mjtGeom.mjGEOM_BOX,
        contype=0,
        conaffinity=0,
    )
    b_mocap.add_site(name="mocap")

    b_gripper_root = gripper.worldbody.first_body()

    b_gripper_root.add_site(name="root")
    b_gripper_root.add_freejoint()

    _f.attach_body(b_gripper_root, prefix="gripper/")

    scene.add_equality(
        name="weld",
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_SITE,
        name1="mocap",
        name2="gripper/root",
        # omit data => MuJoCo uses the current relative pose as the target
        solref=[0.000001, 2],
    )

    g_0 = scene.worldbody.first_geom()

    geoms = scene.worldbody.find_all("geom")
    _g = None
    for g in geoms:
        if "pad" in g.name:
            g.solref = [0.001, 3]

    # <key
    #   time="5.92"
    #   qpos="-0.96232 -0.310315 0.00821729 0.0553001 -0.152186 -0.806324 0.568879 0.99972 0.00400948 0.0120487 -0.019957 0.99958 0.00564608 0.0135266 -0.0250119 0.999213 0.00697393 0.0186555 -0.0343034 0.999314 0.00757918 0.0152461 -0.0329011 0.999681 0.00773247 -0.000154943 -0.0240269 0.999351 0.00721107 -0.0348006 -0.00583124 0.998141 0.00785734 -0.0534318 0.0282443 0.993185 0.0140636 -0.0913325 0.0710282 0.998498 0.0432591 -0.0262267 0.0210157 0.999129 -0.0169444 0.00832389 0.0372118 0.999622 -0.0158279 0.0217098 0.00586508 0.999446 -0.0154785 0.0236398 -0.0175784 0.999309 -0.0144307 0.012933 -0.03171 0.999309 -0.0117984 -0.00763152 -0.0344152 0.999922 -0.00635246 -0.00119965 -0.0106755 0.999976 -0.00314211 0.0054079 0.00312701 0.999966 -0.00198791 0.00724479 0.00332934 0.999977 -0.00156569 0.00662965 4.12157e-05 -1.17796 -0.24808 0.179379 0.00058455 -0.722105 -0.691755 -0.00621466 0.739054 0.000562543 0.730411 -0.736887 0.739054 0.000562827 0.730405 -0.736903"
    #   qvel="0.000524685 0.00139056 0.000110475 0.000387267 -0.00332488 -0.000637018 -0.000340321 -0.00199311 0.00498695 -0.000731889 -0.0048906 0.0161636 0.000420779 -0.00492143 -0.00475585 0.000718906 0.000829923 -0.0131897 0.000452709 0.00508467 -0.0155297 -0.000382475 0.00748081 -0.0113062 -0.000115123 0.00460338 0.000518482 0.00309457 -0.00321484 0.0225712 -0.00127211 0.0529149 0.00934746 0.00357411 -0.0398959 0.0101416 0.00353874 -0.0207134 -0.007091 0.00445331 -0.0052276 -0.0156978 0.00441011 0.00527641 -0.0132248 0.00348936 0.0110195 0.00174422 0.00190069 -0.00575922 0.0171878 0.000878582 5.85321e-05 0.00470361 0.000687971 0.00287952 0.00397919 0.000255688 -0.002059 0.00593209 3.79798e-07 -1.13445e-08 1.93784e-07 5.23767e-05 1.18092e-05 -1.49264e-06 -5.99913e-06 5.67392e-07 -2.22363e-05 -1.32551e-05 -4.98016e-06 -1.69525e-06 1.0342e-05 6.18723e-05"
    #   ctrl="255"
    #   mpos="-1.17796 -0.24808 0.179379"
    #   mquat="0.000584735 -0.722105 -0.691755 -0.00621478"
    # />

    scene.add_key(
        name="test",
        time=5.92,
        qpos=_parse_float_list(
            "-0.96232 -0.310315 0.00821729 0.0553001 -0.152186 -0.806324 0.568879 0.99972 0.00400948 0.0120487 -0.019957 0.99958 0.00564608 0.0135266 -0.0250119 0.999213 0.00697393 0.0186555 -0.0343034 0.999314 0.00757918 0.0152461 -0.0329011 0.999681 0.00773247 -0.000154943 -0.0240269 0.999351 0.00721107 -0.0348006 -0.00583124 0.998141 0.00785734 -0.0534318 0.0282443 0.993185 0.0140636 -0.0913325 0.0710282 0.998498 0.0432591 -0.0262267 0.0210157 0.999129 -0.0169444 0.00832389 0.0372118 0.999622 -0.0158279 0.0217098 0.00586508 0.999446 -0.0154785 0.0236398 -0.0175784 0.999309 -0.0144307 0.012933 -0.03171 0.999309 -0.0117984 -0.00763152 -0.0344152 0.999922 -0.00635246 -0.00119965 -0.0106755 0.999976 -0.00314211 0.0054079 0.00312701 0.999966 -0.00198791 0.00724479 0.00332934 0.999977 -0.00156569 0.00662965 4.12157e-05 -1.17796 -0.24808 0.179379 0.00058455 -0.722105 -0.691755 -0.00621466 0.739054 0.000562543 0.730411 -0.736887 0.739054 0.000562827 0.730405 -0.736903"
        ),
        qvel=_parse_float_list(
            "0.000524685 0.00139056 0.000110475 0.000387267 -0.00332488 -0.000637018 -0.000340321 -0.00199311 0.00498695 -0.000731889 -0.0048906 0.0161636 0.000420779 -0.00492143 -0.00475585 0.000718906 0.000829923 -0.0131897 0.000452709 0.00508467 -0.0155297 -0.000382475 0.00748081 -0.0113062 -0.000115123 0.00460338 0.000518482 0.00309457 -0.00321484 0.0225712 -0.00127211 0.0529149 0.00934746 0.00357411 -0.0398959 0.0101416 0.00353874 -0.0207134 -0.007091 0.00445331 -0.0052276 -0.0156978 0.00441011 0.00527641 -0.0132248 0.00348936 0.0110195 0.00174422 0.00190069 -0.00575922 0.0171878 0.000878582 5.85321e-05 0.00470361 0.000687971 0.00287952 0.00397919 0.000255688 -0.002059 0.00593209 3.79798e-07 -1.13445e-08 1.93784e-07 5.23767e-05 1.18092e-05 -1.49264e-06 -5.99913e-06 5.67392e-07 -2.22363e-05 -1.32551e-05 -4.98016e-06 -1.69525e-06 1.0342e-05 6.18723e-05"
        ),
        ctrl=[255],
        mpos=_parse_float_list("-1.17796 -0.24808 0.179379"),
        mquat=_parse_float_list("0.000584735 -0.722105 -0.691755 -0.00621478"),
    )

    return scene.compile()


m = init()

d = mj.MjData(m)


def cb(key: int) -> None:
    if key is glfw.KEY_SPACE:
        global d
        d.actuator("gripper/fingers_actuator").ctrl = 255


with mujoco.viewer.launch_passive(model=m, data=d, key_callback=cb) as viewer:
    mj.mj_resetDataKeyframe(m, d, 0)

    while viewer.is_running():
        step_start = time.time()

        # step simulation one time step
        mj.mj_step(m, d)

        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
