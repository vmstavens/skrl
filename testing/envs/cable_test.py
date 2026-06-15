import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np

from lauge.dominik.cable import _parse_float_list
from testing.envs.pipe_insert_4 import mjx_cable


def empty() -> mj.MjSpec:

    _XML = """
        <mujoco model="empty scene">

        <compiler angle="radian" autolimits="true" />
        <option timestep="0.002" integrator="implicitfast" solver="Newton" gravity="0 0 -9.82" cone="elliptic" >
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
                    solref="0.001 3"
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


def add_6dof_base_position_actuators(
    spec: mj.MjSpec,
    *,
    base_body: Optional[str] = None,
    prefix: str = "base",
    pos_range: tuple[float, float] = (-0.3, 0.3),
    rot_range: tuple[float, float] = (-np.pi, np.pi),
    kp: float = 1000.0,
    kv: float = 1000.0,
    base_dir: Optional[Union[str, Path]] = None,
    meshdir: Optional[Union[str, Path]] = None,
) -> mj.MjSpec:
    """Add 6-DoF slide/hinge joints at the base body with position actuators.

    This attempts to avoid disk IO by converting the spec to an XML string,
    modifying it in-memory, and returning a new MjSpec from the updated XML.
    """

    def _spec_to_xml_string(spec_obj: mj.MjSpec) -> str:
        for attr in ("to_xml", "to_xml_string", "to_string"):
            fn = getattr(spec_obj, attr, None)
            if callable(fn):
                xml = fn()
                if isinstance(xml, bytes):
                    xml = xml.decode("utf-8")
                return xml
        # Fallback to compiling and saving if no string method exists.
        try:
            model = spec_obj.compile()
            if hasattr(mj, "mj_saveLastXML"):
                import tempfile
                from pathlib import Path

                with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
                    mj.mj_saveLastXML(f.name, model)
                    return Path(f.name).read_text()
        except Exception as exc:  # pragma: no cover - best effort fallback
            raise RuntimeError("Unable to export MjSpec to XML string.") from exc
        raise RuntimeError("Unable to export MjSpec to XML string.")

    xml = _spec_to_xml_string(spec)
    root = ET.fromstring(xml)

    compiler = root.find("compiler")
    if meshdir is not None or base_dir is not None:
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        if meshdir is not None:
            compiler.set("meshdir", str(Path(meshdir)))
        elif base_dir is not None:
            base_dir_path = Path(base_dir)
            meshdir_attr = compiler.get("meshdir")
            if meshdir_attr:
                meshdir_path = Path(meshdir_attr)
                if not meshdir_path.is_absolute():
                    compiler.set("meshdir", str(base_dir_path / meshdir_path))
            else:
                compiler.set("meshdir", str(base_dir_path))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in spec XML.")

    if base_body is None:
        base = worldbody.find("body")
        if base is None:
            raise ValueError("No <body> found under <worldbody>.")
    else:
        base = worldbody.find(f".//body[@name='{base_body}']")
        if base is None:
            raise ValueError(f"Base body '{base_body}' not found.")

    joint_names = {j.get("name") for j in root.findall(".//joint") if j.get("name")}
    actuator_names = {
        a.get("name") for a in root.findall(".//actuator/*") if a.get("name")
    }

    def _add_joint(
        name: str, jtype: str, axis: list[float], jrange: tuple[float, float]
    ):
        if name in joint_names:
            raise ValueError(f"Joint '{name}' already exists.")
        ET.SubElement(
            base,
            "joint",
            name=name,
            type=jtype,
            axis=f"{axis[0]} {axis[1]} {axis[2]}",
            range=f"{jrange[0]} {jrange[1]}",
        )
        joint_names.add(name)

    def _ensure_actuator_root() -> ET.Element:
        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")
        return actuator

    def _add_position_actuator(
        name: str, joint: str, ctrlrange: tuple[float, float]
    ) -> None:
        if name in actuator_names:
            raise ValueError(f"Actuator '{name}' already exists.")
        actuator_root = _ensure_actuator_root()
        ET.SubElement(
            actuator_root,
            "position",
            name=name,
            joint=joint,
            kp=str(kp),
            kv=str(kv),
            ctrlrange=f"{ctrlrange[0]} {ctrlrange[1]}",
        )
        actuator_names.add(name)

    # 3 translational slides.
    _add_joint(f"{prefix}_x", "slide", [1.0, 0.0, 0.0], pos_range)
    _add_joint(f"{prefix}_y", "slide", [0.0, 1.0, 0.0], pos_range)
    _add_joint(f"{prefix}_z", "slide", [0.0, 0.0, 1.0], pos_range)
    # 3 rotational hinges (roll, pitch, yaw).
    _add_joint(f"{prefix}_roll", "hinge", [1.0, 0.0, 0.0], rot_range)
    _add_joint(f"{prefix}_pitch", "hinge", [0.0, 1.0, 0.0], rot_range)
    _add_joint(f"{prefix}_yaw", "hinge", [0.0, 0.0, 1.0], rot_range)

    # Position actuators for all 6 joints.
    _add_position_actuator(f"{prefix}_x", f"{prefix}_x", pos_range)
    _add_position_actuator(f"{prefix}_y", f"{prefix}_y", pos_range)
    _add_position_actuator(f"{prefix}_z", f"{prefix}_z", pos_range)
    _add_position_actuator(f"{prefix}_roll", f"{prefix}_roll", rot_range)
    _add_position_actuator(f"{prefix}_pitch", f"{prefix}_pitch", rot_range)
    _add_position_actuator(f"{prefix}_yaw", f"{prefix}_yaw", rot_range)

    updated_xml = ET.tostring(root, encoding="unicode")
    return mj.MjSpec.from_string(updated_xml)


def init() -> mj.MjModel:
    scene = empty()

    _f = scene.worldbody.add_frame(name="cable", pos=[0, 0, 1])
    _xf = scene.worldbody.add_frame(name="xcable", pos=[0.2, 0, 1])

    cable = mjs_cable(model_name="cable", initial="free", count="5 1 1", size=0.3)
    xcable = mjx_cable(model_name="xcable", initial="free", count="20 1 1", size=1)

    # _f.attach_body(cable.worldbody.first_body(), prefix="cable")
    _xf.attach_body(xcable.worldbody.first_body(), prefix="xcable")

    gripper = mj.MjSpec.from_file("assets/robotiq_hande/hande.xml")

    from robot_descriptions import robotiq_2f85_mj_description

    gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)

    print(robotiq_2f85_mj_description.MJCF_PATH)

    gripper = add_6dof_base_position_actuators(
        gripper,
        base_dir=Path(
            robotiq_2f85_mj_description.MJCF_PATH
        ).parent,  # fixes meshdir resolution
        # base_body="robotiq_2f85_base",  # optional, if you know the base body name
        prefix="gripper_base",
    )

    f_hande = scene.worldbody.add_frame(
        name="hande", pos=[0, 0, 0.1], euler=[0, 3.14, 0]
    )
    f_hande.attach_body(gripper.worldbody.first_body(), prefix="hande")

    scene.compile()

    # print(scene.to_file("testing/envs/assets/cp.xml"))

    return scene.compile()


m = init()

m = mj.MjModel.from_xml_path("testing/envs/assets/test.xml")

d = mj.MjData(m)


def cb(key: int) -> None:
    if key is glfw.KEY_SPACE:
        global d
        d.actuator("handefingers_actuator").ctrl = 255


with mujoco.viewer.launch_passive(model=m, data=d, key_callback=cb) as viewer:
    # input("go?")
    while viewer.is_running():
        step_start = time.time()

        # step simulation one time step
        mj.mj_step(m, d)

        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # input()
