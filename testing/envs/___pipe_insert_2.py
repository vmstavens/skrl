import json
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
        sparse_reward=False,
    )


def add_mesh(
    mesh_path: Union[str, Path],
    pos: list[float] = [0, 0, 0],
    euler: list[float] = [0, 0, 0],
    scale=1,
    density=8000,
    rgba: list[float] = [0.2, 0.2, 0.2, 1],
    contype: int = 1,
    conaffinity: int = 1,
) -> mj.MjSpec:
    """Create a spec containing a single OBJ mesh body."""
    mesh_path = Path(mesh_path)
    mesh_name = mesh_path.stem
    spec = mj.MjSpec()

    # Load mesh
    vertices, faces = parse_obj(mesh_path)

    # Add mesh to spec
    spec.add_mesh(
        name=mesh_name,
        uservert=vertices.flatten(),
        userface=faces.flatten(),
        scale=[scale] * 3,
        # plugin=mj.mjtGeom.mjGEOM_SDF,
    )

    # quat = np.array([1, 0, 0, 0])
    # euler = np.array([1.57, 1.57, 0])

    # Create needle body
    body = spec.worldbody.add_body(name=mesh_name, pos=pos, euler=euler)

    body.add_geom(
        # euler=[1.57, 1.57, 1.57],
        meshname=mesh_name,
        type=mj.mjtGeom.mjGEOM_MESH,
        # type=mj.mjtGeom.mjGEOM_SDF,
        # mesh="test",
        # plugin=mj.mjtObj.mjOBJ_PLUGIN,
        # type=mj.mjtGeom.mjGEOM_MESH,
        contype=contype,
        conaffinity=conaffinity,
        rgba=rgba,
        density=density,
        solref=[0.02, 1],
        solimp=[0.9, 0.95, 0.001, 0.5, 2],
        friction=[0.1, 0.005, 0.0001],
    )

    return spec


def parse_obj(obj_file_path):
    vertices = []
    faces = []

    with open(obj_file_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.split()

            if parts[0] == "v":
                vertices.append(list(map(float, parts[1:4])))
            elif parts[0] == "f":
                face = []
                for part in parts[1:]:
                    # Handle cases like "1", "1/2", or "1/2/3"
                    v = part.split("/")[0]
                    face.append(int(v) - 1)  # Convert to 0-based index

                # Triangulate N-gons on the fly
                if len(face) > 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])
                else:
                    faces.append(face)

    return np.array(vertices), np.array(faces)


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


def pipe(
    inner_radius: float = 0.1,
    outer_radius: float = 0.12,
    length: float = 0.1,
    resolution: int = 40,
    friction: list[float] = [0.2, 0.2, 0.2],
    rgba: list[float] = [0.2, 0.2, 0.2, 1],
    solref: list[float] = [0.00001, 2],
) -> mj.MjSpec:
    if inner_radius >= outer_radius:
        raise ValueError(
            f"pipe requires inner_radius < outer_radius, got {inner_radius} >= {outer_radius}"
        )
    if resolution <= 0:
        raise ValueError(f"pipe requires resolution > 0, got {resolution}")
    if length <= 0:
        raise ValueError(f"pipe requires length > 0, got {length}")
    if len(friction) != 3:
        raise ValueError(f"pipe requires 3 friction values, got {len(friction)}")
    if len(rgba) != 4:
        raise ValueError(f"pipe requires 4 rgba values, got {len(rgba)} ({rgba})")
    if len(solref) != 2:
        raise ValueError(f"pipe requires 2 solref values, got {len(solref)} ({solref})")

    friction_str = " ".join(f"{float(f)}" for f in friction)
    rgba_str = " ".join(f"{float(f)}" for f in rgba)
    solref_str = " ".join(f"{float(f)}" for f in solref)
    angle_deg = 360.0 / resolution
    angle_rad = 2.0 * np.pi / resolution

    wall_thickness = outer_radius - inner_radius
    radial_half = wall_thickness / 2.0
    radius_mid = inner_radius + radial_half
    tangential_half = outer_radius * np.sin(angle_rad / 2.0)
    half_length = length

    _XML = f"""
    <mujoco>
        <worldbody>
            <body euler="0 0 0" pos="0 0 0">
                <replicate sep="hole:" count="{resolution}" euler="0 0 {angle_deg}">
                    <geom
                        type="box"
                        solref="{solref_str}"
                        pos="{radius_mid} 0 0"
                        size="{radial_half} {tangential_half} {half_length}"
                        friction="{friction_str}"
                        rgba="{rgba_str}"
                    />
                </replicate>
                <site
                    name="pipe_entry"
                    pos="0 0 {-half_length}"
                    size="0.002"
                    rgba="0 0 0 0"
                    group="2"
                />
                <site
                    name="pipe_exit"
                    pos="0 0 {half_length}"
                    size="0.002"
                    rgba="1 1 1 0"
                    group="2"
                />
            </body>
        </worldbody>
    </mujoco>
    """
    return mj.MjSpec().from_string(_XML)


def empty() -> mj.MjSpec:
    # <flag multiccd="enable" nativeccd="enable" />
    # gravity="0 0 -9.82"
    # _XML = """
    #     <mujoco model="empty scene">

    #     <compiler angle="radian" autolimits="true" />
    #     <option timestep="0.002"
    #         integrator="implicitfast"
    #         solver="Newton"
    #         gravity="0 0 0"
    #         cone="elliptic"
    #         sdf_iterations="5"
    #         sdf_initpoints="30"
    #         noslip_iterations="2"
    #         ls_iterations="10"
    #     />

    #     <statistic center="0.3 0 0.3" extent="0.8" meansize="0.08" />

    #     <visual>
    #         <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
    #         <rgba haze="0.15 0.25 0.35 1" />
    #         <global azimuth="120" elevation="-20" offwidth="2000" offheight="2000" />

    #     </visual>

    #     <asset>
    #         <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
    #             height="3072" />
    #         <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
    #             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    #         <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
    #             reflectance="0.2" />
    #     </asset>

    #     <worldbody>
    #         <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    #     </worldbody>
    # </mujoco>
    # """
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
            <!-- <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" /> -->

        </worldbody>

    </mujoco>
    """
    return mj.MjSpec().from_string(_XML)
    # <geom name="floor" size="0 0 0.5" type="plane" material="groundplane"
    #     solimp="0.0 0.0 0.0 0.0 1" />


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


def points_inside_cylinder(
    points: Union[list[float], np.ndarray],
    axis_start: Union[list[float], np.ndarray],
    axis_end: Union[list[float], np.ndarray],
    radius: float,
    *,
    inclusive: bool = True,
) -> np.ndarray:
    """Return a boolean mask for points inside a finite cylinder.

    The cylinder is defined by the axis segment from axis_start to axis_end
    and radius. Uses the distance-to-axis formula based on the cross product:
    ||(P-A) x (B-A)|| / ||B-A|| <= R and 0 <= t <= 1 where
    t = ((P-A)·(B-A)) / ||B-A||^2.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]

    a = np.asarray(axis_start, dtype=float)
    b = np.asarray(axis_end, dtype=float)
    v = b - a
    v2 = float(np.dot(v, v))
    if v2 <= 0.0:
        raise ValueError("Cylinder axis endpoints must be distinct.")

    w = pts - a
    t = (w @ v) / v2
    if inclusive:
        within_caps = (t >= 0.0) & (t <= 1.0)
    else:
        within_caps = (t > 0.0) & (t < 1.0)

    cross = np.cross(w, v)
    dist2 = np.einsum("ij,ij->i", cross, cross)
    if inclusive:
        within_radius = dist2 <= (radius * radius) * v2
    else:
        within_radius = dist2 < (radius * radius) * v2

    return within_caps & within_radius


def keypoints_within_pipe(
    keypoints: Union[list[float], np.ndarray],
    pipe_axis_start: Union[list[float], np.ndarray],
    pipe_axis_end: Union[list[float], np.ndarray],
    inner_radius: float,
    *,
    inclusive: bool = True,
) -> bool:
    """Return True if all keypoints lie within the pipe's inner cylinder."""
    inside = points_inside_cylinder(
        keypoints, pipe_axis_start, pipe_axis_end, inner_radius, inclusive=inclusive
    )
    return bool(np.all(inside))


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

        self.sequence = {}
        with open("testing/experiments/pipe_insert/tmp/success_rollout.json", "r") as f:
            self.sequence = json.load(f)
        self.sequence_actions = jp.array(self.sequence.get("actions", []))

        # print(f"{self.sequence_actions.shape=}")
        # quit()

        self.i = 0

        self._mj_model = self._init()
        self.ctrl_scale: float = 1.0
        # self._mj_model = self._init()
        # self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model)

        # self._termination_threshold = 0.05
        self._termination_threshold = 0.04

        self._sparse_reward = config["sparse_reward"]

        self._post_init()

    # def _init(self) -> mj.MjModel:
    #     # root
    #     _HERE = Path(__file__).parent.parent
    #     # scene path

    #     scene = empty()
    #     # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

    #     keyframe_path = _HERE / "envs/tmp/keyframe.xml"
    #     if keyframe_path.exists():
    #         key_kwargs = _load_keyframe_xml(keyframe_path, key_name="bent")
    #         scene.add_key(**key_kwargs)

    #     # keyframe
    #     # <key
    #     #   time="0"
    #     #   qpos="-1.17579e-16 -4.29464e-09 -0.00762523 -5.44483e-09 0.202229 -0.0516089 0.638943 0.302905 -0.302905 0.638943 0.999999 2.2253e-15 0.00142279 5.91223e-12 0.999984 3.96161e-14 0.00570596 2.37104e-11 0.999916 2.59024e-13 0.0129429 5.37814e-11 0.999727 1.10739e-12 0.0233815 9.71439e-11 0.999297 3.83636e-12 0.037491 1.5568e-10 0.998431 1.19919e-11 0.0560022 2.32109e-10 0.996805 3.60452e-11 0.0798733 3.29115e-10 0.993924 1.09549e-10 0.110073 4.45615e-10 0.989142 3.53764e-10 0.14696 5.63023e-10 0.981983 1.26857e-09 0.188971 5.92783e-10 0.977606 4.07614e-10 0.210443 7.76868e-11 0.99128 1.90684e-10 0.13177 -4.76039e-11 0.996672 -1.1612e-09 0.081522 -2.61475e-11 0.998726 -1.95258e-09 0.0504712 1.27808e-10 0.999506 -2.04172e-09 0.0314265 2.16759e-10 0.999806 -1.82159e-09 0.0197197 2.23546e-10 0.999922 -1.52217e-09 0.0124806 1.90193e-10 0.999968 -1.23217e-09 0.00797065 1.47164e-10 0.999987 -9.79397e-10 0.00513794 1.0812e-10 0.999994 -7.69205e-10 0.0033436 7.71299e-11 0.999998 -5.98964e-10 0.00219706 5.41213e-11 0.999999 -4.63361e-10 0.00145792 3.76578e-11 1 -3.56465e-10 0.000977087 2.6121e-11 1 -2.72638e-10 0.000661377 1.81273e-11 1 -2.06959e-10 0.0004521 1.26159e-11 1 -1.55395e-10 0.000311997 8.81829e-12 1 -1.14815e-10 0.000217225 6.1945e-12 1 -8.28998e-11 0.000152406 4.37209e-12 1 -5.79781e-11 0.000107538 3.09667e-12 1 -3.8849e-11 7.60742e-05 2.19542e-12 1 -2.45999e-11 5.36949e-05 1.55121e-12 1 -1.446e-11 3.75401e-05 1.08484e-12 1 -7.69925e-12 2.57135e-05 7.42983e-13 1 -3.58256e-12 1.69658e-05 4.90066e-13 1 -1.37644e-12 1.04883e-05 3.02855e-13 1 -3.95552e-13 5.77704e-06 1.6677e-13 1 -6.98175e-14 2.54067e-06 7.33316e-14 1 -4.45923e-15 6.32989e-07 1.82689e-14"
    #     #   qvel="-9.55857e-16 -3.31243e-08 -1.72337e-09 2.71389e-13 1.64346e-05 -2.22251e-05 -1.26243e-12 -0.000115897 -1.10107e-12 -6.05372e-17 3.60291e-08 2.56484e-15 -1.05208e-15 1.51787e-07 1.03033e-14 -6.7966e-15 3.54527e-07 2.34527e-14 -2.85164e-14 6.43448e-07 4.24595e-14 -9.55509e-14 1.0046e-06 6.82961e-14 -2.79508e-13 1.40362e-06 1.03578e-13 -7.34085e-13 1.7748e-06 1.58204e-13 -1.6758e-12 2.00573e-06 2.7867e-13 -2.61207e-12 1.9214e-06 7.23119e-13 4.56822e-12 1.28628e-06 3.23267e-12 -3.35099e-11 1.27249e-05 -6.86669e-12 4.05836e-11 9.89241e-06 1.27213e-11 8.59296e-11 7.18127e-06 2.53953e-12 7.5729e-11 4.85206e-06 -8.51421e-12 5.92835e-11 2.97601e-06 -1.08683e-11 4.66737e-11 1.54886e-06 -9.51619e-12 3.70647e-11 5.35384e-07 -7.35385e-12 2.94025e-11 -1.13352e-07 -5.36873e-12 2.31852e-11 -4.52606e-07 -3.80363e-12 1.81522e-11 -5.39978e-07 -2.64971e-12 1.41173e-11 -4.32818e-07 -1.82768e-12 1.0916e-11 -1.86256e-07 -1.25241e-12 8.39619e-12 1.48288e-07 -8.52985e-13 6.42132e-12 5.24312e-07 -5.76111e-13 4.87431e-12 9.0117e-07 -3.83842e-13 3.65986e-12 1.24478e-06 -2.49965e-13 2.70415e-12 1.52813e-06 -1.56678e-13 1.95248e-12 1.7316e-06 -9.19709e-14 1.36552e-12 1.8431e-06 -4.77141e-14 9.14989e-13 1.85797e-06 -1.83362e-14 5.79391e-13 1.77878e-06 8.09611e-17 3.40573e-13 1.61493e-06 1.03959e-14 1.8134e-13 1.38203e-06 1.47869e-14 8.43813e-14 1.10121e-06 1.50065e-14 3.24211e-14 7.9834e-07 1.25294e-14 9.3182e-15 5.03136e-07 8.65177e-15 1.64568e-15 2.48278e-07 4.54915e-15 1.05677e-16 6.8541e-08 1.31731e-15"
    #     #   ctrl="0 0 0"
    #     # />
    #     # scene.add_key()

    #     self.pipe_inner_radius = 0.0385
    #     self.pipe_outer_radius = 0.0435
    #     self.pipe_length = 0.09

    #     pip = pipe(
    #         inner_radius=self.pipe_inner_radius,
    #         outer_radius=self.pipe_outer_radius,
    #         length=self.pipe_length,
    #         rgba=[0.2, 0.2, 0.2, 1],
    #     )
    #     # pip = pipe(inner_radius=0.0385, outer_radius=0.0435, length=0.075)

    #     keypoint_height = self.pipe_inner_radius - (
    #         self.pipe_outer_radius - self.pipe_inner_radius
    #     )

    #     pip.worldbody.first_body().add_site(
    #         name="target_1", pos=[0, keypoint_height, -0.05], group=1, rgba=[1, 0, 0, 1]
    #     )
    #     pip.worldbody.first_body().add_site(
    #         name="target_2", pos=[0, keypoint_height, 0], group=1, rgba=[1, 1, 0, 1]
    #     )
    #     pip.worldbody.first_body().add_site(
    #         name="target_3", pos=[0, keypoint_height, 0.05], group=1, rgba=[0, 1, 0, 1]
    #     )
    #     # pip.worldbody.first_body().add_site(
    #     #     name="target_1", pos=[0, 0.015, -0.05], group=1, rgba=[1, 0, 0, 1]
    #     # )
    #     # pip.worldbody.first_body().add_site(
    #     #     name="target_2", pos=[0, 0.015, 0], group=1, rgba=[1, 1, 0, 1]
    #     # )
    #     # pip.worldbody.first_body().add_site(
    #     #     name="target_3", pos=[0, 0.015, 0.05], group=1, rgba=[0, 1, 0, 1]
    #     # )

    #     twist = 60000.0 * 2
    #     bend = 10000000.0 * 2
    #     # twist = 60000.0 * 2
    #     # bend = 10000000.0 * 2

    #     cable = mjx_cable(count=40, twist=twist, bend=bend)
    #     # cable = mjx_cable(count="40 1 1", twist=twist, bend=bend)
    #     # quit()

    #     # cable = mjx_cbale(count="40 1 1", twist=60000.0 * 2, bend=10000000.0 * 2)

    #     cable.body("cable:Bfirst").add_site(
    #         name="keypoint_3", group=1, rgba=[0, 1, 0, 1], pos=[0, -0.01, 0]
    #     )
    #     cable.body("cable:B2").add_site(
    #         name="keypoint_2", group=1, rgba=[1, 1, 0, 1], pos=[0, -0.01, 0]
    #     )
    #     cable.body("cable:B4").add_site(
    #         name="keypoint_1", group=1, rgba=[1, 0, 0, 1], pos=[0, -0.01, 0]
    #     )

    #     # print(cable.to_file("/home/vims/git/skrl/testing/envs/tmp/cable_mj.xml"))

    #     _c = scene.worldbody.add_camera(
    #         name="cam",
    #         pos=[1.2, 0.234, 0.156],
    #         # pos=[0.721, 0.234, 0.156],
    #         xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
    #         resolution=[640, 480],
    #     )

    #     # _b = scene.worldbody.add_body(name="am i falling?!", pos=[0, 0, 0])
    #     # _g = _b.add_geom(
    #     #     name="am i falling?!",
    #     #     type=mj.mjtGeom.mjGEOM_SPHERE,
    #     #     size=[0.02, 0.02, 0.02],
    #     #     rgba=[1, 0, 0, 1],
    #     #     contype=0,
    #     #     conaffinity=0,
    #     # )
    #     # _b.add_freejoint()

    #     # <camera pos="0.721 0.234 0.156" xyaxes="-0.037 0.999 0.000 -0.001 -0.000 1.000"/>

    #     gripper = scene.worldbody.add_body(
    #         # name="gripper",
    #         # pos=[0, 0.4, 0.1],
    #         # euler=[0, 0, 0],
    #         name="gripper",
    #         pos=[0, 0.4, 0.1],
    #         euler=[0, 0, 0],
    #         # euler=[0, 0, 1.57],
    #     )
    #     gripper.add_geom(
    #         type=mj.mjtGeom.mjGEOM_BOX,
    #         size=[0.02, 0.02, 0.02],
    #         contype=0,
    #         conaffinity=0,
    #         # rgba=[0, 0, 0, 0],
    #     )
    #     gripper.add_joint(
    #         name="x",
    #         type=mj.mjtJoint.mjJNT_SLIDE,
    #         axis=[1, 0, 0],
    #         range=self._slide_limits["x"],
    #     )
    #     gripper.add_joint(
    #         name="y",
    #         type=mj.mjtJoint.mjJNT_SLIDE,
    #         axis=[0, 1, 0],
    #         range=self._slide_limits["y"],
    #     )
    #     gripper.add_joint(
    #         name="z",
    #         type=mj.mjtJoint.mjJNT_SLIDE,
    #         axis=[0, 0, 1],
    #         range=self._slide_limits["z"],
    #     )
    #     # after creating `gripper`
    #     gripper.add_site(
    #         name="grip_cable_site",
    #         pos=[0.0, 0.0, 0.0],  # choose where on the gripper
    #         euler=[0.0, 0.0, 1.5707963],  # rotate attachment frame as needed
    #         group=1,
    #         rgba=[1, 0, 1, 1],
    #     )

    #     kp = 1000
    #     kv = 1000
    #     # kp = 100
    #     # kv = 100

    #     scene.add_actuator(
    #         name="x",
    #         target="x",
    #         trntype=mj.mjtTrn.mjTRN_JOINT,
    #         ctrlrange=self._slide_limits["x"],
    #     ).set_to_position(kp=kp, kv=kv)
    #     scene.add_actuator(
    #         name="y",
    #         target="y",
    #         trntype=mj.mjtTrn.mjTRN_JOINT,
    #         ctrlrange=self._slide_limits["y"],
    #     ).set_to_position(kp=kp, kv=kv)
    #     scene.add_actuator(
    #         name="z",
    #         target="z",
    #         trntype=mj.mjtTrn.mjTRN_JOINT,
    #         ctrlrange=self._slide_limits["z"],
    #     ).set_to_position(kp=kp, kv=kv)

    #     scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 3.14]).attach_body(
    #         # scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 0]).attach_body(
    #         pip.worldbody.first_body()
    #     )
    #     scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
    #         # scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
    #         cable.worldbody.first_body()
    #     )

    #     b = None
    #     b0 = cable.worldbody.first_body()
    #     for i in range(10):
    #         b = b0.first_body()
    #         b0 = b

    #     b.add_site(
    #         name="cable_weld_site",
    #         pos=[0.0, 0.0, 0.0],  # adjust if you want an offset on that segment
    #         euler=[0.0, 0.0, 1.57],
    #         group=1,
    #         rgba=[0, 1, 1, 1],
    #     )

    #     scene.add_equality(
    #         name="weld",
    #         type=mj.mjtEq.mjEQ_WELD,
    #         objtype=mj.mjtObj.mjOBJ_SITE,
    #         name1="grip_cable_site",
    #         name2="cable_weld_site",
    #         # omit data => MuJoCo uses the current relative pose as the target
    #         solref=[0.000001, 2],
    #     )

    #     # eq: mj.MjsEquality = scene.add_equality(
    #     #     name="weld",
    #     #     type=mj.mjtEq.mjEQ_WELD,
    #     #     objtype=mj.mjtObj.mjOBJ_BODY,
    #     #     name1="gripper",
    #     #     name2=b.name,
    #     #     # data=[*_rel_pos, 0, 0, 0, 1.0, 0, 0.7071068, 0.0, 0.7071068],
    #     #     # data=[*_rel_pos, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 1.0],
    #     #     # data=[0.0, 0.0, 0.0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 1.0],
    #     #     data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 0.0, 0.0, 1.0],
    #     #     # data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0.0, 0.0, 1.0],
    #     #     solref=[0.000001, 2],
    #     #     # info=[0, 0, 0, 0.7071068, 0, 0, 0.7071068],
    #     # )

    #     self._xml_path = "generated_scene.xml"  # dummy path
    #     return scene.compile()

    def _init(self) -> mj.MjModel:
        # root
        _ROOT = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        keyframe_path = _ROOT / "envs/tmp/keyframe.xml"
        if keyframe_path.exists():
            key_kwargs = _load_keyframe_xml(keyframe_path, key_name="bent")
            scene.add_key(**key_kwargs)

        # mesh_path = _HERE / "assets/surgical_robotics/assets/NeedleCollision.OBJ"

        # keyframe
        # <key
        #   time="0"
        #   qpos="-1.17579e-16 -4.29464e-09 -0.00762523 -5.44483e-09 0.202229 -0.0516089 0.638943 0.302905 -0.302905 0.638943 0.999999 2.2253e-15 0.00142279 5.91223e-12 0.999984 3.96161e-14 0.00570596 2.37104e-11 0.999916 2.59024e-13 0.0129429 5.37814e-11 0.999727 1.10739e-12 0.0233815 9.71439e-11 0.999297 3.83636e-12 0.037491 1.5568e-10 0.998431 1.19919e-11 0.0560022 2.32109e-10 0.996805 3.60452e-11 0.0798733 3.29115e-10 0.993924 1.09549e-10 0.110073 4.45615e-10 0.989142 3.53764e-10 0.14696 5.63023e-10 0.981983 1.26857e-09 0.188971 5.92783e-10 0.977606 4.07614e-10 0.210443 7.76868e-11 0.99128 1.90684e-10 0.13177 -4.76039e-11 0.996672 -1.1612e-09 0.081522 -2.61475e-11 0.998726 -1.95258e-09 0.0504712 1.27808e-10 0.999506 -2.04172e-09 0.0314265 2.16759e-10 0.999806 -1.82159e-09 0.0197197 2.23546e-10 0.999922 -1.52217e-09 0.0124806 1.90193e-10 0.999968 -1.23217e-09 0.00797065 1.47164e-10 0.999987 -9.79397e-10 0.00513794 1.0812e-10 0.999994 -7.69205e-10 0.0033436 7.71299e-11 0.999998 -5.98964e-10 0.00219706 5.41213e-11 0.999999 -4.63361e-10 0.00145792 3.76578e-11 1 -3.56465e-10 0.000977087 2.6121e-11 1 -2.72638e-10 0.000661377 1.81273e-11 1 -2.06959e-10 0.0004521 1.26159e-11 1 -1.55395e-10 0.000311997 8.81829e-12 1 -1.14815e-10 0.000217225 6.1945e-12 1 -8.28998e-11 0.000152406 4.37209e-12 1 -5.79781e-11 0.000107538 3.09667e-12 1 -3.8849e-11 7.60742e-05 2.19542e-12 1 -2.45999e-11 5.36949e-05 1.55121e-12 1 -1.446e-11 3.75401e-05 1.08484e-12 1 -7.69925e-12 2.57135e-05 7.42983e-13 1 -3.58256e-12 1.69658e-05 4.90066e-13 1 -1.37644e-12 1.04883e-05 3.02855e-13 1 -3.95552e-13 5.77704e-06 1.6677e-13 1 -6.98175e-14 2.54067e-06 7.33316e-14 1 -4.45923e-15 6.32989e-07 1.82689e-14"
        #   qvel="-9.55857e-16 -3.31243e-08 -1.72337e-09 2.71389e-13 1.64346e-05 -2.22251e-05 -1.26243e-12 -0.000115897 -1.10107e-12 -6.05372e-17 3.60291e-08 2.56484e-15 -1.05208e-15 1.51787e-07 1.03033e-14 -6.7966e-15 3.54527e-07 2.34527e-14 -2.85164e-14 6.43448e-07 4.24595e-14 -9.55509e-14 1.0046e-06 6.82961e-14 -2.79508e-13 1.40362e-06 1.03578e-13 -7.34085e-13 1.7748e-06 1.58204e-13 -1.6758e-12 2.00573e-06 2.7867e-13 -2.61207e-12 1.9214e-06 7.23119e-13 4.56822e-12 1.28628e-06 3.23267e-12 -3.35099e-11 1.27249e-05 -6.86669e-12 4.05836e-11 9.89241e-06 1.27213e-11 8.59296e-11 7.18127e-06 2.53953e-12 7.5729e-11 4.85206e-06 -8.51421e-12 5.92835e-11 2.97601e-06 -1.08683e-11 4.66737e-11 1.54886e-06 -9.51619e-12 3.70647e-11 5.35384e-07 -7.35385e-12 2.94025e-11 -1.13352e-07 -5.36873e-12 2.31852e-11 -4.52606e-07 -3.80363e-12 1.81522e-11 -5.39978e-07 -2.64971e-12 1.41173e-11 -4.32818e-07 -1.82768e-12 1.0916e-11 -1.86256e-07 -1.25241e-12 8.39619e-12 1.48288e-07 -8.52985e-13 6.42132e-12 5.24312e-07 -5.76111e-13 4.87431e-12 9.0117e-07 -3.83842e-13 3.65986e-12 1.24478e-06 -2.49965e-13 2.70415e-12 1.52813e-06 -1.56678e-13 1.95248e-12 1.7316e-06 -9.19709e-14 1.36552e-12 1.8431e-06 -4.77141e-14 9.14989e-13 1.85797e-06 -1.83362e-14 5.79391e-13 1.77878e-06 8.09611e-17 3.40573e-13 1.61493e-06 1.03959e-14 1.8134e-13 1.38203e-06 1.47869e-14 8.43813e-14 1.10121e-06 1.50065e-14 3.24211e-14 7.9834e-07 1.25294e-14 9.3182e-15 5.03136e-07 8.65177e-15 1.64568e-15 2.48278e-07 4.54915e-15 1.05677e-16 6.8541e-08 1.31731e-15"
        #   ctrl="0 0 0"
        # />
        # scene.add_key()

        self.pipe_inner_radius = 0.0385 / 2
        self.pipe_outer_radius = 0.0435 / 2
        self.pipe_length = 0.121 / 2  # 121 mm from schematic
        # pipe_length = 0.09

        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            # rgba=[0.2, 0.2, 0.2, 1],
        )
        # pip = pipe(inner_radius=0.0385, outer_radius=0.0435, length=0.075)

        mesh_path = _ROOT / "envs" / "assets/DS00240017_E.obj"

        mesh_spec = add_mesh(
            mesh_path=mesh_path,
            scale=0.001,
            euler=[0, 1.57, 0],
            rgba=[0.2, 0.2, 0.2, 1],
            # rgba=[0.2, 0.2, 0.2, 0.2],
            contype=0,
            conaffinity=0,
        )
        mesh_body = mesh_spec.worldbody.first_body()
        scene.worldbody.add_frame(
            pos=[0, -self.pipe_length - 0.01, 0.1], euler=[0, 0, 0]
        ).attach_body(mesh_body)

        keypoint_height = self.pipe_inner_radius - (
            self.pipe_outer_radius - self.pipe_inner_radius
        )

        pip.worldbody.first_body().add_site(
            name="target_1", pos=[0, keypoint_height, -0.05], group=1, rgba=[1, 0, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_2", pos=[0, keypoint_height, 0], group=1, rgba=[1, 1, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_3", pos=[0, keypoint_height, 0.05], group=1, rgba=[0, 1, 0, 1]
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

        cable = mjx_cable(count=40, twist=twist, bend=bend)
        # cable = mjx_cable(count="40 1 1", twist=twist, bend=bend)
        # quit()

        # cable = mjx_cbale(count="40 1 1", twist=60000.0 * 2, bend=10000000.0 * 2)

        cable.body("cable:Bfirst").add_site(
            name="keypoint_3", group=1, rgba=[0, 1, 0, 1], pos=[0, -0.01, 0]
        )
        cable.body("cable:B2").add_site(
            name="keypoint_2", group=1, rgba=[1, 1, 0, 1], pos=[0, -0.01, 0]
        )
        cable.body("cable:B4").add_site(
            name="keypoint_1", group=1, rgba=[1, 0, 0, 1], pos=[0, -0.01, 0]
        )

        # print(cable.to_file("/home/vims/git/skrl/testing/envs/tmp/cable_mj.xml"))

        _c = scene.worldbody.add_camera(
            name="cam",
            pos=[1.2, 0.234, 0.156],
            # pos=[0.721, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
            resolution=[640, 480],
        )

        # _b = scene.worldbody.add_body(name="am i falling?!", pos=[0, 0, 0])
        # _g = _b.add_geom(
        #     name="am i falling?!",
        #     type=mj.mjtGeom.mjGEOM_SPHERE,
        #     size=[0.02, 0.02, 0.02],
        #     rgba=[1, 0, 0, 1],
        #     contype=0,
        #     conaffinity=0,
        # )
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

        kp = 1000
        kv = 1000
        # kp = 100
        # kv = 100

        scene.add_actuator(
            name="x",
            target="x",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=self._slide_limits["x"],
        ).set_to_position(kp=kp, kv=kv)
        scene.add_actuator(
            name="y",
            target="y",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=self._slide_limits["y"],
        ).set_to_position(kp=kp, kv=kv)
        scene.add_actuator(
            name="z",
            target="z",
            trntype=mj.mjtTrn.mjTRN_JOINT,
            ctrlrange=self._slide_limits["z"],
        ).set_to_position(kp=kp, kv=kv)

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
            euler=[0.0, 0.0, 1.57],
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

        # eq: mj.MjsEquality = scene.add_equality(
        #     name="weld",
        #     type=mj.mjtEq.mjEQ_WELD,
        #     objtype=mj.mjtObj.mjOBJ_BODY,
        #     name1="gripper",
        #     name2=b.name,
        #     # data=[*_rel_pos, 0, 0, 0, 1.0, 0, 0.7071068, 0.0, 0.7071068],
        #     # data=[*_rel_pos, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 1.0],
        #     # data=[0.0, 0.0, 0.0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 1.0],
        #     data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, 0.0, 0.0, 1.0],
        #     # data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0.0, 0.0, 1.0],
        #     solref=[0.000001, 2],
        #     # info=[0, 0, 0, 0.7071068, 0, 0, 0.7071068],
        # )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _post_init(self) -> None:
        self.target_ids = [self._mj_model.site(f"target_{i + 1}").id for i in range(3)]
        self.keypoint_ids = [
            self._mj_model.site(f"keypoint_{i + 1}").id for i in range(3)
        ]
        # Initialize default state from the compiled model so it stays consistent
        # with the current kinematic layout (y locked, no hinge joints).
        # d0 = mj.MjData(self._mj_model)
        # mj.mj_forward(self._mj_model, d0)
        # self.QPOS = jp.array(
        #     np.load("testing/experiments/pipe_insert/constants/qpos.npy")
        # )
        # self.QVEL = jp.array(
        #     np.load("testing/experiments/pipe_insert/constants/qvel.npy")
        # )
        # self.CTRL = jp.array(
        #     np.load("testing/experiments/pipe_insert/constants/ctrl.npy")
        # )

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

        # TODO : WE ARE Wokring from here....

        # self.qpos0 = jp.array(np.array([4.11014825e-07, -7.62520820e-03]))
        # self.ctrl0 = jp.array(np.array([0.0, 0.0]))
        # self.qpos1 = jp.array(np.array([0.20210016, -0.05143623]))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Split RNG
        rng, rng_delta = jax.random.split(rng)

        self.i = 0

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
        delta = jax.random.uniform(rng_delta, shape=(3,), minval=-0.3, maxval=0.3)
        # delta = jp.zeros(3)
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

        # print(f"{self.mjx_model.nu=}")
        # print(f"{self.mjx_model.nv=}")
        # print(f"{self.mjx_model.na=}")
        # print(f"{self.mjx_model.nq=}")
        # print(f"{self.mjx_model.nbody=}")
        # print(f"{self.mjx_model.ngeom=}")
        # print(f"{self.mjx_model.njnt=}")

        data = data.replace(qvel=qvel)
        data = data.replace(qacc=qacc)
        data = data.replace(qfrc_applied=qfrc_applied)
        data = data.replace(xfrc_applied=xfrc_applied)

        metrics = {}
        info = {"i": jp.array(0)}
        info = {"rng": rng}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # action = jp.zeros(3)

        # action_index = state.info.get("i", jp.array(0))
        # max_index = jp.asarray(self.sequence_actions.shape[0] - 1)
        # action_index = jp.minimum(action_index, max_index)
        # action = self.sequence_actions[action_index]

        a = state.data.ctrl + (action * self.ctrl_scale)

        # a = state.data.ctrl + self.ctrl_scale * action
        data = mjx_env.step(self.mjx_model, state.data, a, self.n_substeps)

        # Reward: negative distance to target + bonus for being close
        dist_to_target = self._get_distance(data)

        if self._sparse_reward:
            reward = jp.float64(dist_to_target <= self._termination_threshold)
        else:
            reward = -dist_to_target

        obs = self._get_obs(data, state.info)

        # This is to check for stability. If the simulation has turned unstable these will be None
        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        # --- RANDOM TERMINATION (50%) ---
        rng, rng_term = jax.random.split(state.info["rng"])
        random_done = jax.random.bernoulli(rng_term, p=0.1)
        state.info["rng"] = rng
        # Existing failure termination
        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        # Create bool (remember this is not branching so this is legal)
        # if we are within 2 cm of the target, then we are successful
        success = dist_to_target <= self._termination_threshold
        # success = dist_to_target <= 0.02

        # def _print_reward(_):
        #     part1 = dist_to_target
        #     part2 = jp.asarray(self._termination_threshold)
        #     part3 = part1 <= part2

        #     # part4 = self.mjx_model.dof_armature[self._id]
        #     # part5 = self.mjx_model.dof_damping[self._id]

        #     # jax.debug.print("part4 = {part4}", part4=part4)
        #     # jax.debug.print("part5 = {part5}", part5=part5)

        #     jax.debug.print("success success: {success}", success=success)
        #     jax.debug.print("\tsuccess reward: {reward}", reward=reward)
        #     jax.debug.print("\tsuccess part1: {part1}", part1=part1)
        #     jax.debug.print("\tsuccess part2: {part2}", part2=part2)
        #     jax.debug.print("\tsuccess part3: {part3}", part3=part3)
        #     jax.debug.print("\tsuccess action: {action}", action=action)
        #     # jax.debug.print("\tsuccess action: {i}", i=action_index)
        #     return None

        # jax.lax.cond(success, _print_reward, lambda _: None, operand=None)

        # Combine all termination conditions
        # done = jp.logical_or(nan_condition, success, random_done)
        done = nan_condition | success
        # done = nan_condition | success | random_done

        # done = jp.logical_or(nan_condition, success)

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

        info = {**state.info}

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
        # if env._key_id is not None:
        #     data.qpos[:] = np.asarray(model.key_qpos[env._key_id])
        #     data.qvel[:] = np.asarray(model.key_qvel[env._key_id])
        #     data.ctrl[:] = np.asarray(model.key_ctrl[env._key_id])
        # else:
        data.qpos[:] = np.load(env.keys["qpos"])
        data.qvel[:] = np.load(env.keys["qvel"])
        data.ctrl[:] = np.load(env.keys["ctrl"])

    def cb(key: int) -> None:
        if key is glfw.KEY_SPACE:
            mj.mj_setKeyframe(model, data, 0)
            qpos = data.qpos
            qvel = data.qvel
            ctrl = data.ctrl
            np.save("testing/experiments/pipe_insert/constants/v3/qpos.npy", qpos)
            np.save("testing/experiments/pipe_insert/constants/v3/qvel.npy", qvel)
            np.save("testing/experiments/pipe_insert/constants/v3/ctrl.npy", ctrl)
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

        def get_keypoints(data: mj.MjData) -> np.ndarray:
            return np.array([data.site(kp_id).xpos for kp_id in env.keypoint_ids])

        def get_pipe_end_points(data: mj.MjData) -> np.ndarray:
            pipe_points = ["pipe_entry", "pipe_exit"]
            return np.array([data.site(kp_id).xpos for kp_id in pipe_points])

        # input("go?")
        while viewer.is_running():
            step_start = time.time()

            pipe_entry, pipe_exit = get_pipe_end_points(data)

            success = keypoints_within_pipe(
                keypoints=get_keypoints(data),
                pipe_axis_start=pipe_entry,
                pipe_axis_end=pipe_exit,
                inner_radius=env.pipe_inner_radius,
            )

            print(f"{success=}")

            # step simulation one time step
            mj.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
