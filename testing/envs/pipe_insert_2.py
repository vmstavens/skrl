import json
import pickle
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import glfw
import jax
import jax.numpy as jp
import mujoco
import mujoco as mj
import mujoco.viewer
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from utils.mjx import ObjType, get_pose, is_stable


def _ensure_warp_internal_module() -> None:
    """Ensure warp._src is available on the warp package.

    Some warp builds expose the internal package but do not bind it on the
    top-level module, which breaks downstream references like warp._src.types.
    """
    try:
        import importlib

        import warp as wp

        try:
            getattr(wp, "_src")
            return
        except AttributeError:
            pass

        wp_src = importlib.import_module("warp._src")
        setattr(wp, "_src", wp_src)
    except Exception:
        # If warp isn't available or internal layout changed, let callers fail
        # with the original error.
        return


# def domain_randomize(model: mjx.Model, mj_model: mj.MjModel, rng: jax.Array):
#     mj_model = mj_model

#     # Cable joint ids (exclude the free joint) and section properties.
#     cable_joint_ids = np.array(
#         [
#             j
#             for j in range(mj_model.njnt)
#             if mj_model.joint(j).name.startswith("cable:J")
#         ],
#         dtype=np.int32,
#     )
#     cable_geom_id = None
#     for geom_name in ("cable:G1", "cable:G2", "cable:Gfirst", "cable:Glast"):
#         try:
#             cable_geom_id = mj_model.geom(geom_name).id
#             break
#         except KeyError:
#             continue
#     if cable_joint_ids.size and cable_geom_id is None:
#         raise ValueError("Cable joints found but no cable geom to infer stiffness.")

#     if cable_joint_ids.size:
#         geom_size = np.asarray(mj_model.geom_size[cable_geom_id], dtype=np.float64)
#         radius = float(geom_size[0])
#         fromto = np.asarray(mj_model.geom_fromto[cable_geom_id], dtype=np.float64)
#         length = float(np.linalg.norm(fromto[3:] - fromto[:3]))
#         if not np.isfinite(length) or length <= 0:
#             length = float(geom_size[1] * 2.0)
#         length = max(length, 1e-9)
#         # Section properties for capsule/cylinder geometry.
#         cable_j = np.pi * radius**4 / 2.0
#         cable_iy = np.pi * radius**4 / 4.0
#         cable_iz = cable_iy

#         # Base stiffnesses used for this cable (match creation defaults).
#         cable_youngs_base = 10000000.0 * 2
#         cable_shear_base = 60000.0 * 2
#         # Per-env ranges (scale as needed).
#         cable_youngs_range = (
#             0.8 * cable_youngs_base,
#             1.2 * cable_youngs_base,
#         )
#         cable_shear_range = (
#             0.8 * cable_shear_base,
#             1.2 * cable_shear_base,
#         )

#     @jax.vmap
#     def rand(rng):

#         if cable_joint_ids.size:
#             # Sample Young's (bend) and shear (twist) moduli per environment.
#             rng, key = jax.random.split(rng)
#             youngs = jax.random.uniform(
#                 key,
#                 shape=(),
#                 minval=cable_youngs_range[0],
#                 maxval=cable_youngs_range[1],
#             )
#             rng, key = jax.random.split(rng)
#             shear = jax.random.uniform(
#                 key,
#                 shape=(),
#                 minval=cable_shear_range[0],
#                 maxval=cable_shear_range[1],
#             )

#             k_twist = (cable_j * shear) / length
#             k_bend = ((cable_iy + cable_iz) * youngs) / length
#             k_ball = (k_twist + k_bend) / 3.0
#             cable_k = jp.full((cable_joint_ids.shape[0],), k_ball)
#             jnt_stiffness = model.jnt_stiffness.at[cable_joint_ids].set(cable_k)
#         else:
#             jnt_stiffness = model.jnt_stiffness

#         return (
#             geom_friction,
#             body_mass,
#             body_inertia,
#             body_ipos,
#             qpos0,
#             dof_frictionloss,
#             dof_armature,
#             dof_damping,
#             jnt_stiffness,
#             actuator_gainprm,
#             actuator_biasprm,
#         )

#     (
#         geom_friction,
#         body_mass,
#         body_inertia,
#         body_ipos,
#         qpos0,
#         dof_frictionloss,
#         dof_armature,
#         dof_damping,
#         jnt_stiffness,
#         actuator_gainprm,
#         actuator_biasprm,
#     ) = rand(rng)

#     in_axes = jax.tree_util.tree_map(lambda x: None, model)
#     in_axes = in_axes.tree_replace(
#         {
#             "geom_friction": 0,
#             "body_mass": 0,
#             "body_inertia": 0,
#             "body_ipos": 0,
#             "qpos0": 0,
#             "dof_frictionloss": 0,
#             "dof_armature": 0,
#             "dof_damping": 0,
#             "jnt_stiffness": 0,
#             "actuator_gainprm": 0,
#             "actuator_biasprm": 0,
#         }
#     )

#     model = model.tree_replace(
#         {
#             "geom_friction": geom_friction,
#             "body_mass": body_mass,
#             "body_inertia": body_inertia,
#             "body_ipos": body_ipos,
#             "qpos0": qpos0,
#             "dof_frictionloss": dof_frictionloss,
#             "dof_armature": dof_armature,
#             "dof_damping": dof_damping,
#             "jnt_stiffness": jnt_stiffness,
#             "actuator_gainprm": actuator_gainprm,
#             "actuator_biasprm": actuator_biasprm,
#         }
#     )

#     return model, in_axes


def domain_randomize(
    model: mjx.Model, mj_model: mj.MjModel, rng: jax.Array
) -> tuple[mjx.Model, Any]:
    mj_model = mj_model

    # Cable joint ids (exclude the free joint) and section properties.
    cable_joint_ids = np.array(
        [
            j
            for j in range(mj_model.njnt)
            if mj_model.joint(j).name.startswith("cable:J")
        ],
        dtype=np.int32,
    )
    cable_geom_id = None
    for geom_name in ("cable:G1", "cable:G2", "cable:Gfirst", "cable:Glast"):
        try:
            cable_geom_id = mj_model.geom(geom_name).id
            break
        except KeyError:
            continue
    if cable_joint_ids.size and cable_geom_id is None:
        raise ValueError("Cable joints found but no cable geom to infer stiffness.")

    if cable_joint_ids.size:
        geom_size = np.asarray(mj_model.geom_size[cable_geom_id], dtype=np.float64)
        radius = float(geom_size[0])
        half_segment_length = float(geom_size[1])
        segment_length = 2 * half_segment_length

        # Section properties for capsule/cylinder geometry.
        cable_j = np.pi * radius**4 / 2.0
        cable_iy = np.pi * radius**4 / 4.0
        cable_iz = cable_iy

        # Base stiffnesses used for this cable (match creation defaults).
        cable_youngs_base = 10000000.0 * 2
        cable_shear_base = 60000.0 * 2
        # Per-env ranges (scale as needed).
        cable_youngs_range = (
            0.8 * cable_youngs_base,
            1.2 * cable_youngs_base,
        )
        cable_shear_range = (
            0.8 * cable_shear_base,
            1.2 * cable_shear_base,
        )

    geom_friction = model.geom_friction
    body_mass = model.body_mass
    body_inertia = model.body_inertia
    body_ipos = model.body_ipos
    qpos0 = model.qpos0
    dof_frictionloss = model.dof_frictionloss
    dof_armature = model.dof_armature
    dof_damping = model.dof_damping
    actuator_gainprm = model.actuator_gainprm
    actuator_biasprm = model.actuator_biasprm

    def rand(rng):

        if cable_joint_ids.size:
            # Sample Young's (bend) and shear (twist) moduli per environment.
            print(f"{rng=}")
            rng, key = jax.random.split(rng)
            youngs = jax.random.uniform(
                key,
                shape=(),
                minval=cable_youngs_range[0],
                maxval=cable_youngs_range[1],
            )
            rng, key = jax.random.split(rng)
            shear = jax.random.uniform(
                key,
                shape=(),
                minval=cable_shear_range[0],
                maxval=cable_shear_range[1],
            )

            length = max(segment_length, 1e-9)

            k_twist = (cable_j * shear) / length
            k_bend = ((cable_iy + cable_iz) * youngs) / length
            k_ball = (k_twist + k_bend) / 3.0
            cable_k = jp.full((cable_joint_ids.shape[0],), k_ball)
            jnt_stiffness = model.jnt_stiffness.at[cable_joint_ids].set(cable_k)
        else:
            jnt_stiffness = model.jnt_stiffness

        return (
            geom_friction,
            body_mass,
            body_inertia,
            body_ipos,
            qpos0,
            dof_frictionloss,
            dof_armature,
            dof_damping,
            jnt_stiffness,
            actuator_gainprm,
            actuator_biasprm,
        )

    batched = rng.ndim > 1
    if batched:
        outputs = jax.vmap(rand)(rng)
    else:
        outputs = rand(rng)

    (
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        jnt_stiffness,
        actuator_gainprm,
        actuator_biasprm,
    ) = outputs

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    if batched:
        in_axes = in_axes.tree_replace(
            {
                "geom_friction": 0,
                "body_mass": 0,
                "body_inertia": 0,
                "body_ipos": 0,
                "qpos0": 0,
                "dof_frictionloss": 0,
                "dof_armature": 0,
                "dof_damping": 0,
                "jnt_stiffness": 0,
                "actuator_gainprm": 0,
                "actuator_biasprm": 0,
            }
        )

    model = model.tree_replace(
        {
            "geom_friction": geom_friction,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "body_ipos": body_ipos,
            "qpos0": qpos0,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "dof_damping": dof_damping,
            "jnt_stiffness": jnt_stiffness,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )

    return model, in_axes


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
    # segment_size: float = 0.002,
    mass: float = 0.00035,
    rgba: Union[str, list[float]] = "0.2 0.2 0.2 1",
    initial: str = "free",
) -> mj.MjSpec:
    del model_name, curve, vmax
    base_pos = [0.0, 0.0, 0.0]
    damping = 1e-2
    armature = 0.001
    friction = [0.3, 0.3, 0.3]
    solref = [0.00001, 3.0]
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
    solref: list[float] = [0.0001, 3],
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


def _pipe(
    inner_radius: float = 0.1,
    outer_radius: float = 0.12,
    length: float = 0.1,
    resolution: int = 40,
    friction: list[float] = [0.2, 0.2, 0.2],
    rgba: list[float] = [0.2, 0.2, 0.2, 1],
    solref: list[float] = [0.000001, 1],
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

    friction = [float(f) for f in friction]
    rgba = [float(c) for c in rgba]
    solref = [float(s) for s in solref]

    angle_rad = 2.0 * np.pi / resolution
    wall_thickness = outer_radius - inner_radius
    radial_half = wall_thickness / 2.0
    radius_mid = inner_radius + radial_half
    tangential_half = outer_radius * np.sin(angle_rad / 2.0)
    half_length = length

    spec = mj.MjSpec()
    # Match the rest of the scene (and the XML-based pipe): use radians for euler.
    spec.compiler.angle = "radian"
    body = spec.worldbody.add_body(pos=[0.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0])
    geom_size = [radial_half, tangential_half, half_length]

    for i in range(resolution):
        angle = i * angle_rad
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        pos = [radius_mid * cos_a, radius_mid * sin_a, 0.0]
        body.add_geom(
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=pos,
            euler=[0.0, 0.0, angle],
            size=geom_size,
            friction=friction,
            rgba=rgba,
            solref=solref,
        )

    body.add_site(
        name="pipe_entry",
        pos=[0.0, 0.0, -half_length],
        size=[0.002, 0.002, 0.002],
        rgba=[0.0, 0.0, 0.0, 0.0],
        group=2,
    )
    body.add_site(
        name="pipe_exit",
        pos=[0.0, 0.0, half_length],
        size=[0.002, 0.002, 0.002],
        rgba=[1.0, 1.0, 1.0, 0.0],
        group=2,
    )

    return spec


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

    # <custom>
    #     <numeric data="40" name="max_contact_points" />
    #     <numeric data="40" name="max_geom_pairs" />
    # </custom>

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
            ls_iterations="4"
        >
                <flag eulerdamp="enable" />
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
            <!-- <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" /> -->

        </worldbody>

    </mujoco>
    """
    # _XML = """
    #     <mujoco model="empty scene">

    #     <compiler angle="radian" autolimits="true" />
    #     <option timestep="0.002"
    #         integrator="implicitfast"
    #         solver="Newton"
    #         gravity="0 0 -9.82"
    #         cone="elliptic"
    #         sdf_iterations="5"
    #         sdf_initpoints="30"
    #         noslip_iterations="2"
    #         ls_iterations="10"
    #     >
    #         <!-- impratio="100" -->
    #         <!-- mjMAXCONPAIR="10" -->
    #             <flag eulerdamp="disable" />
    #         <!-- <flag nativeccd="enable" /> -->
    #     </option>

    #     <custom>
    #         <numeric data="15" name="max_contact_points" />
    #         <numeric data="15" name="max_geom_pairs" />
    #     </custom>

    #     <extension>
    #         <plugin plugin="mujoco.sensor.touch_grid" />
    #         <!-- <plugin plugin="mujoco.elasticity.solid" /> -->
    #         <!-- <plugin plugin="mujoco.elasticity.shell" /> -->
    #     </extension>

    #     <statistic center="0.3 0 0.3" extent="0.8" meansize="0.08" />

    #     <visual>
    #         <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
    #         <rgba haze="0.15 0.25 0.35 1" />
    #         <global azimuth="120" elevation="-20" offwidth="2000" offheight="2000" />
    #         <!-- <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080" /> -->

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
    #         <!-- <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" /> -->

    #     </worldbody>

    # </mujoco>
    # """
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
        impl: str = "warp",
    ):
        super().__init__(config, config_overrides=config_overrides)
        if impl not in {"jax", "warp", "c"}:
            raise ValueError(f"Invalid MJX impl '{impl}' (expected: jax, warp, c)")
        self._impl = impl
        if self._impl == "warp":
            _ensure_warp_internal_module()

        self._slide_limits = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (-0.3, 0.3),
        }
        self._random_limits = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1),
        }

        self.keys = {
            "qpos": "testing/experiments/pipe_insert/constants/v3/qpos.npy",
            "qvel": "testing/experiments/pipe_insert/constants/v3/qvel.npy",
            "ctrl": "testing/experiments/pipe_insert/constants/v3/ctrl.npy",
        }

        self.i = 0

        self._mj_model = self._init()
        self.ctrl_scale: float = 1.0
        # self._mj_model = self._init()
        # self._mj_model.opt.timestep = self.sim_dt
        self._max_ncon_sim = 50
        self._num_env = 900
        self._max_ncon_total = self._max_ncon_sim * self._num_env
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._impl)
        # data = mjx.make_data(self._mj_model, impl='warp', nconmax=self._max_ncon_total, njmax=self._max_ncon_sim)
        # self._mjx_model: mjx.Model = mjx.put_model(self._mj_model)

        # self._termination_threshold = 0.05
        self._termination_threshold = 0.04

        self._sparse_reward = config["sparse_reward"]

        self._post_init()

    def _init(self) -> mj.MjModel:
        # root
        _ROOT = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        # keyframe_path = _ROOT / "envs/tmp/keyframe.xml"
        # if keyframe_path.exists():
        #     key_kwargs = _load_keyframe_xml(keyframe_path, key_name="bent")
        #     scene.add_key(**key_kwargs)

        self.pipe_inner_radius = 0.0385 / 2
        self.pipe_outer_radius = 0.0435 / 2
        self.pipe_length = 0.121 / 2  # 121 mm from schematic
        # pipe_length = 0.09

        # pip = _pipe(
        #     inner_radius=self.pipe_inner_radius,
        #     outer_radius=self.pipe_outer_radius,
        #     length=self.pipe_length,
        #     rgba=[0.2, 0.2, 0.2, 0.2],
        # )
        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            resolution=10,
            friction=[0.1, 0.1, 0.1],
        )

        # mesh_path = _ROOT / "envs" / "assets/DS00240017_E.obj"

        # mesh_spec = add_mesh(
        #     mesh_path=mesh_path,
        #     scale=0.001,
        #     euler=[0, 1.57, 0],
        #     # euler=[0, 1.57, 0],
        #     rgba=[0.2, 0.2, 0.2, 0],
        #     contype=0,
        #     conaffinity=0,
        # )
        # mesh_body = mesh_spec.worldbody.first_body()
        # scene.worldbody.add_frame(
        #     pos=[0, -self.pipe_length - 0.01, 0.1], euler=[0, 1.57, 0]
        # ).attach_body(mesh_body)

        keypoint_height = self.pipe_inner_radius - (
            self.pipe_outer_radius - self.pipe_inner_radius
        )

        # pip.worldbody.first_body().add_site(
        #     name="target_1", pos=[0, keypoint_height, -0.05], group=1, rgba=[1, 0, 0, 1]
        # )
        # pip.worldbody.first_body().add_site(
        #     name="target_2",
        #     pos=[0, keypoint_height, -0.025],
        #     group=1,
        #     rgba=[1, 1, 0, 1],
        # )
        # pip.worldbody.first_body().add_site(
        #     name="target_3", pos=[0, keypoint_height, 0], group=1, rgba=[0, 1, 0, 1]
        # )
        pip.worldbody.first_body().add_site(
            name="target_1", pos=[0, keypoint_height, -0.05], group=1, rgba=[1, 0, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_2", pos=[0, keypoint_height, 0], group=1, rgba=[1, 1, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_3", pos=[0, keypoint_height, 0.05], group=1, rgba=[0, 1, 0, 1]
        )

        twist = 60000.0 * 2
        bend = 10000000.0 * 2

        cable = mjx_cable(count=40, twist=twist, bend=bend)

        # cable.body("cable:Bfirst").add_site(
        #     name="keypoint_3", group=1, rgba=[0, 1, 0, 1], pos=[0, -0.01, 0]
        # )
        # cable.body("cable:B2").add_site(
        #     name="keypoint_2", group=1, rgba=[1, 1, 0, 1], pos=[0, -0.03, 0]
        # )
        # cable.body("cable:B4").add_site(
        #     name="keypoint_1", group=1, rgba=[1, 0, 0, 1], pos=[0, -0.05, 0]
        # )
        cable.body("cable:Bfirst").add_site(
            name="keypoint_3", group=1, rgba=[0, 1, 0, 1], pos=[0, -0.01, 0]
        )
        cable.body("cable:B2").add_site(
            name="keypoint_2", group=1, rgba=[1, 1, 0, 1], pos=[0, -0.01, 0]
        )
        cable.body("cable:B4").add_site(
            name="keypoint_1", group=1, rgba=[1, 0, 0, 1], pos=[0, -0.01, 0]
        )

        _c = scene.worldbody.add_camera(
            name="cam",
            pos=[1.2, 0.234, 0.156],
            # pos=[0.721, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
            resolution=[640, 480],
        )

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

        # scene.compile()
        # scene.to_file("tmp/pipe_inser_2.xml")

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _post_init(self) -> None:
        self.target_ids = [self._mj_model.site(f"target_{i + 1}").id for i in range(3)]
        self.keypoint_ids = [
            self._mj_model.site(f"keypoint_{i + 1}").id for i in range(3)
        ]
        self.pipe_entry_id = self._mj_model.site("pipe_entry").id
        self.pipe_exit_id = self._mj_model.site("pipe_exit").id
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
        # if self._key_id is not None:
        #     qpos = jp.array(self._mj_model.key_qpos[self._key_id])
        #     qvel = jp.array(self._mj_model.key_qvel[self._key_id])
        #     ctrl = jp.array(self._mj_model.key_ctrl[self._key_id])
        #     if self._mj_model.na:
        #         act = jp.array(self._mj_model.key_act[self._key_id])

        # Match the manual initialization from __main__.
        delta = jax.random.uniform(rng_delta, shape=(3,), minval=-0.3, maxval=0.3)
        delta = jp.zeros(3)
        qpos = qpos.at[:3].set(self.qpos0_gripper + delta)
        qpos = qpos.at[3:6].set(self.qpos0_cable + delta)
        ctrl = ctrl.at[:3].set(self.ctrl0 + delta)

        limits = self._random_limits
        low = jp.array([limits["x"][0], limits["y"][0], limits["z"][0]])
        high = jp.array([limits["x"][1], limits["y"][1], limits["z"][1]])

        qpos = qpos.at[:3].set(jp.clip(qpos[:3], low, high))
        ctrl = ctrl.at[:3].set(jp.clip(ctrl[:3], low, high))

        # Zero velocities if no keyframe is present
        # if self._key_id is None:
        #     qvel = jp.zeros(self.mj_model.nv)
        qacc = jp.zeros(self.mj_model.nv)
        qfrc_applied = jp.zeros(self.mj_model.nv)
        xfrc_applied = jp.zeros((self.mj_model.nbody, 6))

        # Initialize MuJoCo-X data
        data = mjx.make_data(
            self.mj_model,
            impl=self._impl,
            nconmax=self._max_ncon_total,
            njmax=self._max_ncon_sim,
        )
        # data = mjx.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl, act=act)
        # data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl, act=act)

        data = data.replace(qpos=qpos)
        data = data.replace(ctrl=ctrl)
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

        a = state.data.ctrl + (action * self.ctrl_scale)

        # a = state.data.ctrl + self.ctrl_scale * action
        data = mjx_env.step(self.mjx_model, state.data, a, self.n_substeps)

        # Reward: negative distance to target + bonus for being close
        dist_to_target = self._get_distance(data)

        done = self._get_done(data)

        if self._sparse_reward:
            reward = jp.float64(done)
        else:
            reward = -dist_to_target

        obs = self._get_obs(data, state.info)

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
        # done = nan_condition | success

        # done = done.astype(float)

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

    def _points_inside_cylinder(
        self,
        points: jax.Array,
        axis_start: jax.Array,
        axis_end: jax.Array,
        radius: float,
        *,
        inclusive: bool = True,
    ) -> jax.Array:
        """JAX-friendly mask for points inside a finite cylinder."""
        pts = jp.atleast_2d(jp.asarray(points))
        a = jp.asarray(axis_start)
        b = jp.asarray(axis_end)
        v = b - a
        v2 = jp.dot(v, v)
        valid_axis = v2 > 1e-12
        v2_safe = jp.where(valid_axis, v2, 1.0)

        w = pts - a
        t = (w @ v) / v2_safe
        if inclusive:
            within_caps = (t >= 0.0) & (t <= 1.0)
        else:
            within_caps = (t > 0.0) & (t < 1.0)

        cross = jp.cross(w, v)
        dist2 = jp.sum(cross * cross, axis=1)
        if inclusive:
            within_radius = dist2 <= (radius * radius) * v2_safe
        else:
            within_radius = dist2 < (radius * radius) * v2_safe

        return valid_axis & within_caps & within_radius

    def _keypoints_within_pipe(
        self,
        keypoints: jax.Array,
        pipe_axis_start: jax.Array,
        pipe_axis_end: jax.Array,
        inner_radius: float,
        *,
        inclusive: bool = True,
    ) -> jax.Array:
        """Return True if all keypoints lie within the pipe's inner cylinder."""
        inside = self._points_inside_cylinder(
            keypoints,
            pipe_axis_start,
            pipe_axis_end,
            inner_radius,
            inclusive=inclusive,
        )
        return jp.all(inside)

    def _get_done(self, data: mjx.Data) -> float:

        # 1) get the positions of the keypoints
        # 2) get the points at the entry and exit of the pipe
        #
        T_w_pipe_entry = get_pose(
            self._mjx_model, data, self.pipe_entry_id, obj_type=ObjType.SITE
        )
        T_w_pipe_exit = get_pose(
            self._mjx_model, data, self.pipe_exit_id, obj_type=ObjType.SITE
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

        keypoints = jp.stack(
            [
                T_w_keypoint_1.translation(),
                T_w_keypoint_2.translation(),
                T_w_keypoint_3.translation(),
            ],
            axis=0,
        )

        success = self._keypoints_within_pipe(
            keypoints=keypoints,
            pipe_axis_start=T_w_pipe_entry.translation(),
            pipe_axis_end=T_w_pipe_exit.translation(),
            inner_radius=self.pipe_inner_radius,
        )

        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = nan_condition | success
        return done.astype(float)

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

        r1 = self._from_to(data, self.keypoint_ids[0], self.target_ids[0])
        r2 = self._from_to(data, self.keypoint_ids[1], self.target_ids[1])
        r3 = self._from_to(data, self.keypoint_ids[2], self.target_ids[2])

        o = jp.concatenate([r1, r2, r3]).flatten()

        return o

    @property
    def observation_size(self) -> int:
        """Compute observation size by calling _get_obs with dummy data"""
        # Create dummy data for size computation
        dummy_data = mjx.make_data(
            self.mj_model,
            impl=self._impl,
            nconmax=self._max_ncon_total,
            njmax=self._max_ncon_sim,
        )
        dummy_info = {}

        # Get observation and check its shape
        obs = self._get_obs(dummy_data, dummy_info)
        return obs.shape[0]  # Get the last dimension (feature size)

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


if __name__ == "__main__":
    env = PipeInsert2()
    model = env.mj_model
    data = mj.MjData(model)

    joint_ids = [model.joint(name).id for name in ["x", "y", "z"]]
    ctrl_ids = [model.actuator(name).id for name in ["x", "y", "z"]]

    def _load_base_state() -> None:
        data.qpos[:] = np.load(env.keys["qpos"])
        data.qvel[:] = np.load(env.keys["qvel"])
        data.ctrl[:] = np.load(env.keys["ctrl"])

    def _apply_reset_state(rng: jax.Array) -> None:
        state = env.reset(rng)
        data.qpos[:] = np.asarray(state.data.qpos)
        data.qvel[:] = np.asarray(state.data.qvel)
        data.ctrl[:] = np.asarray(state.data.ctrl)
        if model.na:
            data.act[:] = np.asarray(state.data.act)
        mj.mj_forward(model, data)

    def cb(key: int) -> None:
        if key is glfw.KEY_SPACE:
            qpos = data.qpos
            qvel = data.qvel
            ctrl = data.ctrl

            np.save("testing/experiments/pipe_insert/constants/v3/qpos.npy", qpos)
            np.save("testing/experiments/pipe_insert/constants/v3/qvel.npy", qvel)
            np.save("testing/experiments/pipe_insert/constants/v3/ctrl.npy", ctrl)
            print("saved")
        if key is glfw.KEY_PERIOD:
            qpos = np.load("testing/experiments/pipe_insert/constants/v3/qpos.npy")
            qvel = np.load("testing/experiments/pipe_insert/constants/v3/qvel.npy")
            ctrl = np.load("testing/experiments/pipe_insert/constants/v3/ctrl.npy")
            data.qpos = qpos
            data.qvel = qvel
            data.ctrl = ctrl
            print("loaded")

    # delta = np.random.uniform(-0.3, 0.3, 3)
    _load_base_state()
    _apply_reset_state(jax.random.PRNGKey(0))
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

            # pipe_entry, pipe_exit = get_pipe_end_points(data)

            # success = keypoints_within_pipe(
            #     keypoints=get_keypoints(data),
            #     pipe_axis_start=pipe_entry,
            #     pipe_axis_end=pipe_exit,
            #     inner_radius=env.pipe_inner_radius,
            # )

            # step simulation one time step
            mj.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
