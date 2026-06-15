import functools
import json
import math
import os
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
import tqdm
import tyro
from brax.envs.wrappers import training as brax_training
from brax.training import acting
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco import glfw, mjx
from mujoco_playground._src import mjx_env, wrapper

from testing.envs.pipe_insert_2 import parse_obj
from utils.mjx import ObjType, get_pose, is_stable


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

    print(f"{type(in_axes)=}")

    return model, in_axes


def _metric_to_float(value) -> float:
    if value is None:
        return math.nan
    try:
        return float(np.asarray(value))
    except (TypeError, ValueError):
        return math.nan


def _plot_metrics(history: dict[str, list[float]], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = history.get("step", [])
    if not steps:
        return

    plot_specs = [
        ("eval/episode_reward", "Eval episode reward"),
        ("training/total_loss", "Training total loss"),
        ("training/sps", "Training steps/sec"),
    ]

    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(7, 8), sharex=True)
    if len(plot_specs) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, plot_specs):
        values = history.get(key)
        if not values:
            continue
        ax.plot(steps, values, marker="o", linewidth=1.2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _quat_normalize(q: jax.Array, eps: float = 1e-8) -> jax.Array:
    norm = jp.linalg.norm(q)
    return jp.where(norm > eps, q / norm, jp.array([1.0, 0.0, 0.0, 0.0]))


def _quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _rotvec_to_quat(v: jax.Array, eps: float = 1e-6) -> jax.Array:
    angle = jp.linalg.norm(v)
    half = 0.5 * angle
    axis = v / jp.where(angle > eps, angle, 1.0)
    sin_half = jp.sin(half)
    quat = jp.concatenate([jp.cos(half)[None], axis * sin_half], axis=0)
    quat_small = jp.concatenate([jp.ones(1), 0.5 * v], axis=0)
    return _quat_normalize(jp.where(angle > eps, quat, quat_small))


def _save_frames(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    if not frames:
        return
    array = np.stack(frames, axis=0)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, array, fps=fps)
        return
    except Exception:
        pass

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not path.endswith(".png"):
            path = f"{path}.png"
        plt.imsave(path, array[0])
        return
    except Exception:
        pass

    if not path.endswith(".npy"):
        path = f"{path}.npy"
    np.save(path, array)


def _format_eval_path(base_path: str, eval_index: int, step: int) -> str:
    if not base_path:
        return ""
    if "{" in base_path and "}" in base_path:
        try:
            return base_path.format(step=step, eval=eval_index)
        except (KeyError, ValueError):
            pass
    stem, ext = os.path.splitext(base_path)
    suffix = f"_eval{eval_index:03d}_step{step}"
    return f"{stem}{suffix}{ext}"


def _format_final_path(base_path: str) -> str:
    if not base_path:
        return ""
    if "{" in base_path and "}" in base_path:
        try:
            return base_path.format(step="final", eval="final")
        except (KeyError, ValueError):
            pass
    stem, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".gif"
    return f"{stem}_final{ext}"


def _rollout_trajectory(
    env: mjx_env.MjxEnv,
    steps: int,
    seed: int,
    policy=None,
    use_jit: bool = True,
) -> mjx_env.State:
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    def step_fn(carry, _):
        state, key = carry
        key, step_key = jax.random.split(key)
        if policy is None:
            action = jp.zeros((env.action_size,), dtype=jp.float32)
        else:
            action, _ = policy(state.obs, step_key)
        state = env.step(state, action)
        return (state, key), state

    def run_scan(init_state, init_key):
        return jax.lax.scan(step_fn, (init_state, init_key), None, length=steps)

    if use_jit:
        run_scan = jax.jit(run_scan)

    (_final_state, _), traj = run_scan(state, rng)

    # Prepend initial state to match rollout length.
    traj = jax.tree_util.tree_map(
        lambda x0, xs: jp.concatenate([x0[None, ...], xs], axis=0), state, traj
    )
    return traj


def _trajectory_to_list(traj: mjx_env.State) -> list[mjx_env.State]:
    leaves = jax.tree_util.tree_leaves(traj)
    length = int(leaves[0].shape[0]) if leaves else 0
    states: list[mjx_env.State] = []
    for i in range(length):
        states.append(jax.tree_util.tree_map(lambda x: x[i], traj))
    return states


def render_rollout(
    env: mjx_env.MjxEnv,
    steps: int,
    path: str,
    seed: int = 0,
    fps: int = 30,
    policy=None,
    use_jit: bool = True,
    frame_skip: int = 1,
    height: int = 480,
    width: int = 640,
) -> None:
    frame_skip = max(int(frame_skip), 1)
    traj = _rollout_trajectory(env, steps, seed, policy=policy, use_jit=use_jit)
    if frame_skip > 1:
        traj = jax.tree_util.tree_map(lambda x: x[::frame_skip], traj)
    traj = jax.device_get(traj)
    trajectory = _trajectory_to_list(traj)
    frames = env.render(trajectory, height=height, width=width)
    _save_frames(frames, path, fps=fps)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.002,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        sparse_reward=False,
        # grace_steps=5,
        # ctrl_dt=0.02,
        # sim_dt=0.002,
        # episode_length=300,
        # action_repeat=1,
        success_threshold_pos=0.01,
        impl="warp",
        centi=True,
    )


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
    solref = [0.000001, 1.0]
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
            <body name="pipe" euler="0 0 0" pos="0 0 0">
                <replicate sep=":" count="{resolution}" euler="0 0 {angle_deg}">
                    <geom name="pipe"
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
    # _XML = f"""
    # <mujoco>
    #     <worldbody>
    #         <body euler="0 0 0" pos="0 0 0">
    #             <replicate sep="hole:" count="40" euler="0 0 20">
    #                 <geom type="box" solref="0.0000000001 1" pos="0 -0.03 0" size=".008 .001 {length / 2}" friction="0.2 0.2 0.2" />
    #             </replicate>
    #         </body>
    #     </worldbody>
    # </mujoco>
    # """
    # <replicate sep="hole:" count="30" euler="0 0 20">
    # <geom type="box" solref="0.0000000001 1" pos="0 -0.018 0" size=".004 .001 {length / 2}" friction="0.2 0.2 0.2" />
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


def _ensure_warp_internal_module() -> None:
    """Ensure warp._src is available on the warp package."""
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
        return


class PipeInsert2(mjx_env.MjxEnv):
    """Simple 3D position control environment."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)
        if config.impl not in {"jax", "warp", "c"}:
            raise ValueError(
                f"Invalid MJX impl '{config.impl}' (expected: jax, warp, c)"
            )
        if config.impl == "warp":
            _ensure_warp_internal_module()
        self._slide_limits = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (-0.3, 0.3),
        }

        self.impl = config.impl

        self.keys = {
            "qpos": "constants/v4/qpos.npy",
            "qvel": "constants/v4/qvel.npy",
            "ctrl": "constants/v4/ctrl.npy",
        }

        self._mj_model = self._init_mocap_weld_4()
        self.ctrl_scale: float = 1.0
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model, impl=self.impl)

        self._centi = config.centi
        self._termination_threshold = 0.01 if not self._centi else 1

        self._sparse_reward = config["sparse_reward"]
        self._episode_length = config["episode_length"]

        self._post_init()

    def _init_mocap_weld_4(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = empty()

        self.pipe_inner_radius = 0.0385 / 2
        self.pipe_outer_radius = 0.0435 / 2
        self.pipe_length = 0.125 / 2  # 121 mm from schematic

        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            resolution=20,
        )

        # keypoint_height = self.pipe_inner_radius / 2
        keypoint_height = self.pipe_inner_radius - (
            self.pipe_outer_radius - self.pipe_inner_radius
        )

        pip.worldbody.first_body().add_site(
            name="target",
            pos=[0, keypoint_height, -0.065],
            group=1,
            rgba=[1, 0, 0, 1],
            # name="target", pos=[0, keypoint_height, -0.06], group=1, rgba=[1, 0, 0, 1]
        )

        twist = 60000.0 * 5
        bend = 10000000.0 * 5

        cable = mjx_cable(twist=twist, bend=bend, size=0.4, initial="free")

        cable.body("cable:Bfirst").add_site(
            name="keypoint", group=1, rgba=[1, 0, 0, 1], pos=[0, 0, 0]
        )

        _c = scene.worldbody.add_camera(
            name="cam",
            pos=[1.2, 0.234, 0.156],
            # pos=[0.721, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
            resolution=[640, 480],
        )

        gripper_spawn = [0.0, 0.4, 0.4]

        mocap = scene.worldbody.add_body(
            name="mocap", mocap=True, pos=gripper_spawn, euler=[0, 0, 0]
        )
        mocap.add_geom(
            name="mocap",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
            # rgba=[1, 1, 1, 0.2],
        )
        s_mocap_1 = mocap.add_site(name="mocap_site_1", pos=[0, -0.01, 0])
        s_mocap_2 = mocap.add_site(name="mocap_site_2", pos=[0, 0.01, 0])
        s_mocap_3 = mocap.add_site(name="mocap_site_3", pos=[0.01, 0, 0])
        s_mocap_4 = mocap.add_site(name="mocap_site_4", pos=[-0.01, 0, 0])

        scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 3.14]).attach_body(
            # scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 0]).attach_body(
            pip.worldbody.first_body(),
        )
        cable_root = cable.worldbody.first_body()
        scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
            cable_root
        )

        b = None
        b0 = cable_root
        for i in range(10):
            b = b0.first_body()
            b0 = b

        b.add_site(
            name="cable_weld_site_1",
            pos=[0.0, -0.01, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )
        b.add_site(
            name="cable_weld_site_2",
            pos=[0.0, 0.01, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )
        b.add_site(
            name="cable_weld_site_3",
            pos=[0.01, 0, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )
        b.add_site(
            name="cable_weld_site_4",
            pos=[-0.01, 0, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )

        solref = [0.00000002, 1.0]
        solimp = [0.99, 0.9999, 1e-4, 0.0, 10]

        scene.add_equality(
            name="mocap_cable_weld_1",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_1",
            name2="cable_weld_site_1",
            # Keep the current relative pose at creation time.
            solref=solref,
            solimp=solimp,
        )
        scene.add_equality(
            name="mocap_cable_weld_2",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_2",
            name2="cable_weld_site_2",
            # Keep the current relative pose at creation time.
            solref=solref,
            solimp=solimp,
        )
        scene.add_equality(
            name="mocap_cable_weld_3",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_3",
            name2="cable_weld_site_3",
            # Keep the current relative pose at creation time.
            solref=solref,
            solimp=solimp,
        )
        scene.add_equality(
            name="mocap_cable_weld_4",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_4",
            name2="cable_weld_site_4",
            # Keep the current relative pose at creation time.
            solref=solref,
            solimp=solimp,
        )

        scene.add_key(
            name="init",
            time=0,
            qpos=_parse_float_list(
                "-0.00947511 0.180653 -0.0887441 0.851438 0.524092 -0.0102263 -0.0166138 0.999999 -0.00146951 -4.03258e-12 -1.68616e-09 0.999983 -0.00590368 -8.03079e-11 -6.97733e-09 0.999909 -0.0134648 -5.6518e-10 -1.6602e-08 0.999698 -0.02459 -2.51178e-09 -3.21214e-08 0.999194 -0.0401316 -8.61218e-09 -5.60426e-08 0.998108 -0.0614894 -2.48804e-08 -9.04947e-08 0.995881 -0.0906679 -6.23136e-08 -1.30835e-07 0.99151 -0.130033 -1.32017e-07 -1.46397e-07 0.983451 -0.181172 -2.12483e-07 -4.69784e-08"
            ),
            qvel=_parse_float_list(
                "-6.55311e-08 3.12602e-07 4.81199e-07 8.20477e-06 -8.22193e-07 -3.26604e-07 -1.92035e-07 1.12256e-13 3.03087e-09 -7.59123e-07 2.20739e-11 1.25879e-08 -1.67801e-06 3.05132e-10 2.99011e-08 -2.88816e-06 1.87116e-09 5.73999e-08 -4.21045e-06 7.7683e-09 9.88912e-08 -5.14747e-06 2.55825e-08 1.57635e-07 -4.51174e-06 7.129e-08 2.26798e-07 -1.61511e-08 1.7075e-07 2.59132e-07 1.12037e-05 3.46293e-07 1.05207e-07"
            ),
            mpos=_parse_float_list("-0.000600281 0.407976 0.168044"),
            mquat=_parse_float_list("1 0 0 0"),
        )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _from_to(self, data: mjx.Data, _from: int, _to: int) -> jax.Array:
        T_w_from = get_pose(self._mjx_model, data, _from, obj_type=ObjType.SITE)
        T_w_to = get_pose(self._mjx_model, data, _to, obj_type=ObjType.SITE)
        return T_w_to.translation() - T_w_from.translation()

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        d = self._from_to(data, self.keypoint_id, self.target_id)
        return d

    def _get_reward(self, data: mjx.Data) -> float:
        d = -jp.linalg.norm(self._get_obs(data))
        return d if not self._centi else d * 100

    def _get_done(self, data: mjx.Data, info: dict) -> float:
        step_count = info["steps"] if "steps" in info else info["step"]

        is_unstable = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        timeout = step_count >= self._episode_length

        # negative and either in m or cm
        r = self._get_reward(data)

        success = jp.abs(r) < self._termination_threshold

        done = is_unstable | timeout | success
        return done.astype(float)

    def _post_init(self) -> None:
        self.target_id = self._mj_model.site("target").id
        self.keypoint_id = self._mj_model.site("keypoint").id

        self._mocap_id = None
        try:
            self._mocap_id = int(self._mj_model.body("mocap").mocapid)
            if self._mocap_id < 0:
                self._mocap_id = None
        except Exception:
            self._mocap_id = None

        self._key_id: Optional[int] = None
        for key_name in ("bent", "init"):
            try:
                self._key_id = self._mj_model.key(key_name).id
                break
            except Exception:
                continue
        if self._key_id is None and getattr(self._mj_model, "nkey", 0) > 0:
            self._key_id = 0

        if self._key_id is not None:
            self._qpos0 = jp.array(self._mj_model.key_qpos[self._key_id])
            self._qvel0 = jp.array(self._mj_model.key_qvel[self._key_id])
            self._ctrl0 = jp.array(self._mj_model.key_ctrl[self._key_id])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Split RNG (advance per reset even if we don't add noise).
        rng, rng_delta = jax.random.split(rng)

        qpos = self._qpos0
        qvel = self._qvel0
        ctrl = self._ctrl0
        act = jp.zeros(self.mj_model.na)
        mocap_pos = None
        mocap_quat = None
        if self._key_id is not None:
            qpos = jp.array(self._mj_model.key_qpos[self._key_id])
            qvel = jp.array(self._mj_model.key_qvel[self._key_id])
            ctrl = jp.array(self._mj_model.key_ctrl[self._key_id])
            if self._mj_model.na:
                act = jp.array(self._mj_model.key_act[self._key_id])
            if self._mj_model.nmocap:
                mocap_pos = jp.array(self._mj_model.key_mpos[self._key_id])
                mocap_quat = jp.array(self._mj_model.key_mquat[self._key_id])

        # Randomize cable root + mocap position around keyframe.
        delta = jax.random.uniform(rng_delta, (3,), minval=-0.03, maxval=0.03)
        try:
            cable_root_jnt = self._mj_model.joint("cable:free").id
            adr = int(self._mj_model.jnt_qposadr[cable_root_jnt])
            qpos = qpos.at[adr : adr + 3].add(delta)
        except Exception:
            pass
        if mocap_pos is not None and self._mj_model.nmocap:
            mocap_pos = mocap_pos + delta

        # Zero velocities if no keyframe is present
        if self._key_id is None:
            qvel = jp.zeros(self.mj_model.nv)
        qacc = jp.zeros(self.mj_model.nv)
        qfrc_applied = jp.zeros(self.mj_model.nv)
        xfrc_applied = jp.zeros((self.mj_model.nbody, 6))

        # Initialize MJX data without warp (warp requires an MjModel input).
        # data = mjx.make_data(self.mj_model, impl=self.impl, nconmax=4500, njmax=100)
        data = mjx.make_data(self.mj_model, impl=self.impl, nconmax=4500, njmax=200)
        if qpos is not None:
            data = data.replace(qpos=qpos)
        if qvel is not None:
            data = data.replace(qvel=qvel)
        if ctrl is not None:
            data = data.replace(ctrl=ctrl)
        if act is not None:
            data = data.replace(act=act)
        if mocap_pos is not None:
            data = data.replace(mocap_pos=mocap_pos.reshape(self.mjx_model.nmocap, -1))
        if mocap_quat is not None:
            data = data.replace(
                mocap_quat=mocap_quat.reshape(self.mjx_model.nmocap, -1)
            )
        data = mjx.forward(self.mjx_model, data)

        data = data.replace(qvel=qvel)
        data = data.replace(qacc=qacc)
        data = data.replace(qfrc_applied=qfrc_applied)
        data = data.replace(xfrc_applied=xfrc_applied)

        metrics = {
            "reward_metric": jp.array(0.0, dtype=jp.float32),
        }
        info = {"rng": rng, "step": jp.array(0)}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if (
            action is not None
            and action.shape[-1] == 6
            and self._mocap_id is not None
            and self.mj_model.nmocap
        ):
            dpos = action[:3]
            drot = action[3:6]
            target_pos = state.data.mocap_pos[self._mocap_id] + dpos
            current_quat = state.data.mocap_quat[self._mocap_id]
            delta_quat = _rotvec_to_quat(drot)
            target_quat = _quat_normalize(_quat_mul(delta_quat, current_quat))

            data = state.data.replace(
                mocap_pos=state.data.mocap_pos.at[self._mocap_id].set(target_pos),
                mocap_quat=state.data.mocap_quat.at[self._mocap_id].set(target_quat),
            )
            data = mjx_env.step(self.mjx_model, data, data.ctrl, self.n_substeps)
        else:
            a = state.data.ctrl + (action * self.ctrl_scale)
            data = mjx_env.step(self.mjx_model, state.data, a, self.n_substeps)

        done = self._get_done(data, state.info)

        if self._sparse_reward:
            reward = done
        else:
            reward = self._get_reward(data)

        obs = self._get_obs(data)

        rng, _ = jax.random.split(state.info["rng"])
        info = {**state.info}
        info["rng"] = rng
        info["step"] = info["step"] + 1
        metrics = {
            **state.metrics,
            "reward_metric": reward,
        }

        return mjx_env.State(data, obs, reward, done, metrics, info)

    @property
    def observation_size(self) -> int:
        """Compute observation size by calling _get_obs with dummy data"""
        # Create dummy data for size computation
        dummy_data = mjx.make_data(self.mj_model, impl=self.impl)

        # Get observation and check its shape
        obs = self._get_obs(dummy_data)
        # print(f"{obs=}")
        return obs.shape[0]  # Get the last dimension (feature size)
        # return 6  # Get the last dimension (feature size)

    @property
    def action_size(self) -> int:
        if self._mocap_id is not None and self.mj_model.nmocap:
            return 6
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


class SafeAutoResetWrapper(wrapper.Wrapper):
    """Auto-reset wrapper that tolerates non-batched mjx.Data fields."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        state.info["first_state"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, state.info["first_state"], state.data)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(data=data, obs=obs)


def wrap_for_mjx_training(
    env: mjx_env.MjxEnv,
    episode_length: int,
    action_repeat: int,
    randomization_fn=None,
) -> wrapper.Wrapper:
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = SafeAutoResetWrapper(env)
    return env


def view(model: mj.MjModel):
    import glfw
    import mujoco.viewer

    m = model
    d = mj.MjData(m)

    close = False
    rng = np.random.default_rng()

    def _keyframe_id() -> int:
        for key_name in ("init", "bent"):
            try:
                return m.key(key_name).id
            except Exception:
                continue
        return 0 if getattr(m, "nkey", 0) > 0 else -1

    def randomize_state(delta_range: float = 0.05) -> None:
        key_id = _keyframe_id()
        if key_id >= 0:
            d.qpos[:] = m.key_qpos[key_id]
            d.qvel[:] = m.key_qvel[key_id]
            if d.ctrl.size:
                d.ctrl[:] = m.key_ctrl[key_id]
            if m.nmocap:
                d.mocap_pos[:] = m.key_mpos[key_id].reshape(m.nmocap, 3)
                d.mocap_quat[:] = m.key_mquat[key_id].reshape(m.nmocap, 4)

        delta = rng.uniform(-delta_range, delta_range, size=3)
        try:
            cable_root_jnt = m.joint("cable:free").id
            adr = int(m.jnt_qposadr[cable_root_jnt])
            d.qpos[adr : adr + 3] += delta
        except Exception:
            pass

        try:
            mocap_id = int(m.body("mocap").mocapid)
            if mocap_id >= 0 and m.nmocap:
                d.mocap_pos[mocap_id] += delta
        except Exception:
            pass

        mj.mj_forward(m, d)

    def cb(key: int) -> None:
        if key is glfw.KEY_SPACE:
            global close
            close = True
        if key is glfw.KEY_PERIOD:
            d.ctrl = np.array([255])
        if key is glfw.KEY_R:
            randomize_state()

    cable_site_name = "mocap_site_1"
    mocap_site_name = "cable_weld_site_1"

    cn = "keypoint"
    pn = "target"

    with mujoco.viewer.launch_passive(model=m, data=d, key_callback=cb) as viewer:
        while not close:
            step_start = time.time()

            p1 = d.site(cn).xpos
            p2 = d.site(pn).xpos

            print(np.linalg.norm(p1 - p2))

            # step simulation one time step
            mj.mj_step(m, d)

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


@dataclass
class TrainArgs:
    num_timesteps: int = 200_000
    num_evals: int = 10
    episode_length: int = 150
    action_repeat: int = 1
    unroll_length: int = 10
    num_minibatches: int = 16
    num_updates_per_batch: int = 4
    discounting: float = 0.99
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    num_envs: int = 256
    batch_size: int = 4096
    seed: int = 0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    deterministic_eval: bool = False
    metrics_plot_path: str = "testing/envs/mocap_cable_3x_weld.png"
    render_path: str = "testing/envs/mocap_cable_3x_weld_rollout.gif"
    render_steps: int = 150
    render_fps: int = 30
    render_impl: str = "jax"
    render_use_jit: tyro.conf.FlagConversionOff[bool] = True
    render_frame_skip: int = 1
    render_height: int = 480
    render_width: int = 640
    render_eval_rollouts: tyro.conf.FlagConversionOff[bool] = True
    render_eval_path: str = ""
    render_after_training: tyro.conf.FlagConversionOff[bool] = True
    render_final_path: str = ""


def main() -> None:

    VIEW = True

    args = tyro.cli(TrainArgs)

    print("config...")

    cfg = default_config()
    cfg.episode_length = args.episode_length
    cfg.action_repeat = args.action_repeat

    print("env...")
    env = PipeInsert2(cfg)
    render_env = env
    if args.render_impl != cfg.impl:
        render_cfg = default_config()
        render_cfg.ctrl_dt = cfg.ctrl_dt
        render_cfg.sim_dt = cfg.sim_dt
        render_cfg.episode_length = cfg.episode_length
        render_cfg.action_repeat = cfg.action_repeat
        render_cfg.impl = args.render_impl
        render_env = PipeInsert2(render_cfg)

    if VIEW:
        view(env.mj_model)
        quit()

    plot_keys = [
        "eval/episode_reward",
        "training/total_loss",
        "training/sps",
    ]
    metrics_history: dict[str, list[float]] = {key: [] for key in plot_keys}
    metrics_history["step"] = []
    eval_rollout_index = 0

    def progress_fn(step: int, metrics: dict) -> None:
        metrics_history["step"].append(step)
        for key in plot_keys:
            metrics_history[key].append(_metric_to_float(metrics.get(key)))
        _plot_metrics(metrics_history, args.metrics_plot_path)

    def policy_params_fn(step: int, make_policy, params) -> None:
        nonlocal eval_rollout_index
        if not args.render_eval_rollouts:
            return
        if jax.process_index() != 0:
            return
        base_path = args.render_eval_path or args.render_path
        if not base_path:
            return
        eval_path = _format_eval_path(base_path, eval_rollout_index, int(step))
        policy = make_policy(params, deterministic=args.deterministic_eval)
        seed = args.seed + eval_rollout_index
        eval_rollout_index += 1
        render_rollout(
            render_env,
            steps=args.render_steps,
            path=eval_path,
            seed=seed,
            fps=args.render_fps,
            policy=policy,
            use_jit=args.render_use_jit,
            frame_skip=args.render_frame_skip,
            height=args.render_height,
            width=args.render_width,
        )
        print(f"saved eval rollout to {eval_path}")

    print("train def...")
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=args.reward_scaling,
        episode_length=args.episode_length,
        normalize_observations=args.normalize_observations,
        action_repeat=args.action_repeat,
        unroll_length=args.unroll_length,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        discounting=args.discounting,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        seed=args.seed,
        gae_lambda=args.gae_lambda,
        clipping_epsilon=args.clipping_epsilon,
        deterministic_eval=args.deterministic_eval,
        network_factory=ppo_networks.make_ppo_networks,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )

    print("train run...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrap_for_mjx_training,
    )

    print("training metrics:", metrics)

    if args.render_after_training:
        policy = make_inference_fn(params, deterministic=args.deterministic_eval)
        final_path = args.render_final_path or _format_final_path(args.render_path)
        if final_path:
            render_rollout(
                render_env,
                steps=args.render_steps,
                path=final_path,
                seed=args.seed + 10_000,
                fps=args.render_fps,
                policy=policy,
                use_jit=args.render_use_jit,
                frame_skip=args.render_frame_skip,
                height=args.render_height,
                width=args.render_width,
            )
            print(f"saved final rollout to {final_path}")

    # quick deterministic rollout after training
    policy = make_inference_fn(params)
    rng = jax.random.PRNGKey(args.seed)
    eval_env = wrap_for_mjx_training(
        env, episode_length=args.episode_length, action_repeat=args.action_repeat
    )
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, 1)
    state = eval_env.reset(reset_keys)
    for _ in range(args.episode_length):
        rng, step_key = jax.random.split(rng)
        state, _ = acting.actor_step(eval_env, state, policy, step_key)
    final_obs = np.asarray(state.obs[0])
    final_pos_err = np.linalg.norm(final_obs)
    print(f"final errors (unbatched): pos={float(final_pos_err):.4f}")


if __name__ == "__main__":
    main()
