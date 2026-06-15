import csv
import functools
import json
import math
import os
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import mujoco as mj
import numpy as np
import tyro
from brax.envs.wrappers import training as brax_training
from brax.training import acting
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env, wrapper

from utils.mjx import ObjType, get_pose

_REWARD_COMPONENTS_LOG_PATH = Path(__file__).with_name("reward_components.csv")
_REWARD_COMPONENTS_LOG_LOCK = threading.Lock()
_REWARD_COMPONENTS_LOG_FIELDS = [
    "time",
    "reward_weighted_position_error_prev",
    "reward_weighted_rotation_error_prev",
    "reward_weighted_position_error",
    "reward_weighted_rotation_error",
    "reward_delta_pose_prev",
    "reward_delta_pose",
    "near",
    "reward_total",
    "done",
]


def _to_jsonable(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return json.dumps(array.tolist())


def _log_reward_components(
    pos_err0: Any,
    rot_err0: Any,
    pos_err1: Any,
    rot_err1: Any,
    delta_pose_0: Any,
    delta_pose_1: Any,
    near: Any,
    reward: Any,
    done: Any,
) -> None:
    record = {
        "time": time.time(),
        "reward_weighted_position_error_prev": _to_jsonable(pos_err0),
        "reward_weighted_rotation_error_prev": _to_jsonable(rot_err0),
        "reward_weighted_position_error": _to_jsonable(pos_err1),
        "reward_weighted_rotation_error": _to_jsonable(rot_err1),
        "reward_delta_pose_prev": _to_jsonable(delta_pose_0),
        "reward_delta_pose": _to_jsonable(delta_pose_1),
        "near": _to_jsonable(near),
        "reward_total": _to_jsonable(reward),
        "done": _to_jsonable(done),
    }
    with _REWARD_COMPONENTS_LOG_LOCK:
        write_header = (
            not _REWARD_COMPONENTS_LOG_PATH.exists()
            or _REWARD_COMPONENTS_LOG_PATH.stat().st_size == 0
        )
        with _REWARD_COMPONENTS_LOG_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_REWARD_COMPONENTS_LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record)


def reward_metrics_from_error_terms(
    pos_err0: Any,
    rot_err0: Any,
    pos_err1: Any,
    rot_err1: Any,
    done: Any | None,
    termination_threshold_pos: Any,
    termination_threshold_rot: Any,
) -> dict[str, jax.Array]:
    pos_err0 = jp.asarray(pos_err0)
    rot_err0 = jp.asarray(rot_err0)
    pos_err1 = jp.asarray(pos_err1)
    rot_err1 = jp.asarray(rot_err1)
    termination_threshold_pos = jp.asarray(termination_threshold_pos)
    termination_threshold_rot = jp.asarray(termination_threshold_rot)

    # Work in units of the success thresholds so translation and z-axis
    # alignment have comparable scales. rot_err is the z-axis alignment angle,
    # not a full SO(3) pose error.
    eps = jp.asarray(1e-6)
    pos_scale = jp.maximum(termination_threshold_pos, eps)
    rot_scale = jp.maximum(termination_threshold_rot, eps)
    pos0 = pos_err0 / pos_scale
    pos1 = pos_err1 / pos_scale
    rot0 = rot_err0 / rot_scale
    rot1 = rot_err1 / rot_scale

    orientation_weight = jp.asarray(2.0)
    delta_pose_0 = jp.sqrt(pos0 * pos0 + (orientation_weight * rot0) ** 2)
    delta_pose_1 = jp.sqrt(pos1 * pos1 + (orientation_weight * rot1) ** 2)

    pose_progress = jp.clip(delta_pose_0 - delta_pose_1, -5.0, 5.0)
    pos_progress = pos0 - pos1

    # Dense bounded terms keep the gradient useful near the goal without
    # letting far-away states dominate the update.
    position_closeness = jp.exp(-0.5 * pos1)
    z_axis_alignment = 0.5 * (jp.cos(rot_err1) + 1.0)
    rotation_closeness = jp.exp(-0.5 * rot1)
    near = jp.exp(-0.5 * delta_pose_1)

    pose_distance_penalty = -0.02 * delta_pose_1
    step_penalty = jp.asarray(-0.01)

    success = (pos_err1 < termination_threshold_pos) & (
        rot_err1 < termination_threshold_rot
    )
    done_value = jp.array(0.0) if done is None else jp.asarray(done)
    success_bonus = jp.where(success, 50.0, 0.0)
    terminal_penalty = jp.where((done_value > 0.0) & (~success), -10.0, 0.0)

    reward = (
        10.0 * pose_progress
        + 2.0 * position_closeness
        + 2.0 * rotation_closeness
        + near
        + z_axis_alignment
        + pose_distance_penalty
        + step_penalty
        + success_bonus
        + terminal_penalty
    )

    return {
        "position_error_prev": pos_err0,
        "rotation_error_prev": rot_err0,
        "position_error": pos_err1,
        "rotation_error": rot_err1,
        "reward_delta_pose_prev": delta_pose_0,
        "reward_delta_pose": delta_pose_1,
        "reward_position_progress": pos_progress,
        "reward_rotation_distance_component": rotation_closeness,
        "reward_z_axis_alignment": z_axis_alignment,
        "reward_pose_distance_penalty": pose_distance_penalty,
        "reward_step_penalty": step_penalty,
        "near": near,
        "reward_total": reward,
    }


DEFAULT_CABLE_STIFFNESS_MULTIPLIER_RANGE = (0.01, 10.0)


def domain_randomize(
    model: mjx.Model,
    mj_model: mj.MjModel,
    rng: jax.Array,
    cable_youngs_multiplier_range: tuple[
        float, float
    ] = DEFAULT_CABLE_STIFFNESS_MULTIPLIER_RANGE,
    cable_shear_multiplier_range: tuple[
        float, float
    ] = DEFAULT_CABLE_STIFFNESS_MULTIPLIER_RANGE,
    log_uniform: bool = False,
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

        # Base stiffnesses used for this cable (match _init creation defaults).
        cable_youngs_base = 10000000.0 * 5
        cable_shear_base = 60000.0 * 1000
        # Per-env ranges (scale as needed).
        cable_youngs_range = (
            float(cable_youngs_multiplier_range[0]) * cable_youngs_base,
            float(cable_youngs_multiplier_range[1]) * cable_youngs_base,
        )
        cable_shear_range = (
            float(cable_shear_multiplier_range[0]) * cable_shear_base,
            float(cable_shear_multiplier_range[1]) * cable_shear_base,
        )
        if log_uniform and (
            cable_youngs_range[0] <= 0
            or cable_shear_range[0] <= 0
            or cable_youngs_range[1] <= 0
            or cable_shear_range[1] <= 0
        ):
            raise ValueError(
                "log-uniform cable stiffness randomization requires positive ranges."
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
            rng, key = jax.random.split(rng)
            if log_uniform:
                youngs = jp.exp(
                    jax.random.uniform(
                        key,
                        shape=(),
                        minval=jp.log(cable_youngs_range[0]),
                        maxval=jp.log(cable_youngs_range[1]),
                    )
                )
            else:
                youngs = jax.random.uniform(
                    key,
                    shape=(),
                    minval=cable_youngs_range[0],
                    maxval=cable_youngs_range[1],
                )
            rng, key = jax.random.split(rng)
            if log_uniform:
                shear = jp.exp(
                    jax.random.uniform(
                        key,
                        shape=(),
                        minval=jp.log(cable_shear_range[0]),
                        maxval=jp.log(cable_shear_range[1]),
                    )
                )
            else:
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


def _quat_conjugate(q: jax.Array) -> jax.Array:
    return jp.array([q[0], -q[1], -q[2], -q[3]])


def _rotate_vec_by_quat(v: jax.Array, q: jax.Array) -> jax.Array:
    v_quat = jp.concatenate([jp.zeros(1), v], axis=0)
    rotated = _quat_mul(_quat_mul(q, v_quat), _quat_conjugate(q))
    return rotated[1:]


def _cable_ball_joint_qpos_adrs(
    mj_model: mj.MjModel, prefix: str = "cable:"
) -> np.ndarray:
    """Return qpos addresses for cable ball joints, excluding the free joint."""
    adrs = [
        int(mj_model.jnt_qposadr[joint_id])
        for joint_id in range(mj_model.njnt)
        if mj_model.joint(joint_id).name.startswith(f"{prefix}J")
        and mj_model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_BALL
    ]
    print(
        [
            mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, joint_id)
            for joint_id in range(mj_model.njnt)
            if mj_model.joint(joint_id).name.startswith(f"{prefix}J")
            and mj_model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_BALL
        ]
    )
    return np.asarray(sorted(adrs), dtype=np.int32)


def random_cable_curvature_qpos(
    qpos: jax.Array,
    rng: jax.Array,
    cable_joint_qpos_adrs: jax.Array,
    bend_range: tuple[float, float],
    phi_range: tuple[float, float],
    num_cables: int = 1,
) -> jax.Array:
    """Randomly initialize cable ball-joint qpos with constant curvature."""
    num_joints = int(cable_joint_qpos_adrs.shape[0])
    if num_joints == 0:
        return qpos
    if num_cables <= 0:
        raise ValueError(f"num_cables must be > 0, got {num_cables}")
    if num_joints % num_cables != 0:
        raise ValueError(
            f"Expected cable joints ({num_joints}) to be divisible by "
            f"num_cables ({num_cables})."
        )

    joints_per_cable = num_joints // num_cables
    rng_bend, rng_phi = jax.random.split(rng)
    total_bend = jax.random.uniform(
        rng_bend,
        shape=(),
        minval=float(bend_range[0]),
        maxval=float(bend_range[1]),
    )
    azimuth_phi = jax.random.uniform(
        rng_phi,
        shape=(),
        minval=float(phi_range[0]),
        maxval=float(phi_range[1]),
    )

    theta = total_bend / float(joints_per_cable)
    theta_y = theta * jp.cos(azimuth_phi)
    theta_z = theta * jp.sin(azimuth_phi)
    joint_quat = _rotvec_to_quat(jp.array([0.0, theta_y, theta_z]))
    joint_quats = jp.broadcast_to(joint_quat, (num_joints, 4))

    qpos_indices = cable_joint_qpos_adrs[:, None] + jp.arange(4)
    return qpos.at[qpos_indices].set(joint_quats)


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
        a_max=0,
        a_min=0,
        success_threshold_pos=0.01,
        impl="warp",
        randomize_cable_curvature=False,
        cable_curvature_bend_range=(0.0, 0.0),
        cable_curvature_phi_range=(0.0, 2.0 * math.pi),
        cable_curvature_num_cables=1,
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

        self._mj_model = self._init()
        self.ctrl_scale: float = 1.0
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model, impl=self.impl)

        self._termination_threshold_pos = 0.01  # m
        self._termination_threshold_rot = jp.deg2rad(10)  # rad
        self._failed_pos_threshold = 5  # m

        self._termination_threshold = (
            self._termination_threshold_pos,
            self._termination_threshold_rot,
        )

        self._w_pos = 10.0
        self._w_rot = 100.0

        self._sparse_reward = config["sparse_reward"]
        self._episode_length = config["episode_length"]
        self._randomize_cable_curvature = bool(config["randomize_cable_curvature"])
        self._cable_curvature_bend_range = tuple(config["cable_curvature_bend_range"])
        self._cable_curvature_phi_range = tuple(config["cable_curvature_phi_range"])
        self._cable_curvature_num_cables = int(config["cable_curvature_num_cables"])

        self._post_init()

    def _init(self) -> mj.MjModel:
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
        keypoint_height = 0
        # keypoint_height = self.pipe_inner_radius - (
        #     self.pipe_outer_radius - self.pipe_inner_radius
        # )

        pip.worldbody.first_body().add_site(
            name="target",
            pos=[0, keypoint_height, -0.062],
            group=1,
            rgba=[1, 0, 0, 1],
            # name="target", pos=[0, keypoint_height, -0.06], group=1, rgba=[1, 0, 0, 1]
        )

        twist = 60000.0 * 1000
        bend = 10000000.0 * 5
        # twist = 60000.0 * 5
        # bend = 10000000.0 * 5

        cable = mjx_cable(twist=twist, bend=bend, size=0.35, initial="free")
        # cable = mjx_cable(twist=twist, bend=bend, size=0.4, initial="free")

        s_key = cable.body("cable:Bfirst").add_site(
            name="keypoint",
            group=1,
            rgba=[1, 0, 0, 1],
            pos=[0, -0.02, 0],
            # pos=[0, 0, 0],
            quat=[0, 0, -0.7071068, 0.7071068],
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

        solref = [-50_000, -362.71080706]
        solimp = [0.999, 0.9999, 0.0001, 0.5, 1]
        # solref = [0.00000002, 1.0]
        # solimp = [0.99, 0.9999, 1e-4, 0.0, 10]

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

    def _get_obs(
        self,
        data: mjx.Data,
        rng: jax.Array | None = None,
        add_noise: bool = True,
    ) -> jax.Array:
        T_w_target = get_pose(self.mjx_model, data, "target", ObjType.SITE)
        T_w_keypoint = get_pose(self.mjx_model, data, "keypoint", ObjType.SITE)
        R_w_target = T_w_target.rotation().as_matrix()
        R_w_keypoint = T_w_keypoint.rotation().as_matrix()
        e_pos_world = T_w_keypoint.translation() - T_w_target.translation()
        e_pos = jp.linalg.matrix_transpose(R_w_target) @ e_pos_world
        z_rel = jp.linalg.matrix_transpose(R_w_target) @ R_w_keypoint[:, 2]

        if add_noise and rng is not None:
            rng_pos, rng_rot_axis, rng_rot_angle = jax.random.split(rng, 3)

            # e_pos noise +- 3 mm
            e_pos = e_pos + jax.random.uniform(
                rng_pos, (3,), minval=-0.003, maxval=0.003
            )

            # z_rel noise +- 0.05 deg
            rot_axis = jax.random.normal(rng_rot_axis, (3,))
            rot_axis = rot_axis / jp.maximum(jp.linalg.norm(rot_axis), 1e-6)
            rot_angle = jax.random.uniform(
                rng_rot_angle,
                (),
                # minval=-0.05,
                # maxval=0.05,
                minval=-jp.deg2rad(0.05),
                maxval=jp.deg2rad(0.05),
            )
            z_rel = _rotate_vec_by_quat(z_rel, _rotvec_to_quat(rot_axis * rot_angle))
            z_rel = z_rel / jp.maximum(jp.linalg.norm(z_rel), 1e-6)

        return jp.concatenate([e_pos, z_rel], axis=0)

    def _get_reward(
        self, data0: mjx.Data, data1: mjx.Data, done: jax.Array | None = None
    ) -> dict[str, jax.Array]:
        _, pos_err0, rot_err0 = self._get_error_metrics(data0)
        _, pos_err1, rot_err1 = self._get_error_metrics(data1)

        # w_pos_err0, w_rot_err0 = self._get_weighted_error_metrics(
        #     pos_err=pos_err0, rot_err=rot_err0
        # )
        # w_pos_err1, w_rot_err1 = self._get_weighted_error_metrics(
        #     pos_err=pos_err1, rot_err=rot_err1
        # )

        # Normalize by success thresholds so meters and radians are comparable.
        # The orientation multiplier keeps alignment important even when the
        # translational error is still large.
        eps = jp.asarray(1e-6)
        pos_scale = jp.maximum(jp.asarray(self._termination_threshold_pos), eps)
        rot_scale = jp.maximum(jp.asarray(self._termination_threshold_rot), eps)
        pos0 = pos_err0 / pos_scale
        pos1 = pos_err1 / pos_scale
        rot0 = rot_err0 / rot_scale
        rot1 = rot_err1 / rot_scale

        orientation_weight = jp.asarray(10.0)
        delta_pose_0 = pos0 + orientation_weight * rot0
        delta_pose_1 = pos1 + orientation_weight * rot1

        pose_progress = delta_pose_0 - delta_pose_1
        pos_progress = pos0 - pos1
        rot_distance_reward = -orientation_weight * rot1
        near = jp.exp(-delta_pose_1)

        success = (pos_err1 < self._termination_threshold_pos) & (
            rot_err1 < self._termination_threshold_rot
        )
        done_value = jp.array(0.0) if done is None else done
        success_bonus = jp.where(success, 25.0, 0.0)
        terminal_penalty = jp.where((done_value > 0.0) & (~success), -25.0, 0.0)
        # success_bonus = jp.where(success, 500.0, 0.0)
        # terminal_penalty = jp.where((done_value > 0.0) & (~success), -250.0, 0.0)

        reward = pose_progress + near + success_bonus + terminal_penalty

        return {
            "position_error_prev": pos_err0,
            "rotation_error_prev": rot_err0,
            "position_error": pos_err1,
            "rotation_error": rot_err1,
            "reward_delta_pose_prev": delta_pose_0,
            "reward_delta_pose": delta_pose_1,
            "reward_position_progress": pos_progress,
            "reward_rotation_distance_component": rot_distance_reward,
            "near": near,
            "reward_total": reward,
        }

    def get_reward_2(
        self, data0: mjx.Data, data1: mjx.Data, done: jax.Array | None = None
    ) -> dict[str, jax.Array]:
        _, pos_err0, rot_err0 = self._get_error_metrics(data0)
        _, pos_err1, rot_err1 = self._get_error_metrics(data1)

        return reward_metrics_from_error_terms(
            pos_err0=pos_err0,
            rot_err0=rot_err0,
            pos_err1=pos_err1,
            rot_err1=rot_err1,
            done=done,
            termination_threshold_pos=self._termination_threshold_pos,
            termination_threshold_rot=self._termination_threshold_rot,
        )

    def _get_empty_reward_metrics(self) -> dict[str, jax.Array]:
        zero = jp.array(0.0)
        return {
            "position_error_prev": zero,
            "rotation_error_prev": zero,
            "position_error": zero,
            "rotation_error": zero,
            "reward_delta_pose_prev": zero,
            "reward_delta_pose": zero,
            "reward_position_progress": zero,
            "reward_rotation_distance_component": zero,
            "reward_z_axis_alignment": zero,
            "reward_pose_distance_penalty": zero,
            "reward_step_penalty": zero,
            "near": zero,
            "reward_total": zero,
        }

    def _get_error_metrics(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        return position and orientation error in m and rad
        """
        obs = self._get_obs(data, add_noise=False)
        pos_err = jp.linalg.norm(obs[:3])
        z_dot = jp.clip(obs[5], -1.0, 1.0)
        rot_err = jp.arccos(z_dot)
        return obs, pos_err, rot_err

    def _get_weighted_error_metrics(
        self, pos_err: jax.Array, rot_err: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return self._w_pos * pos_err, self._w_rot * rot_err

    def _get_done(self, data: mjx.Data, info: dict) -> float:
        step_count = info["steps"] if "steps" in info else info["step"]

        is_unstable = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        timeout = step_count >= self._episode_length

        _, pos_err, rot_err = self._get_error_metrics(data)

        # Recommended: separate thresholds for translation and rotation
        if isinstance(self._termination_threshold, (tuple, list)):
            pos_thresh, rot_thresh = self._termination_threshold
        else:
            pos_thresh = self._termination_threshold
            rot_thresh = self._termination_threshold

        # Deviated too far away
        # failed = pos_err > self._failed_pos_threshold

        success = (pos_err < pos_thresh) & (rot_err < rot_thresh)

        # jax.debug.print(
        #     (
        #         "done states: \n"
        #         "\tis_unstable={is_unstable}, \n"
        #         "\ttimeout={timeout}, \n"
        #         "\tsuccess={success}, \n"
        #         "\tfailed={failed}, \n"
        #     ),
        #     is_unstable=is_unstable,
        #     timeout=timeout,
        #     success=success,
        #     failed=failed,
        # )

        done = is_unstable | timeout | success
        # done = is_unstable | timeout | success | failed
        return done.astype(float)

    def _post_init(self) -> None:
        self.target_id = self._mj_model.site("target").id
        self.keypoint_id = self._mj_model.site("keypoint").id
        self._cable_joint_qpos_adrs = jp.array(
            _cable_ball_joint_qpos_adrs(self._mj_model)
        )
        if (
            self._randomize_cable_curvature
            and self._cable_joint_qpos_adrs.shape[0] % self._cable_curvature_num_cables
            != 0
        ):
            raise ValueError(
                "cable_curvature_num_cables must divide the number of cable ball "
                f"joints ({self._cable_joint_qpos_adrs.shape[0]})."
            )

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
        rng, rng_delta, rng_curvature, rng_obs = jax.random.split(rng, 4)

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

        # Randomize cable root + mocap position around keyframe before applying
        # precurvature.  The precurvature helper only rewrites internal cable
        # ball-joint quaternions, so the free-joint offset is preserved.
        delta = jax.random.uniform(rng_delta, (3,), minval=-0.3, maxval=0.3) + jp.array(
            [0, 0.1, 0]
        )
        try:
            cable_root_jnt = self._mj_model.joint("cable:free").id
            adr = int(self._mj_model.jnt_qposadr[cable_root_jnt])
            qpos = qpos.at[adr : adr + 3].add(delta)
        except Exception:
            pass
        if mocap_pos is not None and self._mj_model.nmocap:
            mocap_pos = mocap_pos + delta

        if self._randomize_cable_curvature:
            qpos = random_cable_curvature_qpos(
                qpos=qpos,
                rng=rng_curvature,
                cable_joint_qpos_adrs=self._cable_joint_qpos_adrs,
                bend_range=self._cable_curvature_bend_range,
                phi_range=self._cable_curvature_phi_range,
                num_cables=self._cable_curvature_num_cables,
            )
            qvel = jp.zeros(self.mj_model.nv)

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

        obs = self._get_obs(data, rng=rng_obs, add_noise=True)
        _, pos_err, rot_err = self._get_error_metrics(data)
        weighted_pos_err, weighted_rot_err = self._get_weighted_error_metrics(
            pos_err, rot_err
        )
        reward_metrics = self._get_empty_reward_metrics()
        metrics = {
            "position_error": pos_err,
            "rotation_error": rot_err,
            "weighted_position_error": weighted_pos_err,
            "weighted_rotation_error": weighted_rot_err,
            **reward_metrics,
        }
        info = {
            "rng": rng,
            "step": jp.array(0),
            "position_error": pos_err,
            "rotation_error": rot_err,
            "weighted_position_error": weighted_pos_err,
            "weighted_rotation_error": weighted_rot_err,
            **reward_metrics,
        }

        reward, done = jp.zeros(2)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        data0 = state.data

        dpos = action[:3]
        drot = action[3:6]

        target_pos = data0.mocap_pos[self._mocap_id] + dpos

        current_quat = data0.mocap_quat[self._mocap_id]

        delta_quat = _rotvec_to_quat(drot)

        target_quat = _quat_normalize(_quat_mul(delta_quat, current_quat))

        data = data0.replace(
            mocap_pos=data0.mocap_pos.at[self._mocap_id].set(target_pos),
            mocap_quat=data0.mocap_quat.at[self._mocap_id].set(target_quat),
        )
        data = mjx_env.step(self.mjx_model, data, data.ctrl, self.n_substeps)

        rng, rng_obs = jax.random.split(state.info["rng"])
        obs = self._get_obs(data, rng=rng_obs, add_noise=True)
        _, pos_err, rot_err = self._get_error_metrics(data)

        weighted_pos_err, weighted_rot_err = self._get_weighted_error_metrics(
            pos_err, rot_err
        )
        done = self._get_done(data, state.info)
        reward_metrics = self._get_empty_reward_metrics()

        if self._sparse_reward:
            reward = done
            reward_metrics = {
                **reward_metrics,
                "reward_total": reward,
            }
        else:
            # reward_metrics = self.get_reward_2(data0, data, done)
            reward_metrics = self._get_reward(data0, data, done)
            reward = reward_metrics["reward_total"]

        info = {**state.info}
        info["rng"] = rng
        info["step"] = info["step"] + 1
        info["position_error"] = pos_err
        info["rotation_error"] = rot_err
        info["weighted_position_error"] = weighted_pos_err
        info["weighted_rotation_error"] = weighted_rot_err
        info.update(reward_metrics)

        metrics = {
            **state.metrics,
            "position_error": pos_err,
            "rotation_error": rot_err,
            "weighted_position_error": weighted_pos_err,
            "weighted_rotation_error": weighted_rot_err,
            **reward_metrics,
        }

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def se3_z_axis_observation(
        R_current: jp.ndarray,
        t_current: jp.ndarray,
        R_target: jp.ndarray,
        t_target: jp.ndarray,
        include_z_dot: bool = True,
    ) -> jp.ndarray:
        """
        Compute an SE(3)-style observation in the target frame where
        translation is standard and orientation only measures z-axis alignment.

        Args:
            R_current: (3, 3) current/keypoint rotation matrix
            t_current: (3,) current/keypoint position
            R_target:  (3, 3) target rotation matrix
            t_target:  (3,) target position
            include_z_dot: if True, include the z-axis cosine term for robustness

        Returns:
            obs:
                if include_z_dot:
                    shape (6,) = [dp_x, dp_y, dp_z, z_rel_x, z_rel_y, z_rel_z]
                else:
                    shape (5,) = [dp_x, dp_y, dp_z, z_rel_x, z_rel_y]
        """
        # Position error in target frame.
        dp_world = t_current - t_target
        dp = R_target.T @ dp_world

        # Relative keypoint z-axis expressed in the target frame.
        # This is the 3rd column of R_target.T @ R_current.
        z_rel = R_target.T @ R_current[:, 2]

        if include_z_dot:
            obs = jp.concatenate([dp, z_rel], axis=0)
        else:
            obs = jp.concatenate([dp, z_rel[:2]], axis=0)

        return obs

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
