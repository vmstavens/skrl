import functools
import math
import os
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

from testing.envs.pipe_insert_2 import parse_obj
from utils.mjx import ObjType, get_pose

# <key
#   time="24.352"
#   qpos="1.36004e-07 0.4 0.399964 0.0007742 1 5.10054e-07 1.49254e-07 0.705048 0.000203406 0.694227 -0.701813 0.705048 0.000206316 0.694225 -0.701839 0.00218251 0.242195 0.0903311 0.648398 0.407118 -0.547032 -0.338514 0.999999 -0.000209236 -5.9603e-06 0.00137112 0.999984 -0.000889182 -6.22239e-06 0.0055497 0.999918 -0.00205618 -6.35398e-06 0.0126689 0.999725 -0.00377208 -8.02477e-06 0.0231287 0.999269 -0.00616777 -1.55575e-05 0.0377267 0.998285 -0.00946107 -3.85047e-05 0.0577762 0.996269 -0.0139576 -9.40666e-05 0.0851656 0.992304 -0.0200178 -0.000213389 0.122196 0.984944 -0.027851 -0.000441306 0.170613 0.972913 -0.0369859 -0.000644665 0.228194 0.992152 -0.0160525 -2.53628e-06 0.124004 0.997345 -0.00897996 3.58934e-06 0.0722653 0.999234 -0.00439477 2.32971e-05 0.038896 0.999853 -0.00136528 3.92105e-05 0.0171049 0.999997 0.000728434 4.95125e-05 0.00236156 0.999964 0.0022576 5.41419e-05 -0.00813067 0.999865 0.00345771 5.31513e-05 -0.01608 0.999737 0.00444589 4.79055e-05 -0.0224962 0.999596 0.00529425 4.19877e-05 -0.0279421 0.999447 0.00601181 3.88485e-05 -0.0326958 0.9993 0.00659028 4.14671e-05 -0.0368293 0.999165 0.00702228 4.61888e-05 -0.0402504 0.999059 0.00722337 5.53289e-05 -0.0427775 0.998997 0.00713513 7.32397e-05 -0.0441983 0.999003 0.00667051 0.000101378 -0.0441411 0.999092 0.00571423 0.000140736 -0.0422308 0.999269 0.00413733 0.000190595 -0.0379978 0.999519 0.00184447 0.000230379 -0.0309477 0.999789 -0.001437 0.000242094 -0.020502 0.99996 -0.00691183 0.000194048 -0.00559836 0.999964 -0.00834044 0.000130477 -0.00150954 0.999999 0.00126059 9.73332e-05 0.000228067 0.999998 0.00198394 7.54568e-05 0.000355323 0.999996 0.00264783 5.52334e-05 0.000496258 0.999994 0.00346091 3.77568e-05 0.000645757 0.999993 0.00359892 2.42041e-05 0.000671576 0.999995 0.00323031 1.43891e-05 0.000600423 0.999997 0.00238307 8.07263e-06 0.000443923 0.999999 0.0012134 3.82412e-06 0.000236578"
#   qvel="5.33691e-08 -1.7121e-06 2.82668e-07 -0.000344996 7.06013e-06 8.05372e-06 -6.21541e-07 -1.57996e-06 -3.18236e-06 6.92397e-06 -3.92801e-07 5.22924e-07 1.92425e-06 3.2255e-07 -0.000133793 -9.54591e-05 6.81012e-06 -0.000563511 -0.00187129 0.000285827 -2.03781e-05 8.73249e-06 1.61228e-05 -1.05112e-05 9.23232e-06 3.71629e-06 1.84928e-05 8.99986e-06 -1.26548e-05 6.08968e-05 8.69917e-06 -2.58534e-05 0.000119548 1.0802e-05 -4.5903e-05 0.000202842 2.13483e-05 -8.46529e-05 0.000313088 5.09971e-05 -0.000138879 0.000453326 0.000121336 -0.000205302 0.000581225 0.000278285 -0.000217178 0.000407486 0.000347011 0.00102666 -0.000402769 0.000142698 0.00133963 -0.00031519 0.000163477 0.00120856 -0.000206618 0.000145187 0.000910056 -9.71032e-05 0.000124652 0.000546469 -4.32525e-06 0.000110176 0.000191035 8.04691e-05 0.00010408 -0.000160049 0.000142971 0.000106742 -0.000496744 0.0002025 0.000116364 -0.000815937 0.000250701 0.000127388 -0.00111774 0.000307838 0.000135839 -0.00138428 0.000369907 0.000138919 -0.00159516 0.000392448 0.00014004 -0.00174696 0.000416441 0.000137579 -0.00179739 0.000418147 0.00012681 -0.00162385 0.000394045 0.000105739 -0.00125535 0.000359531 7.3441e-05 -0.000659565 0.00031901 2.85588e-05 0.000125493 0.00017826 -1.19449e-05 0.00113713 3.36564e-05 -3.36494e-05 0.00248928 -0.000718188 -6.86367e-05 0.00415896 -0.000243119 -5.72125e-05 0.00171137 -0.000412314 -3.68994e-05 0.00186791 -0.000255963 -2.63524e-05 0.00121703 -0.000221835 -1.84138e-05 0.0010312 -0.000101426 -1.31763e-05 0.000500452 -0.000119414 -9.45405e-06 0.000316422 -0.000105454 -5.59891e-06 0.000335345 5.1126e-05 -5.51806e-06 -0.000347723 -5.81844e-06 -2.56083e-06 6.5512e-06"
#   ctrl="255"
#   mpos="0 0.4 0.4"
#   mquat="0.000796327 1 0 0"
# />


# <key
#   time="112.672"
#   qpos="-7.0802e-08 0.4 0.399964 0.000799541 1 2.99191e-07 4.02526e-06 0.705055 0.000209212 0.694236 -0.701859 0.705055 0.000211376 0.694236 -0.701882 0.00180079 0.242155 0.0902499 0.647417 0.405906 -0.548272 -0.339838 0.999999 -0.00022072 1.40805e-13 0.00138287 0.999984 -0.000886664 2.55741e-12 0.00555516 0.999918 -0.0020218 1.66775e-11 0.012667 0.999726 -0.00369068 6.80645e-11 0.023123 0.99927 -0.00601949 2.13093e-10 0.0377135 0.998289 -0.00921636 5.66729e-10 0.0577427 0.99628 -0.0135827 1.35572e-09 0.0850987 0.992329 -0.0194854 3.0116e-09 0.122081 0.984986 -0.0272096 6.29496e-09 0.170475 0.972771 -0.0365304 1.23275e-08 0.228872 0.967532 -0.0398372 -8.39201e-09 0.24959 0.98963 -0.0226396 -2.65502e-09 0.141842 0.996766 -0.0126654 -8.21613e-10 0.0793519 0.998985 -0.00709956 -2.5615e-10 0.0444804 0.999676 -0.00401032 -8.12448e-11 0.0251256 0.999895 -0.00228713 -2.63037e-11 0.0143294 0.999965 -0.00131791 -8.70428e-12 0.00825702 0.999988 -0.000767587 -2.94588e-12 0.00480911 0.999996 -0.000451992 -1.01991e-12 0.00283183 0.999999 -0.000269152 -3.61076e-13 0.0016863 0.999999 -0.000162115 -1.3048e-13 0.00101569 1 -9.87893e-05 -4.78776e-14 0.000618938 1 -6.09186e-05 -1.75971e-14 0.000381669 1 -3.80218e-05 -6.24914e-15 0.000238215 1 -2.40233e-05 -1.91545e-15 0.000150512 1 -1.53673e-05 -2.43678e-16 9.62796e-05 1 -9.9521e-06 3.92943e-16 6.23522e-05 1 -6.52343e-06 6.16927e-16 4.08708e-05 1 -4.32513e-06 6.72253e-16 2.70979e-05 1 -2.89674e-06 6.56892e-16 1.81487e-05 1 -1.95509e-06 6.1116e-16 1.22491e-05 1 -1.32436e-06 5.5206e-16 8.29741e-06 1 -8.94412e-07 4.86947e-16 5.60369e-06 1 -5.95882e-07 4.19083e-16 3.73333e-06 1 -3.85046e-07 3.49948e-16 2.4124e-06 1 -2.34627e-07 2.80234e-16 1.46999e-06 1 -1.28099e-07 2.10267e-16 8.0257e-07 1 -5.60893e-08 1.40201e-16 3.51412e-07 1 -1.39556e-08 7.01034e-17 8.74352e-08"
#   qvel="-1.5665e-10 -4.19492e-11 3.25121e-11 1.14502e-09 4.34549e-09 -1.38611e-09 9.13508e-09 6.81017e-09 1.21335e-08 -5.48219e-08 8.98244e-09 6.69328e-09 1.28613e-08 -5.54774e-08 -3.20096e-08 -1.08279e-08 -2.59575e-08 1.01681e-08 6.2781e-08 1.1853e-09 -1.24528e-10 1.63355e-13 -2.25804e-11 -5.00409e-10 2.60871e-12 -9.07271e-11 -1.14217e-09 1.32464e-11 -2.07006e-10 -2.08891e-09 4.30119e-11 -3.78297e-10 -3.41681e-09 1.11691e-10 -6.17907e-10 -5.25081e-09 2.56595e-10 -9.47303e-10 -7.76871e-09 5.48696e-10 -1.39595e-09 -1.11761e-08 1.11545e-09 -1.99465e-09 -1.55969e-08 2.14649e-09 -2.75121e-09 -2.07904e-08 3.77936e-09 -3.59283e-09 -2.09687e-08 6.37227e-09 -2.66002e-09 -1.2093e-08 2.03217e-09 -1.5252e-09 -6.69773e-09 6.25029e-10 -8.43136e-10 -3.68855e-09 1.92764e-10 -4.63927e-10 -2.04269e-09 6.03678e-11 -2.56781e-10 -1.14169e-09 1.9278e-11 -1.43456e-10 -6.44826e-10 6.28772e-12 -8.09922e-11 -3.68239e-10 2.09623e-12 -4.62342e-11 -2.12693e-10 7.14701e-13 -2.66939e-11 -1.24287e-10 2.49291e-13 -1.55924e-11 -7.34945e-11 8.89699e-14 -9.21646e-12 -4.39895e-11 3.24767e-14 -5.51406e-12 -2.6657e-11 1.2106e-14 -3.34001e-12 -1.63585e-11 4.58802e-15 -2.04874e-12 -1.01679e-11 1.7486e-15 -1.27293e-12 -6.40235e-12 6.52264e-16 -8.01167e-13 -4.08399e-12 2.20833e-16 -5.10857e-13 -2.63868e-12 4.91661e-17 -3.29945e-13 -1.72588e-12 -1.85205e-17 -2.15748e-13 -1.14141e-12 -4.35543e-17 -1.4265e-13 -7.6157e-13 -5.06536e-17 -9.51783e-14 -5.10654e-13 -5.00236e-17 -6.38197e-14 -3.41886e-13 -4.59795e-17 -4.2733e-14 -2.26165e-13 -4.03913e-17 -2.82832e-14 -1.45351e-13 -3.40893e-17 -1.81846e-14 -8.82279e-14 -2.74524e-17 -1.10466e-14 -4.8052e-14 -2.06582e-17 -6.0172e-15 -2.10118e-14 -1.3793e-17 -2.63466e-15 -5.22509e-15 -6.90011e-18 -6.55429e-16"
#   ctrl="255"
#   mpos="0 0.4 0.4"
#   mquat="0.000796327 1 0 0"
# />


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
        action_scale_pos=0.01,
        success_threshold_pos=0.02,
        impl="warp",
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
        solimp=[0.9, 0.95, 0.001, 0.5, 1],
        # solimp=[0.9, 0.95, 0.001, 0.5, 1],
        friction=[0.1, 0.005, 0.0001],
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
    # noslip_iterations="2"
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
    # <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" />
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
            # "qpos": "tmp/qpos.npy",
            # "qvel": "tmp/qvel.npy",
            # "ctrl": "tmp/ctrl.npy",
            "qpos": "constants/v4/qpos.npy",
            "qvel": "constants/v4/qvel.npy",
            "ctrl": "constants/v4/ctrl.npy",
        }

        # self.sequence = {}
        # with open("testing/experiments/pipe_insert/tmp/success_rollout.json", "r") as f:
        #     self.sequence = json.load(f)
        # self.sequence_actions = jp.array(self.sequence.get("actions", []))

        # print(f"{self.sequence_actions.shape=}")
        # quit()

        self.i = 0

        # self._mj_model = self._init_mocap_attach()
        # self._mj_model = self._init_mocap_weld()
        # self._mj_model = self._init_mocap_weld_2()
        # self._mj_model = self._init_mocap_weld_3()
        self._mj_model = self._init_mocap_weld_4()
        self.ctrl_scale: float = 1.0
        # self._mj_model = self._init()
        # self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model, impl=self.impl)

        # self._termination_threshold = 0.05
        self._termination_threshold = 0.04

        self._sparse_reward = config["sparse_reward"]
        self._episode_length = config["episode_length"]
        self._grace_steps = int(config.get("grace_steps", 0))

        self._post_init()

    def _init_mocap_attach(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        # Do not load keyframes from file; use keys defined in this init.

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
        self.pipe_length = 0.125 / 2  # 121 mm from schematic
        # self.pipe_length = 0.121 / 2  # 121 mm from schematic
        # pipe_length = 0.09

        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            resolution=10,
            # resolution=20,
            # rgba=[0.2, 0.2, 0.2, 1],
        )
        # pip = pipe(inner_radius=0.0385, outer_radius=0.0435, length=0.075)

        mesh_path = Path("assets/DS00240017_E.obj")
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

        twist = 60000.0 * 2
        bend = 10000000.0 * 2

        cable = mjx_cable(
            twist=twist, bend=bend, segment_size=0.002, size=0.4, initial="none"
        )

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

        gripper_spawn = [0.0, 0.4, 0.4]

        mocap = scene.worldbody.add_body(
            name="mocap", mocap=True, pos=gripper_spawn, euler=[3.14, 0, 0]
        )
        mocap.add_geom(
            name="mocap",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
            # rgba=[1, 1, 1, ],
        )
        s_mocap = mocap.add_site(name="mocap_site", pos=[0, 0, 0])

        scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 3.14]).attach_body(
            # scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 0]).attach_body(
            pip.worldbody.first_body(),
        )
        cable_root = cable.worldbody.first_body()
        s_mocap.attach_body(cable_root)

        # b = None
        # b0 = cable.worldbody.first_body()
        # for i in range(10):
        #     b = b0.first_body()
        #     b0 = b

        # b.add_site(
        #     name="cable_weld_site",
        #     pos=[0.0, 0.0, 0.0],  # adjust if you want an offset on that segment
        #     euler=[0.0, 0.0, 1.57],
        #     group=1,
        #     rgba=[0, 1, 1, 1],
        # )

        # scene.add_equality(
        #     name="mocap_cable_weld",
        #     type=mj.mjtEq.mjEQ_WELD,
        #     objtype=mj.mjtObj.mjOBJ_SITE,
        #     name1="mocap_site",
        #     name2="cable_weld_site",
        #     # Keep the current relative pose at creation time.
        #     solref=[0.000001, 2],
        # )

        # scene.add_key(
        #     name="bent",
        #     time=112.672,
        #     qpos=_parse_float_list(
        #         "-7.0802e-08 0.4 0.399964 0.000799541 1 2.99191e-07 4.02526e-06 0.705055 0.000209212 0.694236 -0.701859 0.705055 0.000211376 0.694236 -0.701882 0.00180079 0.242155 0.0902499 0.647417 0.405906 -0.548272 -0.339838 0.999999 -0.00022072 1.40805e-13 0.00138287 0.999984 -0.000886664 2.55741e-12 0.00555516 0.999918 -0.0020218 1.66775e-11 0.012667 0.999726 -0.00369068 6.80645e-11 0.023123 0.99927 -0.00601949 2.13093e-10 0.0377135 0.998289 -0.00921636 5.66729e-10 0.0577427 0.99628 -0.0135827 1.35572e-09 0.0850987 0.992329 -0.0194854 3.0116e-09 0.122081 0.984986 -0.0272096 6.29496e-09 0.170475 0.972771 -0.0365304 1.23275e-08 0.228872 0.967532 -0.0398372 -8.39201e-09 0.24959 0.98963 -0.0226396 -2.65502e-09 0.141842 0.996766 -0.0126654 -8.21613e-10 0.0793519 0.998985 -0.00709956 -2.5615e-10 0.0444804 0.999676 -0.00401032 -8.12448e-11 0.0251256 0.999895 -0.00228713 -2.63037e-11 0.0143294 0.999965 -0.00131791 -8.70428e-12 0.00825702 0.999988 -0.000767587 -2.94588e-12 0.00480911 0.999996 -0.000451992 -1.01991e-12 0.00283183 0.999999 -0.000269152 -3.61076e-13 0.0016863 0.999999 -0.000162115 -1.3048e-13 0.00101569 1 -9.87893e-05 -4.78776e-14 0.000618938 1 -6.09186e-05 -1.75971e-14 0.000381669 1 -3.80218e-05 -6.24914e-15 0.000238215 1 -2.40233e-05 -1.91545e-15 0.000150512 1 -1.53673e-05 -2.43678e-16 9.62796e-05 1 -9.9521e-06 3.92943e-16 6.23522e-05 1 -6.52343e-06 6.16927e-16 4.08708e-05 1 -4.32513e-06 6.72253e-16 2.70979e-05 1 -2.89674e-06 6.56892e-16 1.81487e-05 1 -1.95509e-06 6.1116e-16 1.22491e-05 1 -1.32436e-06 5.5206e-16 8.29741e-06 1 -8.94412e-07 4.86947e-16 5.60369e-06 1 -5.95882e-07 4.19083e-16 3.73333e-06 1 -3.85046e-07 3.49948e-16 2.4124e-06 1 -2.34627e-07 2.80234e-16 1.46999e-06 1 -1.28099e-07 2.10267e-16 8.0257e-07 1 -5.60893e-08 1.40201e-16 3.51412e-07 1 -1.39556e-08 7.01034e-17 8.74352e-08"
        #     ),
        #     qvel=_parse_float_list(
        #         "-1.5665e-10 -4.19492e-11 3.25121e-11 1.14502e-09 4.34549e-09 -1.38611e-09 9.13508e-09 6.81017e-09 1.21335e-08 -5.48219e-08 8.98244e-09 6.69328e-09 1.28613e-08 -5.54774e-08 -3.20096e-08 -1.08279e-08 -2.59575e-08 1.01681e-08 6.2781e-08 1.1853e-09 -1.24528e-10 1.63355e-13 -2.25804e-11 -5.00409e-10 2.60871e-12 -9.07271e-11 -1.14217e-09 1.32464e-11 -2.07006e-10 -2.08891e-09 4.30119e-11 -3.78297e-10 -3.41681e-09 1.11691e-10 -6.17907e-10 -5.25081e-09 2.56595e-10 -9.47303e-10 -7.76871e-09 5.48696e-10 -1.39595e-09 -1.11761e-08 1.11545e-09 -1.99465e-09 -1.55969e-08 2.14649e-09 -2.75121e-09 -2.07904e-08 3.77936e-09 -3.59283e-09 -2.09687e-08 6.37227e-09 -2.66002e-09 -1.2093e-08 2.03217e-09 -1.5252e-09 -6.69773e-09 6.25029e-10 -8.43136e-10 -3.68855e-09 1.92764e-10 -4.63927e-10 -2.04269e-09 6.03678e-11 -2.56781e-10 -1.14169e-09 1.9278e-11 -1.43456e-10 -6.44826e-10 6.28772e-12 -8.09922e-11 -3.68239e-10 2.09623e-12 -4.62342e-11 -2.12693e-10 7.14701e-13 -2.66939e-11 -1.24287e-10 2.49291e-13 -1.55924e-11 -7.34945e-11 8.89699e-14 -9.21646e-12 -4.39895e-11 3.24767e-14 -5.51406e-12 -2.6657e-11 1.2106e-14 -3.34001e-12 -1.63585e-11 4.58802e-15 -2.04874e-12 -1.01679e-11 1.7486e-15 -1.27293e-12 -6.40235e-12 6.52264e-16 -8.01167e-13 -4.08399e-12 2.20833e-16 -5.10857e-13 -2.63868e-12 4.91661e-17 -3.29945e-13 -1.72588e-12 -1.85205e-17 -2.15748e-13 -1.14141e-12 -4.35543e-17 -1.4265e-13 -7.6157e-13 -5.06536e-17 -9.51783e-14 -5.10654e-13 -5.00236e-17 -6.38197e-14 -3.41886e-13 -4.59795e-17 -4.2733e-14 -2.26165e-13 -4.03913e-17 -2.82832e-14 -1.45351e-13 -3.40893e-17 -1.81846e-14 -8.82279e-14 -2.74524e-17 -1.10466e-14 -4.8052e-14 -2.06582e-17 -6.0172e-15 -2.10118e-14 -1.3793e-17 -2.63466e-15 -5.22509e-15 -6.90011e-18 -6.55429e-16"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )
        # scene.add_key(
        #     name="bent",
        #     time=24.352,
        #     qpos=_parse_float_list(
        #         "1.36004e-07 0.4 0.399964 0.0007742 1 5.10054e-07 1.49254e-07 0.705048 0.000203406 0.694227 -0.701813 0.705048 0.000206316 0.694225 -0.701839 0.00218251 0.242195 0.0903311 0.648398 0.407118 -0.547032 -0.338514 0.999999 -0.000209236 -5.9603e-06 0.00137112 0.999984 -0.000889182 -6.22239e-06 0.0055497 0.999918 -0.00205618 -6.35398e-06 0.0126689 0.999725 -0.00377208 -8.02477e-06 0.0231287 0.999269 -0.00616777 -1.55575e-05 0.0377267 0.998285 -0.00946107 -3.85047e-05 0.0577762 0.996269 -0.0139576 -9.40666e-05 0.0851656 0.992304 -0.0200178 -0.000213389 0.122196 0.984944 -0.027851 -0.000441306 0.170613 0.972913 -0.0369859 -0.000644665 0.228194 0.992152 -0.0160525 -2.53628e-06 0.124004 0.997345 -0.00897996 3.58934e-06 0.0722653 0.999234 -0.00439477 2.32971e-05 0.038896 0.999853 -0.00136528 3.92105e-05 0.0171049 0.999997 0.000728434 4.95125e-05 0.00236156 0.999964 0.0022576 5.41419e-05 -0.00813067 0.999865 0.00345771 5.31513e-05 -0.01608 0.999737 0.00444589 4.79055e-05 -0.0224962 0.999596 0.00529425 4.19877e-05 -0.0279421 0.999447 0.00601181 3.88485e-05 -0.0326958 0.9993 0.00659028 4.14671e-05 -0.0368293 0.999165 0.00702228 4.61888e-05 -0.0402504 0.999059 0.00722337 5.53289e-05 -0.0427775 0.998997 0.00713513 7.32397e-05 -0.0441983 0.999003 0.00667051 0.000101378 -0.0441411 0.999092 0.00571423 0.000140736 -0.0422308 0.999269 0.00413733 0.000190595 -0.0379978 0.999519 0.00184447 0.000230379 -0.0309477 0.999789 -0.001437 0.000242094 -0.020502 0.99996 -0.00691183 0.000194048 -0.00559836 0.999964 -0.00834044 0.000130477 -0.00150954 0.999999 0.00126059 9.73332e-05 0.000228067 0.999998 0.00198394 7.54568e-05 0.000355323 0.999996 0.00264783 5.52334e-05 0.000496258 0.999994 0.00346091 3.77568e-05 0.000645757 0.999993 0.00359892 2.42041e-05 0.000671576 0.999995 0.00323031 1.43891e-05 0.000600423 0.999997 0.00238307 8.07263e-06 0.000443923 0.999999 0.0012134 3.82412e-06 0.000236578"
        #     ),
        #     qvel=_parse_float_list(
        #         "5.33691e-08 -1.7121e-06 2.82668e-07 -0.000344996 7.06013e-06 8.05372e-06 -6.21541e-07 -1.57996e-06 -3.18236e-06 6.92397e-06 -3.92801e-07 5.22924e-07 1.92425e-06 3.2255e-07 -0.000133793 -9.54591e-05 6.81012e-06 -0.000563511 -0.00187129 0.000285827 -2.03781e-05 8.73249e-06 1.61228e-05 -1.05112e-05 9.23232e-06 3.71629e-06 1.84928e-05 8.99986e-06 -1.26548e-05 6.08968e-05 8.69917e-06 -2.58534e-05 0.000119548 1.0802e-05 -4.5903e-05 0.000202842 2.13483e-05 -8.46529e-05 0.000313088 5.09971e-05 -0.000138879 0.000453326 0.000121336 -0.000205302 0.000581225 0.000278285 -0.000217178 0.000407486 0.000347011 0.00102666 -0.000402769 0.000142698 0.00133963 -0.00031519 0.000163477 0.00120856 -0.000206618 0.000145187 0.000910056 -9.71032e-05 0.000124652 0.000546469 -4.32525e-06 0.000110176 0.000191035 8.04691e-05 0.00010408 -0.000160049 0.000142971 0.000106742 -0.000496744 0.0002025 0.000116364 -0.000815937 0.000250701 0.000127388 -0.00111774 0.000307838 0.000135839 -0.00138428 0.000369907 0.000138919 -0.00159516 0.000392448 0.00014004 -0.00174696 0.000416441 0.000137579 -0.00179739 0.000418147 0.00012681 -0.00162385 0.000394045 0.000105739 -0.00125535 0.000359531 7.3441e-05 -0.000659565 0.00031901 2.85588e-05 0.000125493 0.00017826 -1.19449e-05 0.00113713 3.36564e-05 -3.36494e-05 0.00248928 -0.000718188 -6.86367e-05 0.00415896 -0.000243119 -5.72125e-05 0.00171137 -0.000412314 -3.68994e-05 0.00186791 -0.000255963 -2.63524e-05 0.00121703 -0.000221835 -1.84138e-05 0.0010312 -0.000101426 -1.31763e-05 0.000500452 -0.000119414 -9.45405e-06 0.000316422 -0.000105454 -5.59891e-06 0.000335345 5.1126e-05 -5.51806e-06 -0.000347723 -5.81844e-06 -2.56083e-06 6.5512e-06"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _init_mocap_weld(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        # Do not load keyframes from file; use keys defined in this init.

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
        self.pipe_length = 0.125 / 2  # 121 mm from schematic
        # self.pipe_length = 0.121 / 2  # 121 mm from schematic
        # pipe_length = 0.09

        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            resolution=20,
            # rgba=[0.2, 0.2, 0.2, 1],
        )
        # pip = pipe(inner_radius=0.0385, outer_radius=0.0435, length=0.075)

        mesh_path = Path("assets/DS00240017_E.obj")
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

        twist = 60000.0 * 2
        bend = 10000000.0 * 2

        cable = mjx_cable(
            twist=twist, bend=bend, segment_size=0.002, size=0.4, initial="free"
        )

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

        gripper_spawn = [0.0, 0.4, 0.4]

        mocap = scene.worldbody.add_body(
            name="mocap", mocap=True, pos=gripper_spawn, euler=[3.14, 0, 0]
        )
        mocap.add_geom(
            name="mocap",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
            rgba=[1, 1, 1, 0.2],
        )
        s_mocap = mocap.add_site(name="mocap_site", pos=[0, 0, 0])

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
            name="cable_weld_site",
            pos=[0.0, 0.0, 0.0],  # adjust if you want an offset on that segment
            euler=[0.0, 0.0, 1.57],
            group=1,
            rgba=[0, 1, 1, 1],
        )

        scene.add_equality(
            name="mocap_cable_weld",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site",
            name2="cable_weld_site",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
            solimp=[0.95, 0.99, 0.001, 0.1, 6],
            # solimp=[0.95, 0.99, 0.001, 0.1, 6],
        )

        # scene.add_key(
        #     name="bent",
        #     time=112.672,
        #     qpos=_parse_float_list(
        #         "-7.0802e-08 0.4 0.399964 0.000799541 1 2.99191e-07 4.02526e-06 0.705055 0.000209212 0.694236 -0.701859 0.705055 0.000211376 0.694236 -0.701882 0.00180079 0.242155 0.0902499 0.647417 0.405906 -0.548272 -0.339838 0.999999 -0.00022072 1.40805e-13 0.00138287 0.999984 -0.000886664 2.55741e-12 0.00555516 0.999918 -0.0020218 1.66775e-11 0.012667 0.999726 -0.00369068 6.80645e-11 0.023123 0.99927 -0.00601949 2.13093e-10 0.0377135 0.998289 -0.00921636 5.66729e-10 0.0577427 0.99628 -0.0135827 1.35572e-09 0.0850987 0.992329 -0.0194854 3.0116e-09 0.122081 0.984986 -0.0272096 6.29496e-09 0.170475 0.972771 -0.0365304 1.23275e-08 0.228872 0.967532 -0.0398372 -8.39201e-09 0.24959 0.98963 -0.0226396 -2.65502e-09 0.141842 0.996766 -0.0126654 -8.21613e-10 0.0793519 0.998985 -0.00709956 -2.5615e-10 0.0444804 0.999676 -0.00401032 -8.12448e-11 0.0251256 0.999895 -0.00228713 -2.63037e-11 0.0143294 0.999965 -0.00131791 -8.70428e-12 0.00825702 0.999988 -0.000767587 -2.94588e-12 0.00480911 0.999996 -0.000451992 -1.01991e-12 0.00283183 0.999999 -0.000269152 -3.61076e-13 0.0016863 0.999999 -0.000162115 -1.3048e-13 0.00101569 1 -9.87893e-05 -4.78776e-14 0.000618938 1 -6.09186e-05 -1.75971e-14 0.000381669 1 -3.80218e-05 -6.24914e-15 0.000238215 1 -2.40233e-05 -1.91545e-15 0.000150512 1 -1.53673e-05 -2.43678e-16 9.62796e-05 1 -9.9521e-06 3.92943e-16 6.23522e-05 1 -6.52343e-06 6.16927e-16 4.08708e-05 1 -4.32513e-06 6.72253e-16 2.70979e-05 1 -2.89674e-06 6.56892e-16 1.81487e-05 1 -1.95509e-06 6.1116e-16 1.22491e-05 1 -1.32436e-06 5.5206e-16 8.29741e-06 1 -8.94412e-07 4.86947e-16 5.60369e-06 1 -5.95882e-07 4.19083e-16 3.73333e-06 1 -3.85046e-07 3.49948e-16 2.4124e-06 1 -2.34627e-07 2.80234e-16 1.46999e-06 1 -1.28099e-07 2.10267e-16 8.0257e-07 1 -5.60893e-08 1.40201e-16 3.51412e-07 1 -1.39556e-08 7.01034e-17 8.74352e-08"
        #     ),
        #     qvel=_parse_float_list(
        #         "-1.5665e-10 -4.19492e-11 3.25121e-11 1.14502e-09 4.34549e-09 -1.38611e-09 9.13508e-09 6.81017e-09 1.21335e-08 -5.48219e-08 8.98244e-09 6.69328e-09 1.28613e-08 -5.54774e-08 -3.20096e-08 -1.08279e-08 -2.59575e-08 1.01681e-08 6.2781e-08 1.1853e-09 -1.24528e-10 1.63355e-13 -2.25804e-11 -5.00409e-10 2.60871e-12 -9.07271e-11 -1.14217e-09 1.32464e-11 -2.07006e-10 -2.08891e-09 4.30119e-11 -3.78297e-10 -3.41681e-09 1.11691e-10 -6.17907e-10 -5.25081e-09 2.56595e-10 -9.47303e-10 -7.76871e-09 5.48696e-10 -1.39595e-09 -1.11761e-08 1.11545e-09 -1.99465e-09 -1.55969e-08 2.14649e-09 -2.75121e-09 -2.07904e-08 3.77936e-09 -3.59283e-09 -2.09687e-08 6.37227e-09 -2.66002e-09 -1.2093e-08 2.03217e-09 -1.5252e-09 -6.69773e-09 6.25029e-10 -8.43136e-10 -3.68855e-09 1.92764e-10 -4.63927e-10 -2.04269e-09 6.03678e-11 -2.56781e-10 -1.14169e-09 1.9278e-11 -1.43456e-10 -6.44826e-10 6.28772e-12 -8.09922e-11 -3.68239e-10 2.09623e-12 -4.62342e-11 -2.12693e-10 7.14701e-13 -2.66939e-11 -1.24287e-10 2.49291e-13 -1.55924e-11 -7.34945e-11 8.89699e-14 -9.21646e-12 -4.39895e-11 3.24767e-14 -5.51406e-12 -2.6657e-11 1.2106e-14 -3.34001e-12 -1.63585e-11 4.58802e-15 -2.04874e-12 -1.01679e-11 1.7486e-15 -1.27293e-12 -6.40235e-12 6.52264e-16 -8.01167e-13 -4.08399e-12 2.20833e-16 -5.10857e-13 -2.63868e-12 4.91661e-17 -3.29945e-13 -1.72588e-12 -1.85205e-17 -2.15748e-13 -1.14141e-12 -4.35543e-17 -1.4265e-13 -7.6157e-13 -5.06536e-17 -9.51783e-14 -5.10654e-13 -5.00236e-17 -6.38197e-14 -3.41886e-13 -4.59795e-17 -4.2733e-14 -2.26165e-13 -4.03913e-17 -2.82832e-14 -1.45351e-13 -3.40893e-17 -1.81846e-14 -8.82279e-14 -2.74524e-17 -1.10466e-14 -4.8052e-14 -2.06582e-17 -6.0172e-15 -2.10118e-14 -1.3793e-17 -2.63466e-15 -5.22509e-15 -6.90011e-18 -6.55429e-16"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )
        # scene.add_key(
        #     name="bent",
        #     time=24.352,
        #     qpos=_parse_float_list(
        #         "1.36004e-07 0.4 0.399964 0.0007742 1 5.10054e-07 1.49254e-07 0.705048 0.000203406 0.694227 -0.701813 0.705048 0.000206316 0.694225 -0.701839 0.00218251 0.242195 0.0903311 0.648398 0.407118 -0.547032 -0.338514 0.999999 -0.000209236 -5.9603e-06 0.00137112 0.999984 -0.000889182 -6.22239e-06 0.0055497 0.999918 -0.00205618 -6.35398e-06 0.0126689 0.999725 -0.00377208 -8.02477e-06 0.0231287 0.999269 -0.00616777 -1.55575e-05 0.0377267 0.998285 -0.00946107 -3.85047e-05 0.0577762 0.996269 -0.0139576 -9.40666e-05 0.0851656 0.992304 -0.0200178 -0.000213389 0.122196 0.984944 -0.027851 -0.000441306 0.170613 0.972913 -0.0369859 -0.000644665 0.228194 0.992152 -0.0160525 -2.53628e-06 0.124004 0.997345 -0.00897996 3.58934e-06 0.0722653 0.999234 -0.00439477 2.32971e-05 0.038896 0.999853 -0.00136528 3.92105e-05 0.0171049 0.999997 0.000728434 4.95125e-05 0.00236156 0.999964 0.0022576 5.41419e-05 -0.00813067 0.999865 0.00345771 5.31513e-05 -0.01608 0.999737 0.00444589 4.79055e-05 -0.0224962 0.999596 0.00529425 4.19877e-05 -0.0279421 0.999447 0.00601181 3.88485e-05 -0.0326958 0.9993 0.00659028 4.14671e-05 -0.0368293 0.999165 0.00702228 4.61888e-05 -0.0402504 0.999059 0.00722337 5.53289e-05 -0.0427775 0.998997 0.00713513 7.32397e-05 -0.0441983 0.999003 0.00667051 0.000101378 -0.0441411 0.999092 0.00571423 0.000140736 -0.0422308 0.999269 0.00413733 0.000190595 -0.0379978 0.999519 0.00184447 0.000230379 -0.0309477 0.999789 -0.001437 0.000242094 -0.020502 0.99996 -0.00691183 0.000194048 -0.00559836 0.999964 -0.00834044 0.000130477 -0.00150954 0.999999 0.00126059 9.73332e-05 0.000228067 0.999998 0.00198394 7.54568e-05 0.000355323 0.999996 0.00264783 5.52334e-05 0.000496258 0.999994 0.00346091 3.77568e-05 0.000645757 0.999993 0.00359892 2.42041e-05 0.000671576 0.999995 0.00323031 1.43891e-05 0.000600423 0.999997 0.00238307 8.07263e-06 0.000443923 0.999999 0.0012134 3.82412e-06 0.000236578"
        #     ),
        #     qvel=_parse_float_list(
        #         "5.33691e-08 -1.7121e-06 2.82668e-07 -0.000344996 7.06013e-06 8.05372e-06 -6.21541e-07 -1.57996e-06 -3.18236e-06 6.92397e-06 -3.92801e-07 5.22924e-07 1.92425e-06 3.2255e-07 -0.000133793 -9.54591e-05 6.81012e-06 -0.000563511 -0.00187129 0.000285827 -2.03781e-05 8.73249e-06 1.61228e-05 -1.05112e-05 9.23232e-06 3.71629e-06 1.84928e-05 8.99986e-06 -1.26548e-05 6.08968e-05 8.69917e-06 -2.58534e-05 0.000119548 1.0802e-05 -4.5903e-05 0.000202842 2.13483e-05 -8.46529e-05 0.000313088 5.09971e-05 -0.000138879 0.000453326 0.000121336 -0.000205302 0.000581225 0.000278285 -0.000217178 0.000407486 0.000347011 0.00102666 -0.000402769 0.000142698 0.00133963 -0.00031519 0.000163477 0.00120856 -0.000206618 0.000145187 0.000910056 -9.71032e-05 0.000124652 0.000546469 -4.32525e-06 0.000110176 0.000191035 8.04691e-05 0.00010408 -0.000160049 0.000142971 0.000106742 -0.000496744 0.0002025 0.000116364 -0.000815937 0.000250701 0.000127388 -0.00111774 0.000307838 0.000135839 -0.00138428 0.000369907 0.000138919 -0.00159516 0.000392448 0.00014004 -0.00174696 0.000416441 0.000137579 -0.00179739 0.000418147 0.00012681 -0.00162385 0.000394045 0.000105739 -0.00125535 0.000359531 7.3441e-05 -0.000659565 0.00031901 2.85588e-05 0.000125493 0.00017826 -1.19449e-05 0.00113713 3.36564e-05 -3.36494e-05 0.00248928 -0.000718188 -6.86367e-05 0.00415896 -0.000243119 -5.72125e-05 0.00171137 -0.000412314 -3.68994e-05 0.00186791 -0.000255963 -2.63524e-05 0.00121703 -0.000221835 -1.84138e-05 0.0010312 -0.000101426 -1.31763e-05 0.000500452 -0.000119414 -9.45405e-06 0.000316422 -0.000105454 -5.59891e-06 0.000335345 5.1126e-05 -5.51806e-06 -0.000347723 -5.81844e-06 -2.56083e-06 6.5512e-06"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _init_mocap_weld_2(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        # Do not load keyframes from file; use keys defined in this init.

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
        self.pipe_length = 0.125 / 2  # 121 mm from schematic
        # self.pipe_length = 0.121 / 2  # 121 mm from schematic
        # pipe_length = 0.09

        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            resolution=20,
            # rgba=[0.2, 0.2, 0.2, 1],
        )
        # pip = pipe(inner_radius=0.0385, outer_radius=0.0435, length=0.075)

        mesh_path = Path("assets/DS00240017_E.obj")
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

        twist = 60000.0 * 5
        bend = 10000000.0 * 5

        cable = mjx_cable(
            twist=twist,
            bend=bend,
            size=0.4,
            initial="free",
            # twist=twist, bend=bend, segment_size=0.002, size=0.4, initial="free"
        )

        cable.body("cable:Bfirst").add_site(
            name="keypoint_3", group=1, rgba=[0, 1, 0, 1], pos=[0, 0, 0]
        )
        cable.body("cable:B2").add_site(
            name="keypoint_2", group=1, rgba=[1, 1, 0, 1], pos=[0, 0, 0]
        )
        cable.body("cable:B4").add_site(
            name="keypoint_1", group=1, rgba=[1, 0, 0, 1], pos=[0, 0, 0]
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
            rgba=[1, 1, 1, 0.2],
        )
        s_mocap_1 = mocap.add_site(name="mocap_site_1", pos=[0, -0.01, 0])
        s_mocap_2 = mocap.add_site(name="mocap_site_2", pos=[0, 0.01, 0])

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

        scene.add_equality(
            name="mocap_cable_weld_1",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_1",
            name2="cable_weld_site_1",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
        )
        scene.add_equality(
            name="mocap_cable_weld_2",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_2",
            name2="cable_weld_site_2",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
        )

        # scene.add_key(
        #     name="bent",
        #     time=112.672,
        #     qpos=_parse_float_list(
        #         "-7.0802e-08 0.4 0.399964 0.000799541 1 2.99191e-07 4.02526e-06 0.705055 0.000209212 0.694236 -0.701859 0.705055 0.000211376 0.694236 -0.701882 0.00180079 0.242155 0.0902499 0.647417 0.405906 -0.548272 -0.339838 0.999999 -0.00022072 1.40805e-13 0.00138287 0.999984 -0.000886664 2.55741e-12 0.00555516 0.999918 -0.0020218 1.66775e-11 0.012667 0.999726 -0.00369068 6.80645e-11 0.023123 0.99927 -0.00601949 2.13093e-10 0.0377135 0.998289 -0.00921636 5.66729e-10 0.0577427 0.99628 -0.0135827 1.35572e-09 0.0850987 0.992329 -0.0194854 3.0116e-09 0.122081 0.984986 -0.0272096 6.29496e-09 0.170475 0.972771 -0.0365304 1.23275e-08 0.228872 0.967532 -0.0398372 -8.39201e-09 0.24959 0.98963 -0.0226396 -2.65502e-09 0.141842 0.996766 -0.0126654 -8.21613e-10 0.0793519 0.998985 -0.00709956 -2.5615e-10 0.0444804 0.999676 -0.00401032 -8.12448e-11 0.0251256 0.999895 -0.00228713 -2.63037e-11 0.0143294 0.999965 -0.00131791 -8.70428e-12 0.00825702 0.999988 -0.000767587 -2.94588e-12 0.00480911 0.999996 -0.000451992 -1.01991e-12 0.00283183 0.999999 -0.000269152 -3.61076e-13 0.0016863 0.999999 -0.000162115 -1.3048e-13 0.00101569 1 -9.87893e-05 -4.78776e-14 0.000618938 1 -6.09186e-05 -1.75971e-14 0.000381669 1 -3.80218e-05 -6.24914e-15 0.000238215 1 -2.40233e-05 -1.91545e-15 0.000150512 1 -1.53673e-05 -2.43678e-16 9.62796e-05 1 -9.9521e-06 3.92943e-16 6.23522e-05 1 -6.52343e-06 6.16927e-16 4.08708e-05 1 -4.32513e-06 6.72253e-16 2.70979e-05 1 -2.89674e-06 6.56892e-16 1.81487e-05 1 -1.95509e-06 6.1116e-16 1.22491e-05 1 -1.32436e-06 5.5206e-16 8.29741e-06 1 -8.94412e-07 4.86947e-16 5.60369e-06 1 -5.95882e-07 4.19083e-16 3.73333e-06 1 -3.85046e-07 3.49948e-16 2.4124e-06 1 -2.34627e-07 2.80234e-16 1.46999e-06 1 -1.28099e-07 2.10267e-16 8.0257e-07 1 -5.60893e-08 1.40201e-16 3.51412e-07 1 -1.39556e-08 7.01034e-17 8.74352e-08"
        #     ),
        #     qvel=_parse_float_list(
        #         "-1.5665e-10 -4.19492e-11 3.25121e-11 1.14502e-09 4.34549e-09 -1.38611e-09 9.13508e-09 6.81017e-09 1.21335e-08 -5.48219e-08 8.98244e-09 6.69328e-09 1.28613e-08 -5.54774e-08 -3.20096e-08 -1.08279e-08 -2.59575e-08 1.01681e-08 6.2781e-08 1.1853e-09 -1.24528e-10 1.63355e-13 -2.25804e-11 -5.00409e-10 2.60871e-12 -9.07271e-11 -1.14217e-09 1.32464e-11 -2.07006e-10 -2.08891e-09 4.30119e-11 -3.78297e-10 -3.41681e-09 1.11691e-10 -6.17907e-10 -5.25081e-09 2.56595e-10 -9.47303e-10 -7.76871e-09 5.48696e-10 -1.39595e-09 -1.11761e-08 1.11545e-09 -1.99465e-09 -1.55969e-08 2.14649e-09 -2.75121e-09 -2.07904e-08 3.77936e-09 -3.59283e-09 -2.09687e-08 6.37227e-09 -2.66002e-09 -1.2093e-08 2.03217e-09 -1.5252e-09 -6.69773e-09 6.25029e-10 -8.43136e-10 -3.68855e-09 1.92764e-10 -4.63927e-10 -2.04269e-09 6.03678e-11 -2.56781e-10 -1.14169e-09 1.9278e-11 -1.43456e-10 -6.44826e-10 6.28772e-12 -8.09922e-11 -3.68239e-10 2.09623e-12 -4.62342e-11 -2.12693e-10 7.14701e-13 -2.66939e-11 -1.24287e-10 2.49291e-13 -1.55924e-11 -7.34945e-11 8.89699e-14 -9.21646e-12 -4.39895e-11 3.24767e-14 -5.51406e-12 -2.6657e-11 1.2106e-14 -3.34001e-12 -1.63585e-11 4.58802e-15 -2.04874e-12 -1.01679e-11 1.7486e-15 -1.27293e-12 -6.40235e-12 6.52264e-16 -8.01167e-13 -4.08399e-12 2.20833e-16 -5.10857e-13 -2.63868e-12 4.91661e-17 -3.29945e-13 -1.72588e-12 -1.85205e-17 -2.15748e-13 -1.14141e-12 -4.35543e-17 -1.4265e-13 -7.6157e-13 -5.06536e-17 -9.51783e-14 -5.10654e-13 -5.00236e-17 -6.38197e-14 -3.41886e-13 -4.59795e-17 -4.2733e-14 -2.26165e-13 -4.03913e-17 -2.82832e-14 -1.45351e-13 -3.40893e-17 -1.81846e-14 -8.82279e-14 -2.74524e-17 -1.10466e-14 -4.8052e-14 -2.06582e-17 -6.0172e-15 -2.10118e-14 -1.3793e-17 -2.63466e-15 -5.22509e-15 -6.90011e-18 -6.55429e-16"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )
        # scene.add_key(
        #     name="bent",
        #     time=24.352,
        #     qpos=_parse_float_list(
        #         "1.36004e-07 0.4 0.399964 0.0007742 1 5.10054e-07 1.49254e-07 0.705048 0.000203406 0.694227 -0.701813 0.705048 0.000206316 0.694225 -0.701839 0.00218251 0.242195 0.0903311 0.648398 0.407118 -0.547032 -0.338514 0.999999 -0.000209236 -5.9603e-06 0.00137112 0.999984 -0.000889182 -6.22239e-06 0.0055497 0.999918 -0.00205618 -6.35398e-06 0.0126689 0.999725 -0.00377208 -8.02477e-06 0.0231287 0.999269 -0.00616777 -1.55575e-05 0.0377267 0.998285 -0.00946107 -3.85047e-05 0.0577762 0.996269 -0.0139576 -9.40666e-05 0.0851656 0.992304 -0.0200178 -0.000213389 0.122196 0.984944 -0.027851 -0.000441306 0.170613 0.972913 -0.0369859 -0.000644665 0.228194 0.992152 -0.0160525 -2.53628e-06 0.124004 0.997345 -0.00897996 3.58934e-06 0.0722653 0.999234 -0.00439477 2.32971e-05 0.038896 0.999853 -0.00136528 3.92105e-05 0.0171049 0.999997 0.000728434 4.95125e-05 0.00236156 0.999964 0.0022576 5.41419e-05 -0.00813067 0.999865 0.00345771 5.31513e-05 -0.01608 0.999737 0.00444589 4.79055e-05 -0.0224962 0.999596 0.00529425 4.19877e-05 -0.0279421 0.999447 0.00601181 3.88485e-05 -0.0326958 0.9993 0.00659028 4.14671e-05 -0.0368293 0.999165 0.00702228 4.61888e-05 -0.0402504 0.999059 0.00722337 5.53289e-05 -0.0427775 0.998997 0.00713513 7.32397e-05 -0.0441983 0.999003 0.00667051 0.000101378 -0.0441411 0.999092 0.00571423 0.000140736 -0.0422308 0.999269 0.00413733 0.000190595 -0.0379978 0.999519 0.00184447 0.000230379 -0.0309477 0.999789 -0.001437 0.000242094 -0.020502 0.99996 -0.00691183 0.000194048 -0.00559836 0.999964 -0.00834044 0.000130477 -0.00150954 0.999999 0.00126059 9.73332e-05 0.000228067 0.999998 0.00198394 7.54568e-05 0.000355323 0.999996 0.00264783 5.52334e-05 0.000496258 0.999994 0.00346091 3.77568e-05 0.000645757 0.999993 0.00359892 2.42041e-05 0.000671576 0.999995 0.00323031 1.43891e-05 0.000600423 0.999997 0.00238307 8.07263e-06 0.000443923 0.999999 0.0012134 3.82412e-06 0.000236578"
        #     ),
        #     qvel=_parse_float_list(
        #         "5.33691e-08 -1.7121e-06 2.82668e-07 -0.000344996 7.06013e-06 8.05372e-06 -6.21541e-07 -1.57996e-06 -3.18236e-06 6.92397e-06 -3.92801e-07 5.22924e-07 1.92425e-06 3.2255e-07 -0.000133793 -9.54591e-05 6.81012e-06 -0.000563511 -0.00187129 0.000285827 -2.03781e-05 8.73249e-06 1.61228e-05 -1.05112e-05 9.23232e-06 3.71629e-06 1.84928e-05 8.99986e-06 -1.26548e-05 6.08968e-05 8.69917e-06 -2.58534e-05 0.000119548 1.0802e-05 -4.5903e-05 0.000202842 2.13483e-05 -8.46529e-05 0.000313088 5.09971e-05 -0.000138879 0.000453326 0.000121336 -0.000205302 0.000581225 0.000278285 -0.000217178 0.000407486 0.000347011 0.00102666 -0.000402769 0.000142698 0.00133963 -0.00031519 0.000163477 0.00120856 -0.000206618 0.000145187 0.000910056 -9.71032e-05 0.000124652 0.000546469 -4.32525e-06 0.000110176 0.000191035 8.04691e-05 0.00010408 -0.000160049 0.000142971 0.000106742 -0.000496744 0.0002025 0.000116364 -0.000815937 0.000250701 0.000127388 -0.00111774 0.000307838 0.000135839 -0.00138428 0.000369907 0.000138919 -0.00159516 0.000392448 0.00014004 -0.00174696 0.000416441 0.000137579 -0.00179739 0.000418147 0.00012681 -0.00162385 0.000394045 0.000105739 -0.00125535 0.000359531 7.3441e-05 -0.000659565 0.00031901 2.85588e-05 0.000125493 0.00017826 -1.19449e-05 0.00113713 3.36564e-05 -3.36494e-05 0.00248928 -0.000718188 -6.86367e-05 0.00415896 -0.000243119 -5.72125e-05 0.00171137 -0.000412314 -3.68994e-05 0.00186791 -0.000255963 -2.63524e-05 0.00121703 -0.000221835 -1.84138e-05 0.0010312 -0.000101426 -1.31763e-05 0.000500452 -0.000119414 -9.45405e-06 0.000316422 -0.000105454 -5.59891e-06 0.000335345 5.1126e-05 -5.51806e-06 -0.000347723 -5.81844e-06 -2.56083e-06 6.5512e-06"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )

        scene.add_key(
            name="init",
            time=0,
            qpos=_parse_float_list(
                "-0.00178983 0.265561 0.0928174 0.755305 0.655325 0.00531633 0.00594599 0.999999 -0.0011596 -1.50973e-07 -2.58863e-05 0.999989 -0.00468051 -2.6357e-06 -0.000102329 0.99994 -0.0109141 -1.68553e-05 -0.000250369 0.999782 -0.0208733 -6.92685e-05 -0.000541302 0.999326 -0.0367043 -0.000226958 -0.0011333 0.998046 -0.0624409 -0.00066422 -0.00232277 0.994444 -0.105151 -0.00184404 -0.00453926 0.98436 -0.175926 -0.00491532 -0.00784959 0.957916 -0.28665 -0.0119359 -0.00926809"
            ),
            qvel=_parse_float_list(
                "-0.000999776 2.95313e-05 -4.62839e-05 -0.000177692 -0.0170984 -0.00223061 8.27751e-06 8.36807e-08 2.11904e-05 2.06721e-05 1.48882e-06 8.25273e-05 3.06709e-05 9.66566e-06 0.000195584 4.77091e-05 3.86553e-05 0.000403653 9.17848e-05 0.000117716 0.000802283 0.000176729 0.000305613 0.00156849 0.000261393 0.000725882 0.00297842 0.000114886 0.001673 0.00529442 -0.000944918 0.00409848 0.00808618"
            ),
            mpos=_parse_float_list("0 0.4 0.4"),
            mquat=_parse_float_list("1 0 0 0"),
        )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _init_mocap_weld_3(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        # Do not load keyframes from file; use keys defined in this init.

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
        self.pipe_length = 0.125 / 2  # 121 mm from schematic
        # self.pipe_length = 0.121 / 2  # 121 mm from schematic
        # pipe_length = 0.09

        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            resolution=20,
            # rgba=[0.2, 0.2, 0.2, 1],
        )
        # pip = pipe(inner_radius=0.0385, outer_radius=0.0435, length=0.075)

        mesh_path = Path("assets/DS00240017_E.obj")
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

        twist = 60000.0 * 5
        bend = 10000000.0 * 5

        cable = mjx_cable(
            twist=twist,
            bend=bend,
            size=0.4,
            initial="free",
            # twist=twist, bend=bend, segment_size=0.002, size=0.4, initial="free"
        )

        cable.body("cable:Bfirst").add_site(
            name="keypoint_3", group=1, rgba=[0, 1, 0, 1], pos=[0, 0, 0]
        )
        cable.body("cable:B2").add_site(
            name="keypoint_2", group=1, rgba=[1, 1, 0, 1], pos=[0, 0, 0]
        )
        cable.body("cable:B4").add_site(
            name="keypoint_1", group=1, rgba=[1, 0, 0, 1], pos=[0, 0, 0]
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
            rgba=[1, 1, 1, 0.2],
        )
        s_mocap_1 = mocap.add_site(name="mocap_site_1", pos=[0, -0.01, 0])
        s_mocap_2 = mocap.add_site(name="mocap_site_2", pos=[0, 0.01, 0])
        s_mocap_3 = mocap.add_site(name="mocap_site_3", pos=[0.01, 0, 0])

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

        scene.add_equality(
            name="mocap_cable_weld_1",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_1",
            name2="cable_weld_site_1",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
            solimp=[0.95, 0.99, 0.001, 0.1, 6],
        )
        scene.add_equality(
            name="mocap_cable_weld_2",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_2",
            name2="cable_weld_site_2",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
            solimp=[0.95, 0.99, 0.001, 0.1, 6],
        )
        scene.add_equality(
            name="mocap_cable_weld_3",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site_3",
            name2="cable_weld_site_3",
            # Keep the current relative pose at creation time.
            solref=[0.00000001, 1],
            solimp=[0.95, 0.99, 0.001, 0.1, 6],
        )

        # scene.add_key(
        #     name="bent",
        #     time=112.672,
        #     qpos=_parse_float_list(
        #         "-7.0802e-08 0.4 0.399964 0.000799541 1 2.99191e-07 4.02526e-06 0.705055 0.000209212 0.694236 -0.701859 0.705055 0.000211376 0.694236 -0.701882 0.00180079 0.242155 0.0902499 0.647417 0.405906 -0.548272 -0.339838 0.999999 -0.00022072 1.40805e-13 0.00138287 0.999984 -0.000886664 2.55741e-12 0.00555516 0.999918 -0.0020218 1.66775e-11 0.012667 0.999726 -0.00369068 6.80645e-11 0.023123 0.99927 -0.00601949 2.13093e-10 0.0377135 0.998289 -0.00921636 5.66729e-10 0.0577427 0.99628 -0.0135827 1.35572e-09 0.0850987 0.992329 -0.0194854 3.0116e-09 0.122081 0.984986 -0.0272096 6.29496e-09 0.170475 0.972771 -0.0365304 1.23275e-08 0.228872 0.967532 -0.0398372 -8.39201e-09 0.24959 0.98963 -0.0226396 -2.65502e-09 0.141842 0.996766 -0.0126654 -8.21613e-10 0.0793519 0.998985 -0.00709956 -2.5615e-10 0.0444804 0.999676 -0.00401032 -8.12448e-11 0.0251256 0.999895 -0.00228713 -2.63037e-11 0.0143294 0.999965 -0.00131791 -8.70428e-12 0.00825702 0.999988 -0.000767587 -2.94588e-12 0.00480911 0.999996 -0.000451992 -1.01991e-12 0.00283183 0.999999 -0.000269152 -3.61076e-13 0.0016863 0.999999 -0.000162115 -1.3048e-13 0.00101569 1 -9.87893e-05 -4.78776e-14 0.000618938 1 -6.09186e-05 -1.75971e-14 0.000381669 1 -3.80218e-05 -6.24914e-15 0.000238215 1 -2.40233e-05 -1.91545e-15 0.000150512 1 -1.53673e-05 -2.43678e-16 9.62796e-05 1 -9.9521e-06 3.92943e-16 6.23522e-05 1 -6.52343e-06 6.16927e-16 4.08708e-05 1 -4.32513e-06 6.72253e-16 2.70979e-05 1 -2.89674e-06 6.56892e-16 1.81487e-05 1 -1.95509e-06 6.1116e-16 1.22491e-05 1 -1.32436e-06 5.5206e-16 8.29741e-06 1 -8.94412e-07 4.86947e-16 5.60369e-06 1 -5.95882e-07 4.19083e-16 3.73333e-06 1 -3.85046e-07 3.49948e-16 2.4124e-06 1 -2.34627e-07 2.80234e-16 1.46999e-06 1 -1.28099e-07 2.10267e-16 8.0257e-07 1 -5.60893e-08 1.40201e-16 3.51412e-07 1 -1.39556e-08 7.01034e-17 8.74352e-08"
        #     ),
        #     qvel=_parse_float_list(
        #         "-1.5665e-10 -4.19492e-11 3.25121e-11 1.14502e-09 4.34549e-09 -1.38611e-09 9.13508e-09 6.81017e-09 1.21335e-08 -5.48219e-08 8.98244e-09 6.69328e-09 1.28613e-08 -5.54774e-08 -3.20096e-08 -1.08279e-08 -2.59575e-08 1.01681e-08 6.2781e-08 1.1853e-09 -1.24528e-10 1.63355e-13 -2.25804e-11 -5.00409e-10 2.60871e-12 -9.07271e-11 -1.14217e-09 1.32464e-11 -2.07006e-10 -2.08891e-09 4.30119e-11 -3.78297e-10 -3.41681e-09 1.11691e-10 -6.17907e-10 -5.25081e-09 2.56595e-10 -9.47303e-10 -7.76871e-09 5.48696e-10 -1.39595e-09 -1.11761e-08 1.11545e-09 -1.99465e-09 -1.55969e-08 2.14649e-09 -2.75121e-09 -2.07904e-08 3.77936e-09 -3.59283e-09 -2.09687e-08 6.37227e-09 -2.66002e-09 -1.2093e-08 2.03217e-09 -1.5252e-09 -6.69773e-09 6.25029e-10 -8.43136e-10 -3.68855e-09 1.92764e-10 -4.63927e-10 -2.04269e-09 6.03678e-11 -2.56781e-10 -1.14169e-09 1.9278e-11 -1.43456e-10 -6.44826e-10 6.28772e-12 -8.09922e-11 -3.68239e-10 2.09623e-12 -4.62342e-11 -2.12693e-10 7.14701e-13 -2.66939e-11 -1.24287e-10 2.49291e-13 -1.55924e-11 -7.34945e-11 8.89699e-14 -9.21646e-12 -4.39895e-11 3.24767e-14 -5.51406e-12 -2.6657e-11 1.2106e-14 -3.34001e-12 -1.63585e-11 4.58802e-15 -2.04874e-12 -1.01679e-11 1.7486e-15 -1.27293e-12 -6.40235e-12 6.52264e-16 -8.01167e-13 -4.08399e-12 2.20833e-16 -5.10857e-13 -2.63868e-12 4.91661e-17 -3.29945e-13 -1.72588e-12 -1.85205e-17 -2.15748e-13 -1.14141e-12 -4.35543e-17 -1.4265e-13 -7.6157e-13 -5.06536e-17 -9.51783e-14 -5.10654e-13 -5.00236e-17 -6.38197e-14 -3.41886e-13 -4.59795e-17 -4.2733e-14 -2.26165e-13 -4.03913e-17 -2.82832e-14 -1.45351e-13 -3.40893e-17 -1.81846e-14 -8.82279e-14 -2.74524e-17 -1.10466e-14 -4.8052e-14 -2.06582e-17 -6.0172e-15 -2.10118e-14 -1.3793e-17 -2.63466e-15 -5.22509e-15 -6.90011e-18 -6.55429e-16"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )
        # scene.add_key(
        #     name="bent",
        #     time=24.352,
        #     qpos=_parse_float_list(
        #         "1.36004e-07 0.4 0.399964 0.0007742 1 5.10054e-07 1.49254e-07 0.705048 0.000203406 0.694227 -0.701813 0.705048 0.000206316 0.694225 -0.701839 0.00218251 0.242195 0.0903311 0.648398 0.407118 -0.547032 -0.338514 0.999999 -0.000209236 -5.9603e-06 0.00137112 0.999984 -0.000889182 -6.22239e-06 0.0055497 0.999918 -0.00205618 -6.35398e-06 0.0126689 0.999725 -0.00377208 -8.02477e-06 0.0231287 0.999269 -0.00616777 -1.55575e-05 0.0377267 0.998285 -0.00946107 -3.85047e-05 0.0577762 0.996269 -0.0139576 -9.40666e-05 0.0851656 0.992304 -0.0200178 -0.000213389 0.122196 0.984944 -0.027851 -0.000441306 0.170613 0.972913 -0.0369859 -0.000644665 0.228194 0.992152 -0.0160525 -2.53628e-06 0.124004 0.997345 -0.00897996 3.58934e-06 0.0722653 0.999234 -0.00439477 2.32971e-05 0.038896 0.999853 -0.00136528 3.92105e-05 0.0171049 0.999997 0.000728434 4.95125e-05 0.00236156 0.999964 0.0022576 5.41419e-05 -0.00813067 0.999865 0.00345771 5.31513e-05 -0.01608 0.999737 0.00444589 4.79055e-05 -0.0224962 0.999596 0.00529425 4.19877e-05 -0.0279421 0.999447 0.00601181 3.88485e-05 -0.0326958 0.9993 0.00659028 4.14671e-05 -0.0368293 0.999165 0.00702228 4.61888e-05 -0.0402504 0.999059 0.00722337 5.53289e-05 -0.0427775 0.998997 0.00713513 7.32397e-05 -0.0441983 0.999003 0.00667051 0.000101378 -0.0441411 0.999092 0.00571423 0.000140736 -0.0422308 0.999269 0.00413733 0.000190595 -0.0379978 0.999519 0.00184447 0.000230379 -0.0309477 0.999789 -0.001437 0.000242094 -0.020502 0.99996 -0.00691183 0.000194048 -0.00559836 0.999964 -0.00834044 0.000130477 -0.00150954 0.999999 0.00126059 9.73332e-05 0.000228067 0.999998 0.00198394 7.54568e-05 0.000355323 0.999996 0.00264783 5.52334e-05 0.000496258 0.999994 0.00346091 3.77568e-05 0.000645757 0.999993 0.00359892 2.42041e-05 0.000671576 0.999995 0.00323031 1.43891e-05 0.000600423 0.999997 0.00238307 8.07263e-06 0.000443923 0.999999 0.0012134 3.82412e-06 0.000236578"
        #     ),
        #     qvel=_parse_float_list(
        #         "5.33691e-08 -1.7121e-06 2.82668e-07 -0.000344996 7.06013e-06 8.05372e-06 -6.21541e-07 -1.57996e-06 -3.18236e-06 6.92397e-06 -3.92801e-07 5.22924e-07 1.92425e-06 3.2255e-07 -0.000133793 -9.54591e-05 6.81012e-06 -0.000563511 -0.00187129 0.000285827 -2.03781e-05 8.73249e-06 1.61228e-05 -1.05112e-05 9.23232e-06 3.71629e-06 1.84928e-05 8.99986e-06 -1.26548e-05 6.08968e-05 8.69917e-06 -2.58534e-05 0.000119548 1.0802e-05 -4.5903e-05 0.000202842 2.13483e-05 -8.46529e-05 0.000313088 5.09971e-05 -0.000138879 0.000453326 0.000121336 -0.000205302 0.000581225 0.000278285 -0.000217178 0.000407486 0.000347011 0.00102666 -0.000402769 0.000142698 0.00133963 -0.00031519 0.000163477 0.00120856 -0.000206618 0.000145187 0.000910056 -9.71032e-05 0.000124652 0.000546469 -4.32525e-06 0.000110176 0.000191035 8.04691e-05 0.00010408 -0.000160049 0.000142971 0.000106742 -0.000496744 0.0002025 0.000116364 -0.000815937 0.000250701 0.000127388 -0.00111774 0.000307838 0.000135839 -0.00138428 0.000369907 0.000138919 -0.00159516 0.000392448 0.00014004 -0.00174696 0.000416441 0.000137579 -0.00179739 0.000418147 0.00012681 -0.00162385 0.000394045 0.000105739 -0.00125535 0.000359531 7.3441e-05 -0.000659565 0.00031901 2.85588e-05 0.000125493 0.00017826 -1.19449e-05 0.00113713 3.36564e-05 -3.36494e-05 0.00248928 -0.000718188 -6.86367e-05 0.00415896 -0.000243119 -5.72125e-05 0.00171137 -0.000412314 -3.68994e-05 0.00186791 -0.000255963 -2.63524e-05 0.00121703 -0.000221835 -1.84138e-05 0.0010312 -0.000101426 -1.31763e-05 0.000500452 -0.000119414 -9.45405e-06 0.000316422 -0.000105454 -5.59891e-06 0.000335345 5.1126e-05 -5.51806e-06 -0.000347723 -5.81844e-06 -2.56083e-06 6.5512e-06"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )

        scene.add_key(
            name="init",
            time=0,
            qpos=_parse_float_list(
                "-0.00178983 0.265561 0.0928174 0.755305 0.655325 0.00531633 0.00594599 0.999999 -0.0011596 -1.50973e-07 -2.58863e-05 0.999989 -0.00468051 -2.6357e-06 -0.000102329 0.99994 -0.0109141 -1.68553e-05 -0.000250369 0.999782 -0.0208733 -6.92685e-05 -0.000541302 0.999326 -0.0367043 -0.000226958 -0.0011333 0.998046 -0.0624409 -0.00066422 -0.00232277 0.994444 -0.105151 -0.00184404 -0.00453926 0.98436 -0.175926 -0.00491532 -0.00784959 0.957916 -0.28665 -0.0119359 -0.00926809"
            ),
            qvel=_parse_float_list(
                "-0.000999776 2.95313e-05 -4.62839e-05 -0.000177692 -0.0170984 -0.00223061 8.27751e-06 8.36807e-08 2.11904e-05 2.06721e-05 1.48882e-06 8.25273e-05 3.06709e-05 9.66566e-06 0.000195584 4.77091e-05 3.86553e-05 0.000403653 9.17848e-05 0.000117716 0.000802283 0.000176729 0.000305613 0.00156849 0.000261393 0.000725882 0.00297842 0.000114886 0.001673 0.00529442 -0.000944918 0.00409848 0.00808618"
            ),
            mpos=_parse_float_list("0 0.4 0.4"),
            mquat=_parse_float_list("1 0 0 0"),
        )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _init_mocap_weld_4(self) -> mj.MjModel:
        # root
        _HERE = Path(__file__).parent.parent
        # scene path

        scene = empty()
        # scene = mj.MjSpec().from_file((_HERE / "scenes/empty.xml").as_posix())

        # Do not load keyframes from file; use keys defined in this init.

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
        self.pipe_length = 0.125 / 2  # 121 mm from schematic
        # self.pipe_length = 0.121 / 2  # 121 mm from schematic
        # pipe_length = 0.09

        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            resolution=20,
            # rgba=[0.2, 0.2, 0.2, 1],
        )
        # pip = pipe(inner_radius=0.0385, outer_radius=0.0435, length=0.075)

        mesh_path = Path("assets/DS00240017_E.obj")
        mesh_spec = add_mesh(
            mesh_path=mesh_path,
            scale=0.001,
            euler=[0, 1.57, 0],
            rgba=[0.2, 0.2, 0.2, 0.1],
            # rgba=[0.2, 0.2, 0.2, 1],
            # rgba=[0.2, 0.2, 0.2, 0.2],
            contype=0,
            conaffinity=0,
        )
        mesh_body = mesh_spec.worldbody.first_body()
        scene.worldbody.add_frame(
            pos=[0, -self.pipe_length - 0.01, 0.1], euler=[0, 0, 0]
        ).attach_body(mesh_body)

        keypoint_height = self.pipe_inner_radius / 2
        # keypoint_height = self.pipe_inner_radius - (
        #     self.pipe_outer_radius - self.pipe_inner_radius
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

        twist = 60000.0 * 5
        bend = 10000000.0 * 5

        cable = mjx_cable(
            twist=twist,
            bend=bend,
            size=0.4,
            initial="free",
            # twist=twist, bend=bend, segment_size=0.002, size=0.4, initial="free"
        )

        cable.body("cable:Bfirst").add_site(
            name="keypoint_3", group=1, rgba=[0, 1, 0, 1], pos=[0, 0, 0]
        )
        cable.body("cable:B2").add_site(
            name="keypoint_2", group=1, rgba=[1, 1, 0, 1], pos=[0, 0, 0]
        )
        cable.body("cable:B4").add_site(
            name="keypoint_1", group=1, rgba=[1, 0, 0, 1], pos=[0, 0, 0]
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

        # solref = [0.0005, 1.0]
        # solimp = [0.999, 0.999999, 2e-5, 0.0, 20]

        # solimp = [0.95, 0.99, 0.001, 0.001, 6]
        # solref = [0.00000001, 1]
        # solimp = [0.95, 0.99, 0.001, 0.1, 6]
        # solimp = [0.95, 0.99, 0.001, 0.1, 6]

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

        # scene.add_key(
        #     name="bent",
        #     time=112.672,
        #     qpos=_parse_float_list(
        #         "-7.0802e-08 0.4 0.399964 0.000799541 1 2.99191e-07 4.02526e-06 0.705055 0.000209212 0.694236 -0.701859 0.705055 0.000211376 0.694236 -0.701882 0.00180079 0.242155 0.0902499 0.647417 0.405906 -0.548272 -0.339838 0.999999 -0.00022072 1.40805e-13 0.00138287 0.999984 -0.000886664 2.55741e-12 0.00555516 0.999918 -0.0020218 1.66775e-11 0.012667 0.999726 -0.00369068 6.80645e-11 0.023123 0.99927 -0.00601949 2.13093e-10 0.0377135 0.998289 -0.00921636 5.66729e-10 0.0577427 0.99628 -0.0135827 1.35572e-09 0.0850987 0.992329 -0.0194854 3.0116e-09 0.122081 0.984986 -0.0272096 6.29496e-09 0.170475 0.972771 -0.0365304 1.23275e-08 0.228872 0.967532 -0.0398372 -8.39201e-09 0.24959 0.98963 -0.0226396 -2.65502e-09 0.141842 0.996766 -0.0126654 -8.21613e-10 0.0793519 0.998985 -0.00709956 -2.5615e-10 0.0444804 0.999676 -0.00401032 -8.12448e-11 0.0251256 0.999895 -0.00228713 -2.63037e-11 0.0143294 0.999965 -0.00131791 -8.70428e-12 0.00825702 0.999988 -0.000767587 -2.94588e-12 0.00480911 0.999996 -0.000451992 -1.01991e-12 0.00283183 0.999999 -0.000269152 -3.61076e-13 0.0016863 0.999999 -0.000162115 -1.3048e-13 0.00101569 1 -9.87893e-05 -4.78776e-14 0.000618938 1 -6.09186e-05 -1.75971e-14 0.000381669 1 -3.80218e-05 -6.24914e-15 0.000238215 1 -2.40233e-05 -1.91545e-15 0.000150512 1 -1.53673e-05 -2.43678e-16 9.62796e-05 1 -9.9521e-06 3.92943e-16 6.23522e-05 1 -6.52343e-06 6.16927e-16 4.08708e-05 1 -4.32513e-06 6.72253e-16 2.70979e-05 1 -2.89674e-06 6.56892e-16 1.81487e-05 1 -1.95509e-06 6.1116e-16 1.22491e-05 1 -1.32436e-06 5.5206e-16 8.29741e-06 1 -8.94412e-07 4.86947e-16 5.60369e-06 1 -5.95882e-07 4.19083e-16 3.73333e-06 1 -3.85046e-07 3.49948e-16 2.4124e-06 1 -2.34627e-07 2.80234e-16 1.46999e-06 1 -1.28099e-07 2.10267e-16 8.0257e-07 1 -5.60893e-08 1.40201e-16 3.51412e-07 1 -1.39556e-08 7.01034e-17 8.74352e-08"
        #     ),
        #     qvel=_parse_float_list(
        #         "-1.5665e-10 -4.19492e-11 3.25121e-11 1.14502e-09 4.34549e-09 -1.38611e-09 9.13508e-09 6.81017e-09 1.21335e-08 -5.48219e-08 8.98244e-09 6.69328e-09 1.28613e-08 -5.54774e-08 -3.20096e-08 -1.08279e-08 -2.59575e-08 1.01681e-08 6.2781e-08 1.1853e-09 -1.24528e-10 1.63355e-13 -2.25804e-11 -5.00409e-10 2.60871e-12 -9.07271e-11 -1.14217e-09 1.32464e-11 -2.07006e-10 -2.08891e-09 4.30119e-11 -3.78297e-10 -3.41681e-09 1.11691e-10 -6.17907e-10 -5.25081e-09 2.56595e-10 -9.47303e-10 -7.76871e-09 5.48696e-10 -1.39595e-09 -1.11761e-08 1.11545e-09 -1.99465e-09 -1.55969e-08 2.14649e-09 -2.75121e-09 -2.07904e-08 3.77936e-09 -3.59283e-09 -2.09687e-08 6.37227e-09 -2.66002e-09 -1.2093e-08 2.03217e-09 -1.5252e-09 -6.69773e-09 6.25029e-10 -8.43136e-10 -3.68855e-09 1.92764e-10 -4.63927e-10 -2.04269e-09 6.03678e-11 -2.56781e-10 -1.14169e-09 1.9278e-11 -1.43456e-10 -6.44826e-10 6.28772e-12 -8.09922e-11 -3.68239e-10 2.09623e-12 -4.62342e-11 -2.12693e-10 7.14701e-13 -2.66939e-11 -1.24287e-10 2.49291e-13 -1.55924e-11 -7.34945e-11 8.89699e-14 -9.21646e-12 -4.39895e-11 3.24767e-14 -5.51406e-12 -2.6657e-11 1.2106e-14 -3.34001e-12 -1.63585e-11 4.58802e-15 -2.04874e-12 -1.01679e-11 1.7486e-15 -1.27293e-12 -6.40235e-12 6.52264e-16 -8.01167e-13 -4.08399e-12 2.20833e-16 -5.10857e-13 -2.63868e-12 4.91661e-17 -3.29945e-13 -1.72588e-12 -1.85205e-17 -2.15748e-13 -1.14141e-12 -4.35543e-17 -1.4265e-13 -7.6157e-13 -5.06536e-17 -9.51783e-14 -5.10654e-13 -5.00236e-17 -6.38197e-14 -3.41886e-13 -4.59795e-17 -4.2733e-14 -2.26165e-13 -4.03913e-17 -2.82832e-14 -1.45351e-13 -3.40893e-17 -1.81846e-14 -8.82279e-14 -2.74524e-17 -1.10466e-14 -4.8052e-14 -2.06582e-17 -6.0172e-15 -2.10118e-14 -1.3793e-17 -2.63466e-15 -5.22509e-15 -6.90011e-18 -6.55429e-16"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )
        # scene.add_key(
        #     name="bent",
        #     time=24.352,
        #     qpos=_parse_float_list(
        #         "1.36004e-07 0.4 0.399964 0.0007742 1 5.10054e-07 1.49254e-07 0.705048 0.000203406 0.694227 -0.701813 0.705048 0.000206316 0.694225 -0.701839 0.00218251 0.242195 0.0903311 0.648398 0.407118 -0.547032 -0.338514 0.999999 -0.000209236 -5.9603e-06 0.00137112 0.999984 -0.000889182 -6.22239e-06 0.0055497 0.999918 -0.00205618 -6.35398e-06 0.0126689 0.999725 -0.00377208 -8.02477e-06 0.0231287 0.999269 -0.00616777 -1.55575e-05 0.0377267 0.998285 -0.00946107 -3.85047e-05 0.0577762 0.996269 -0.0139576 -9.40666e-05 0.0851656 0.992304 -0.0200178 -0.000213389 0.122196 0.984944 -0.027851 -0.000441306 0.170613 0.972913 -0.0369859 -0.000644665 0.228194 0.992152 -0.0160525 -2.53628e-06 0.124004 0.997345 -0.00897996 3.58934e-06 0.0722653 0.999234 -0.00439477 2.32971e-05 0.038896 0.999853 -0.00136528 3.92105e-05 0.0171049 0.999997 0.000728434 4.95125e-05 0.00236156 0.999964 0.0022576 5.41419e-05 -0.00813067 0.999865 0.00345771 5.31513e-05 -0.01608 0.999737 0.00444589 4.79055e-05 -0.0224962 0.999596 0.00529425 4.19877e-05 -0.0279421 0.999447 0.00601181 3.88485e-05 -0.0326958 0.9993 0.00659028 4.14671e-05 -0.0368293 0.999165 0.00702228 4.61888e-05 -0.0402504 0.999059 0.00722337 5.53289e-05 -0.0427775 0.998997 0.00713513 7.32397e-05 -0.0441983 0.999003 0.00667051 0.000101378 -0.0441411 0.999092 0.00571423 0.000140736 -0.0422308 0.999269 0.00413733 0.000190595 -0.0379978 0.999519 0.00184447 0.000230379 -0.0309477 0.999789 -0.001437 0.000242094 -0.020502 0.99996 -0.00691183 0.000194048 -0.00559836 0.999964 -0.00834044 0.000130477 -0.00150954 0.999999 0.00126059 9.73332e-05 0.000228067 0.999998 0.00198394 7.54568e-05 0.000355323 0.999996 0.00264783 5.52334e-05 0.000496258 0.999994 0.00346091 3.77568e-05 0.000645757 0.999993 0.00359892 2.42041e-05 0.000671576 0.999995 0.00323031 1.43891e-05 0.000600423 0.999997 0.00238307 8.07263e-06 0.000443923 0.999999 0.0012134 3.82412e-06 0.000236578"
        #     ),
        #     qvel=_parse_float_list(
        #         "5.33691e-08 -1.7121e-06 2.82668e-07 -0.000344996 7.06013e-06 8.05372e-06 -6.21541e-07 -1.57996e-06 -3.18236e-06 6.92397e-06 -3.92801e-07 5.22924e-07 1.92425e-06 3.2255e-07 -0.000133793 -9.54591e-05 6.81012e-06 -0.000563511 -0.00187129 0.000285827 -2.03781e-05 8.73249e-06 1.61228e-05 -1.05112e-05 9.23232e-06 3.71629e-06 1.84928e-05 8.99986e-06 -1.26548e-05 6.08968e-05 8.69917e-06 -2.58534e-05 0.000119548 1.0802e-05 -4.5903e-05 0.000202842 2.13483e-05 -8.46529e-05 0.000313088 5.09971e-05 -0.000138879 0.000453326 0.000121336 -0.000205302 0.000581225 0.000278285 -0.000217178 0.000407486 0.000347011 0.00102666 -0.000402769 0.000142698 0.00133963 -0.00031519 0.000163477 0.00120856 -0.000206618 0.000145187 0.000910056 -9.71032e-05 0.000124652 0.000546469 -4.32525e-06 0.000110176 0.000191035 8.04691e-05 0.00010408 -0.000160049 0.000142971 0.000106742 -0.000496744 0.0002025 0.000116364 -0.000815937 0.000250701 0.000127388 -0.00111774 0.000307838 0.000135839 -0.00138428 0.000369907 0.000138919 -0.00159516 0.000392448 0.00014004 -0.00174696 0.000416441 0.000137579 -0.00179739 0.000418147 0.00012681 -0.00162385 0.000394045 0.000105739 -0.00125535 0.000359531 7.3441e-05 -0.000659565 0.00031901 2.85588e-05 0.000125493 0.00017826 -1.19449e-05 0.00113713 3.36564e-05 -3.36494e-05 0.00248928 -0.000718188 -6.86367e-05 0.00415896 -0.000243119 -5.72125e-05 0.00171137 -0.000412314 -3.68994e-05 0.00186791 -0.000255963 -2.63524e-05 0.00121703 -0.000221835 -1.84138e-05 0.0010312 -0.000101426 -1.31763e-05 0.000500452 -0.000119414 -9.45405e-06 0.000316422 -0.000105454 -5.59891e-06 0.000335345 5.1126e-05 -5.51806e-06 -0.000347723 -5.81844e-06 -2.56083e-06 6.5512e-06"
        #     ),
        #     ctrl=_parse_float_list("255"),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("0.000796327 1 0 0"),
        # )

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

        # scene.add_key(
        #     name="init",
        #     time=0,
        #     qpos=_parse_float_list(
        #         "-0.00178983 0.265561 0.0928174 0.755305 0.655325 0.00531633 0.00594599 0.999999 -0.0011596 -1.50973e-07 -2.58863e-05 0.999989 -0.00468051 -2.6357e-06 -0.000102329 0.99994 -0.0109141 -1.68553e-05 -0.000250369 0.999782 -0.0208733 -6.92685e-05 -0.000541302 0.999326 -0.0367043 -0.000226958 -0.0011333 0.998046 -0.0624409 -0.00066422 -0.00232277 0.994444 -0.105151 -0.00184404 -0.00453926 0.98436 -0.175926 -0.00491532 -0.00784959 0.957916 -0.28665 -0.0119359 -0.00926809"
        #     ),
        #     qvel=_parse_float_list(
        #         "-0.000999776 2.95313e-05 -4.62839e-05 -0.000177692 -0.0170984 -0.00223061 8.27751e-06 8.36807e-08 2.11904e-05 2.06721e-05 1.48882e-06 8.25273e-05 3.06709e-05 9.66566e-06 0.000195584 4.77091e-05 3.86553e-05 0.000403653 9.17848e-05 0.000117716 0.000802283 0.000176729 0.000305613 0.00156849 0.000261393 0.000725882 0.00297842 0.000114886 0.001673 0.00529442 -0.000944918 0.00409848 0.00808618"
        #     ),
        #     mpos=_parse_float_list("0 0.4 0.4"),
        #     mquat=_parse_float_list("1 0 0 0"),
        # )

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _post_init(self) -> None:
        self.target_ids = [self._mj_model.site(f"target_{i + 1}").id for i in range(3)]
        self.keypoint_ids = [
            self._mj_model.site(f"keypoint_{i + 1}").id for i in range(3)
        ]
        try:
            self.pipe_entry_id = self._mj_model.site("pipe_entry").id
            self.pipe_exit_id = self._mj_model.site("pipe_exit").id
        except Exception:
            self.pipe_entry_id = None
            self.pipe_exit_id = None
        # Mid-pipe segment (closer to the pipe center) for success checks.
        self.pipe_mid_start = jp.array(
            [1.35525272e-20, 6.24999802e-02, 9.99502296e-02]
        ) - jp.array([0, 0.04, 0])
        self.pipe_mid_end = jp.array(
            [-1.35525272e-20, -6.24999802e-02, 1.00049770e-01]
        ) + jp.array([0, 0.04, 0])
        self._mocap_id = None
        try:
            self._mocap_id = int(self._mj_model.body("mocap").mocapid)
            if self._mocap_id < 0:
                self._mocap_id = None
        except Exception:
            self._mocap_id = None
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

        # self._qpos0 = jp.array(np.load(self.keys["qpos"]))
        # self._qvel0 = jp.array(np.load(self.keys["qvel"]))
        # self._ctrl0 = jp.array(np.load(self.keys["ctrl"]))
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

        self.i = 0

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

        metrics = {}
        info = {"rng": rng, "step": jp.array(0)}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        jax.debug.print(
            "[pipe_insert_2_new] step={s} action={a}",
            s=state.info.get("step", 0),
            a=jp.max(action),
        )
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
            reward = -self._get_distance(data)

        obs = self._get_obs(data, state.info)

        rng, _ = jax.random.split(state.info["rng"])
        info = {**state.info}
        info["rng"] = rng
        info["step"] = info["step"] + 1

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

    def _points_inside_cylinder(
        self,
        points: jax.Array,
        axis_start: jax.Array,
        axis_end: jax.Array,
        radius: float,
        *,
        inclusive: bool = True,
    ) -> jax.Array:
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
        inside = self._points_inside_cylinder(
            keypoints,
            pipe_axis_start,
            pipe_axis_end,
            inner_radius,
            inclusive=inclusive,
        )
        return jp.all(inside)

    def _get_done(self, data: mjx.Data, info: dict) -> float:
        step_count = info["steps"] if "steps" in info else info["step"]
        if self.pipe_entry_id is None or self.pipe_exit_id is None:
            nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
            timeout = step_count >= self._episode_length
            done = nan_condition | timeout
            # jax.debug.print(
            #     "[pipe_insert_2_new] done check (no pipe ids) step={s} timeout={t} nan={n} done={d}",
            #     s=step_count,
            #     t=timeout,
            #     n=nan_condition,
            #     d=done,
            # )
            return done.astype(float)

        T_w_keypoint_2 = get_pose(
            self._mjx_model, data, self.keypoint_ids[1], obj_type=ObjType.SITE
        )
        keypoints = jp.stack(
            [
                T_w_keypoint_2.translation(),
            ],
            axis=0,
        )

        success = self._keypoints_within_pipe(
            keypoints=keypoints,
            pipe_axis_start=self.pipe_mid_start,
            pipe_axis_end=self.pipe_mid_end,
            inner_radius=self.pipe_inner_radius,
        )
        if self._grace_steps > 0:
            success = jp.where(step_count >= self._grace_steps, success, False)

        timeout = step_count >= self._episode_length
        nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = nan_condition | success | timeout
        # jax.debug.print(
        #     "[pipe_insert_2_new] done check step={s} timeout={t} nan={n} success={su} done={d}",
        #     s=step_count,
        #     t=timeout,
        #     n=nan_condition,
        #     su=success,
        #     d=done,
        # )
        return done.astype(float)

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
        dummy_data = mjx.make_data(self.mj_model, impl=self.impl)
        dummy_info = {}

        # Get observation and check its shape
        obs = self._get_obs(dummy_data, dummy_info)
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

    with mujoco.viewer.launch_passive(model=m, data=d, key_callback=cb) as viewer:
        while not close:
            step_start = time.time()

            p1 = d.site(cable_site_name).xpos
            p2 = d.site(mocap_site_name).xpos

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
        render_cfg.action_scale_pos = cfg.action_scale_pos
        render_cfg.success_threshold_pos = cfg.success_threshold_pos
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


# import glfw
# import mujoco.viewer

# trigger = False

# if __name__ == "__main__":
#     MJX_RENDER = False
#     n_render_steps = 1000

#     env = PipeInsert2()
#     mj_model = env.mj_model
#     data = mj.MjData(mj_model)

#     # mjx_model, in_axes = domain_randomize(env.mjx_model, env.mj_model, rng)

#     def _load_base_state() -> None:
#         if env._key_id is not None:
#             # Initialize directly from the keyframe (includes qpos/qvel/ctrl/mocap).
#             mj.mj_resetDataKeyframe(mj_model, data, env._key_id)
#             mj.mj_forward(mj_model, data)
#             return
#         # data.qpos[:] = np.load(env.keys["qpos"])
#         # data.qvel[:] = np.load(env.keys["qvel"])
#         # data.ctrl[:] = np.load(env.keys["ctrl"])
#         mj.mj_forward(mj_model, data)

#     def cb(key: int) -> None:
#         global trigger
#         if key == glfw.KEY_SPACE:
#             trigger = True
#         if key == glfw.KEY_PERIOD:
#             data.ctrl = np.array([255])

#     # rng, reset_key = jax.random.split(rng)
#     # state = env.reset(reset_key)
#     # mjx.get_data_into(data, mj_model, state.data)
#     # mj.mj_forward(mj_model, data)

#     def get_keypoints(data: mj.MjData) -> np.ndarray:
#         return np.array([data.site(kp_id).xpos for kp_id in env.keypoint_ids])

#     def get_pipe_end_points(data: mj.MjData) -> np.ndarray:
#         pipe_points = ["pipe_entry", "pipe_exit"]
#         return np.array([data.site(kp_id).xpos for kp_id in pipe_points])

#     i = 0

#     with mujoco.viewer.launch_passive(
#         model=mj_model, data=data, key_callback=cb
#     ) as viewer:
#         # set gui camera to the specified in the model
#         viewer.cam.azimuth = mj_model.vis.global_.azimuth
#         viewer.cam.elevation = mj_model.vis.global_.elevation
#         viewer.cam.lookat = mj_model.stat.center
#         viewer.cam.distance = mj_model.stat.extent

#         # mj.mj_forward(model, data)

#         _scene = viewer.user_scn

#         # input("go?")
#         while viewer.is_running():
#             step_start = time.time()

#             # pipe_entry, pipe_exit = get_pipe_end_points(data)

#             # success = keypoints_within_pipe(
#             #     keypoints=get_keypoints(data),
#             #     pipe_axis_start=pipe_entry,
#             #     pipe_axis_end=pipe_exit,
#             #     inner_radius=env.pipe_inner_radius,
#             # )

#             # print(f"{success=}")

#             if trigger:
#                 # if trigger and (i % 10 == 0):
#                 data.ctrl[-1] -= 0.001

#             # print(data.ncon)

#             # step simulation one time step
#             mj.mj_step(mj_model, data)

#             viewer.sync()

#             time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 time.sleep(time_until_next_step)

#             i += 1
#             # input()
