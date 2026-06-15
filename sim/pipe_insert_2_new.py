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
from testing.envs.pipe_insert_2_new import PipeInsert2, default_config
from utils.mjx import ObjType, get_pose, is_stable


def view(model: mj.MjModel):
    import glfw
    import mujoco.viewer

    m = model
    d = mj.MjData(m)

    close = False
    rng = np.random.default_rng()
    t0 = time.time()

    def _keyframe_id() -> int:
        for key_name in ("init", "bent"):
            try:
                return m.key(key_name).id
            except Exception:
                continue
        return 0 if getattr(m, "nkey", 0) > 0 else -1

    def _apply_keyframe() -> None:
        key_id = _keyframe_id()
        if key_id < 0:
            return
        # Let MuJoCo populate all keyframe state (qpos/qvel/ctrl/mocap/act)
        mj.mj_resetDataKeyframe(m, d, key_id)
        mj.mj_forward(m, d)

    def randomize_state(delta_range: float = 0.05) -> None:
        _apply_keyframe()
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
        nonlocal is_recording, sine_enabled
        if key is glfw.KEY_SPACE:
            global close
            close = True
        if key is glfw.KEY_PERIOD:
            d.ctrl = np.array([255])
        if key is glfw.KEY_R:
            if is_recording:
                is_recording = False
                sine_enabled = False
                _save_logs()
        if key is glfw.KEY_S:
            _apply_keyframe()
            _capture_mocap_base()
            _capture_weld_refs()
            mj.mj_step(m, d)
            sine_enabled = True
            is_recording = True

    cable_site_name = "cable_weld_site_1"
    mocap_site_name = "mocap_site_1"

    logs = {
        "e_x": [],
        "e_y": [],
        "e_z": [],
        "e_roll": [],
        "e_pitch": [],
        "e_yaw": [],
        "weld_names": [],
        "weld_pos_err": [],
        "weld_rot_err": [],
    }
    is_recording = False
    log_path = Path("sim/data/test_new.json")
    sine_enabled = False
    sine_pos_amp = np.array([0.02, 0.02, 0.02], dtype=np.float64)
    sine_pos_freq_hz = np.array([0.2, 0.25, 0.15], dtype=np.float64)
    sine_rpy_amp = np.array([0.2, 0.2, 0.2], dtype=np.float64)
    sine_rpy_freq_hz = np.array([0.2, 0.25, 0.15], dtype=np.float64)
    base_mocap_pos = None
    base_mocap_quat = None
    base_mocap_rpy = None
    weld_eq_ids = []
    weld_ref_pos = []
    weld_ref_rot = []

    def _rpy_from_mat(rmat: np.ndarray) -> np.ndarray:
        # XYZ (roll, pitch, yaw) convention
        roll = math.atan2(rmat[2, 1], rmat[2, 2])
        pitch = math.atan2(-rmat[2, 0], math.sqrt(rmat[2, 1] ** 2 + rmat[2, 2] ** 2))
        yaw = math.atan2(rmat[1, 0], rmat[0, 0])
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def _quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return np.array([qw, qx, qy, qz], dtype=np.float64)

    def _mat_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat_wxyz
        n = qw * qw + qx * qx + qy * qy + qz * qz
        if n <= 0.0:
            return np.eye(3, dtype=np.float64)
        s = 2.0 / n
        x = qx * s
        y = qy * s
        z = qz * s
        wx = qw * x
        wy = qw * y
        wz = qw * z
        xx = qx * x
        xy = qx * y
        xz = qx * z
        yy = qy * y
        yz = qy * z
        zz = qz * z
        return np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dtype=np.float64,
        )

    def _rpy_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat_wxyz
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def _get_pose(o_type: int, o_id: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if o_type == mj.mjtObj.mjOBJ_SITE:
            return d.site_xpos[o_id], d.site_xmat[o_id].reshape(3, 3)
        if o_type == mj.mjtObj.mjOBJ_BODY:
            return d.xpos[o_id], d.xmat[o_id].reshape(3, 3)
        if o_type == mj.mjtObj.mjOBJ_GEOM:
            return d.geom_xpos[o_id], d.geom_xmat[o_id].reshape(3, 3)
        if o_type == mj.mjtObj.mjOBJ_CAMERA:
            return d.cam_xpos[o_id], d.cam_xmat[o_id].reshape(3, 3)
        return None

    def _capture_mocap_base() -> None:
        nonlocal base_mocap_pos, base_mocap_quat, base_mocap_rpy
        if mocap_id < 0:
            base_mocap_pos = None
            base_mocap_quat = None
            base_mocap_rpy = None
            return
        base_mocap_pos = d.mocap_pos[mocap_id].copy()
        base_mocap_quat = d.mocap_quat[mocap_id].copy()
        base_mocap_rpy = _rpy_from_quat(base_mocap_quat)

    def _relative_pose_for_eq(eq_id: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        obj_type_arr = getattr(m, "eq_objtype", None)
        obj_type = (
            obj_type_arr[eq_id] if obj_type_arr is not None else mj.mjtObj.mjOBJ_SITE
        )
        obj1_id = m.eq_obj1id[eq_id]
        obj2_id = m.eq_obj2id[eq_id]
        pose1 = _get_pose(obj_type, obj1_id)
        pose2 = _get_pose(obj_type, obj2_id)
        if pose1 is None or pose2 is None:
            return None
        p1, r1 = pose1
        p2, r2 = pose2
        r_rel = r1.T @ r2
        p_rel = r1.T @ (p2 - p1)
        return p_rel, r_rel

    def _capture_weld_refs() -> None:
        nonlocal weld_ref_pos, weld_ref_rot
        weld_ref_pos = []
        weld_ref_rot = []
        for eq_id in weld_eq_ids:
            rel = _relative_pose_for_eq(eq_id)
            if rel is None:
                weld_ref_pos.append(None)
                weld_ref_rot.append(None)
            else:
                p_rel, r_rel = rel
                weld_ref_pos.append(p_rel.copy())
                weld_ref_rot.append(r_rel.copy())

    def _save_logs() -> None:
        if not any(logs.values()):
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
        print(f" > Saved to {log_path}")

    mocap_body_name = "mocap"
    try:
        mocap_id = int(m.body(mocap_body_name).mocapid)
    except Exception:
        mocap_id = -1

    weld_eq_ids = [i for i in range(m.neq) if m.eq_type[i] == mj.mjtEq.mjEQ_WELD]
    weld_eq_names = [
        mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, i) or f"eq_{i}" for i in weld_eq_ids
    ]
    logs["weld_names"] = weld_eq_names

    with mujoco.viewer.launch_passive(model=m, data=d, key_callback=cb) as viewer:
        _apply_keyframe()
        while not close:
            step_start = time.time()

            if (
                sine_enabled
                and mocap_id >= 0
                and base_mocap_pos is not None
                and base_mocap_rpy is not None
            ):
                t = time.time() - t0
                pos_offset = sine_pos_amp * np.sin(2.0 * math.pi * sine_pos_freq_hz * t)
                rpy_offset = sine_rpy_amp * np.sin(2.0 * math.pi * sine_rpy_freq_hz * t)
                d.mocap_pos[mocap_id] = base_mocap_pos + pos_offset
                d.mocap_quat[mocap_id] = _quat_from_rpy(base_mocap_rpy + rpy_offset)

            p1 = d.site(cable_site_name).xpos
            p2 = d.site(mocap_site_name).xpos
            e_pos = p1 - p2

            if is_recording:
                r1 = d.site(cable_site_name).xmat.reshape(3, 3)
                r2 = d.site(mocap_site_name).xmat.reshape(3, 3)
                r_err = r1 @ r2.T
                e_rpy = _rpy_from_mat(r_err)

                logs["e_x"].append(float(e_pos[0]))
                logs["e_y"].append(float(e_pos[1]))
                logs["e_z"].append(float(e_pos[2]))
                logs["e_roll"].append(float(e_rpy[0]))
                logs["e_pitch"].append(float(e_rpy[1]))
                logs["e_yaw"].append(float(e_rpy[2]))

                if weld_eq_ids:
                    pos_errs = []
                    rot_errs = []
                    for idx, eq_id in enumerate(weld_eq_ids):
                        obj_type_arr = getattr(m, "eq_objtype", None)
                        obj_type = (
                            obj_type_arr[eq_id]
                            if obj_type_arr is not None
                            else mj.mjtObj.mjOBJ_SITE
                        )
                        obj1_id = m.eq_obj1id[eq_id]
                        obj2_id = m.eq_obj2id[eq_id]

                        pose1 = _get_pose(obj_type, obj1_id)
                        pose2 = _get_pose(obj_type, obj2_id)
                        if pose1 is None or pose2 is None:
                            pos_errs.append([])
                            rot_errs.append([])
                            continue

                        p1, r1 = pose1
                        p2, r2 = pose2
                        r_rel = r1.T @ r2
                        p_rel = r1.T @ (p2 - p1)

                        p_ref = (
                            weld_ref_pos[idx]
                            if idx < len(weld_ref_pos) and weld_ref_pos[idx] is not None
                            else np.zeros(3, dtype=np.float64)
                        )
                        r_ref = (
                            weld_ref_rot[idx]
                            if idx < len(weld_ref_rot) and weld_ref_rot[idx] is not None
                            else np.eye(3, dtype=np.float64)
                        )

                        p_err = p_rel - p_ref
                        r_err = r_rel @ r_ref.T
                        rpy_err = _rpy_from_mat(r_err)

                        pos_errs.append([float(v) for v in p_err])
                        rot_errs.append([float(v) for v in rpy_err])
                    logs["weld_pos_err"].append(pos_errs)
                    logs["weld_rot_err"].append(rot_errs)

            # step simulation one time step
            viewer.sync()

            mj.mj_step(m, d)

            # input()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    import glfw
    import mujoco.viewer

    cfg = default_config()

    env = PipeInsert2(cfg)

    m = env.mj_model
    d = mj.MjData(m)

    close = False
    rng = np.random.default_rng()
    view(m)
