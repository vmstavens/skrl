from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
import pyspacemouse
import torch

from testing.envs.pipe_insert_2_new import PipeInsert2
from testing.experiments.pipe_insert.exp_utils import get_dp_config, setup_environment
from testing.mj import ObjType, get_names, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

from .physics import amperes_law, biot_savarts_law


def _quat_normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _rotvec_to_quat(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    angle = np.linalg.norm(v)
    if angle < eps:
        return _quat_normalize(
            np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]], dtype=np.float64)
        )
    axis = v / angle
    half = 0.5 * angle
    sin_half = np.sin(half)
    return _quat_normalize(
        np.array(
            [np.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
            dtype=np.float64,
        )
    )


def _spacemouse_rotvec(state) -> np.ndarray:
    for attrs in (("ry", "rx", "rz"), ("pitch", "roll", "yaw")):
        if all(hasattr(state, name) for name in attrs):
            values = [getattr(state, n) for n in attrs]
            values[0] = -values[0]
            return np.array(values, dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


class Test:
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        self.keys = {
            "qpos": "testing/experiments/pipe_insert/constants/v4/qpos.npy",
            "qvel": "testing/experiments/pipe_insert/constants/v4/qvel.npy",
            "ctrl": "testing/experiments/pipe_insert/constants/v4/ctrl.npy",
            "mpos": "testing/experiments/pipe_insert/constants/v4/mpos.npy",
            "mquat": "testing/experiments/pipe_insert/constants/v4/mquat.npy",
        }
        Path(self.keys["qpos"]).resolve().parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.cameraid = -1

        self.qpos0 = None
        self.qpos0_root = None
        self.mocap_pos0 = None
        self.mocap_quat0 = None
        self.mocap_low = None
        self.mocap_high = None
        self.mocap_range = np.array([0.3, 0.3, 0.3], dtype=np.float64)
        self.mocap_id = None
        try:
            self.mocap_id = int(model.body("mocap").mocapid)
            if self.mocap_id < 0:
                self.mocap_id = None
        except Exception:
            self.mocap_id = None
        self.cable_root_qpos_adr = None
        try:
            cable_root_jnt = model.joint("cable:free").id
            self.cable_root_qpos_adr = int(model.jnt_qposadr[cable_root_jnt])
        except Exception:
            self.cable_root_qpos_adr = None

        self.m: mj.MjModel = model
        self.d = data
        self.trigger = False
        self.recording = False
        self.record_states_sparse: List[List[float]] = []
        self.record_states_dense: List[List[float]] = []
        self.record_actions: List[List[float]] = []
        self.record_reward_dense: List[float] = []
        self.record_reward_sparse: List[float] = []
        self.record_dir = Path(__file__).resolve().parent / "data_collection_2_mocap"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.mocap_gain = 0.01 / 4
        self.mocap_rot_gain = 0.01 / 4
        self.success_thresh = 0.01
        self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.spacemouse_available = False

        while not self.spacemouse_available:
            self.spacemouse_available = pyspacemouse.open()
            time.sleep(1)
            print("waiting for space mouse")

    def cb(self, key: int) -> None:
        # NOTE: viewer key_callback calls with raw key codes; we only use key here.
        if key == glfw.KEY_SPACE:
            if len(self.obs_history) == 0:
                obs = self.get_obs_sparse()
                for _ in range(self.agent.config["obs_horizon"]):
                    self.obs_history.append(obs)
            obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)
            actions, _, _ = self.agent.act(obs_seq)
            start = self.agent.config["obs_horizon"] - 1
            end = start + self.agent.config["action_horizon"]
            action_seq = actions[0, start:end, :].detach().cpu().numpy()
            action_seq = action_seq * self.action_scale
            self.action_queue = deque(action_seq)

        elif key == glfw.KEY_PERIOD:
            np.save(self.keys["qpos"], self.d.qpos)
            np.save(self.keys["qvel"], self.d.qvel)
            if self.d.ctrl.size:
                np.save(self.keys["ctrl"], self.d.ctrl)
            if self.d.mocap_pos.size:
                np.save(self.keys["mpos"], self.d.mocap_pos)
            if self.d.mocap_quat.size:
                np.save(self.keys["mquat"], self.d.mocap_quat)
            # self.d.qpos[:] = np.load(self.keys["qpos"])
            # self.d.qvel[:] = np.load(self.keys["qvel"])
            # self.d.ctrl[:] = np.load(self.keys["ctrl"])
            # mj.mj_forward(self.m, self.d)
            # print("Loaded qpos/qvel/ctrl.")

        elif key == glfw.KEY_ENTER:
            self.trigger = True

        elif key == glfw.KEY_ESCAPE:
            # viewer handles close; we can just print
            self.randomize_state()

        elif key == glfw.KEY_R:
            self.start_recording()

        elif key == glfw.KEY_S:
            self.save_recording()

    def get_obs_sparse(self) -> torch.Tensor:
        # Your handcrafted obs: distances between keypoints and targets.
        T_w_keypoint_1 = get_pose(self.m, self.d, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.m, self.d, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.m, self.d, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.m, self.d, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.m, self.d, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.m, self.d, "target_3", ObjType.SITE)

        d1 = np.linalg.norm(T_w_target_1.t - T_w_keypoint_1.t)
        d2 = np.linalg.norm(T_w_target_2.t - T_w_keypoint_2.t)
        d3 = np.linalg.norm(T_w_target_3.t - T_w_keypoint_3.t)

        d = np.array([d1, d2, d3], dtype=np.float32).flatten()
        return torch.from_numpy(d)  # [o_dim]

    def get_obs_dense(self) -> torch.Tensor:
        # Your handcrafted obs: distances between keypoints and targets.
        T_w_keypoint_1 = get_pose(self.m, self.d, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.m, self.d, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.m, self.d, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.m, self.d, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.m, self.d, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.m, self.d, "target_3", ObjType.SITE)

        d1 = T_w_target_1.t - T_w_keypoint_1.t
        d2 = T_w_target_2.t - T_w_keypoint_2.t
        d3 = T_w_target_3.t - T_w_keypoint_3.t

        d = np.array([d1, d2, d3], dtype=np.float32).flatten()
        return torch.from_numpy(d)  # [o_dim]

    def get_reward_sparse(self) -> float:
        """Placeholder dense reward encouraging smaller distances."""
        d = 0
        # I_wire = 1.0 * 10

        # t_py = time.perf_counter()
        # I_py = amperes_law(cable_pts, I_wire, pipe_center)
        # self._homotopy_dt_py = perf_counter() - t_py

        # t_jax = perf_counter()
        # I_jax, _ = self._jax_ampere_fn(
        #     jnp.asarray(cable_pts), jnp.asarray(pipe_center), I_wire
        # I_val = float(I_jax) if np.isfinite(I_jax) else float(I_py)

        # self.cable_through_pipe = np.abs(I_val) > I_wire * 0.5
        return float(-d)

    def get_reward_dense(self) -> float:
        """norm of dist vector D = [L2(t1,k1), L2(t2,k2), L2(t3,k3)], reward = -|D|"""
        o = self.get_obs_sparse()
        return float(-np.linalg.norm(o))

    def randomize_state(self) -> None:
        """Randomize qpos/mocap the same way as the original trigger logic."""
        print(0)
        if self.qpos0 is None or self.mocap_pos0 is None or self.mocap_quat0 is None:
            raise RuntimeError("Base state not initialized. Call run() first.")

        # delta = np.random.uniform(-0.4, 0.4, 3)
        delta = np.random.uniform(-0.3, 0.3, 3)
        # delta = np.array([0.1, 0, 0])
        self.d.qpos[:] = self.qpos0
        if self.cable_root_qpos_adr is not None and self.qpos0_root is not None:
            adr = self.cable_root_qpos_adr
            self.d.qpos[adr : adr + 3] = self.qpos0_root + delta
        self.d.qvel[:] = 0.0
        if self.mocap_id is not None:
            target = self.mocap_pos0 + delta
            if self.mocap_low is not None and self.mocap_high is not None:
                target = np.clip(target, self.mocap_low, self.mocap_high)
            self.d.mocap_pos[self.mocap_id] = target
            self.d.mocap_quat[self.mocap_id] = self.mocap_quat0
        print(delta)
        # mj.mj_forward(self.m, self.d)

    def start_recording(self) -> None:
        """Start a new demo: randomize initial state and clear buffers."""
        self.randomize_state()
        self.record_states_sparse.clear()
        self.record_states_dense.clear()
        self.record_actions.clear()
        self.record_reward_dense.clear()
        self.record_reward_sparse.clear()
        self.recording = True
        print("Recording started (press 'S' to save).")
        self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def save_recording(self) -> None:
        """Persist the current recording to disk."""
        if not self.recording:
            print("No active recording to save.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        actions = list(self.record_actions)
        n = len(actions)
        states_sparse = list(self.record_states_sparse)[:n]
        states_dense = list(self.record_states_dense)[:n]
        rewards_dense = list(self.record_reward_dense)[:n]
        rewards_sparse = list(self.record_reward_sparse)[:n]
        if n > 0:
            next_states_sparse = states_sparse[1:] + [states_sparse[-1]]
            next_states_dense = states_dense[1:] + [states_dense[-1]]
            terminated = [0] * (n - 1) + [1]
        else:
            next_states_sparse = []
            next_states_dense = []
            terminated = []

        payload = {
            "timestamp": ts,
            "timestep": float(self.m.opt.timestep),
            "states_sparse": states_sparse,
            "states_dense": states_dense,
            "actions": actions,
            "next_states_sparse": next_states_sparse,
            "next_states_dense": next_states_dense,
            "rewards_dense": rewards_dense,
            "rewards_sparse": rewards_sparse,
            "terminated": terminated,
        }
        path = self.record_dir / f"{ts}.json"
        with path.open("w") as f:
            json.dump(payload, f, indent=4)
        print(f"Saved recording to {path}")
        self.recording = False

    def update_mocap_from_spacemouse(self) -> None:
        """Update mocap position and orientation from the SpaceMouse."""
        if not self.spacemouse_available:
            return
        if self.mocap_id is None:
            return
        state = pyspacemouse.read()
        if state is None:
            self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            return
        delta_raw = np.array([state.x, state.y, state.z], dtype=np.float64)
        # delta_raw = np.array([-state.x, -state.y], dtype=np.float64)
        # delta_raw = np.array([state.x, state.z], dtype=np.float64)
        delta_pos = self.mocap_gain * delta_raw
        delta_rot = self.mocap_rot_gain * _spacemouse_rotvec(state)

        moved = False
        if not np.allclose(delta_pos, 0.0, atol=1e-6):
            mocap_pos = self.d.mocap_pos[self.mocap_id].copy()
            target = mocap_pos + delta_pos
            if self.mocap_low is not None and self.mocap_high is not None:
                target = np.clip(target, self.mocap_low, self.mocap_high)
            self.d.mocap_pos[self.mocap_id] = target
            moved = True

        if not np.allclose(delta_rot, 0.0, atol=1e-6):
            current = self.d.mocap_quat[self.mocap_id].copy()
            delta_quat = _rotvec_to_quat(delta_rot)
            target_quat = _quat_mul(delta_quat, current)
            self.d.mocap_quat[self.mocap_id] = _quat_normalize(target_quat)
            moved = True

        if moved:
            self.last_action = np.concatenate([delta_pos, delta_rot]).tolist()
        else:
            self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def run(self):
        # Load initial state
        if self.m.nkey:
            key_id = None
            for key_name in ("init", "bent"):
                try:
                    key_id = self.m.key(key_name).id
                    break
                except Exception:
                    continue
            if key_id is None:
                key_id = 0
            self.d.qpos[:] = self.m.key_qpos[key_id]
            self.d.qvel[:] = self.m.key_qvel[key_id]
            if self.d.ctrl.size:
                self.d.ctrl[:] = self.m.key_ctrl[key_id]
            if self.m.nmocap:
                self.d.mocap_pos[:] = self.m.key_mpos[key_id].reshape(self.m.nmocap, 3)
                self.d.mocap_quat[:] = self.m.key_mquat[key_id].reshape(
                    self.m.nmocap, 4
                )
        else:
            if Path(self.keys["qpos"]).exists():
                self.d.qpos[:] = np.load(self.keys["qpos"])
            if Path(self.keys["qvel"]).exists():
                self.d.qvel[:] = np.load(self.keys["qvel"])
            if self.d.ctrl.size and Path(self.keys["ctrl"]).exists():
                self.d.ctrl[:] = np.load(self.keys["ctrl"])
            if self.m.nmocap and Path(self.keys["mpos"]).exists():
                self.d.mocap_pos[:] = np.load(self.keys["mpos"])
            if self.m.nmocap and Path(self.keys["mquat"]).exists():
                self.d.mocap_quat[:] = np.load(self.keys["mquat"])

        if self.mocap_id is None and self.m.nmocap:
            self.mocap_id = 0

        self.qpos0 = self.d.qpos.copy()
        if self.cable_root_qpos_adr is not None:
            adr = self.cable_root_qpos_adr
            self.qpos0_root = self.d.qpos[adr : adr + 3].copy()
        if self.mocap_id is not None and self.m.nmocap:
            self.mocap_pos0 = self.d.mocap_pos[self.mocap_id].copy()
            self.mocap_quat0 = self.d.mocap_quat[self.mocap_id].copy()
            self.mocap_low = self.mocap_pos0 - self.mocap_range
            self.mocap_high = self.mocap_pos0 + self.mocap_range
        mj.mj_forward(self.m, self.d)

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            viewer.user_scn

            while viewer.is_running():
                step_start = time.time()

                self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                self.update_mocap_from_spacemouse()
                mj.mj_step(self.m, self.d)
                viewer.sync()

                if self.trigger:
                    self.randomize_state()
                    self.trigger = False

                if self.recording:
                    obs_dense = self.get_obs_dense()
                    obs_sparse = self.get_obs_sparse()
                    reward_dense = self.get_reward_dense()
                    reward_sparse = self.get_reward_sparse()

                    self.record_states_sparse.append(obs_sparse.numpy().tolist())
                    self.record_states_dense.append(obs_dense.numpy().tolist())
                    self.record_actions.append(self.last_action)
                    self.record_reward_sparse.append(reward_sparse)
                    self.record_reward_dense.append(reward_dense)

                # real-time pacing
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    _env = PipeInsert2()

    m = _env.mj_model
    d = mj.MjData(m)

    sim = Test(m, d)
    sim.run()


if __name__ == "__main__":
    main()
