from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
import pyspacemouse

from testing.experiments.pipe_insert.V3.env import PipeInsert2
from testing.mj import ObjType, get_pose


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
    def __init__(self, env: PipeInsert2, model: mj.MjModel, data: mj.MjData):
        self.keys = {
            "qpos": "testing/experiments/pipe_insert/constants/v4/qpos.npy",
            "qvel": "testing/experiments/pipe_insert/constants/v4/qvel.npy",
            "ctrl": "testing/experiments/pipe_insert/constants/v4/ctrl.npy",
            "mpos": "testing/experiments/pipe_insert/constants/v4/mpos.npy",
            "mquat": "testing/experiments/pipe_insert/constants/v4/mquat.npy",
        }
        Path(self.keys["qpos"]).resolve().parent.mkdir(parents=True, exist_ok=True)
        model.vis.global_.cameraid = -1

        self.env = env
        self.qpos0 = None
        self.qvel0 = None
        self.ctrl0 = None
        self.qpos0_root = None
        self.mocap_pos0 = None
        self.mocap_quat0 = None
        self.mocap_low = None
        self.mocap_high = None
        self.mocap_range = np.array([0.3, 0.3, 0.3], dtype=np.float64)
        self.mocap_rot_range = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        # self.mocap_rot_range = np.array([0.3, 0.3, 0.3], dtype=np.float64)
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
        self.record_states: List[List[float]] = []
        self.record_actions: List[List[float]] = []
        self.record_rewards: List[float] = []
        self.record_terminations: List[bool] = []
        self.record_position_errors: List[float] = []
        self.record_orientation_errors: List[float] = []
        self.record_time: List[float] = []
        self.record_dir = Path(__file__).resolve().parent / "demos/full_se3"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.mocap_gain = 0.01 / 4
        self.mocap_rot_gain = 0.01 / 4
        self.episode_step = 0
        self.episode_time0 = 0.0
        self._centi = bool(self.env._centi)
        self._w_pos = float(self.env._w_pos)
        self._w_rot = float(self.env._w_rot)
        self._sparse_reward = bool(self.env._sparse_reward)
        self._episode_length = int(self.env._episode_length)
        self._termination_threshold = tuple(
            float(x) for x in self.env._termination_threshold
        )
        self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.spacemouse_available = False

        while not self.spacemouse_available:
            self.spacemouse_available = pyspacemouse.open()
            time.sleep(1)
            print("waiting for space mouse")

    def cb(self, key: int) -> None:
        # NOTE: viewer key_callback calls with raw key codes; we only use key here.
        if key == glfw.KEY_PERIOD:
            np.save(self.keys["qpos"], self.d.qpos)
            np.save(self.keys["qvel"], self.d.qvel)
            if self.d.ctrl.size:
                np.save(self.keys["ctrl"], self.d.ctrl)
            if self.d.mocap_pos.size:
                np.save(self.keys["mpos"], self.d.mocap_pos)
            if self.d.mocap_quat.size:
                np.save(self.keys["mquat"], self.d.mocap_quat)

        elif key == glfw.KEY_ENTER:
            self.trigger = True

        elif key == glfw.KEY_ESCAPE:
            self.randomize_state()

        elif key == glfw.KEY_R:
            self.start_recording()

        elif key == glfw.KEY_S:
            self.save_recording()

    def get_obs(self) -> np.ndarray:
        def _e_pos() -> np.ndarray:
            # position of target relative to keypoint
            T_w_target = get_pose(self.m, self.d, "target", ObjType.SITE)
            T_w_keypoint = get_pose(self.m, self.d, "keypoint", ObjType.SITE)
            return T_w_target.t - T_w_keypoint.t

        def _angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            denom = max(norm_a * norm_b, 1e-8)
            cos_theta = np.dot(a, b) / denom
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return float(np.arccos(cos_theta))

        def _e_rot(data) -> float:
            T_w_target = get_pose(self.m, data, "target", ObjType.SITE)
            T_w_keypoint = get_pose(self.m, data, "keypoint", ObjType.SITE)

            R_w_target = T_w_target.R
            R_w_keypoint = T_w_keypoint.R

            z_w_target = R_w_target[:, -1]
            z_w_keypoint = R_w_keypoint[:, -1]
            return _angle_between_vectors(z_w_target, z_w_keypoint)

        e_pos = _e_pos()
        e_rot = _e_rot(self.d)

        if self._centi:
            e_pos = e_pos * 100.0

        return np.concatenate([e_pos, np.array([e_rot], dtype=np.float64)], axis=0)

    def get_reward(self, obs: np.ndarray, terminated: bool) -> float:
        if self._sparse_reward:
            return float(terminated)
        e_pos = obs[:3]
        rot_err = abs(float(obs[3]))
        return float(-(self._w_pos * np.linalg.norm(e_pos) + self._w_rot * rot_err))

    def get_error_terms(self, obs: np.ndarray) -> tuple[float, float]:
        pos_err = float(np.linalg.norm(obs[:3]))
        orient_err = abs(float(obs[3]))
        return pos_err, orient_err

    def get_termination(self, obs: np.ndarray) -> bool:
        is_unstable = np.isnan(self.d.qpos).any() or np.isnan(self.d.qvel).any()
        # timeout = self.episode_step >= self._episode_length

        pos_err, orient_err = self.get_error_terms(obs)
        pos_thresh, rot_thresh = self._termination_threshold

        print(
            pos_err,
            pos_thresh,
            orient_err,
            rot_thresh,
        )

        success = (pos_err < pos_thresh) and (orient_err < rot_thresh)

        return bool(is_unstable or success)

    def randomize_state(self) -> None:
        """Randomize qpos/mocap the same way as PipeInsert2.reset()."""
        if self.qpos0 is None or self.qvel0 is None:
            raise RuntimeError("Base state not initialized. Call run() first.")
        delta = np.random.uniform(-0.3, 0.3, 3)
        delta_rot = np.random.uniform(-self.mocap_rot_range, self.mocap_rot_range)
        # delta = np.random.uniform(-0.03, 0.03, 3)
        self.d.qpos[:] = self.qpos0
        self.d.qvel[:] = self.qvel0
        if self.ctrl0 is not None and self.d.ctrl.size:
            self.d.ctrl[:] = self.ctrl0
        if self.cable_root_qpos_adr is not None and self.qpos0_root is not None:
            adr = self.cable_root_qpos_adr
            self.d.qpos[adr : adr + 3] = self.qpos0_root + delta
        if (
            self.mocap_id is not None
            and self.mocap_pos0 is not None
            and self.mocap_quat0 is not None
        ):
            target = self.mocap_pos0 + delta
            if self.mocap_low is not None and self.mocap_high is not None:
                target = np.clip(target, self.mocap_low, self.mocap_high)
            self.d.mocap_pos[self.mocap_id] = target
            delta_quat = _rotvec_to_quat(delta_rot)
            target_quat = _quat_mul(delta_quat, self.mocap_quat0)
            self.d.mocap_quat[self.mocap_id] = _quat_normalize(target_quat)
        mj.mj_forward(self.m, self.d)

    def start_recording(self) -> None:
        """Start a new demo: randomize initial state and clear buffers."""
        self.randomize_state()
        self.record_states.clear()
        self.record_actions.clear()
        self.record_rewards.clear()
        self.record_terminations.clear()
        self.record_position_errors.clear()
        self.record_orientation_errors.clear()
        self.record_time.clear()
        self.episode_step = 0
        self.episode_time0 = float(self.d.time)
        self.recording = True
        print("Recording started (press 'S' to save).")
        self.last_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def save_recording(self) -> None:
        """Persist the current recording to disk."""
        if not self.recording:
            print("No active recording to save.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        n = min(
            len(self.record_states),
            len(self.record_actions),
            len(self.record_rewards),
            len(self.record_terminations),
            len(self.record_position_errors),
            len(self.record_orientation_errors),
            len(self.record_time),
        )
        states = list(self.record_states)[:n]
        actions = list(self.record_actions)[:n]
        rewards = list(self.record_rewards)[:n]
        terminated = [int(flag) for flag in self.record_terminations[:n]]
        position_errors = list(self.record_position_errors)[:n]
        orientation_errors = list(self.record_orientation_errors)[:n]
        time = list(self.record_time)[:n]
        if n > 0:
            next_states = states[1:] + [states[-1]]
        else:
            next_states = []

        payload = {
            "timestamp": ts,
            "timestep": float(self.m.opt.timestep),
            "observations": states,
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "terminations": terminated,
            "terminated": terminated,
            "position_errors": position_errors,
            "orientation_errors": orientation_errors,
            "time": time,
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
            for key_name in ("bent", "init"):
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
        self.qvel0 = self.d.qvel.copy()
        if self.d.ctrl.size:
            self.ctrl0 = self.d.ctrl.copy()
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
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

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
                    obs = self.get_obs()
                    position_error, orientation_error = self.get_error_terms(obs)
                    terminated = self.get_termination(obs)
                    reward = self.get_reward(obs, terminated)

                    self.record_states.append(obs.tolist())
                    self.record_actions.append(self.last_action)
                    self.record_rewards.append(reward)
                    self.record_terminations.append(terminated)
                    self.record_position_errors.append(position_error)
                    self.record_orientation_errors.append(orientation_error)
                    self.record_time.append(float(self.d.time) - self.episode_time0)
                    self.episode_step += 1
                    if terminated:
                        self.save_recording()
                        self.start_recording()

                # real-time pacing
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    _env = PipeInsert2()

    m = _env.mj_model
    d = mj.MjData(m)

    sim = Test(_env, m, d)
    sim.run()


if __name__ == "__main__":
    main()
