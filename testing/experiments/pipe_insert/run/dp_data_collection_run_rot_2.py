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

from testing.envs.pipe_insert import PipeInsert
from testing.envs.pipe_insert_2 import PipeInsert2
from testing.envs.pipe_insert_3 import PipeInsert3, keypoints_within_pipe
from testing.experiments.pipe_insert.exp_utils import get_dp_config, setup_environment
from testing.mj import ObjType, get_names, get_pose, id2name
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

from .physics import amperes_law, biot_savarts_law


class Test:
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        self.keys = {
            "qpos": "tmp/v4/qpos.npy",
            "qvel": "tmp/v4/qvel.npy",
            "ctrl": "tmp/v4/ctrl.npy",
        }
        model.vis.global_.cameraid = -1

        # gripper
        # self.qpos0 = np.array([4.11014825e-07, -7.62520820e-03])
        # self.qpos0 = np.array([-6.67943402e-07, -2.35101053e-10, -0.00762527])
        self.qpos0_gripper = None

        # self.ctrl0 = np.array([0.0, 0.0, 0.0])
        # self.ctrl0 = np.array([-6.67943402e-07, -2.35101053e-10, -0.00762527])
        self.ctrl0 = None
        # cable
        # self.qpos1 = np.array([0.20210016, -0.05143623])
        # self.qpos1 = np.array([-1.61630601e-04, 2.01762220e-01, -5.09595182e-02])
        self.qpos0_cable = None

        self.model: mj.MjModel = model
        self.data = data
        self.trigger = False
        self.recording = False
        self.record_states_sparse: List[List[float]] = []
        self.record_states_dense: List[List[float]] = []
        self.record_actions: List[List[float]] = []
        self.record_reward_dense: List[float] = []
        self.record_reward_sparse: List[float] = []
        self.record_dir = Path(__file__).resolve().parent / "data_collection_2_rot"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.ctrl_gain = 0.01 / 4
        self.rot_gain = 0.01 / 8
        self.success_thresh = 0.01

        self.pipe_inner_radius = 0.0385 / 2
        self.pipe_outer_radius = 0.0435 / 2
        self.pipe_length = 0.121 / 2  # 121 mm from schematic

        # Clamp ctrl using actuator ctrlrange.
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]
        actuator_names = get_names(self.model, ObjType.ACTUATOR)
        actuator_name_to_id = {
            name: idx for idx, name in enumerate(actuator_names) if name
        }
        self.pos_actuator_ids = [
            actuator_name_to_id[name]
            for name in ("x", "y", "z")
            if name in actuator_name_to_id
        ]
        if len(self.pos_actuator_ids) != 3:
            self.pos_actuator_ids = list(range(min(3, self.model.nu)))
        self.rot_actuator_ids = [
            actuator_name_to_id[name]
            for name in ("roll", "pitch", "yaw")
            if name in actuator_name_to_id
        ]
        self.action_dim = len(self.pos_actuator_ids) + len(self.rot_actuator_ids)
        self.last_action = [0.0] * self.action_dim

        self.spacemouse_available = False

        while not self.spacemouse_available:
            self.spacemouse_available = pyspacemouse.open()
            time.sleep(1)
            print("waiting for space mouse")

        a_dim = 3
        o_dim = 3

        env = setup_environment(batch_size=1)

        self.action_scale = 1.0

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
            np.save("tmp/v4/qpos.npy", self.data.qpos)
            np.save("tmp/v4/qvel.npy", self.data.qvel)
            np.save("tmp/v4/ctrl.npy", self.data.ctrl)
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
            # self.randomize_state()
            self.start_recording()

        elif key == glfw.KEY_S:
            self.save_recording()

    def get_obs_sparse(self) -> torch.Tensor:
        # Your handcrafted obs: distances between keypoints and targets.
        T_w_keypoint_1 = get_pose(self.model, self.data, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.model, self.data, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.model, self.data, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.model, self.data, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.model, self.data, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.model, self.data, "target_3", ObjType.SITE)

        d1 = np.linalg.norm(T_w_target_1.t - T_w_keypoint_1.t)
        d2 = np.linalg.norm(T_w_target_2.t - T_w_keypoint_2.t)
        d3 = np.linalg.norm(T_w_target_3.t - T_w_keypoint_3.t)

        d = np.array([d1, d2, d3], dtype=np.float32).flatten()
        return torch.from_numpy(d)  # [o_dim]

    def get_obs_dense(self) -> torch.Tensor:
        # Your handcrafted obs: distances between keypoints and targets.
        T_w_keypoint_1 = get_pose(self.model, self.data, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.model, self.data, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.model, self.data, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.model, self.data, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.model, self.data, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.model, self.data, "target_3", ObjType.SITE)

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
        """Randomize qpos/ctrl the same way as the original trigger logic."""
        if self.qpos0_gripper is None or self.qpos0_cable is None or self.ctrl0 is None:
            raise RuntimeError("Base state not initialized. Call run() first.")

        pos_lim = (-0.3, 0.3)
        delta_pos = np.random.uniform(pos_lim[0], pos_lim[1], size=3)

        def set_joint_qpos(joint_name: str, value: float) -> bool:
            try:
                joint_id = self.model.joint(joint_name).id
            except Exception:
                return False
            qposadr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qposadr] = value
            return True

        def set_free_joint_pos(joint_name: str, value: np.ndarray) -> bool:
            try:
                joint_id = self.model.joint(joint_name).id
            except Exception:
                return False
            qposadr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qposadr : qposadr + 3] = value
            return True

        # Gripper position (joint-addressed).
        qpos0_pos = self.qpos0_gripper[:3]
        for i, name in enumerate(("x", "y", "z")):
            set_joint_qpos(name, qpos0_pos[i] + delta_pos[i])

        # Reset gripper orientation joints to their base (no randomization).
        if self.qpos0_gripper.shape[0] >= 6:
            qpos0_ori = self.qpos0_gripper[3:6]
            for i, name in enumerate(("roll", "pitch", "yaw")):
                set_joint_qpos(name, qpos0_ori[i])

        if self.qpos0_cable is not None and self.qpos0_cable.shape[0] >= 3:
            cable_pos = self.qpos0_cable[:3] + delta_pos
        else:
            cable_pos = delta_pos
        if not set_free_joint_pos("cable:free", cable_pos):
            if self.data.qpos.shape[0] >= 9:
                self.data.qpos[6:9] = cable_pos

        # for i in range(self.model.njnt):
        #     name = id2name(self.model, i, ObjType.JOINT)
        #     print(name)
        # quit()

        # Update ctrl for position actuators.
        ctrl = self.data.ctrl.copy()
        if self.pos_actuator_ids:
            pos_ids = np.array(self.pos_actuator_ids, dtype=int)
            base = self.ctrl0[pos_ids]
            ctrl[pos_ids] = np.clip(
                base + delta_pos[: len(pos_ids)],
                self.ctrl_low[pos_ids],
                self.ctrl_high[pos_ids],
            )
        if self.rot_actuator_ids:
            rot_ids = np.array(self.rot_actuator_ids, dtype=int)
            ctrl[rot_ids] = np.clip(
                self.ctrl0[rot_ids], self.ctrl_low[rot_ids], self.ctrl_high[rot_ids]
            )

        self.data.ctrl[:] = ctrl

        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)

    def start_recording(self) -> None:
        """Start a new demo: randomize initial state and clear buffers."""
        print(1)
        if self.recording:
            print("Recording restarted (previous demo discarded).")
        print(2)
        self.randomize_state()
        print(3)
        self.record_states_sparse.clear()
        self.record_states_dense.clear()
        self.record_actions.clear()
        self.record_reward_dense.clear()
        self.record_reward_sparse.clear()
        self.recording = True
        print("Recording started (press 'S' to save).")
        self.last_action = [0.0] * self.action_dim

    def save_recording(self) -> None:
        """Persist the current recording to disk."""
        if not self.recording:
            print("No active recording to save.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        actions = list(self.record_actions)
        states_sparse = list(self.record_states_sparse)
        states_dense = list(self.record_states_dense)
        next_states_sparse = states_sparse[1:] + [states_sparse[-1]]
        next_states_dense = states_dense[1:] + [states_dense[-1]]
        rewards_dense = list(self.record_reward_dense)
        rewards_sparse = list(self.record_reward_sparse)
        terminated = [0] * (len(states_dense) - 1) + [1]

        payload = {
            "timestamp": ts,
            "timestep": float(self.model.opt.timestep),
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

    def update_ctrl_from_spacemouse(self) -> None:
        """Update ctrl by integrating delta from the SpaceMouse."""
        if not self.spacemouse_available:
            return
        state = pyspacemouse.read()
        if state is None:
            self.last_action = [0.0] * self.action_dim
            return
        delta_pos_raw = np.array([state.x, state.y, state.z], dtype=np.float64)
        pos_len = len(self.pos_actuator_ids)
        delta_pos = self.ctrl_gain * delta_pos_raw[:pos_len]
        delta_rot = np.zeros(0, dtype=np.float64)
        rot_len = len(self.rot_actuator_ids)
        if rot_len:
            delta_rot_raw = np.array(
                [-state.pitch, state.roll, -state.yaw],
                dtype=np.float64,
                # [state.roll, state.pitch, state.yaw], dtype=np.float64
            )
            delta_rot = self.rot_gain * delta_rot_raw[:rot_len]
        delta_action = np.concatenate([delta_pos, delta_rot])
        if np.allclose(delta_action, 0.0, atol=1e-6):
            self.last_action = [0.0] * self.action_dim
            return
        ctrl = self.data.ctrl.copy()

        if self.pos_actuator_ids:
            pos_ids = np.array(self.pos_actuator_ids, dtype=int)
            ctrl[pos_ids] = np.clip(
                ctrl[pos_ids] + delta_pos[: len(pos_ids)],
                self.ctrl_low[pos_ids],
                self.ctrl_high[pos_ids],
            )
        if self.rot_actuator_ids:
            rot_ids = np.array(self.rot_actuator_ids, dtype=int)
            ctrl[rot_ids] = np.clip(
                ctrl[rot_ids] + delta_rot[: len(rot_ids)],
                self.ctrl_low[rot_ids],
                self.ctrl_high[rot_ids],
            )
        # print(ctrl)
        self.data.ctrl[:] = ctrl
        self.last_action = delta_action.tolist()

    def run(self):
        # Load initial state
        self.data.qpos[:] = np.load(self.keys["qpos"])
        self.data.qvel[:] = np.load(self.keys["qvel"])
        self.data.ctrl[:] = np.load(self.keys["ctrl"])

        gripper_vals: list[float] = []
        for name in ("x", "y", "z", "roll", "pitch", "yaw"):
            try:
                joint_id = self.model.joint(name).id
            except Exception:
                continue
            qposadr = self.model.jnt_qposadr[joint_id]
            gripper_vals.append(float(self.data.qpos[qposadr]))
        if gripper_vals:
            self.qpos0_gripper = np.array(gripper_vals, dtype=np.float64)
        else:
            self.qpos0_gripper = self.data.qpos[:6].copy()

        try:
            cable_id = self.model.joint("cable:free").id
            cable_qposadr = self.model.jnt_qposadr[cable_id]
            self.qpos0_cable = self.data.qpos[cable_qposadr : cable_qposadr + 3].copy()
        except Exception:
            self.qpos0_cable = self.data.qpos[6:9].copy()

        self.ctrl0 = self.data.ctrl.copy()
        mj.mj_forward(self.model, self.data)

        self.target_ids = [self.model.site(f"target_{i + 1}").id for i in range(3)]
        self.keypoint_ids = [self.model.site(f"keypoint_{i + 1}").id for i in range(3)]

        def get_keypoints(data: mj.MjData) -> np.ndarray:
            return np.array([data.site(kp_id).xpos for kp_id in self.keypoint_ids])

        def get_pipe_end_points(data: mj.MjData) -> np.ndarray:
            pipe_points = ["pipe_entry", "pipe_exit"]
            return np.array([data.site(kp_id).xpos for kp_id in pipe_points])

        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.cb
        ) as viewer:
            viewer.user_scn

            while viewer.is_running():
                step_start = time.time()

                # pipe_entry, pipe_exit = get_pipe_end_points(self.data)

                # success = keypoints_within_pipe(
                #     keypoints=get_keypoints(self.data),
                #     pipe_axis_start=pipe_entry,
                #     pipe_axis_end=pipe_exit,
                #     inner_radius=self.pipe_inner_radius,
                # )

                # input()

                self.last_action = [0.0] * self.action_dim
                self.update_ctrl_from_spacemouse()
                mj.mj_step(self.model, self.data)
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
                    self.record_actions.append(list(self.last_action))
                    self.record_reward_sparse.append(reward_sparse)
                    self.record_reward_dense.append(reward_dense)

                # real-time pacing
                time_until_next_step = self.model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    _env = PipeInsert3()

    m = _env.mj_model
    d = mj.MjData(m)

    sim = Test(m, d)
    sim.run()


if __name__ == "__main__":
    main()
