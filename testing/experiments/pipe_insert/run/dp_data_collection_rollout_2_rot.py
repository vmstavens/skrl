from __future__ import annotations

import json
import sys
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

from performance import timer
from testing.envs.pipe_insert_3 import PipeInsert3, keypoints_within_pipe
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_dp_config,
    setup_environment,
)
from testing.mj import ObjType, get_names, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


class Test:
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        seed = int(np.random.uniform() * 1000)
        exp_set_seed(seed)
        self.keys = {
            "qpos": "tmp/v4/qpos.npy",
            "qvel": "tmp/v4/qvel.npy",
            "ctrl": "tmp/v4/ctrl.npy",
        }
        model.vis.global_.cameraid = -1

        self.qpos0_gripper = None
        self.qpos0_cable = None
        self.ctrl0 = None

        self.model: mj.MjModel = model
        self.data = data
        self.trigger = False
        self.recording = False
        self.record_qpos: List[np.ndarray] = []
        self.record_qvel: List[np.ndarray] = []
        self.record_ctrl: List[np.ndarray] = []
        self.record_states: List[List[float]] = []
        self.record_actions: List[List[float]] = []
        self.record_reward_dense: List[float] = []
        self.record_reward_sparse: List[float] = []
        # self.record_dir = Path(__file__).resolve().parent / "data_collection"
        # self.record_dir.mkdir(parents=True, exist_ok=True)
        self.ctrl_gain = 0.01 / 2
        self.action_gain = 1.0
        # self.action_gain = 0.3
        # self.action_gain = 0.17
        # self.action_gain = 0.17
        self.success_thresh = 0.01
        self.pipe_inner_radius = 0.0385 / 2
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

        a_dim = self.action_dim
        o_dim = 9

        self.start = False

        env = setup_environment(batch_size=1)

        dp_config = get_dp_config(exp_name="pipe_insert_2_rot", env=env, wandb=False)
        # dp_config["pred_horizon"] = 12

        dp_config["obs_dim"] = o_dim
        dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]

        dp_models: dict = {}
        dp_models["model"] = ConditionalUnet1D(
            a_dim=a_dim, o_dim=o_dim, config=dp_config
        )
        ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
        dp_models["ema_model"] = ConditionalUnet1D(
            a_dim=a_dim, o_dim=o_dim, config=dp_config
        )

        self.agent = DiffusionPolicy(
            a_dim=a_dim,
            o_dim=o_dim,
            models=dp_models,
            ema=ema,
            device=env.device,
            config=dp_config,
        )
        self.agent = self.agent.load(
            "testing/experiments/pipe_insert/.runs/data_100_2_1_action_horizon_rot_20260215_160452/models/latest_model.pth",
            # "testing/experiments/pipe_insert/.runs/finetune_100_2_1_action_horizon_rot_20260215_132427/models/latest_model.pth",
            # "testing/experiments/pipe_insert/.runs/data_100_2_1_action_horizon_rot_20260214_192555/models/latest_model.pth",
            # "testing/experiments/pipe_insert/.runs/data_100_2_1_action_horizon_rot/models/latest_model.pth",
            device=env.device,
        )

        # performance enhancer -------------------------
        self.agent._num_diffusion_iters = 8
        self.agent._pred_horizon = 8
        # self.agent._pred_horizon = 16
        # self.agent._num_diffusion_iters = 20
        # self.agent._num_diffusion_iters = 3
        # end - performance enhancer -------------------------

        self.agent.to(env.device)
        self.action = np.zeros(self.action_dim, dtype=np.float64)
        self.dp = False
        self.action_queue: deque[np.ndarray] = deque()
        self.obs_history: deque[torch.Tensor] = deque(
            maxlen=self.agent.config["obs_horizon"]
        )
        # self.action_scale = 0.3
        # self.action_interval_steps = 1
        self.action_interval_steps = 1
        self.step_idx = 0

    def cb(self, key: int) -> None:
        # NOTE: viewer key_callback calls with raw key codes; we only use key here.
        if key == glfw.KEY_PERIOD:
            self.data.qpos[:] = np.load(self.keys["qpos"])
            self.data.qvel[:] = np.load(self.keys["qvel"])
            self.data.ctrl[:] = np.load(self.keys["ctrl"])
            mj.mj_forward(self.model, self.data)
            print("Loaded qpos/qvel/ctrl.")

        elif key == glfw.KEY_ENTER:
            self.trigger = True

        elif key == glfw.KEY_SPACE:
            # viewer handles close; we can just print
            print("SPACE pressed.")
            self.randomize_state()
            # self.start = True

        elif key == glfw.KEY_R:
            self.start_recording()

        elif key == glfw.KEY_S:
            self.save_recording()

    def get_obs(self) -> torch.Tensor:
        # Dense obs: keypoint-target vector deltas (3 targets * 3 dims = 9)
        T_w_keypoint_1 = get_pose(self.model, self.data, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.model, self.data, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.model, self.data, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.model, self.data, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.model, self.data, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.model, self.data, "target_3", ObjType.SITE)

        d1 = T_w_target_1.t - T_w_keypoint_1.t
        d2 = T_w_target_2.t - T_w_keypoint_2.t
        d3 = T_w_target_3.t - T_w_keypoint_3.t

        d = np.concatenate([d1, d2, d3], axis=0).astype(np.float32)
        return torch.from_numpy(d)  # [o_dim]

    def compute_dense_reward(self) -> float:
        """Placeholder dense reward encouraging smaller distances."""
        d = torch.linalg.norm(self.get_obs())
        return float(-d)

    def compute_sparse_reward(self, obs: np.ndarray) -> float:
        """Placeholder sparse reward for success when within threshold."""
        return float(np.max(obs) < self.success_thresh)

    def randomize_state(self) -> None:
        """Randomize qpos/ctrl the same way as the original trigger logic."""
        if self.qpos0_gripper is None or self.qpos0_cable is None or self.ctrl0 is None:
            raise RuntimeError("Base state not initialized. Call run() first.")

        pos_lim = (-0.05, 0.05)
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

        self.data.ctrl[:] = ctrl

        self.data.qvel[:] = 0.0

    # def randomize_state(self) -> None:
    #     """Randomize qpos/ctrl the same way as the original trigger logic."""
    #     if self.qpos0_gripper is None or self.qpos0_cable is None or self.ctrl0 is None:
    #         raise RuntimeError("Base state not initialized. Call run() first.")

    #     # delta = np.random.uniform(-0.4, 0.4, 3)
    #     delta = np.random.uniform(-0.3, 0.3, 3)
    #     # delta = np.array([0.1, 0, 0])
    #     self.d.qpos[:3] = self.qpos0_gripper + delta
    #     self.d.ctrl[:3] = self.ctrl0 + delta
    #     # self.d.qpos[3:6] = self.qpos0_cable + delta
    #     self.d.qpos[3:6] = self.qpos0_cable + delta
    #     # self.d.qpos[3:6] = self.qpos0_cable + np.array([-delta[1], delta[0], delta[2]])
    #     self.d.qvel[:] = 0.0
    #     # delta = np.random.uniform(-0.1, 0.1, 2)
    #     # self.d.qpos[:2] = self.qpos0 + delta
    #     # self.d.ctrl[:2] = self.ctrl0 + delta
    #     # self.d.qpos[3:5] = self.qpos1 + delta
    #     # self.d.qvel[:] = 0.0
    #     # mj.mj_forward(self.m, self.d)

    def start_recording(self) -> None:
        """Start a new demo: randomize initial state and clear buffers."""
        self.randomize_state()
        self.record_qpos.clear()
        self.record_qvel.clear()
        self.record_ctrl.clear()
        self.record_states.clear()
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
        if len(self.record_qpos) == 0:
            print("Recording is empty; nothing saved.")
            self.recording = False
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        states = list(self.record_states)
        actions = list(self.record_actions)
        next_states = states[1:] + [states[-1]]
        rewards_dense = list(self.record_reward_dense)
        rewards_sparse = list(self.record_reward_sparse)
        rewards = rewards_dense  # legacy key
        terminated = [0] * (len(states) - 1) + [1]

        payload = {
            "timestamp": ts,
            "timestep": float(self.model.opt.timestep),
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
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
        delta_pos_raw = np.array([state.y, -state.x, state.z], dtype=np.float64)
        # delta_raw = np.array([-state.x, -state.y], dtype=np.float64)
        # delta_raw = np.array([state.x, state.z], dtype=np.float64)
        pos_len = len(self.pos_actuator_ids)
        delta_pos = self.ctrl_gain * delta_pos_raw[:pos_len]
        delta_rot = np.zeros(0, dtype=np.float64)
        rot_len = len(self.rot_actuator_ids)
        if rot_len:
            delta_rot_raw = np.array(
                [-state.pitch, state.roll, -state.yaw], dtype=np.float64
            )
            delta_rot = self.ctrl_gain * delta_rot_raw[:rot_len]
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
        self.data.ctrl[:] = ctrl
        self.last_action = delta_action.tolist()

    def run(self):

        RENDER = False

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
        self.randomize_state()
        mj.mj_forward(self.model, self.data)
        # Load initial state
        # self.data.qpos[:] = np.load(self.keys["qpos"])
        # self.data.qvel[:] = np.load(self.keys["qvel"])
        # self.data.ctrl[:] = np.load(self.keys["ctrl"])
        # # mj.mj_forward(self.m, self.d)

        # self.qpos0_gripper = self.data.qpos[:3].copy()
        # self.qpos0_cable = self.data.qpos[3:6].copy()
        # self.ctrl0 = self.data.ctrl[:3].copy()

        # # self.randomize_state()
        # self.trigger = True
        # self.start = True
        # mj.mj_forward(self.model, self.data)

        self.keypoint_ids = [self.model.site(f"keypoint_{i + 1}").id for i in range(3)]
        self.pipe_end_ids = [
            self.model.site(name).id for name in ("pipe_entry", "pipe_exit")
        ]

        def get_keypoints(data: mj.MjData) -> np.ndarray:
            return np.array([data.site(kp_id).xpos for kp_id in self.keypoint_ids])

        def get_pipe_end_points(data: mj.MjData) -> np.ndarray:
            return np.array([data.site(pid).xpos for pid in self.pipe_end_ids])

        i = 0

        data = {"states": [], "actions": []}

        # with open("testing/experiments/pipe_insert/tmp/success_rollout.json", "r") as f:
        #     data = json.load(f)

        obs_seq = None

        if RENDER:
            with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.cb
            ) as viewer:
                viewer.user_scn

                while viewer.is_running():
                    step_start = time.time()

                    pipe_entry, pipe_exit = get_pipe_end_points(self.data)
                    success = keypoints_within_pipe(
                        keypoints=get_keypoints(self.data),
                        pipe_axis_start=pipe_entry,
                        pipe_axis_end=pipe_exit,
                        inner_radius=self.pipe_inner_radius,
                    )
                    # print(f"{success=}")

                    if success:
                        print("success...", i)
                        sys.exit(i)  # success exit code

                    if i > 10_000:
                        print("failed...")
                        sys.exit(-1)  # failure exit code

                    i += 1

                    self.last_action = [0.0] * self.action_dim
                    self.update_ctrl_from_spacemouse()
                    mj.mj_step(self.model, self.data)
                    viewer.sync()

                    # r = self.compute_dense_reward()
                    # dist_to_target = -r
                    # threshold = 0.04
                    # success = dist_to_target <= threshold

                    # print(success, dist_to_target, threshold)

                    # # input()

                    # print(i)
                    # if i > 2500:
                    #     with open(
                    #         "testing/experiments/pipe_insert/tmp/success_rollout.json", "w"
                    #     ) as f:
                    #         json.dump(data, f, indent=4)
                    #     quit()
                    # i += 1

                    # if not self.start:
                    #     continue

                    if self.action_queue:
                        print(f" {i=} {len(data['actions'])=}")
                    # take an action every timestep
                    if len(self.obs_history) == 0:
                        obs = self.get_obs()
                        for _ in range(self.agent.config["obs_horizon"]):
                            self.obs_history.append(obs)
                    obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)

                    with timer(log=False):
                        actions, _, _ = self.agent.act(obs_seq)
                    start = 0
                    # start = self.agent.config["obs_horizon"] - 1
                    action = actions[0, start, :].detach().cpu().numpy()

                    # action = action
                    action = action * self.action_gain
                    # action_full = np.zeros(self.action_dim, dtype=np.float64)
                    # copy_len = min(self.action_dim, action.shape[0])
                    # action_full[:copy_len] = action[:copy_len]
                    # print(f"{action=}")
                    # print(f"{action_full=}")
                    # print(f"{copy_len=}")

                    # input()
                    # data["actions"].append(action.tolist())
                    # data["states"].append(obs_seq.tolist())
                    ctrl = self.data.ctrl.copy()
                    if self.pos_actuator_ids:
                        pos_ids = np.array(self.pos_actuator_ids, dtype=int)
                        pos_len = len(pos_ids)
                        ctrl[pos_ids] = np.clip(
                            ctrl[pos_ids] + action[:pos_len],
                            # ctrl[pos_ids] + action_full[:pos_len],
                            self.ctrl_low[pos_ids],
                            self.ctrl_high[pos_ids],
                        )
                    if self.rot_actuator_ids:
                        rot_ids = np.array(self.rot_actuator_ids, dtype=int)
                        rot_start = len(self.pos_actuator_ids)
                        rot_end = rot_start + len(rot_ids)
                        ctrl[rot_ids] = np.clip(
                            ctrl[rot_ids] + action[rot_start:rot_end],
                            # ctrl[rot_ids] + action_full[rot_start:rot_end],
                            self.ctrl_low[rot_ids],
                            self.ctrl_high[rot_ids],
                        )
                    self.data.ctrl[:] = ctrl
                    self.last_action = action.tolist()

                    if self.trigger:
                        # self.randomize_state()
                        self.trigger = False

                    self.obs_history.append(self.get_obs())
                    self.step_idx += 1

                    # real-time pacing
                    # time_until_next_step = self.model.opt.timestep - (
                    #     time.time() - step_start
                    # )
                    # if time_until_next_step > 0:
                    #     time.sleep(time_until_next_step)

        else:
            while True:
                pipe_entry, pipe_exit = get_pipe_end_points(self.data)
                success = keypoints_within_pipe(
                    keypoints=get_keypoints(self.data),
                    pipe_axis_start=pipe_entry,
                    pipe_axis_end=pipe_exit,
                    inner_radius=self.pipe_inner_radius,
                )
                # print(f"{success=}")

                if success:
                    print("success...", i)
                    sys.exit(i)  # success exit code

                if i > 10_000:
                    print("failed...")
                    sys.exit(-1)  # failure exit code

                print(f"{(i / 10_000) * 100}", end="\r")

                i += 1

                self.last_action = [0.0] * self.action_dim
                self.update_ctrl_from_spacemouse()
                mj.mj_step(self.model, self.data)

                if self.action_queue:
                    print(f" {i=} {len(data['actions'])=}")
                # take an action every timestep
                if len(self.obs_history) == 0:
                    obs = self.get_obs()
                    for _ in range(self.agent.config["obs_horizon"]):
                        self.obs_history.append(obs)
                obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)

                actions, _, _ = self.agent.act(obs_seq)
                start = 0
                # start = self.agent.config["obs_horizon"] - 1
                action = actions[0, start, :].detach().cpu().numpy()

                # action = action
                action = action * self.action_gain

                ctrl = self.data.ctrl.copy()
                if self.pos_actuator_ids:
                    pos_ids = np.array(self.pos_actuator_ids, dtype=int)
                    pos_len = len(pos_ids)
                    ctrl[pos_ids] = np.clip(
                        ctrl[pos_ids] + action[:pos_len],
                        # ctrl[pos_ids] + action_full[:pos_len],
                        self.ctrl_low[pos_ids],
                        self.ctrl_high[pos_ids],
                    )
                if self.rot_actuator_ids:
                    rot_ids = np.array(self.rot_actuator_ids, dtype=int)
                    rot_start = len(self.pos_actuator_ids)
                    rot_end = rot_start + len(rot_ids)
                    ctrl[rot_ids] = np.clip(
                        ctrl[rot_ids] + action[rot_start:rot_end],
                        # ctrl[rot_ids] + action_full[rot_start:rot_end],
                        self.ctrl_low[rot_ids],
                        self.ctrl_high[rot_ids],
                    )
                self.data.ctrl[:] = ctrl
                self.last_action = action.tolist()

                if self.trigger:
                    # self.randomize_state()
                    self.trigger = False

                self.obs_history.append(self.get_obs())
                self.step_idx += 1


def main():
    _env = PipeInsert3()

    m = _env.mj_model
    d = mj.MjData(m)

    sim = Test(m, d)
    sim.run()


if __name__ == "__main__":
    main()
