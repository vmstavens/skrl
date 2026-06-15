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

from performance import timer
from testing.envs.pipe_insert_2 import PipeInsert2
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_dp_config,
    setup_environment,
)
from testing.mj import ObjType, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


class Test:
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        exp_set_seed()
        self.keys = {
            "qpos": "testing/experiments/pipe_insert/constants/v3/qpos.npy",
            "qvel": "testing/experiments/pipe_insert/constants/v3/qvel.npy",
            "ctrl": "testing/experiments/pipe_insert/constants/v3/ctrl.npy",
        }
        model.vis.global_.cameraid = -1

        self.qpos0_gripper = None
        self.qpos0_cable = None
        self.ctrl0 = None

        self.m: mj.MjModel = model
        self.d = data
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
        self.action_gain = 0.17
        self.success_thresh = 0.01
        self.last_action = [0.0, 0.0, 0.0]

        # Clamp ctrl using actuator ctrlrange (expects two actuators).
        self.ctrl_low = self.m.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.m.actuator_ctrlrange[:, 1]

        self.spacemouse_available = False

        while not self.spacemouse_available:
            self.spacemouse_available = pyspacemouse.open()
            time.sleep(1)
            print("waiting for space mouse")

        a_dim = 3
        o_dim = 9

        self.start = False

        env = setup_environment(batch_size=1)

        dp_config = get_dp_config(exp_name="pipe_insert", env=env, wandb=False)

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
            "testing/experiments/pipe_insert/.runs/data_100_2_1_action_horizon/models/latest_model.pth",
            # "testing/experiments/pipe_insert/.runs/data_100_2/models/model_epoch_90.pth",
            device=env.device,
        )
        # performance enhancer -------------------------
        self.agent._num_diffusion_iters = 5
        # self.agent._num_diffusion_iters = 20
        # self.agent._num_diffusion_iters = 3
        # end - performance enhancer -------------------------

        self.agent.to(env.device)
        self.action = np.array([0, 0, 0])
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
            self.d.qpos[:] = np.load(self.keys["qpos"])
            self.d.qvel[:] = np.load(self.keys["qvel"])
            self.d.ctrl[:] = np.load(self.keys["ctrl"])
            mj.mj_forward(self.m, self.d)
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
        T_w_keypoint_1 = get_pose(self.m, self.d, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.m, self.d, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.m, self.d, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.m, self.d, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.m, self.d, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.m, self.d, "target_3", ObjType.SITE)

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

        # delta = np.random.uniform(-0.4, 0.4, 3)
        delta = np.random.uniform(-0.3, 0.3, 3)
        # delta = np.array([0.1, 0, 0])
        self.d.qpos[:3] = self.qpos0_gripper + delta
        self.d.ctrl[:3] = self.ctrl0 + delta
        # self.d.qpos[3:6] = self.qpos0_cable + delta
        self.d.qpos[3:6] = self.qpos0_cable + delta
        # self.d.qpos[3:6] = self.qpos0_cable + np.array([-delta[1], delta[0], delta[2]])
        self.d.qvel[:] = 0.0
        # delta = np.random.uniform(-0.1, 0.1, 2)
        # self.d.qpos[:2] = self.qpos0 + delta
        # self.d.ctrl[:2] = self.ctrl0 + delta
        # self.d.qpos[3:5] = self.qpos1 + delta
        # self.d.qvel[:] = 0.0
        # mj.mj_forward(self.m, self.d)

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
        self.last_action = [0.0, 0.0]

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
            "timestep": float(self.m.opt.timestep),
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
            self.last_action = [0.0, 0.0]
            return
        delta_raw = np.array([state.y, -state.x, state.z], dtype=np.float64)
        # delta_raw = np.array([-state.x, -state.y], dtype=np.float64)
        # delta_raw = np.array([state.x, state.z], dtype=np.float64)
        delta_ctrl = self.ctrl_gain * delta_raw
        if np.allclose(delta_ctrl, 0.0, atol=1e-6):
            self.last_action = [0.0, 0.0]
            return
        ctrl = self.d.ctrl.copy()
        ctrl[:3] = np.clip(ctrl[:3] + delta_ctrl, self.ctrl_low, self.ctrl_high)
        self.d.ctrl[:] = ctrl
        self.last_action = delta_ctrl.tolist()

    def run(self):
        # Load initial state
        self.d.qpos[:] = np.load(self.keys["qpos"])
        self.d.qvel[:] = np.load(self.keys["qvel"])
        self.d.ctrl[:] = np.load(self.keys["ctrl"])
        # mj.mj_forward(self.m, self.d)

        self.qpos0_gripper = self.d.qpos[:3].copy()
        self.qpos0_cable = self.d.qpos[3:6].copy()
        self.ctrl0 = self.d.ctrl[:3].copy()

        # self.randomize_state()
        self.trigger = True
        self.start = True
        mj.mj_forward(self.m, self.d)

        i = 0

        data = {"states": [], "actions": []}

        # with open("testing/experiments/pipe_insert/tmp/success_rollout.json", "r") as f:
        #     data = json.load(f)

        obs_seq = None

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            viewer.user_scn

            while viewer.is_running():
                step_start = time.time()

                self.last_action = [0.0, 0.0]
                self.update_ctrl_from_spacemouse()
                mj.mj_step(self.m, self.d)
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

                if not self.start:
                    continue

                if self.action_queue:
                    print(f" {i=} {len(data["actions"])=}")
                # take an action every timestep
                if len(self.obs_history) == 0:
                    obs = self.get_obs()
                    for _ in range(self.agent.config["obs_horizon"]):
                        self.obs_history.append(obs)
                obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)

                with timer(log=False):
                    actions, _, _ = self.agent.act(obs_seq)
                start = self.agent.config["obs_horizon"] - 1
                action = actions[0, start, :].detach().cpu().numpy()

                # action = action
                action = action * self.action_gain
                # data["actions"].append(action.tolist())
                # data["states"].append(obs_seq.tolist())
                self.d.ctrl[:3] = np.clip(
                    self.d.ctrl[:3] + action, self.ctrl_low, self.ctrl_high
                )

                if self.trigger:
                    # self.randomize_state()
                    self.trigger = False

                if self.recording:
                    self.record_qpos.append(self.d.qpos.copy())
                    self.record_qvel.append(self.d.qvel.copy())
                    self.record_ctrl.append(self.d.ctrl.copy())
                    obs_np = self.get_obs().numpy()
                    self.record_states.append(obs_np.tolist())
                    self.record_actions.append(list(self.last_action))
                    self.record_reward_dense.append(self.compute_dense_reward(obs_np))
                    self.record_reward_sparse.append(self.compute_sparse_reward(obs_np))

                self.obs_history.append(self.get_obs())
                self.step_idx += 1

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
