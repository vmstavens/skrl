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
from testing.experiments.pipe_insert.exp_utils import get_dp_config, setup_environment
from testing.mj import ObjType, get_names, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

from .physics import amperes_law, biot_savarts_law


class Test:
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        self.keys = {
            "qpos": "testing/experiments/pipe_insert/constants/v3/qpos.npy",
            "qvel": "testing/experiments/pipe_insert/constants/v3/qvel.npy",
            "ctrl": "testing/experiments/pipe_insert/constants/v3/ctrl.npy",
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

        self.m: mj.MjModel = model
        self.d = data
        self.trigger = False
        self.recording = False
        self.record_states_sparse: List[List[float]] = []
        self.record_states_dense: List[List[float]] = []
        self.record_actions: List[List[float]] = []
        self.record_reward_dense: List[float] = []
        self.record_reward_sparse: List[float] = []
        self.record_dir = Path(__file__).resolve().parent / "data_collection_2"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.ctrl_gain = 0.01 / 4
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
        o_dim = 3

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
            "testing/experiments/pipe_insert/.runs/data_100_2/models/model_epoch_90.pth",
            device=env.device,
        )
        self.action = np.array([0, 0, 0])
        self.dp = False
        self.action_queue: deque[np.ndarray] = deque()
        self.obs_history: deque[torch.Tensor] = deque(
            maxlen=self.agent.config["obs_horizon"]
        )
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
            np.save(
                "testing/experiments/pipe_insert/constants/v3/qpos.npy", self.d.qpos
            )
            np.save(
                "testing/experiments/pipe_insert/constants/v3/qvel.npy", self.d.qvel
            )
            np.save(
                "testing/experiments/pipe_insert/constants/v3/ctrl.npy", self.d.ctrl
            )
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
        """Randomize qpos/ctrl the same way as the original trigger logic."""
        print(0)
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
        print(delta)
        # mj.mj_forward(self.m, self.d)

    def start_recording(self) -> None:
        """Start a new demo: randomize initial state and clear buffers."""
        self.randomize_state()
        self.record_states_sparse.clear()
        self.record_actions.clear()
        self.record_reward_dense.clear()
        self.record_reward_sparse.clear()
        self.recording = True
        print("Recording started (press 'S' to save).")
        self.last_action = [0.0, 0.0, 0.0]

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

    def update_ctrl_from_spacemouse(self) -> None:
        """Update ctrl by integrating delta from the SpaceMouse."""
        if not self.spacemouse_available:
            return
        state = pyspacemouse.read()
        if state is None:
            self.last_action = [0.0, 0.0, 0.0]
            return
        delta_raw = np.array([state.x, state.y, state.z], dtype=np.float64)
        # delta_raw = np.array([-state.x, -state.y], dtype=np.float64)
        # delta_raw = np.array([state.x, state.z], dtype=np.float64)
        delta_ctrl = self.ctrl_gain * delta_raw
        if np.allclose(delta_ctrl, 0.0, atol=1e-6):
            self.last_action = [0.0, 0.0, 0.0]
            return
        ctrl = self.d.ctrl.copy()

        ctrl[:3] = np.clip(ctrl[:3] + delta_ctrl, self.ctrl_low, self.ctrl_high)
        # print(ctrl)
        self.d.ctrl[:] = ctrl
        self.last_action = delta_ctrl.tolist()

    def run(self):
        # Load initial state
        self.d.qpos[:] = np.load(self.keys["qpos"])
        self.d.qvel[:] = np.load(self.keys["qvel"])
        self.d.ctrl[:] = np.load(self.keys["ctrl"])

        self.qpos0_gripper = self.d.qpos[:3].copy()
        self.qpos0_cable = self.d.qpos[3:6].copy()
        self.ctrl0 = self.d.ctrl[:3].copy()
        mj.mj_forward(self.m, self.d)

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            viewer.user_scn

            while viewer.is_running():
                step_start = time.time()

                self.last_action = [0.0, 0.0, 0.0]
                self.update_ctrl_from_spacemouse()
                mj.mj_step(self.m, self.d)
                viewer.sync()

                if self.action_queue:
                    action = self.action_queue.popleft()
                    self.d.ctrl[:3] = np.clip(
                        self.d.ctrl[:3] + action, self.ctrl_low, self.ctrl_high
                    )
                    # OBS: ---- experimental
                    self.action_queue.clear()
                    print(f"moving {action}")

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

                self.obs_history.append(self.get_obs_sparse())

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
