from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
import torch
import tyro

from testing.experiments.pipe_insert.exp_utils import get_dp_config, setup_environment
from testing.experiments.pipe_insert.V2.env import PipeInsert2
from testing.mj import ObjType, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


def _latest_checkpoint(runs_dir: Path) -> Path:
    pattern = "data_mocap_x4/models/latest_model.pth"
    # pattern = "data_mocap_*/models/latest_model.pth"
    candidates = list(runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {runs_dir.as_posix()} matching {pattern}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


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


@dataclass
class Args:
    checkpoint: Optional[Path] = None
    max_steps: int = 1000
    action_scale: float = 1.0
    render: bool = True
    stop_on_success: bool = True
    success_thresh: float = 0.01
    num_diffusion_iters: Optional[int] = None
    pred_horizon: Optional[int] = None
    record_path: Optional[Path] = Path(
        "testing/experiments/pipe_insert/V2/data/rollout_dp.json"
    )
    randomize_start: bool = False


class DPRollout:
    def __init__(
        self, model: mj.MjModel, data: mj.MjData, policy: DiffusionPolicy, args: Args
    ):
        self.m = model
        self.d = data
        self.policy = policy
        self.args = args

        self.mocap_id = None
        try:
            self.mocap_id = int(self.m.body("mocap").mocapid)
            if self.mocap_id < 0:
                self.mocap_id = None
        except Exception:
            self.mocap_id = None

        self.cable_root_qpos_adr = None
        try:
            cable_root_jnt = self.m.joint("cable:free").id
            self.cable_root_qpos_adr = int(self.m.jnt_qposadr[cable_root_jnt])
        except Exception:
            self.cable_root_qpos_adr = None

        self.qpos0 = None
        self.qpos0_root = None
        self.mocap_pos0 = None
        self.mocap_quat0 = None
        self.mocap_range = np.array([0.3, 0.3, 0.3], dtype=np.float64)
        self.mocap_low = None
        self.mocap_high = None

        self.keys = {
            "qpos": "testing/experiments/pipe_insert/constants/v4/qpos.npy",
            "qvel": "testing/experiments/pipe_insert/constants/v4/qvel.npy",
            "ctrl": "testing/experiments/pipe_insert/constants/v4/ctrl.npy",
            "mpos": "testing/experiments/pipe_insert/constants/v4/mpos.npy",
            "mquat": "testing/experiments/pipe_insert/constants/v4/mquat.npy",
        }

        self.obs_history: Deque[torch.Tensor] = deque(
            maxlen=self.policy.config["obs_horizon"]
        )
        self.action_queue: Deque[np.ndarray] = deque()

        self.rollout_states: List[List[float]] = []
        self.rollout_actions: List[List[float]] = []
        self.rollout_rewards: List[float] = []
        self.rollout_time: List[float] = []

        self._exit = False
        self._reset = False
        self._saved = False

    def cb(self, key: int) -> None:
        if key == glfw.KEY_Q:
            self._exit = True
        elif key == glfw.KEY_R:
            self._reset = True
        elif key == glfw.KEY_S and self.args.record_path is not None:
            self._save_rollout(self.args.record_path)

    def _init_state(self) -> None:
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

    def _randomize_state(self) -> None:
        if self.qpos0 is None or self.mocap_pos0 is None or self.mocap_quat0 is None:
            raise RuntimeError("Base state not initialized. Call run() first.")
        delta = np.random.uniform(-0, 0, 3)
        # delta = np.random.uniform(-0.03, 0.03, 3)
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
        mj.mj_forward(self.m, self.d)

    def _reset_episode(self) -> None:
        if self.args.randomize_start:
            self._randomize_state()
        self.obs_history.clear()
        self.action_queue.clear()
        obs = self.get_obs()
        for _ in range(self.policy.config["obs_horizon"]):
            self.obs_history.append(obs)

    def get_obs(self) -> torch.Tensor:
        T_w_keypoint = get_pose(self.m, self.d, "keypoint", ObjType.SITE)
        T_w_target = get_pose(self.m, self.d, "target", ObjType.SITE)
        d = (T_w_target.t - T_w_keypoint.t).astype(np.float32)
        return torch.from_numpy(d)

    def _compute_reward(self, obs: torch.Tensor) -> float:
        return float(-torch.linalg.norm(obs).item())

    def _apply_action(self, action: np.ndarray) -> None:
        if self.mocap_id is None:
            return
        pos_delta = action[:3]
        rot_delta = action[3:6]

        target_pos = self.d.mocap_pos[self.mocap_id] + pos_delta
        if self.mocap_low is not None and self.mocap_high is not None:
            target_pos = np.clip(target_pos, self.mocap_low, self.mocap_high)
        self.d.mocap_pos[self.mocap_id] = target_pos

        current = self.d.mocap_quat[self.mocap_id].copy()
        delta_quat = _rotvec_to_quat(rot_delta)
        target_quat = _quat_mul(delta_quat, current)
        self.d.mocap_quat[self.mocap_id] = _quat_normalize(target_quat)

    def _maybe_enqueue_actions(self) -> None:
        if self.action_queue:
            return

        obs_seq = (
            torch.stack(list(self.obs_history))
            .unsqueeze(0)
            .to(self.policy.device, dtype=torch.float32)
        )
        with torch.no_grad():
            actions, _, _ = self.policy.act(states=obs_seq)
        obs_horizon = self.policy.config["obs_horizon"]
        action_horizon = self.policy.config.get(
            "action_horizon", self.policy.config.get("pred_horizon", 1)
        )
        start = obs_horizon - 1
        end = start + action_horizon
        action_seq = actions[0, start:end, :].detach().cpu().numpy()
        for act in action_seq:
            self.action_queue.append(act)

    def _record_step(
        self, obs: torch.Tensor, action: np.ndarray, reward: float
    ) -> None:
        self.rollout_states.append(obs.detach().cpu().numpy().tolist())
        self.rollout_actions.append(action.tolist())
        self.rollout_rewards.append(float(reward))
        self.rollout_time.append(float(self.d.time))

    def _save_rollout(self, path: Path) -> None:
        actions = list(self.rollout_actions)
        n = len(actions)
        states = list(self.rollout_states)[:n]
        rewards = list(self.rollout_rewards)[:n]
        time_steps = list(self.rollout_time)[:n]
        if n > 0:
            next_states = states[1:] + [states[-1]]
            terminated = [0] * (n - 1) + [1]
        else:
            next_states = []
            terminated = []

        payload = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestep": float(self.m.opt.timestep),
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "terminated": terminated,
            "time": time_steps,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"Saved rollout data to {path.as_posix()}")
        self._saved = True

    def run(self) -> None:
        self._init_state()
        self._reset_episode()

        if not self.args.render:
            for _ in range(self.args.max_steps):
                self._maybe_enqueue_actions()
                action = self.action_queue.popleft()
                action = action * self.args.action_scale
                self._apply_action(action)
                mj.mj_step(self.m, self.d)
                obs = self.get_obs()
                reward = self._compute_reward(obs)
                self._record_step(obs, action, reward)
                self.obs_history.append(obs)
                if self.args.stop_on_success and reward >= -self.args.success_thresh:
                    if self.args.record_path is not None:
                        self._save_rollout(self.args.record_path)
                    break
            return

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            input("ready?")
            i = 0
            while viewer.is_running() and not self._exit:
                step_start = time.time()

                if self._reset:
                    self._reset = False
                    self._reset_episode()

                self._maybe_enqueue_actions()
                action = self.action_queue.popleft()
                action = action * self.args.action_scale
                self._apply_action(action)

                print(i)
                i += 1
                mj.mj_step(self.m, self.d)
                viewer.sync()

                obs = self.get_obs()
                reward = self._compute_reward(obs)
                self._record_step(obs, action, reward)
                self.obs_history.append(obs)

                if self.args.stop_on_success and reward >= -self.args.success_thresh:
                    print("Success threshold reached.")
                    if self.args.record_path is not None:
                        self._save_rollout(self.args.record_path)
                    break

                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main(args: Optional[Args] = None) -> None:
    if args is None:
        args = tyro.cli(Args)

    runs_dir = Path(__file__).parent / ".runs"
    checkpoint_path = args.checkpoint or _latest_checkpoint(runs_dir)
    env = setup_environment(batch_size=1)
    dp_config = get_dp_config(
        exp_name="pipe_insert_2_rot_mocap_rollout", env=env, wandb=False
    )
    dp_config["obs_dim"] = env.observation_space.shape[0]
    dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]

    a_dim = env.action_space.shape[0]
    o_dim = dp_config["obs_dim"]
    dp_models: dict = {}
    dp_models["model"] = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config)
    ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
    dp_models["ema_model"] = ConditionalUnet1D(
        a_dim=a_dim, o_dim=o_dim, config=dp_config
    )
    policy = DiffusionPolicy(
        a_dim=a_dim,
        o_dim=o_dim,
        models=dp_models,
        ema=ema,
        device=env.device,
        config=dp_config,
    )
    policy = policy.load(checkpoint_path.as_posix(), device=env.device)
    policy.to(env.device)

    if args.num_diffusion_iters is None:
        policy._num_diffusion_iters = 4
    else:
        policy._num_diffusion_iters = int(args.num_diffusion_iters)

    if args.pred_horizon is None:
        policy._pred_horizon = 8
    else:
        policy._pred_horizon = int(args.pred_horizon)

    sim_env = PipeInsert2()
    m = sim_env.mj_model
    d = mj.MjData(m)

    rollout = DPRollout(m, d, policy, args)
    rollout.run()

    if args.record_path is not None and not rollout._saved:
        rollout._save_rollout(args.record_path)


if __name__ == "__main__":
    main()
