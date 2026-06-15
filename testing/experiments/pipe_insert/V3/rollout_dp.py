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

from testing.experiments.pipe_insert.V3.env import PipeInsert2, default_config
from testing.mj import ObjType, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy

EXPECTED_RUN_TAG = "align_z_and_pos_x3_16"
EXPECTED_EXPERIMENT_NAME = "pipe_insert_reduced_se3"
EXPECTED_OBS_DIM = 6
EXPECTED_ACTION_DIM = 6


def _latest_checkpoint(runs_dir: Path) -> Path:
    pattern = f"{EXPECTED_RUN_TAG}/models/latest_model.pth"
    candidates = list(runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {runs_dir.as_posix()} matching {pattern}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _validate_policy(policy: DiffusionPolicy, checkpoint_path: Path) -> None:
    experiment_name = policy.config.get("experiment", {}).get("experiment_name")
    if experiment_name != EXPECTED_EXPERIMENT_NAME:
        raise ValueError(
            f"Expected DP experiment '{EXPECTED_EXPERIMENT_NAME}', got "
            f"'{experiment_name}' from {checkpoint_path.as_posix()}"
        )
    if policy._o_dim != EXPECTED_OBS_DIM:
        raise ValueError(
            f"Expected obs dim {EXPECTED_OBS_DIM}, got {policy._o_dim} from "
            f"{checkpoint_path.as_posix()}"
        )
    if policy._a_dim != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"Expected action dim {EXPECTED_ACTION_DIM}, got {policy._a_dim} from "
            f"{checkpoint_path.as_posix()}"
        )
    if policy.stats is None:
        raise ValueError(f"DP checkpoint has no normalization stats: {checkpoint_path}")
    obs_stats = policy.stats.get("obs", {})
    action_stats = policy.stats.get("action", {})
    if tuple(obs_stats.get("min", ()).shape) != (EXPECTED_OBS_DIM,):
        raise ValueError(
            f"Expected obs stats shape ({EXPECTED_OBS_DIM},), got "
            f"{getattr(obs_stats.get('min'), 'shape', None)}"
        )
    if tuple(action_stats.get("min", ()).shape) != (EXPECTED_ACTION_DIM,):
        raise ValueError(
            f"Expected action stats shape ({EXPECTED_ACTION_DIM},), got "
            f"{getattr(action_stats.get('min'), 'shape', None)}"
        )


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
    num_diffusion_iters: Optional[int] = 4
    pred_horizon: Optional[int] = 16
    record_path: Optional[Path] = Path(
        "testing/experiments/pipe_insert/V3/data/rollout_dp_reduced_se3.json"
    )
    randomize_start: bool = True
    impl: str = "warp"
    device: Optional[str] = None


class DPRollout:
    def __init__(
        self,
        env: PipeInsert2,
        model: mj.MjModel,
        data: mj.MjData,
        policy: DiffusionPolicy,
        args: Args,
    ):
        self.env = env
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
        self.qvel0 = None
        self.ctrl0 = None
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
        self.rollout_terminations: List[bool] = []
        self.rollout_position_errors: List[float] = []
        self.rollout_orientation_errors: List[float] = []
        self.rollout_time: List[float] = []
        self.episode_step = 0
        self.episode_time0 = 0.0

        self._w_pos = float(self.env._w_pos)
        self._w_rot = float(self.env._w_rot)
        self._sparse_reward = bool(self.env._sparse_reward)
        self._episode_length = int(self.env._episode_length)
        self._termination_threshold = tuple(
            float(x) for x in self.env._termination_threshold
        )

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

    def _randomize_state(self) -> None:
        if self.qpos0 is None or self.qvel0 is None:
            raise RuntimeError("Base state not initialized. Call run() first.")
        delta = np.random.uniform(-0.1, 0.1, 3)
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
            self.d.mocap_quat[self.mocap_id] = self.mocap_quat0
        mj.mj_forward(self.m, self.d)

    def _reset_episode(self) -> None:
        if self.args.randomize_start:
            self._randomize_state()
        self.rollout_states.clear()
        self.rollout_actions.clear()
        self.rollout_rewards.clear()
        self.rollout_terminations.clear()
        self.rollout_position_errors.clear()
        self.rollout_orientation_errors.clear()
        self.rollout_time.clear()
        self.episode_step = 0
        self.episode_time0 = float(self.d.time)
        self.obs_history.clear()
        self.action_queue.clear()
        obs = self.get_obs()
        for _ in range(self.policy.config["obs_horizon"]):
            self.obs_history.append(obs)

    def get_obs(self) -> torch.Tensor:
        T_w_target = get_pose(self.m, self.d, "target", ObjType.SITE)
        T_w_keypoint = get_pose(self.m, self.d, "keypoint", ObjType.SITE)
        dp_world = T_w_target.t - T_w_keypoint.t
        dp = T_w_keypoint.R.T @ dp_world
        z_rel = T_w_keypoint.R.T @ T_w_target.R[:, 2]
        obs = np.concatenate([dp, z_rel], axis=0)
        return torch.from_numpy(obs.astype(np.float32))

    def _compute_reward(self, obs: torch.Tensor, terminated: bool) -> float:
        if self._sparse_reward:
            return float(terminated)
        e_pos = obs[:3]
        rot_err = self._z_axis_alignment_error(obs)
        return float(
            -(self._w_pos * torch.linalg.norm(e_pos).item() + self._w_rot * rot_err)
        )

    def _z_axis_alignment_error(self, obs: torch.Tensor) -> float:
        z_dot = float(torch.clamp(obs[5], -1.0, 1.0).item())
        return float(np.arccos(z_dot))

    def _get_error_terms(self, obs: torch.Tensor) -> tuple[float, float]:
        pos_err = float(torch.linalg.norm(obs[:3]).item())
        rot_err = self._z_axis_alignment_error(obs)
        return pos_err, rot_err

    def _get_termination(self, obs: torch.Tensor) -> bool:
        is_unstable = np.isnan(self.d.qpos).any() or np.isnan(self.d.qvel).any()
        timeout = self.episode_step >= self._episode_length
        print(self.episode_step, self._episode_length)

        pos_err, rot_err = self._get_error_terms(obs)
        pos_thresh, rot_thresh = self._termination_threshold
        # pos_thresh = 2
        # rot_thresh = np.deg2rad(20)
        success = (pos_err < pos_thresh) and (rot_err < rot_thresh)

        print(
            f"{pos_err:.3f}",
            f"{rot_err:.3f}",
            (pos_err < pos_thresh),
            (rot_err < rot_thresh),
        )

        return bool(is_unstable or timeout or success)

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
        self, obs: torch.Tensor, action: np.ndarray, reward: float, terminated: bool
    ) -> None:
        pos_err, rot_err = self._get_error_terms(obs)
        self.rollout_states.append(obs.detach().cpu().numpy().tolist())
        self.rollout_actions.append(action.tolist())
        self.rollout_rewards.append(float(reward))
        self.rollout_terminations.append(bool(terminated))
        self.rollout_position_errors.append(pos_err)
        self.rollout_orientation_errors.append(rot_err)
        self.rollout_time.append(float(self.d.time) - self.episode_time0)

    def _save_rollout(self, path: Path) -> None:
        n = min(
            len(self.rollout_states),
            len(self.rollout_actions),
            len(self.rollout_rewards),
            len(self.rollout_terminations),
            len(self.rollout_position_errors),
            len(self.rollout_orientation_errors),
            len(self.rollout_time),
        )
        states = list(self.rollout_states)[:n]
        actions = list(self.rollout_actions)[:n]
        rewards = list(self.rollout_rewards)[:n]
        terminated = [int(flag) for flag in self.rollout_terminations[:n]]
        position_errors = list(self.rollout_position_errors)[:n]
        orientation_errors = list(self.rollout_orientation_errors)[:n]
        time_steps = list(self.rollout_time)[:n]
        if n > 0:
            next_states = states[1:] + [states[-1]]
        else:
            next_states = []

        payload = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestep": float(self.m.opt.timestep),
            "observation_type": "align_z_and_pos",
            "include_z_dot": True,
            "observations": states,
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "terminations": terminated,
            "terminated": terminated,
            "position_errors": position_errors,
            "orientation_errors": orientation_errors,
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
                terminated = self._get_termination(obs)
                reward = self._compute_reward(obs, terminated)
                self._record_step(obs, action, reward, terminated)
                self.obs_history.append(obs)
                self.episode_step += 1
                if terminated:
                    if self.args.record_path is not None:
                        self._save_rollout(self.args.record_path)
                    break
            return

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running() and not self._exit:
                step_start = time.time()

                if self._reset:
                    self._reset = False
                    self._reset_episode()

                self._maybe_enqueue_actions()
                action = self.action_queue.popleft()
                action = action * self.args.action_scale
                self._apply_action(action)

                mj.mj_step(self.m, self.d)
                viewer.sync()

                obs = self.get_obs()
                terminated = self._get_termination(obs)
                reward = self._compute_reward(obs, terminated)
                self._record_step(obs, action, reward, terminated)
                self.obs_history.append(obs)
                self.episode_step += 1

                if terminated:
                    print("Termination reached.")
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
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    policy = DiffusionPolicy.load(checkpoint_path.as_posix(), device=device)
    policy.to(device)
    policy.set_mode("eval")
    _validate_policy(policy, checkpoint_path)
    print(
        "Loaded DP checkpoint "
        f"{checkpoint_path.as_posix()} "
        f"(experiment={policy.config['experiment']['experiment_name']}, "
        f"obs_dim={policy._o_dim}, action_dim={policy._a_dim}, "
        f"obs_horizon={policy.config['obs_horizon']}, "
        f"pred_horizon={policy.config['pred_horizon']}, "
        f"action_horizon={policy.config['action_horizon']})"
    )

    if args.num_diffusion_iters is not None:
        policy._num_diffusion_iters = int(args.num_diffusion_iters)
        policy.config["num_diffusion_iters"] = policy._num_diffusion_iters

    if args.pred_horizon is not None:
        policy._pred_horizon = int(args.pred_horizon)
        policy.config["pred_horizon"] = policy._pred_horizon

    sim_cfg = default_config()
    sim_cfg.impl = args.impl
    sim_cfg.episode_length = args.max_steps
    sim_env = PipeInsert2(config=sim_cfg)
    m = sim_env.mj_model
    d = mj.MjData(m)

    rollout = DPRollout(sim_env, m, d, policy, args)
    rollout.run()

    if args.record_path is not None and not rollout._saved:
        rollout._save_rollout(args.record_path)


if __name__ == "__main__":
    main()
