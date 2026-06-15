from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, REPO_ROOT.as_posix())

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
import torch
import tyro

from skrl.resources.preprocessors.torch.running_standard_scaler import (
    RunningStandardScaler,
)
from testing.experiments.pipe_insert.V4.env import (
    PipeInsert2,
    default_config,
    reward_metrics_from_error_terms,
)
from testing.experiments.pipe_insert.V4.train_drlr_both_dr import (
    load_trained_dp_policy,
    override_action_space,
    setup_mocap_environment,
)
from testing.experiments.pipe_insert.V4.utils import (
    exp_set_seed,
    get_expert_memory_2,
    get_ibrl_sac_dp_config,
    get_memory,
    get_sac_models,
)
from testing.mj import ObjType, get_pose
from testing.shen.drlr_sac_o_o2_v2 import DRLR

# /home/vims/git/skrl/testing/experiments/pipe_insert/V4/.runs/train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260420_08_50_29/models/train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260420_08_50_29/checkpoints/best_agent.pt
# testing/experiments/pipe_insert/V4/.runs/train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260420_08_50_29/models/train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260420_08_50_29/checkpoints/agent_80000.pt
DEFAULT_DRLR_RUN = Path(
    "testing/experiments/pipe_insert/V4/.runs/"
    "train_drlr_both_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260512_13_25_48"  # other
    # "train_drlr_both_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260511_20_14_11"  # other
    # "train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260420_08_50_29"  # other
    # "train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260421_13_58_52"  # orange
    # "train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260424_08_55_59"  # green
    # "train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260424_13_08_28"
    # "train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260421_09_27_05"
    # "train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260423_09_06_34"
)
# train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260424_13_08_28
DEFAULT_DRLR_CHECKPOINT = (
    DEFAULT_DRLR_RUN
    / "models"
    / DEFAULT_DRLR_RUN.name
    / "checkpoints"
    # / "agent_80000.pt"
    / "agent_70000.pt"
    # / "best_agent.pt"
)
DEFAULT_DP_CHECKPOINT = Path(
    "testing/experiments/pipe_insert/V4/.runs/align_z_and_pos_x3_16/models/latest_model.pth"
)


def _latest_drlr_checkpoint(runs_dir: Path) -> Path:
    patterns = [
        "train_drlr_both*/models/*/checkpoints/best_agent.pt",
        "train_drlr_both*/models/*/checkpoints/agent_*.pt",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No DRLR checkpoints found in {runs_dir.as_posix()} matching {patterns}"
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


def _as_float(value: Any) -> float:
    array = np.asarray(value)
    if array.shape:
        array = array.reshape(-1)
        if array.size != 1:
            raise ValueError(f"Expected scalar reward metric, got shape {array.shape}")
        return float(array[0])
    return float(array.item())


# DRLR: testing/experiments/pipe_insert/V4/.runs/train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260421_09_27_05/models/train_drlr_both_dr_actor_both_align_z_and_pos_warmup_timesteps_2000_reward_scale_0.1_z_align_1_20260421_09_27_05/checkpoints/best_agent.pt
# DP: testing/experiments/pipe_insert/V4/.runs/align_z_and_pos_x3_16/models/latest_model.pth


@dataclass
class Args:
    checkpoint: Optional[Path] = DEFAULT_DRLR_CHECKPOINT
    dp_checkpoint: Optional[Path] = DEFAULT_DP_CHECKPOINT
    expert_data_dir: Path = Path(
        "testing/experiments/pipe_insert/V4/datasets/align_z_and_pos"
    )
    max_steps: int = 15000
    # max_steps: int = 1500
    render: bool = True
    debug_termination: bool = True
    stop_on_success: bool = True
    success_thresh: float = 0.01
    record_path: Optional[Path] = Path(
        "testing/experiments/pipe_insert/V4/data/rollout_drlr.json"
    )
    randomize_start: bool = False
    impl: str = "warp"
    actor: str = "both"
    dp_iters: int = 4
    dp_pred_horizon: int = 16
    warmup_timesteps: int = 2_000
    soft_update_beta: float = 0.2
    decision_block: bool = True
    action_dim_override: int = 6
    a_max_lim: float = 1.0
    a_min_lim: float = -1.0
    action_trans_high: float = 0.005
    action_trans_low: float = -0.005
    action_rot_high: float = 0.007
    action_rot_low: float = -0.007
    cable_stiffness_min_scale: float = 0.01
    cable_stiffness_max_scale: float = 10.0
    cable_stiffness_log_uniform: bool = True
    cable_length_min_scale: float = 0.8
    cable_length_max_scale: float = 1.2


def _build_agent(args: Args) -> DRLR:
    exp_set_seed(10)
    # exp_set_seed(100)

    env = setup_mocap_environment(
        batch_size=1,
        episode_length=args.max_steps,
        auto_reset=True,
        action_repeat=1,
        impl=args.impl,
        cable_stiffness_min_scale=args.cable_stiffness_min_scale,
        cable_stiffness_max_scale=args.cable_stiffness_max_scale,
        cable_stiffness_log_uniform=args.cable_stiffness_log_uniform,
        cable_length_min_scale=args.cable_length_min_scale,
        cable_length_max_scale=args.cable_length_max_scale,
    )
    env = override_action_space(
        env=env,
        action_dim=args.action_dim_override,
        low=[args.a_min_lim] * args.action_dim_override,
        high=[args.a_max_lim] * args.action_dim_override,
        num_envs=1,
    )

    runs_dir = Path(__file__).parent / ".runs"
    exp_name = "rollout_drlr"

    cfg = get_ibrl_sac_dp_config(exp_name=exp_name, env=env, wandb=False)
    cfg["actor"] = args.actor
    cfg["warmup_timesteps"] = args.warmup_timesteps
    cfg["decision_block"] = args.decision_block
    cfg["soft_update_beta"] = args.soft_update_beta
    cfg["action_trans_high"] = args.action_trans_high
    cfg["action_trans_low"] = args.action_trans_low
    cfg["action_rot_high"] = args.action_rot_high
    cfg["action_rot_low"] = args.action_rot_low
    cfg["a_max_lim"] = args.a_max_lim
    cfg["a_min_lim"] = args.a_min_lim
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": env.device,
    }
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0

    dp_ckpt = args.dp_checkpoint or DEFAULT_DP_CHECKPOINT
    dp_policy = load_trained_dp_policy(
        checkpoint_path=dp_ckpt,
        device=env.device,
    )
    if dp_policy._a_dim != env.action_space.shape[0]:
        raise ValueError(
            f"DP action dim {dp_policy._a_dim} does not match env action dim "
            f"{env.action_space.shape[0]}"
        )
    if dp_policy._o_dim != env.observation_space.shape[0]:
        raise ValueError(
            f"DP observation dim {dp_policy._o_dim} does not match env observation dim "
            f"{env.observation_space.shape[0]}"
        )
    dp_policy._num_diffusion_iters = args.dp_iters
    dp_policy._pred_horizon = args.dp_pred_horizon
    dp_policy._act_horizon = 1
    dp_policy.config["num_diffusion_iters"] = dp_policy._num_diffusion_iters
    dp_policy.config["pred_horizon"] = dp_policy._pred_horizon
    dp_policy.config["action_horizon"] = dp_policy._act_horizon

    rl_models = get_sac_models(env)
    memory = get_memory(env, capacity=max(args.max_steps, 1))
    expert_memory = get_expert_memory_2(
        expert_data_dir=str(args.expert_data_dir),
        states_label="states",
        actions_label="actions",
        rewards_label="rewards",
        next_states_label="next_states",
        dones_label="terminated",
    )

    agent = DRLR(
        models=rl_models,
        models_il={"policy": dp_policy},
        memory=memory,
        expert_memory=expert_memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    agent.init(trainer_cfg={"timesteps": 1})

    checkpoint_path = args.checkpoint or _latest_drlr_checkpoint(runs_dir)

    # print(f"{checkpoint_path=}")
    # quit()

    agent.load(checkpoint_path.as_posix())
    agent.set_mode("eval")
    return agent


class DRLRRollout:
    def __init__(
        self,
        env: PipeInsert2,
        model: mj.MjModel,
        data: mj.MjData,
        agent: DRLR,
        args: Args,
    ):
        self.env = env
        self.m = model
        self.d = data
        self.agent = agent
        self.args = args

        self.agent._warmup_timesteps = 0

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

        self.rollout_states: List[List[float]] = []
        self.rollout_actions: List[List[float]] = []
        self.rollout_rewards: List[float] = []
        self.rollout_reward_metrics: List[Dict[str, float]] = []
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
        delta = np.random.uniform(-0.03, 0.03, 3)
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

    def _reset_episode(self) -> torch.Tensor:
        if self.args.randomize_start:
            self._randomize_state()
        self.rollout_states.clear()
        self.rollout_actions.clear()
        self.rollout_rewards.clear()
        self.rollout_reward_metrics.clear()
        self.rollout_terminations.clear()
        self.rollout_position_errors.clear()
        self.rollout_orientation_errors.clear()
        self.rollout_time.clear()
        self.episode_step = 0
        self.episode_time0 = float(self.d.time)
        obs = self.get_obs()
        self.agent._states = None
        self.agent._prev_states = None
        return obs

    def get_obs(self) -> torch.Tensor:
        t_w_target = get_pose(self.m, self.d, "target", ObjType.SITE)
        t_w_keypoint = get_pose(self.m, self.d, "keypoint", ObjType.SITE)

        e_pos_world = t_w_keypoint.t - t_w_target.t
        e_pos = t_w_target.R.T @ e_pos_world
        z_rel = t_w_target.R.T @ t_w_keypoint.R[:, 2]
        z_rel = z_rel / max(np.linalg.norm(z_rel), 1e-6)

        obs = np.concatenate([e_pos, z_rel], axis=0)
        return torch.from_numpy(obs.astype(np.float32))

    def _compute_reward(
        self, prev_obs: torch.Tensor, obs: torch.Tensor, terminated: bool
    ) -> float:
        return self._compute_reward_metrics(prev_obs, obs, terminated)["reward_total"]

    def _compute_reward_metrics(
        self, prev_obs: torch.Tensor, obs: torch.Tensor, terminated: bool
    ) -> Dict[str, float]:
        pos_err0, rot_err0 = self._get_error_terms(prev_obs)
        pos_err1, rot_err1 = self._get_error_terms(obs)
        metrics = {
            key: _as_float(value)
            for key, value in reward_metrics_from_error_terms(
                pos_err0=pos_err0,
                rot_err0=rot_err0,
                pos_err1=pos_err1,
                rot_err1=rot_err1,
                done=float(terminated),
                termination_threshold_pos=self._termination_threshold[0],
                termination_threshold_rot=self._termination_threshold[1],
            ).items()
        }
        if self._sparse_reward:
            metrics = {key: 0.0 for key in metrics}
            metrics["reward_total"] = float(terminated)
        return metrics

    def _get_error_terms(self, obs: torch.Tensor) -> tuple[float, float]:
        pos_err = float(torch.linalg.norm(obs[:3]).item())
        z_dot = float(torch.clamp(obs[5], -1.0, 1.0).item())
        rot_err = float(np.arccos(z_dot))
        return pos_err, rot_err

    def _get_termination(self, obs: torch.Tensor) -> bool:
        is_unstable = np.isnan(self.d.qpos).any() or np.isnan(self.d.qvel).any()

        pos_err, rot_err = self._get_error_terms(obs)
        pos_thresh, rot_thresh = self._termination_threshold
        success = (pos_err < pos_thresh) and (rot_err < rot_thresh)

        if self.args.debug_termination and (is_unstable or success):
            reasons = []
            if is_unstable:
                reasons.append("unstable")
            if success:
                reasons.append(
                    "success "
                    f"(pos_err={pos_err:.6f} < {pos_thresh:.6f}, "
                    f"rot_err={rot_err:.6f} < {rot_thresh:.6f})"
                )
            print(
                "Termination reason: "
                + ", ".join(reasons)
                + f" at episode_step={self.episode_step}"
            )

        return bool(is_unstable or success)

    def _should_stop(self, terminated: bool, obs: torch.Tensor) -> bool:
        if not terminated:
            return False
        is_unstable = np.isnan(self.d.qpos).any() or np.isnan(self.d.qvel).any()
        if is_unstable:
            return True

        pos_err, rot_err = self._get_error_terms(obs)
        pos_thresh, rot_thresh = self._termination_threshold
        success = (pos_err < pos_thresh) and (rot_err < rot_thresh)
        return self.args.stop_on_success or not success

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

    def _record_step(
        self,
        obs: torch.Tensor,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        reward_metrics: Dict[str, float],
    ) -> None:
        pos_err, rot_err = self._get_error_terms(obs)
        self.rollout_states.append(obs.detach().cpu().numpy().tolist())
        self.rollout_actions.append(action.tolist())
        self.rollout_rewards.append(float(reward))
        self.rollout_reward_metrics.append(dict(reward_metrics))
        self.rollout_terminations.append(bool(terminated))
        self.rollout_position_errors.append(pos_err)
        self.rollout_orientation_errors.append(rot_err)
        self.rollout_time.append(float(self.d.time) - self.episode_time0)

    def _save_rollout(self, path: Path) -> None:
        n = min(
            len(self.rollout_states),
            len(self.rollout_actions),
            len(self.rollout_rewards),
            len(self.rollout_reward_metrics),
            len(self.rollout_terminations),
            len(self.rollout_position_errors),
            len(self.rollout_orientation_errors),
            len(self.rollout_time),
        )
        states = list(self.rollout_states)[:n]
        actions = list(self.rollout_actions)[:n]
        rewards = list(self.rollout_rewards)[:n]
        reward_metric_records = list(self.rollout_reward_metrics)[:n]
        terminated = [int(flag) for flag in self.rollout_terminations[:n]]
        position_errors = list(self.rollout_position_errors)[:n]
        orientation_errors = list(self.rollout_orientation_errors)[:n]
        time_steps = list(self.rollout_time)[:n]
        reward_metric_names = []
        seen_reward_metrics = set()
        for record in reward_metric_records:
            for name in record:
                if name not in seen_reward_metrics:
                    seen_reward_metrics.add(name)
                    reward_metric_names.append(name)
        reward_metrics = {
            name: [float(record.get(name, 0.0)) for record in reward_metric_records]
            for name in reward_metric_names
        }
        if n > 0:
            next_states = states[1:] + [states[-1]]
        else:
            next_states = []

        payload = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestep": float(self.m.opt.timestep),
            "observations": states,
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "reward_function": "testing.experiments.pipe_insert.V4.env.get_reward_2",
            "reward_metrics": reward_metrics,
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

    def _policy_action(self, obs: torch.Tensor, step: int) -> np.ndarray:
        obs_batch = obs.unsqueeze(0).to(self.agent.device, dtype=torch.float32)
        self.agent.pre_interaction(obs_batch, step, self.args.max_steps)
        with torch.no_grad():
            actions, _, _ = self.agent.act(obs_batch, step, self.args.max_steps)
        return actions[0].detach().cpu().numpy()

    def run(self) -> None:
        self._init_state()
        obs = self._reset_episode()

        if not self.args.render:
            for step in range(self.args.max_steps):
                action = self._policy_action(obs, step)
                self._apply_action(action)
                mj.mj_step(self.m, self.d)
                next_obs = self.get_obs()
                terminated = self._get_termination(next_obs)
                reward_metrics = self._compute_reward_metrics(
                    obs, next_obs, terminated
                )
                reward = reward_metrics["reward_total"]
                self._record_step(
                    next_obs, action, reward, terminated, reward_metrics
                )
                obs = next_obs
                self.episode_step += 1
                if self._should_stop(terminated, next_obs):
                    if self.args.record_path is not None:
                        self._save_rollout(self.args.record_path)
                    break
            return

        with mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.cb
        ) as viewer:
            # print(self.d.mocap_pos)

            # input("ready?")
            step = 0
            while viewer.is_running() and not self._exit and step < self.args.max_steps:
                step_start = time.time()

                if self._reset:
                    self._reset = False
                    obs = self._reset_episode()
                    step = 0

                action = self._policy_action(obs, step)
                print(action)
                self._apply_action(action)

                mj.mj_step(self.m, self.d)
                viewer.sync()

                next_obs = self.get_obs()
                terminated = self._get_termination(next_obs)
                reward_metrics = self._compute_reward_metrics(
                    obs, next_obs, terminated
                )
                reward = reward_metrics["reward_total"]
                # print(next_obs)
                # input()
                self._record_step(
                    next_obs, action, reward, terminated, reward_metrics
                )
                obs = next_obs
                step += 1
                self.episode_step += 1

                if self._should_stop(terminated, next_obs):
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

    agent = _build_agent(args)

    sim_cfg = default_config()
    sim_cfg.impl = args.impl
    sim_env = PipeInsert2(config=sim_cfg)
    model = sim_env.mj_model
    data = mj.MjData(model)

    rollout = DRLRRollout(sim_env, model, data, agent, args)
    rollout.run()

    if args.record_path is not None and not rollout._saved:
        rollout._save_rollout(args.record_path)


if __name__ == "__main__":
    main()
