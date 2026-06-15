from __future__ import annotations

import json
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import time
from collections import deque
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np
import torch

from testing.envs.pipe_insert_2_new import PipeInsert2, keypoints_within_pipe
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_dp_config,
    setup_environment,
)
from testing.mj import ObjType, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


def _latest_checkpoint(runs_dir: Path) -> Path:
    patterns = (
        "data_mocap_*/models/latest_model.pth",
        "finetune_mocap_*/models/latest_model.pth",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No mocap checkpoints found in {runs_dir.as_posix()} (patterns: {patterns})"
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


class Test:
    def __init__(self, model: mj.MjModel, data: mj.MjData, checkpoint_path: Path):
        seed = int(np.random.uniform() * 1000)
        exp_set_seed(seed)

        self.model: mj.MjModel = model
        self.data = data
        self.model.vis.global_.cameraid = -1

        self.mocap_id = None
        try:
            self.mocap_id = int(self.model.body("mocap").mocapid)
            if self.mocap_id < 0:
                self.mocap_id = None
        except Exception:
            self.mocap_id = None

        self.cable_root_qpos_adr = None
        try:
            cable_root_jnt = self.model.joint("cable:free").id
            self.cable_root_qpos_adr = int(self.model.jnt_qposadr[cable_root_jnt])
        except Exception:
            self.cable_root_qpos_adr = None

        self.qpos0 = None
        self.qpos0_root = None
        self.mocap_pos0 = None
        self.mocap_quat0 = None
        self.mocap_range = np.array([0.3, 0.3, 0.3], dtype=np.float64)
        self.mocap_low = None
        self.mocap_high = None

        self.pipe_inner_radius = 0.0385 / 2
        self.pipe_mid_start = np.array(
            [1.35525272e-20, 6.24999802e-02, 9.99502296e-02], dtype=np.float64
        ) - np.array([0, 0.04, 0])
        self.pipe_mid_end = np.array(
            [-1.35525272e-20, -6.24999802e-02, 1.00049770e-01], dtype=np.float64
        ) + np.array([0, 0.04, 0])

        self.action_dim = 6
        self.action_gain = 1.0
        # self.action_gain = 4.0
        # self.action_gain = 1.0
        self.last_action = [0.0] * self.action_dim

        a_dim = self.action_dim
        o_dim = 9

        env = setup_environment(batch_size=1)

        dp_config = get_dp_config(
            exp_name="pipe_insert_2_rot_mocap_rollout", env=env, wandb=False
        )
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
        self.agent = self.agent.load(checkpoint_path.as_posix(), device=env.device)

        # performance tweak (match other rollout scripts)
        self.agent._num_diffusion_iters = 4
        self.agent._pred_horizon = 8
        # self.agent._pred_horizon = 8

        self.agent.to(env.device)
        self.action = np.zeros(self.action_dim, dtype=np.float64)
        self.obs_history: deque[torch.Tensor] = deque(
            maxlen=self.agent.config["obs_horizon"]
        )
        self.step_idx = 0
        self.rollout_states: list[list[float]] = []
        self.rollout_actions: list[list[float]] = []
        self.rollout_rewards: list[float] = []
        self.rollout_time: list[float] = []
        self.rollout_path = (
            Path(__file__).with_name("dp_rollout_states_actions.json").resolve()
        )

    def get_obs(self) -> torch.Tensor:
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
        return torch.from_numpy(d)

    def _record(self, obs: torch.Tensor, action: np.ndarray, reward: float) -> None:
        self.rollout_states.append(obs.detach().cpu().numpy().tolist())
        self.rollout_actions.append(action.tolist())
        self.rollout_rewards.append(float(reward))
        self.rollout_time.append(float(self.data.time))

    def _save_rollout(self) -> None:
        payload = {
            "states": self.rollout_states,
            "actions": self.rollout_actions,
            "rewards": self.rollout_rewards,
            "rewards_dense": self.rollout_rewards,
            "time": self.rollout_time,
        }
        with self.rollout_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"saved rollout data to {self.rollout_path.as_posix()}")

    def randomize_state(self) -> None:
        if self.qpos0 is None or self.mocap_pos0 is None or self.mocap_quat0 is None:
            raise RuntimeError("Base state not initialized. Call run() first.")

        delta = np.random.uniform(-0.3, 0.3, 3)
        self.data.qpos[:] = self.qpos0
        if self.cable_root_qpos_adr is not None and self.qpos0_root is not None:
            adr = self.cable_root_qpos_adr
            self.data.qpos[adr : adr + 3] = self.qpos0_root + delta
        self.data.qvel[:] = 0.0
        if self.mocap_id is not None:
            target = self.mocap_pos0 + delta
            if self.mocap_low is not None and self.mocap_high is not None:
                target = np.clip(target, self.mocap_low, self.mocap_high)
            self.data.mocap_pos[self.mocap_id] = target
            self.data.mocap_quat[self.mocap_id] = self.mocap_quat0

    def _apply_action(self, action: np.ndarray) -> None:
        if self.mocap_id is None:
            return
        pos_delta = action[:3]
        rot_delta = action[3:6]

        target_pos = self.data.mocap_pos[self.mocap_id] + pos_delta
        if self.mocap_low is not None and self.mocap_high is not None:
            target_pos = np.clip(target_pos, self.mocap_low, self.mocap_high)
        self.data.mocap_pos[self.mocap_id] = target_pos

        current = self.data.mocap_quat[self.mocap_id].copy()
        delta_quat = _rotvec_to_quat(rot_delta)
        target_quat = _quat_mul(delta_quat, current)
        self.data.mocap_quat[self.mocap_id] = _quat_normalize(target_quat)

    def _compute_reward(self, obs: torch.Tensor) -> float:
        # Dense reward: negative distance to targets.
        return float(-torch.linalg.norm(obs).item())

    def run(self) -> None:
        render = True

        if self.model.nkey:
            key_id = None
            for key_name in ("init", "bent"):
                try:
                    key_id = self.model.key(key_name).id
                    break
                except Exception:
                    continue
            if key_id is None:
                key_id = 0
            self.data.qpos[:] = self.model.key_qpos[key_id]
            self.data.qvel[:] = self.model.key_qvel[key_id]
            if self.data.ctrl.size:
                self.data.ctrl[:] = self.model.key_ctrl[key_id]
            if self.model.nmocap:
                self.data.mocap_pos[:] = self.model.key_mpos[key_id].reshape(
                    self.model.nmocap, 3
                )
                self.data.mocap_quat[:] = self.model.key_mquat[key_id].reshape(
                    self.model.nmocap, 4
                )

        if self.mocap_id is None and self.model.nmocap:
            self.mocap_id = 0

        self.qpos0 = self.data.qpos.copy()
        if self.cable_root_qpos_adr is not None:
            adr = self.cable_root_qpos_adr
            self.qpos0_root = self.data.qpos[adr : adr + 3].copy()
        if self.mocap_id is not None and self.model.nmocap:
            self.mocap_pos0 = self.data.mocap_pos[self.mocap_id].copy()
            self.mocap_quat0 = self.data.mocap_quat[self.mocap_id].copy()
            self.mocap_low = self.mocap_pos0 - self.mocap_range
            self.mocap_high = self.mocap_pos0 + self.mocap_range

        mj.mj_forward(self.model, self.data)

        self.keypoint_ids = [self.model.site(f"keypoint_{i + 1}").id for i in range(3)]

        def get_keypoint_2(data: mj.MjData) -> np.ndarray:
            return np.array([data.site(self.keypoint_ids[1]).xpos])

        i = 0

        if render:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.user_scn

                # print(self.data.site(self.pipe_end_ids[0]).xpos)
                # print(self.data.site(self.pipe_end_ids[1]).xpos)
                # # print(self.data.site(self.pipe_end_ids).xpos)
                # quit()
                input("ready?")
                while viewer.is_running():
                    step_start = time.time()

                    success = keypoints_within_pipe(
                        keypoints=get_keypoint_2(self.data),
                        pipe_axis_start=self.pipe_mid_start,
                        pipe_axis_end=self.pipe_mid_end,
                        inner_radius=self.pipe_inner_radius,
                    )
                    print(success)
                    if success:
                        print("success...", i)
                        self._save_rollout()
                        sys.exit(i)
                    if i > 10_000:
                        print("failed...")
                        self._save_rollout()
                        sys.exit(-1)

                    obs = self.get_obs()
                    if len(self.obs_history) == 0:
                        for _ in range(self.agent.config["obs_horizon"]):
                            self.obs_history.append(obs)
                    obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)

                    actions, _, _ = self.agent.act(obs_seq)
                    action = actions[0, 0, :].detach().cpu().numpy()
                    action = action * self.action_gain
                    print(f"{action=}")
                    self._apply_action(action)
                    self.last_action = action.tolist()
                    mj.mj_step(self.model, self.data)
                    next_obs = self.get_obs()
                    reward = self._compute_reward(next_obs)
                    self._record(obs, action, reward)
                    viewer.sync()

                    self.obs_history.append(next_obs)
                    self.step_idx += 1
                    i += 1
                    # print(i)

                    # time_until_next_step = self.model.opt.timestep - (
                    #     time.time() - step_start
                    # )
                    # if time_until_next_step > 0:
                    #     time.sleep(time_until_next_step)
        else:
            while True:
                success = keypoints_within_pipe(
                    keypoints=get_keypoint_2(self.data),
                    pipe_axis_start=self.pipe_mid_start,
                    pipe_axis_end=self.pipe_mid_end,
                    inner_radius=self.pipe_inner_radius,
                )
                if success:
                    print("success...", i)
                    self._save_rollout()
                    sys.exit(i)
                if i > 10_000:
                    print("failed...")
                    self._save_rollout()
                    sys.exit(-1)

                obs = self.get_obs()
                if len(self.obs_history) == 0:
                    for _ in range(self.agent.config["obs_horizon"]):
                        self.obs_history.append(obs)
                obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)

                actions, _, _ = self.agent.act(obs_seq)
                action = actions[0, 0, :].detach().cpu().numpy()
                action = action * self.action_gain
                self._apply_action(action)
                self.last_action = action.tolist()
                mj.mj_step(self.model, self.data)
                next_obs = self.get_obs()
                reward = self._compute_reward(next_obs)
                self._record(obs, action, reward)
                self.obs_history.append(next_obs)
                self.step_idx += 1
                i += 1


def main() -> None:
    _env = PipeInsert2()

    model = _env.mj_model
    data = mj.MjData(model)

    runs_dir = Path(__file__).resolve().parents[1] / ".runs"
    checkpoint_path = _latest_checkpoint(runs_dir)

    sim = Test(model, data, checkpoint_path=checkpoint_path)

    sim.run()


if __name__ == "__main__":
    main()
