from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np
import torch
from gym import spaces
from gym.vector import utils as gym_utils

from skrl.envs.wrappers.torch import wrap_env
from testing import wrappers as wrap
from testing.envs.pipe_insert_2_new import (
    PipeInsert2,
    default_config,
    keypoints_within_pipe,
)
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_dp_config,
    get_drlr_config,
    get_expert_memory_2,
    get_memory,
    get_td3_models,
)
from testing.mj import ObjType, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel
from testing.shen.ibrl_sac_o_o2_clean_test import IBRL


def setup_mocap_environment(
    batch_size: int,
    episode_length: int,
    auto_reset: bool,
    action_repeat: int,
    impl: str,
):
    cfg = default_config()
    cfg.impl = impl
    cfg.episode_length = episode_length
    cfg.action_repeat = action_repeat
    env = PipeInsert2(config=cfg)
    env = wrap.create(
        env,
        batch_size=batch_size,
        episode_length=episode_length,
        auto_reset=auto_reset,
        action_repeat=action_repeat,
    )
    env = wrap_env(env, wrapper="playground")
    return env


def override_action_space(
    env, action_dim: int, low: np.ndarray, high: np.ndarray, num_envs: int
):
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.asarray(high, dtype=np.float32)
    if low_arr.shape != (action_dim,):
        if low_arr.size == 1:
            low_arr = np.full((action_dim,), float(low_arr), dtype=np.float32)
        else:
            raise ValueError(f"low shape {low_arr.shape} != ({action_dim},)")
    if high_arr.shape != (action_dim,):
        if high_arr.size == 1:
            high_arr = np.full((action_dim,), float(high_arr), dtype=np.float32)
        else:
            raise ValueError(f"high shape {high_arr.shape} != ({action_dim},)")
    base_space = spaces.Box(
        low=low_arr,
        high=high_arr,
        shape=(action_dim,),
        dtype="float32",
    )
    batched_space = gym_utils.batch_space(base_space, num_envs)
    try:
        env._env.action_space = batched_space
    except Exception:
        pass
    try:
        env._env.env.action_space = batched_space
    except Exception:
        pass


def _latest_dp_checkpoint(runs_dir: Path) -> Path:
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


def _latest_ibrl_checkpoint(checkpoints_dir: Path) -> Path:
    best_path = checkpoints_dir / "best_agent.pt"
    if best_path.exists():
        return best_path
    candidates = list(checkpoints_dir.glob("agent_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No IBRL checkpoints found in {checkpoints_dir.as_posix()}"
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
    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        dp_checkpoint_path: Path,
        ibrl_checkpoint_path: Path,
    ):
        # seed = int(np.random.uniform() * 1000)
        # exp_set_seed(seed)
        exp_set_seed()

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
        )
        self.pipe_mid_end = np.array(
            [-1.35525272e-20, -6.24999802e-02, 1.00049770e-01], dtype=np.float64
        )
        self.grace_steps = int(default_config().get("grace_steps", 0))

        self.action_dim = 6
        self.action_gain = 1.0
        # self.action_gain = 4.0
        self.last_action = [0.0] * self.action_dim
        # Action limits aligned with ibrl_sac_dp.py defaults.
        self.action_low = np.array(
            [-0.001, -0.001, -0.001, -0.002, -0.002, -0.002], dtype=np.float32
        )
        self.action_high = np.array(
            [0.001, 0.001, 0.001, 0.002, 0.002, 0.002], dtype=np.float32
        )

        env = setup_mocap_environment(
            batch_size=1,
            episode_length=2000,
            auto_reset=True,
            action_repeat=1,
            impl="warp",
        )
        override_action_space(
            env,
            action_dim=self.action_dim,
            low=self.action_low,
            high=self.action_high,
            num_envs=1,
        )
        self.device = env.device

        dp_config = get_dp_config(exp_name="ibrl_rollout", env=env, wandb=False)
        dp_config["obs_dim"] = env.observation_space.shape[0]
        dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]
        dp_config["pred_horizon"] = 8
        dp_config["action_horizon"] = 1

        dp_models: dict = {}
        dp_models["model"] = ConditionalUnet1D(a_dim=6, o_dim=9, config=dp_config)
        ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
        dp_models["ema_model"] = ConditionalUnet1D(a_dim=6, o_dim=9, config=dp_config)

        dp_policy = DiffusionPolicy(
            a_dim=6,
            o_dim=9,
            models=dp_models,
            ema=ema,
            device=self.device,
            config=dp_config,
        )
        dp_policy = dp_policy.load(dp_checkpoint_path.as_posix(), device=self.device)
        dp_policy._num_diffusion_iters = 4
        dp_policy._pred_horizon = 8
        dp_policy._act_horizon = 1
        dp_policy.to(self.device)

        drlr_config = get_drlr_config(exp_name="ibrl_rollout", env=env, wandb=False)
        # drlr_config["actor"] = "il"

        expert_memory = get_expert_memory_2(
            expert_data_dir="testing/experiments/pipe_insert/run/data_collection_2_mocap",
            states_label="states_dense",
            actions_label="actions",
            rewards_label="rewards_dense",
            next_states_label="next_states_dense",
            dones_label="terminated",
        )
        memory = get_memory(env, capacity=100_000)
        rl_models = get_td3_models(env)

        self.agent = IBRL(
            models=rl_models,
            models_il={"policy": dp_policy},
            memory=memory,
            expert_memory=expert_memory,
            cfg=drlr_config,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=self.device,
        )

        self.agent.init(trainer_cfg={"timesteps": 1})
        self.agent.load(ibrl_checkpoint_path.as_posix())
        self.agent.set_mode("eval")

        self.agent._actor = "both"
        self.agent._warmup_timesteps = 0
        # self.agent._actor = "rl"
        # self.agent._actor = "il"

        self._prev_states: torch.Tensor | None = None
        self.step_idx = 0
        self.rollout_states: list[list[float]] = []
        self.rollout_actions: list[list[float]] = []
        self.rollout_rewards: list[float] = []
        self.rollout_rewards_dense: list[float] = []
        self.time: list[float] = []
        self.rollout_path = Path("testing/experiments/pipe_insert/tmp/ibrl.json")
        # print("---------------------------------------------------------")
        # print(f"{self.agent._actor=}")
        # print("---------------------------------------------------------")
        # quit()

    def _record(
        self,
        obs: torch.Tensor,
        action: np.ndarray,
        reward: float,
        reward_dense: float | None = None,
        timestamp: float | None = None,
    ) -> None:
        self.rollout_states.append(obs.detach().cpu().numpy().tolist())
        self.rollout_actions.append(action.tolist())
        self.rollout_rewards.append(float(reward))
        if reward_dense is None:
            reward_dense = reward
        self.rollout_rewards_dense.append(float(reward_dense))
        # if timestamp is None:
        #     timestamp = time.time()
        self.time.append(self.data.time)

    def _save_rollout(self) -> None:
        payload = {
            "states": self.rollout_states,
            "actions": self.rollout_actions,
            "rewards": self.rollout_rewards,
            "rewards_dense": self.rollout_rewards_dense,
            "time": self.time,
        }
        with self.rollout_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"saved rollout data to {self.rollout_path.as_posix()}")

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
        saved = False

        def _handle_sigint(_sig, _frame) -> None:
            nonlocal saved
            if saved:
                return
            saved = True
            print("caught ctrl+c, saving rollout...")
            self._save_rollout()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_sigint)

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

        def is_success(step_count: int) -> bool:
            success = keypoints_within_pipe(
                keypoints=get_keypoint_2(self.data),
                pipe_axis_start=self.pipe_mid_start,
                pipe_axis_end=self.pipe_mid_end,
                inner_radius=self.pipe_inner_radius,
            )
            if self.grace_steps > 0 and step_count < self.grace_steps:
                return False
            return success

        def step_policy() -> None:
            nonlocal i
            obs = self.get_obs().to(self.device)
            states = obs.unsqueeze(0)
            if self._prev_states is None:
                self._prev_states = states
            self.agent._states = states
            self.agent._prev_states = self._prev_states

            actions, _, _ = self.agent.act(states, timestep=i, timesteps=1_000_000)
            action = actions[0].detach().cpu().numpy() * self.action_gain
            self._apply_action(action)
            self.last_action = action.tolist()
            mj.mj_step(self.model, self.data)
            next_obs = self.get_obs().to(self.device)
            reward = self._compute_reward(next_obs)
            self._record(obs, action, reward)
            next_obs = next_obs.unsqueeze(0)
            self._prev_states = next_obs
            i += 1

        if render:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                input("ready?")
                while viewer.is_running():
                    step_start = time.time()

                    success = is_success(i)
                    print(i)

                    # input()
                    if success:
                        print("success...", i)
                        self._save_rollout()
                        sys.exit(i)
                    if i > 10_000:
                        print("failed...")
                        self._save_rollout()
                        sys.exit(-1)

                    step_policy()
                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            while True:
                success = is_success(i)
                if success:
                    print("success...", i)
                    self._save_rollout()
                    sys.exit(i)
                if i > 10_000:
                    print("failed...")
                    self._save_rollout()
                    sys.exit(-1)

                step_policy()


def main() -> None:
    _env = PipeInsert2()

    model = _env.mj_model
    data = mj.MjData(model)

    runs_dir = Path(__file__).resolve().parents[1] / ".runs"
    dp_default = (
        Path(
            "testing/experiments/pipe_insert/.runs/data_mocap_100_20260228_124142/models"
        )
        / "latest_model.pth"
    )
    dp_checkpoint_path = (
        dp_default if dp_default.exists() else _latest_dp_checkpoint(runs_dir)
    )

    chpts = {
        "il": "testing/experiments/pipe_insert/.runs/ibrl_sac_dp_actor_il_20260305_22_45_31/models/ibrl_sac_dp_actor_il_20260305_22_45_31/checkpoints",
        "rl": "testing/experiments/pipe_insert/.runs/ibrl_sac_dp_actor_rl_20260306_14_35_22/models/ibrl_sac_dp_actor_rl_20260306_14_35_22/checkpoints",
        "ibrl": "testing/experiments/pipe_insert/.runs/ibrl_sac_dp_actor_both_20260306_08_21_19/models/ibrl_sac_dp_actor_both_20260306_08_21_19/checkpoints",
        "ibrl_64": "testing/experiments/pipe_insert/.runs/ibrl_sac_dp_actor_both_num_envs_64_20260310_09_22_44/models/ibrl_sac_dp_actor_both_num_envs_64_20260310_09_22_44/checkpoints",
        "ibrl_64_no_warmup": "testing/experiments/pipe_insert/.runs/ibrl_sac_dp_actor_both_num_envs_64_warmup_timesteps_0_20260310_14_16_06/models/ibrl_sac_dp_actor_both_num_envs_64_warmup_timesteps_0_20260310_14_16_06/checkpoints",
        "drlr": "testing/experiments/pipe_insert/.runs/ibrl_sac_dp_actor_both_num_envs_64_warmup_timesteps_0_drlr_True_20260310_18_09_35/models/ibrl_sac_dp_actor_both_num_envs_64_warmup_timesteps_0_drlr_True_20260310_18_09_35/checkpoints",
    }

    # ibrl_checkpoints_dir = Path(
    #     "testing/experiments/pipe_insert/.runs/ibrl_sac_dp_dpiter_4_actor_il_dense_20260301_09_37_36/models/ibrl_sac_dp_dpiter_4_actor_il_dense_20260301_09_37_36/checkpoints"
    # )

    ibrl_checkpoints_dir = Path(chpts["ibrl"])

    ibrl_checkpoint_path = _latest_ibrl_checkpoint(ibrl_checkpoints_dir)

    sim = Test(
        model,
        data,
        dp_checkpoint_path=dp_checkpoint_path,
        ibrl_checkpoint_path=ibrl_checkpoint_path,
    )

    sim.run()


if __name__ == "__main__":
    main()
