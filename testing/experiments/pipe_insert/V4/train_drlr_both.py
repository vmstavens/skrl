"""Clean IBRL + DP training script using V4 PipeInsert2."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from gym import spaces
from gym.vector import utils as gym_utils

from performance import save_timings, timer
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch.running_standard_scaler import (
    RunningStandardScaler,
)
from testing import wrappers as wrap
from testing.experiments.pipe_insert.V4.env import PipeInsert2, default_config
from testing.experiments.pipe_insert.V4.utils import (
    exp_set_seed,
    get_expert_memory_2,
    get_ibrl_sac_dp_config,
    get_memory,
    get_sac_models,
    get_trainer,
)
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.drlr_sac_o_o2_v2 import DRLR

# from testing.shen.ibrl_sac_o_o2_clean_test import IBRL

# Environment vars for JAX / XLA
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".50")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


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
    # MjxWrapper wraps TorchWrapper(VectorGymWrapper). Set on both to be safe.
    try:
        env._env.action_space = batched_space
    except Exception:
        pass
    try:
        env._env.env.action_space = batched_space
    except Exception:
        pass

    return env


def latest_checkpoint(runs_dir: Path) -> Path:
    pattern = "align_z_and_pos_x3_16/models/latest_model.pth"
    candidates = list(runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {runs_dir.as_posix()} matching {pattern}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_trained_dp_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> DiffusionPolicy:
    agent = DiffusionPolicy.load(checkpoint_path.as_posix(), device=device)
    agent.to(device)
    agent.set_mode("eval")
    return agent


@dataclass
class Args:
    """IBRL + Diffusion Policy on V4 PipeInsert2."""

    expert_data_dir: Path = Path(
        "testing/experiments/pipe_insert/V4/datasets/align_z_and_pos"
    )
    dp_checkpoint: Optional[Path] = None
    num_envs: int = field(
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "64"))
        default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "200"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "1000"))
    )
    training_timesteps: int = 100_000
    # training_timesteps: int = 200_000
    episode_length: int = 1000  # 1500
    # episode_length: int = 2000  # 1500
    # episode_length: int = 1000  # 1500
    # episode_length: int = 1500  # 1500
    rollout_every_episodes: int = 1
    # rollout_every_episodes: int = 10
    run_rollout_only: bool = False
    # impl: Literal["jax", "warp", "c"] = "warp"
    impl: Literal["jax", "warp", "c"] = "jax"
    render_impl: Literal["jax", "warp", "c"] = "jax"
    render_num_envs: int = 1
    actor: Literal["rl", "il", "both"] = "both"
    # actor: Literal["rl", "il", "both"] = "both"
    dp_iters: int = 4
    dp_pred_horizon: int = 8
    # warmup_timesteps: int = 0
    warmup_timesteps: int = 2_000
    # warmup_timesteps: int = 2000
    # warmup_timesteps: int = 4000
    # warmup_timesteps: int = 3000
    # warmup_timesteps: int = 3000
    # warmup_timesteps: int = 10_000
    # warmup_timesteps: int = 10_000
    # wandb: bool = False
    wandb: bool = False
    # wandb: bool = True
    action_dim_override: Optional[int] = 6

    a_max_lim: float = 1
    a_min_lim: float = -1

    # action_trans_high: Optional[float] = 0.005
    # action_trans_low: Optional[float] = -0.005
    action_trans_high: Optional[float] = 0.007
    action_trans_low: Optional[float] = -0.007
    # action_trans_high: Optional[float] = 0.001
    # action_trans_low: Optional[float] = -0.001
    action_rot_high: Optional[float] = 0.005
    action_rot_low: Optional[float] = -0.005
    # action_rot_high: Optional[float] = 0.01
    # action_rot_low: Optional[float] = -0.01

    decision_block: bool = True
    # decision_block: bool = False

    soft_update_beta: float = 0.2


def main(args: Optional[Args] = None) -> None:
    if args is None:
        args = tyro.cli(Args)

    # v
    exp_set_seed()

    # v
    env = setup_mocap_environment(
        batch_size=args.num_envs,
        episode_length=args.episode_length,
        auto_reset=True,
        action_repeat=1,
        impl=args.impl,
    )

    env = override_action_space(
        env=env,
        action_dim=args.action_dim_override,
        low=[args.a_min_lim] * 6,
        high=[args.a_max_lim] * 6,
        # low=[-0.007, -0.007, -0.007, -0.01, -0.01, -0.01],
        # high=[0.007, 0.007, 0.007, 0.01, 0.01, 0.01],
        num_envs=args.num_envs,
    )

    # v
    exp_name = (
        Path(__file__).stem
        + f"_actor_{args.actor}_"
        + "align_z_and_pos_"
        + f"warmup_timesteps_{args.warmup_timesteps}_"
        + f"reward_scale_{0.1}_z_align_1_"
        + datetime.now().strftime("%Y%m%d_%H_%M_%S")
    )

    #     "soft_update_beta": 0.2,         # use small for good IL (small ~0.2) original IBRL used 10

    exp_dir = Path(__file__).parent

    # v
    cfg = get_ibrl_sac_dp_config(exp_name=exp_name, env=env, wandb=args.wandb)
    # drlr_config = get_drlr_config(exp_name=exp_name, env=env, wandb=args.wandb)

    # v
    cfg["actor"] = args.actor
    cfg["warmup_timesteps"] = args.warmup_timesteps
    cfg["decision_block"] = args.decision_block
    cfg["soft_update_beta"] = args.soft_update_beta

    cfg["action_trans_high"] = args.action_trans_high
    cfg["action_trans_low"] = args.action_trans_low
    cfg["action_rot_high"] = args.action_rot_high
    cfg["action_rot_low"] = args.action_rot_low

    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": env.device,
    }

    cfg["a_max_lim"] = args.a_max_lim
    cfg["a_min_lim"] = args.a_min_lim

    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]

    # v
    if a_dim == 0:
        raise ValueError(
            "V4 PipeInsert2 reports action_dim=0. "
            "IBRL needs a non-zero action space. "
            "If you intend to control mocap, expose a 6D action interface in the env "
            "or pass --action-dim-override 6."
        )

    if args.dp_checkpoint is None:
        dp_checkpoint = latest_checkpoint(exp_dir / ".runs")
    else:
        dp_checkpoint = args.dp_checkpoint

    dp_policy = load_trained_dp_policy(
        checkpoint_path=dp_checkpoint,
        device=env.device,
    )

    if dp_policy._a_dim != a_dim:
        raise ValueError(
            f"DP action dim {dp_policy._a_dim} does not match env action dim {a_dim}"
        )
    if dp_policy._o_dim != o_dim:
        raise ValueError(
            f"DP observation dim {dp_policy._o_dim} does not match env observation dim {o_dim}"
        )

    dp_policy._num_diffusion_iters = args.dp_iters
    dp_policy._pred_horizon = args.dp_pred_horizon
    dp_policy._act_horizon = 1
    dp_policy.config["num_diffusion_iters"] = dp_policy._num_diffusion_iters
    dp_policy.config["pred_horizon"] = dp_policy._pred_horizon
    dp_policy.config["action_horizon"] = dp_policy._act_horizon

    il_models = {"policy": dp_policy}

    # v
    expert_memory = get_expert_memory_2(
        expert_data_dir=str(args.expert_data_dir),
        states_label="states",
        actions_label="actions",
        rewards_label="rewards",
        next_states_label="next_states",
        dones_label="terminated",
    )

    # v
    memory = get_memory(env, capacity=100_000)
    rl_models = get_sac_models(env)

    agent = DRLR(
        models=rl_models,
        models_il=il_models,
        memory=memory,
        expert_memory=expert_memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    rollout_video_dir = exp_dir / f".runs/{exp_name}/media"
    trainer = get_trainer(
        env,
        agent,
        timesteps=args.training_timesteps,
        trainer_cfg={
            "rollout_video_every_episodes": args.rollout_every_episodes,
            "rollout_video_num_steps": args.episode_length,
            "rollout_video_dir": rollout_video_dir.as_posix(),
            "rollout_video_prefix": "train_rollout",
            "rollout_video_env_index": 0,
            "log_rollout_path": "testing/experiments/pipe_insert/tmp/success_ibrl_mjx.json",
            "log_rollout_steps": args.episode_length,
            "log_rollout_exit": False,
        },
    )

    if args.run_rollout_only:
        print("recording rollout video for env 0...")
        trainer._record_rollout_video(0)
        print("rollout complete. exiting.")
        sys.exit(0)

    print("training...")
    with timer("train"):
        trainer.train()

    save_timings(exp_dir / ".runs" / exp_name / "performance.json")


if __name__ == "__main__":
    main()
