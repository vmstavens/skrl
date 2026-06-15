"""Clean IBRL + DP training script using V2 PipeInsert2."""

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
from testing import wrappers as wrap
from testing.experiments.pipe_insert.V2.env import PipeInsert2, default_config
from testing.experiments.pipe_insert.V2.utils import (
    exp_set_seed,
    get_dp_config,
    get_expert_memory_2,
    get_ibrl_sac_dp_config,
    get_memory,
    get_sac_models,
    get_trainer,
)
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel
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


# def latest_checkpoint(runs_dir: Path) -> Path:
#     patterns = (
#         "data_mocap/models/latest_model.pth",
#         "data_mocap_*/models/latest_model.pth",
#         "finetune_mocap_*/models/latest_model.pth",
#     )
#     candidates: list[Path] = []
#     for pattern in patterns:
#         candidates.extend(runs_dir.glob(pattern))
#     if not candidates:
#         raise FileNotFoundError(
#             f"No mocap checkpoints found in {runs_dir.as_posix()} (patterns: {patterns})"
#         )
#     return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_checkpoint(runs_dir: Path) -> Path:
    pattern = "data_mocap_x4/models/latest_model.pth"
    # pattern = "data_mocap_*/models/latest_model.pth"
    candidates = list(runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {runs_dir.as_posix()} matching {pattern}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_trained_dp_policy(
    checkpoint_path: Path,
    device: torch.device,
    config: dict,
    a_dim: int,
    o_dim: int,
) -> DiffusionPolicy:
    dp_models: dict = {}
    dp_models["model"] = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=config)
    ema = EMAModel(dp_models["model"].parameters(), power=config["ema_power"])
    dp_models["ema_model"] = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=config)

    agent = DiffusionPolicy(
        a_dim=a_dim,
        o_dim=o_dim,
        models=dp_models,
        ema=ema,
        device=device,
        config=config,
    )

    agent = agent.load(checkpoint_path.as_posix(), device=device)
    agent.set_mode("eval")

    return agent


@dataclass
class Args:
    """IBRL + Diffusion Policy on V2 PipeInsert2."""

    expert_data_dir: Path = Path(
        "testing/experiments/pipe_insert/V2/datasets/subsample_x4"
    )
    # expert_data_dir: Path = Path("testing/experiments/pipe_insert/V2/demos")
    dp_checkpoint: Optional[Path] = None
    num_envs: int = field(
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "64"))
        default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
        # default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "1000"))
    )
    training_timesteps: int = 500_000
    # training_timesteps: int = 200_000
    episode_length: int = 1000  # 1500
    # episode_length: int = 1500  # 1500
    rollout_every_episodes: int = 0
    run_rollout_only: bool = False
    impl: Literal["jax", "warp", "c"] = "warp"
    # impl: Literal["jax", "warp", "c"] = "warp"
    # impl: Literal["jax", "warp", "c"] = "warp"
    # impl: Literal["jax", "warp", "c"] = "warp"
    render_impl: Literal["jax", "warp", "c"] = "jax"
    render_num_envs: int = 1
    actor: Literal["rl", "il", "both"] = "both"
    # actor: Literal["rl", "il", "both"] = "both"
    dp_iters: int = 4
    dp_pred_horizon: int = 8
    # il_action_scale: float = 3.0  # 3 mm
    # rl_action_scale: float = 3.0  # 3 mm
    # il_action_scale: float = 1.0  # 1 mm
    # rl_action_scale: float = 1.0  # 1 mm
    # warmup_timesteps: int = 0
    warmup_timesteps: int = 2000
    # warmup_timesteps: int = 3000
    # warmup_timesteps: int = 10_000
    # warmup_timesteps: int = 10_000
    wandb: bool = True
    action_dim_override: Optional[int] = 6
    action_trans_high: Optional[float] = 0.003  # 3 mm
    action_trans_low: Optional[float] = -0.003  # 3 mm
    action_rot_high: Optional[float] = 0.002  # ~1 degree
    action_rot_low: Optional[float] = -0.002  # ~1 degree
    # action_rot_high: Optional[float] = 0.015 # ~1 degree
    # action_rot_low: Optional[float] = -0.015 # ~1 degree

    a_max_lim: float = 1
    a_min_lim: float = -1

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
        num_envs=args.num_envs,
    )

    # v
    exp_name = (
        Path(__file__).stem
        + f"_actor_{args.actor}_"
        + f"_warmup_timesteps_{args.warmup_timesteps}_"
        + f"_soft_update_beta_{args.soft_update_beta}_"
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

    # v
    dp_config = get_dp_config(exp_name=exp_name, env=env)
    dp_config["pred_horizon"] = args.dp_pred_horizon
    dp_config["action_horizon"] = 1

    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]

    # v
    if a_dim == 0:
        raise ValueError(
            "V2 PipeInsert2 reports action_dim=0. "
            "IBRL needs a non-zero action space. "
            "If you intend to control mocap, expose a 6D action interface in the env "
            "or pass --action-dim-override 6."
        )

    if args.dp_checkpoint is None:
        dp_checkpoint = exp_dir / ".runs/data_mocap_x4/models/latest_model.pth"
        if not dp_checkpoint.exists():
            raise FileNotFoundError(
                f"Expected DP checkpoint not found: {dp_checkpoint.as_posix()}"
            )
    else:
        dp_checkpoint = args.dp_checkpoint

    dp_policy = load_trained_dp_policy(
        checkpoint_path=dp_checkpoint,
        device=env.device,
        config=dp_config,
        a_dim=a_dim,
        o_dim=o_dim,
    )
    dp_policy._num_diffusion_iters = args.dp_iters
    dp_policy._pred_horizon = args.dp_pred_horizon
    dp_policy._act_horizon = 1

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
    # rl_models = get_td3_models(env)

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
