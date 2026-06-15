"""Pure SAC training script using the same V4 PipeInsert2 setup as DRLR."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import tyro
from gym import spaces
from gym.vector import utils as gym_utils

from performance import save_timings, timer
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch.running_standard_scaler import (
    RunningStandardScaler,
)
from testing import wrappers as wrap
from testing.experiments.pipe_insert.V4.env import PipeInsert2, default_config
from testing.experiments.pipe_insert.V4.utils import (
    exp_set_seed,
    get_memory,
    get_sac_models,
    get_trainer,
)

EXPECTED_OBS_DIM = 6
EXPECTED_ACTION_DIM = 6

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


def validate_v3_env(env) -> None:
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    if obs_dim != EXPECTED_OBS_DIM:
        raise ValueError(
            f"Expected V4 PipeInsert2 observation dim {EXPECTED_OBS_DIM}, got {obs_dim}"
        )
    if action_dim != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"Expected V4 PipeInsert2 action dim {EXPECTED_ACTION_DIM}, got {action_dim}"
        )


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


def get_comparable_sac_config(exp_name: str, env, wandb: bool, warmup_timesteps: int):
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)

    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": env.device,
    }

    cfg["gradient_steps"] = 1
    cfg["batch_size"] = 256
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005
    cfg["actor_learning_rate"] = 3e-4
    cfg["critic_learning_rate"] = 3e-4
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["warmup_timesteps"] = warmup_timesteps
    cfg["grad_norm_clip"] = 0.1
    cfg["learning_rate"] = 3e-4
    cfg["learn_entropy"] = False
    cfg["entropy_learning_rate"] = 3e-4
    cfg["initial_entropy_value"] = 0.001

    model_path = Path(__file__).parent / f".runs/{exp_name}/models"
    model_path.mkdir(parents=True, exist_ok=True)

    cfg["experiment"]["directory"] = model_path.as_posix()
    cfg["experiment"]["experiment_name"] = exp_name
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 1000
    cfg["experiment"]["wandb"] = wandb
    cfg["experiment"].setdefault("wandb_kwargs", {})
    cfg["experiment"]["wandb_kwargs"].setdefault("project", "pipe-insert-v3")
    cfg["experiment"]["wandb_kwargs"].setdefault("resume", "never")

    return cfg


def log_ignored_drlr_args(args: "Args") -> None:
    ignored = [
        "expert_data_dir",
        "dp_checkpoint",
        "actor",
        "dp_iters",
        "dp_pred_horizon",
        "decision_block",
        "soft_update_beta",
        "action_trans_high",
        "action_trans_low",
        "action_rot_high",
        "action_rot_low",
    ]
    print(
        "Pure SAC keeps DRLR CLI parity; these arguments are accepted but unused: "
        + ", ".join(ignored)
    )


@dataclass
class Args:
    """Pure SAC on V4 PipeInsert2 with DRLR-compatible CLI parameters."""

    expert_data_dir: Path = Path(
        "testing/experiments/pipe_insert/V4/datasets/subsample_x4"
    )
    dp_checkpoint: Optional[Path] = None
    num_envs: int = field(
        default_factory=lambda: int(os.environ.get("PIPE_INSERT_NUM_ENVS", "100"))
    )
    training_timesteps: int = 1_000_000
    episode_length: int = 1000
    rollout_every_episodes: int = 0
    run_rollout_only: bool = False
    impl: Literal["jax", "warp", "c"] = "warp"
    render_impl: Literal["jax", "warp", "c"] = "jax"
    render_num_envs: int = 1
    actor: Literal["rl", "il", "both"] = "both"
    dp_iters: int = 4
    dp_pred_horizon: int = 8
    warmup_timesteps: int = 2_000
    wandb: bool = True
    action_dim_override: Optional[int] = 6
    action_trans_high: Optional[float] = 0.007
    action_trans_low: Optional[float] = -0.007
    action_rot_high: Optional[float] = 0.01
    action_rot_low: Optional[float] = -0.01

    a_max_lim: float = 1
    a_min_lim: float = -1

    decision_block: bool = True
    soft_update_beta: float = 0.2


def main(args: Optional[Args] = None) -> None:
    if args is None:
        args = tyro.cli(Args)

    exp_set_seed()

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
        low=[-0.007, -0.007, -0.007, -0.01, -0.01, -0.01],
        high=[0.007, 0.007, 0.007, 0.01, 0.01, 0.01],
        # low=[args.a_min_lim] * 6,
        # high=[args.a_max_lim] * 6,
        num_envs=args.num_envs,
    )
    validate_v3_env(env)

    exp_name = (
        Path(__file__).stem
        + "_actor_sac_"
        + "align_z_and_pos_"
        + f"_warmup_timesteps_{args.warmup_timesteps}_"
        + f"reward_scale_{0.1}_z_align_1_"
        + datetime.now().strftime("%Y%m%d_%H_%M_%S")
    )

    exp_dir = Path(__file__).parent
    cfg = get_comparable_sac_config(
        exp_name=exp_name,
        env=env,
        wandb=args.wandb,
        warmup_timesteps=args.warmup_timesteps,
    )

    a_dim = env.action_space.shape[0]
    if a_dim == 0:
        raise ValueError(
            "V4 PipeInsert2 reports action_dim=0. "
            "SAC needs a non-zero action space. "
            "If you intend to control mocap, expose a 6D action interface in the env "
            "or pass --action-dim-override 6."
        )

    log_ignored_drlr_args(args)

    memory = get_memory(env, capacity=100_000)
    models = get_sac_models(env)
    for model in models.values():
        model.to(env.device)

    agent = SAC(
        models=models,
        memory=memory,
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
