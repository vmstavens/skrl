"""Clean IBRL + DP training script using V3 PipeInsert2 with domain randomization."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import jax
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
from testing.experiments.pipe_insert.V3.env import (
    DEFAULT_CABLE_STIFFNESS_MULTIPLIER_RANGE,
    PipeInsert2,
    default_config,
    domain_randomize,
    wrap_for_mjx_training,
)
from testing.experiments.pipe_insert.V3.utils import (
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


class DomainRandomizationBatchWrapper:
    """Adapts MJX domain-randomized vmap envs to the local skrl wrapper."""

    def __init__(self, env, batch_size: int):
        self.env = env
        self.batch_size = batch_size

    def reset(self, rng):
        if rng.ndim == 1:
            rng = jax.random.split(rng, self.batch_size)
        return self.env.reset(rng)

    def step(self, state, action):
        return self.env.step(state, action)

    def __getattr__(self, name):
        return getattr(self.env, name)


def cable_randomization_stiffness_range(
    mj_model,
    multiplier_range: tuple[float, float] = DEFAULT_CABLE_STIFFNESS_MULTIPLIER_RANGE,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Returns the same cable joint stiffness span used by V3/dr.ipynb."""
    cable_joint_ids = np.array(
        [
            j
            for j in range(mj_model.njnt)
            if mj_model.joint(j).name.startswith("cable:J")
        ],
        dtype=np.int32,
    )
    cable_geom_id = None
    for geom_name in ("cable:G1", "cable:G2", "cable:Gfirst", "cable:Glast"):
        try:
            cable_geom_id = mj_model.geom(geom_name).id
            break
        except KeyError:
            continue
    if cable_joint_ids.size and cable_geom_id is None:
        raise ValueError("Cable joints found but no cable geom to infer stiffness.")
    if not cable_joint_ids.size:
        return cable_joint_ids, (float("nan"), float("nan"))

    geom_size = np.asarray(mj_model.geom_size[cable_geom_id], dtype=np.float64)
    radius = float(geom_size[0])
    segment_length = max(2 * float(geom_size[1]), 1e-9)

    cable_j = np.pi * radius**4 / 2.0
    cable_iy = np.pi * radius**4 / 4.0
    cable_iz = cable_iy

    # Match testing.experiments.pipe_insert.V3.env.domain_randomize and dr.ipynb.
    cable_youngs_base = 10000000.0 * 5
    cable_shear_base = 60000.0 * 1000
    youngs_range = (
        float(multiplier_range[0]) * cable_youngs_base,
        float(multiplier_range[1]) * cable_youngs_base,
    )
    shear_range = (
        float(multiplier_range[0]) * cable_shear_base,
        float(multiplier_range[1]) * cable_shear_base,
    )

    def stiffness(youngs: float, shear: float) -> float:
        k_twist = (cable_j * shear) / segment_length
        k_bend = ((cable_iy + cable_iz) * youngs) / segment_length
        return float((k_twist + k_bend) / 3.0)

    return cable_joint_ids, (
        stiffness(youngs_range[0], shear_range[0]),
        stiffness(youngs_range[1], shear_range[1]),
    )


def cable_randomization_length_spec(
    mj_model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns cable body, joint, and geom ids needed for length randomization."""
    cable_body_ids = np.array(
        [
            b
            for b in range(mj_model.nbody)
            if mj_model.body(b).name.startswith("cable:B")
        ],
        dtype=np.int32,
    )
    cable_joint_ids = np.array(
        [
            j
            for j in range(mj_model.njnt)
            if mj_model.joint(j).name.startswith("cable:J")
        ],
        dtype=np.int32,
    )
    cable_geom_ids = np.array(
        [
            g
            for g in range(mj_model.ngeom)
            if mj_model.geom(g).name.startswith("cable:G")
        ],
        dtype=np.int32,
    )
    if not cable_geom_ids.size:
        return cable_body_ids, cable_joint_ids, cable_geom_ids, float("nan")
    segment_length = 2.0 * float(mj_model.geom_size[int(cable_geom_ids[0]), 1])
    return cable_body_ids, cable_joint_ids, cable_geom_ids, segment_length


def randomize_cable_length(
    mjx_model,
    mj_model,
    rng,
    scale_range: tuple[float, float],
    in_axes=None,
):
    """Randomize the V3 cable length by scaling each segment in MJX model fields."""
    min_scale, max_scale = scale_range
    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("Cable length scales must be positive.")
    if min_scale > max_scale:
        raise ValueError("cable_length_min_scale must be <= cable_length_max_scale.")

    cable_body_ids, cable_joint_ids, cable_geom_ids, _ = (
        cable_randomization_length_spec(mj_model)
    )
    if not cable_geom_ids.size or min_scale == max_scale == 1.0:
        if in_axes is None:
            in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
        return mjx_model, in_axes

    body_pos_base = mjx_model.body_pos
    jnt_pos_base = mjx_model.jnt_pos
    geom_size_base = mjx_model.geom_size
    geom_rbound_base = mjx_model.geom_rbound
    jnt_stiffness_base = mjx_model.jnt_stiffness

    def rand(key, jnt_stiffness):
        scale = jax.random.uniform(
            key,
            shape=(),
            minval=float(min_scale),
            maxval=float(max_scale),
        )
        body_pos = body_pos_base.at[cable_body_ids, 1].set(
            body_pos_base[cable_body_ids, 1] * scale
        )
        jnt_pos = jnt_pos_base.at[cable_joint_ids, 1].set(
            jnt_pos_base[cable_joint_ids, 1] * scale
        )
        geom_size = geom_size_base.at[cable_geom_ids, 1].set(
            geom_size_base[cable_geom_ids, 1] * scale
        )
        geom_rbound = geom_rbound_base.at[cable_geom_ids].set(
            geom_size_base[cable_geom_ids, 0]
            + geom_size_base[cable_geom_ids, 1] * scale
        )
        if cable_joint_ids.size:
            jnt_stiffness = jnt_stiffness.at[cable_joint_ids].set(
                jnt_stiffness[cable_joint_ids] / scale
            )
        return body_pos, jnt_pos, geom_size, geom_rbound, jnt_stiffness

    batched = rng.ndim > 1
    if batched:
        if jnt_stiffness_base.ndim > 1:
            outputs = jax.vmap(rand)(rng, jnt_stiffness_base)
        else:
            outputs = jax.vmap(lambda key: rand(key, jnt_stiffness_base))(rng)
    else:
        outputs = rand(rng, jnt_stiffness_base)

    body_pos, jnt_pos, geom_size, geom_rbound, jnt_stiffness = outputs
    if in_axes is None:
        in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
    if batched:
        in_axes = in_axes.tree_replace(
            {
                "body_pos": 0,
                "jnt_pos": 0,
                "geom_size": 0,
                "geom_rbound": 0,
                "jnt_stiffness": 0,
            }
        )

    mjx_model = mjx_model.tree_replace(
        {
            "body_pos": body_pos,
            "jnt_pos": jnt_pos,
            "geom_size": geom_size,
            "geom_rbound": geom_rbound,
            "jnt_stiffness": jnt_stiffness,
        }
    )
    return mjx_model, in_axes


def setup_mocap_environment(
    batch_size: int,
    episode_length: int,
    auto_reset: bool,
    action_repeat: int,
    impl: str,
    cable_stiffness_min_scale: float,
    cable_stiffness_max_scale: float,
    cable_stiffness_log_uniform: bool,
    cable_length_min_scale: float,
    cable_length_max_scale: float,
):
    cfg = default_config()
    cfg.impl = impl
    cfg.episode_length = episode_length
    cfg.action_repeat = action_repeat
    env = PipeInsert2(config=cfg)
    del auto_reset
    if cable_stiffness_min_scale <= 0 or cable_stiffness_max_scale <= 0:
        raise ValueError("Cable stiffness scales must be positive.")
    if cable_stiffness_min_scale >= cable_stiffness_max_scale:
        raise ValueError(
            "cable_stiffness_min_scale must be smaller than cable_stiffness_max_scale."
        )
    cable_stiffness_multiplier_range = (
        cable_stiffness_min_scale,
        cable_stiffness_max_scale,
    )
    cable_joint_ids, stiffness_range = cable_randomization_stiffness_range(
        env.mj_model,
        multiplier_range=cable_stiffness_multiplier_range,
    )
    print(
        "Cable stiffness domain randomization: "
        f"{cable_joint_ids.size} joints, "
        f"multiplier_range={cable_stiffness_multiplier_range}, "
        f"joint_stiffness_range=[{stiffness_range[0]:.3e}, "
        f"{stiffness_range[1]:.3e}], "
        f"log_uniform={cable_stiffness_log_uniform}"
    )
    if cable_length_min_scale <= 0 or cable_length_max_scale <= 0:
        raise ValueError("Cable length scales must be positive.")
    if cable_length_min_scale > cable_length_max_scale:
        raise ValueError("cable_length_min_scale must be <= cable_length_max_scale.")
    cable_length_scale_range = (cable_length_min_scale, cable_length_max_scale)
    cable_body_ids, cable_length_joint_ids, cable_geom_ids, segment_length = (
        cable_randomization_length_spec(env.mj_model)
    )
    total_length_range = (
        cable_length_min_scale * segment_length * cable_geom_ids.size,
        cable_length_max_scale * segment_length * cable_geom_ids.size,
    )
    print(
        "Cable length domain randomization: "
        f"{cable_body_ids.size} bodies, "
        f"{cable_length_joint_ids.size} joints, "
        f"{cable_geom_ids.size} geoms, "
        f"scale_range={cable_length_scale_range}, "
        f"total_length_range=[{total_length_range[0]:.4f}, "
        f"{total_length_range[1]:.4f}]"
    )

    def randomization_fn(mjx_model):
        rng = jax.random.PRNGKey(0)
        stiffness_rng, length_rng = jax.random.split(rng)
        randomized_model, in_axes = domain_randomize(
            mjx_model,
            env.mj_model,
            jax.random.split(stiffness_rng, batch_size),
            cable_youngs_multiplier_range=cable_stiffness_multiplier_range,
            cable_shear_multiplier_range=cable_stiffness_multiplier_range,
            log_uniform=cable_stiffness_log_uniform,
        )
        return randomize_cable_length(
            randomized_model,
            env.mj_model,
            jax.random.split(length_rng, batch_size),
            scale_range=cable_length_scale_range,
            in_axes=in_axes,
        )

    env = wrap_for_mjx_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
    )
    env = DomainRandomizationBatchWrapper(env, batch_size=batch_size)
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
    """IBRL + Diffusion Policy on V3 PipeInsert2."""

    expert_data_dir: Path = Path(
        "testing/experiments/pipe_insert/V3/datasets/align_z_and_pos"
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
    rollout_every_episodes: int = 0
    # rollout_every_episodes: int = 10
    run_rollout_only: bool = False
    impl: Literal["jax", "warp", "c"] = "warp"
    # impl: Literal["jax", "warp", "c"] = "jax"
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
    wandb: bool = True
    # wandb: bool = True
    # wandb: bool = True
    action_dim_override: Optional[int] = 6

    a_max_lim: float = 1
    a_min_lim: float = -1

    # v3 1
    # action_trans_high: Optional[float] = 0.007
    # action_trans_low: Optional[float] = -0.007
    # action_rot_high: Optional[float] = 0.01
    # action_rot_low: Optional[float] = -0.01
    # v3 2
    action_trans_high: Optional[float] = 0.005
    action_trans_low: Optional[float] = -0.005
    action_rot_high: Optional[float] = 0.007
    action_rot_low: Optional[float] = -0.007

    decision_block: bool = True
    # decision_block: bool = False

    soft_update_beta: float = 0.2

    cable_stiffness_min_scale: float = DEFAULT_CABLE_STIFFNESS_MULTIPLIER_RANGE[0]
    cable_stiffness_max_scale: float = DEFAULT_CABLE_STIFFNESS_MULTIPLIER_RANGE[1]
    cable_stiffness_log_uniform: bool = True
    cable_length_min_scale: float = 1.0
    cable_length_max_scale: float = 1.0


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
        cable_stiffness_min_scale=args.cable_stiffness_min_scale,
        cable_stiffness_max_scale=args.cable_stiffness_max_scale,
        cable_stiffness_log_uniform=args.cable_stiffness_log_uniform,
        cable_length_min_scale=args.cable_length_min_scale,
        cable_length_max_scale=args.cable_length_max_scale,
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
            "V3 PipeInsert2 reports action_dim=0. "
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
