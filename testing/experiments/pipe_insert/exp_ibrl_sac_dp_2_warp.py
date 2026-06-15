import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from performance import save_timings, timer
from testing.experiments.pipe_insert.exp_utils_warp import (
    exp_set_seed,
    get_dp_config,
    get_drlr_config,
    get_expert_memory_2,
    get_memory,
    get_td3_models,
    get_trainer,
    setup_environment,
)
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel
from testing.shen.ibrl_sac_o_o2_clean_test import IBRL

# Source - https://stackoverflow.com/a/74483570
# Posted by Dev Gurung
# Retrieved 2026-02-09, License - CC BY-SA 4.0

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


expert_data_dir = "testing/experiments/pipe_insert/run/data_collection_2"

date_time = datetime.now()
time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")

actor = "il"
rollout_every_episodes = 1
run_rollout_only = False
dpiter = 3
num_envs = 100
episode_length = 3500
rl_il_equal_scale = True

wandb = False
il_action_scale = 0.17
warmup_timesteps = 100_000

dctrl_rl = [-0.005, 0.005]

reward_str = "dense"

exp_name = (
    Path(__file__).stem
    + f"_dpiter_{dpiter}"
    + f"_actor_{actor}"
    + f"_{reward_str}_"
    + time_stamp
)
exp_dir = Path(__file__).parent


def load_trained_dp_policy(
    checkpoint_path: Path, device: torch.device, config: dict, a_dim: int, o_dim: int
) -> DiffusionPolicy:
    """Load a trained diffusion policy for IL guidance."""

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
    agent = agent.load(checkpoint_path, device=device)

    return agent


exp_set_seed()

env = setup_environment(batch_size=num_envs, version=2, episode_length=episode_length)

drlr_config = get_drlr_config(exp_name, env, wandb=wandb)
drlr_config["actor"] = actor
drlr_config["il_ctrl_scale"] = il_action_scale
drlr_config["warmup_timesteps"] = warmup_timesteps
drlr_config["rl_il_equal_scale"] = rl_il_equal_scale

dp_config = get_dp_config(exp_name=exp_name, env=env)

rl_models = get_td3_models(env)

a_dim = env.action_space.shape[0]
o_dim = env.observation_space.shape[0]

dp_checkpoint = "testing/experiments/pipe_insert/.runs/data_100_2_1_action_horizon/models/latest_model.pth"
dp_policy = load_trained_dp_policy(
    checkpoint_path=dp_checkpoint,
    device=env.device,
    config=dp_config,
    a_dim=a_dim,
    o_dim=o_dim,
)

dp_policy._num_diffusion_iters = dpiter
dp_policy._act_horizon = 1

il_models = {"policy": dp_policy}

expert_memory = get_expert_memory_2(
    expert_data_dir=expert_data_dir,
    states_label="states_dense",
    actions_label="actions",
    rewards_label="rewards_dense",
    next_states_label="next_states_dense",
    dones_label="dones",
)

memory = get_memory(env, capacity=100_000)

device = env.device

agent = IBRL(
    models=rl_models,
    models_il=il_models,
    memory=memory,
    expert_memory=expert_memory,
    cfg=drlr_config,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


rollout_video_dir = Path("/home/vims/git/skrl/testing/experiments/pipe_insert/media")
print("trainer...")
trainer = get_trainer(
    env,
    agent,
    timesteps=1_000_000,
    trainer_cfg={
        "rollout_video_every_episodes": rollout_every_episodes,
        "rollout_video_num_steps": episode_length,
        "rollout_video_dir": rollout_video_dir.as_posix(),
        "rollout_video_prefix": "train_rollout",
        "rollout_video_env_index": 0,
        "log_rollout_path": "testing/experiments/pipe_insert/tmp/success_ibrl_mjx.json",
        "log_rollout_steps": episode_length,
        "log_rollout_exit": False,
    },
)
if run_rollout_only:
    print("recording rollout video for env 0...")
    trainer._record_rollout_video(0)
    print("rollout complete. exiting.")
    sys.exit(0)

print("training...")
with timer("train"):
    trainer.train()

save_timings(exp_dir / "results/models" / exp_name / "performance.json")
