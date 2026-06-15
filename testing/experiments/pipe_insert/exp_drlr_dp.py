from datetime import datetime
from pathlib import Path

import torch

from performance import save_timings, timer
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_drlr_config,
    get_expert_memory2,
    get_expert_memory_2,
    get_ibrl_config,
    get_memory,
    get_td3_models,
    get_trainer,
    setup_environment,
)
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

# from testing.shen.drlr import DRLR
from testing.shen.ibrl_sac_o_o2_clean import IBRL

processed_data_dir = Path(
    "testing/experiments/pipe_insert/run/data_collection_2"
).as_posix()

date_time = datetime.now()
time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")

exp_name = Path(__file__).stem + time_stamp
exp_dir = Path(__file__).parent


def load_trained_dp_policy(
    checkpoint_path: Path, device: torch.device
) -> DiffusionPolicy:
    """Load a trained diffusion policy for IL guidance."""
    checkpoint: dict = torch.load(checkpoint_path, map_location=device)

    if "config" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing the DP config")

    dp_config = checkpoint["config"]
    a_dim = checkpoint.get("a_dim")
    o_dim = checkpoint.get("o_dim")
    if a_dim is None or o_dim is None:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing action/obs dims")

    dp_models = {
        "model": ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(
            device
        ),
        "ema_model": ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(
            device
        ),
    }
    ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])

    policy = DiffusionPolicy(
        a_dim=a_dim,
        o_dim=o_dim,
        models=dp_models,
        ema=ema,
        device=device,
        config=dp_config,
        stats=checkpoint.get("stats"),
    )
    policy.model.load_state_dict(checkpoint["model_state_dict"])
    policy.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    if checkpoint.get("optimizer_state_dict"):
        policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scheduler_state_dict"):
        policy.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Keep EMA shadow params aligned with loaded weights before switching to eval
    ema.shadow_params = [
        param.clone().detach() for param in policy.ema_model.parameters()
    ]
    policy.eval()
    return policy


# 0) set seed
exp_set_seed()

# 1) setup the environment
env = setup_environment(batch_size=1, version=2)

# 2) setup drlr models
rl_models = get_td3_models(env)

dp_checkpoint = (
    "testing/experiments/pipe_insert/.runs/data_100_2/models/latest_model.pth"
)
dp_policy = load_trained_dp_policy(dp_checkpoint, env.device)
il_models = {"policy": dp_policy}

# 3) get bc config
drlr_config = get_drlr_config(exp_name, env, wandb=True)

# 4) get expert data and memory
expert_memory = get_expert_memory_2(expert_data_dir=processed_data_dir)

memory = get_memory(env, capacity=100_000)

# 5) get device
device = env.device

# 6) define agent
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


# Configure and instantiate the RL trainer
trainer = get_trainer(env, agent, timesteps=1_000_000)
with timer("train"):
    # start training
    trainer.train()

save_timings(exp_dir / "results/models" / exp_name / "performance.json")
