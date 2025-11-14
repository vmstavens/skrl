from datetime import datetime
from pathlib import Path

import torch

from performance import save_timings, timer
from skrl.trainers.torch.sequential import SequentialTrainer
from testing.experiments.exp_utils import (
    exp_set_seed,
    get_bc_config,
    get_bc_models,
    get_drlr_config,
    get_expert_memory,
    get_memory,
    get_td3_models,
    get_trainer,
    setup_environment,
)
from testing.shen.drlr import DRLR

# 0) set seed
exp_set_seed()

# 1) setup the environment
env = setup_environment()

# 2) setup drlr models
rl_models = get_td3_models(env)
il_models = get_bc_models(env)

# Load the full saved state
saved_state = torch.load(
    "testing/experiments/results/models/train_exp_bc/checkpoints/best_agent.pt"
)
# Extract just the policy weights
policy_state = saved_state["policy"]
il_models["policy"].load_state_dict(policy_state)
il_models["policy"].eval()

# 3) get bc config

date_time = datetime.now()
time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")

exp_name = Path(__file__).stem + "_" + time_stamp
exp_dir = Path(__file__).parent

drlr_config = get_drlr_config(exp_name, env, wandb=True)

# 4) get expert data and memory
expert_memory = get_expert_memory()
memory = get_memory(env)

# 5) get device
device = env.device


# 6) define agent
agent = DRLR(
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
