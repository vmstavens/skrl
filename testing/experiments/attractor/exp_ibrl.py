from pathlib import Path

import torch

from skrl.trainers.torch.sequential import SequentialTrainer
from testing.experiments.exp_utils import (
    exp_set_seed,
    get_bc_models,
    get_expert_memory,
    get_ibrl_config,
    get_memory,
    get_td3_models,
    get_trainer,
    setup_environment,
)
from testing.shen.ibrl import IBRL

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
ibrl_config = get_ibrl_config(Path(__file__).stem, env)

# 4) get expert data and memory
expert_memory = get_expert_memory()
memory = get_memory(env)

# 5) get device
device = env.device

# 6) define agent
agent = IBRL(
    models=rl_models,
    models_il=il_models,
    memory=memory,
    expert_memory=expert_memory,
    cfg=ibrl_config,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# Configure and instantiate the RL trainer
trainer = get_trainer(env, agent)

# start training
trainer.train()
