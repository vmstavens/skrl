from datetime import datetime
from pathlib import Path

import torch

from skrl.trainers.torch.sequential import SequentialTrainer
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_memory,
    get_ppo_config,
    get_ppo_memory,
    get_td3_config,
    get_td3_models,
    get_trainer,
    setup_environment,
)
from testing.shen.ppo import PPO
from testing.shen.td3 import TD3

# 0) set seed
exp_set_seed()

# 1) setup the environment
env = setup_environment(batch_size=100)

# 2) setup ppo models
rl_models = get_td3_models(env)


# 3) get bc config
date_time = datetime.now()
time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")

exp_name = Path(__file__).stem + "_" + time_stamp
exp_dir = Path(__file__).parent

td3_config = get_td3_config(exp_name, env, wandb=True)
# ppo_config = get_ppo_config(exp_name, env, wandb=False)

# 4) get expert data
memory = get_memory(env, capacity=35_000)
# expert_memory = get_memory(env, capacity=100)

# 5) get device
device = env.device

# 6) define agent
agent = TD3(
    models=rl_models,
    memory=memory,
    # expert_memory=expert_memory,
    cfg=td3_config,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

print("training...")
# Configure and instantiate the RL trainer
trainer = get_trainer(env, agent)

# start training
trainer.train()
