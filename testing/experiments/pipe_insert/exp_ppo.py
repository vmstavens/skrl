from datetime import datetime
from pathlib import Path

import torch

from skrl.trainers.torch.sequential import SequentialTrainer
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_memory,
    get_ppo_config,
    get_ppo_memory,
    get_ppo_models,
    get_trainer,
    setup_environment,
)
from testing.shen.ppo import PPO

# 0) set seed
exp_set_seed()

# 1) setup the environment
env = setup_environment(batch_size=1)

# 2) setup ppo models
rl_models = get_ppo_models(env)


# 3) get bc config
date_time = datetime.now()
time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")

exp_name = Path(__file__).stem + "_" + time_stamp
exp_dir = Path(__file__).parent

ppo_config = get_ppo_config(exp_name, env, wandb=False)

# 4) get expert data
memory = get_ppo_memory(env)
expert_memory = get_memory(env, capacity=100)

# 5) get device
device = env.device

# 6) define agent
agent = PPO(
    models=rl_models,
    memory=memory,
    expert_memory=expert_memory,
    cfg=ppo_config,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

print("training...")
# Configure and instantiate the RL trainer
trainer = get_trainer(env, agent)

# start training
trainer.train()
