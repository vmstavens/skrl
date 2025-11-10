from pathlib import Path

import torch

from skrl.trainers.torch.sequential import SequentialTrainer
from testing.experiments.exp_utils import (
    exp_set_seed,
    get_bc_config,
    get_bc_models,
    get_drlr_config,
    get_expert_memory,
    get_ibrl_config,
    get_memory,
    get_td3_models,
    get_trainer,
    rollout,
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

# 3) get bc config
drlr_config = get_ibrl_config(Path(__file__).stem, env)

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
    cfg=drlr_config,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

agent.load("testing/experiments/results/models/exp_ibrl/checkpoints/best_agent.pt")

_EXPERIMENTS = Path(__file__).parent.parent
_VIDEO_PATH = _EXPERIMENTS / "media" / (Path(__file__).stem + ".mp4")

# 7) generate a rollout video
rollout(file_name=_VIDEO_PATH.as_posix(), env=env, agent=agent)

print(f"Saved video to {_VIDEO_PATH.as_posix()}")
