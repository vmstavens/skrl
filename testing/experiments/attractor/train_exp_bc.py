from datetime import datetime
from pathlib import Path

from skrl.trainers.torch.sequential import SequentialTrainer
from testing.experiments.exp_utils import (
    create_data,
    exp_set_seed,
    get_bc_config,
    get_bc_models,
    get_expert_memory,
    rollout,
    setup_environment,
)
from testing.shen.BC import BC

raw_data_dir = "data/norm_data"
processed_data_dir = "data/test"

if False:
    # -1) generate data
    create_data(input_dir=raw_data_dir, output_dir=processed_data_dir)

# 0) set seed
exp_set_seed()

# 1) setup the environment
env = setup_environment()

# 2) setup bc model
bc_models = get_bc_models(env)

# 3) get bc config
date_time = datetime.now()
time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")

exp_name = Path(__file__).stem + "_" + time_stamp
exp_dir = Path(__file__).parent

bc_config = get_bc_config(exp_name, env, wandb=False)

# 4) get expert data
expert_memory = get_expert_memory(expert_data_dir=processed_data_dir)

# 5) get device
device = env.device

# 6) define agent
agent = BC(
    models=bc_models,
    expert_memory=expert_memory,
    cfg=bc_config,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# Configure and instantiate the RL trainer
cfg = {"timesteps": 2000, "headless": True}
trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent)

# start training
trainer.train()

agent.load("testing/experiments/results/models/train_exp_bc/checkpoints/best_agent.pt")

_EXPERIMENTS = Path(__file__).parent.parent
_VIDEO_PATH = _EXPERIMENTS / "media" / (Path(__file__).stem + ".mp4")

# 7) generate a rollout video
rollout(file_name=_VIDEO_PATH.as_posix(), env=env, agent=agent, end_on_terminate=True)

print(f"Saved video to {_VIDEO_PATH.as_posix()}")
