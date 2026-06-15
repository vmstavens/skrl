from datetime import datetime
from pathlib import Path

from testing.experiments.pipe_insert.exp_utils import (
    get_dp_config,
    get_dp_models,
    rollout_history,
    setup_environment,
)
from testing.shen.diffusion_policy_state import DiffusionPolicy

file_name = "media/dp/test.mp4"

# 0) env setup
env = setup_environment(batch_size=1)

# 1) config setup

date_time = datetime.now()
time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")

exp_name = Path(__file__).stem + "_" + time_stamp
exp_dir = Path(__file__).parent

dp_config = get_dp_config(exp_name=exp_name, env=env, wandb=False)

# 2) model setup

dp_models, ema = get_dp_models(env, config=dp_config)

agent = DiffusionPolicy(
    models=dp_models,
    ema=ema,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=env.device,
)


rollout_history(
    file_name=file_name, env=env, agent=agent, obs_horizon=dp_config["obs_horizon"]
)
