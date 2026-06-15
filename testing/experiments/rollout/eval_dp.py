# quit()
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.markov import MarkovDataset
from datasets.pushert import download_dataset
from skrl import logger
from testing.experiments.pipe_insert import exp_utils
from testing.shen.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    DiffusionPolicy,
)
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

env = exp_utils.setup_environment(batch_size=1)

a_dim: int = env.action_space.shape[0]
o_dim: int = env.observation_space.shape[0]

dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG

# Setup paths
model_path = Path(__file__).parent / ".runs"
model_path.mkdir(parents=True, exist_ok=True)
media_path = model_path / "media2"
media_path.mkdir(exist_ok=True)

dp_config["experiment"]["directory"] = model_path.as_posix()
dp_config["experiment"]["experiment_name"] = "diffusion_policy_trajectory_2"
dp_config["experiment"]["wandb"] = False
# trainer_config["write_interval"] = 1
# trainer_config["checkpoint_interval"] = 10
# trainer_config["batch_size"] = 32
# trainer_config["epochs"] = 100
# trainer_config["shuffle"] = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# Build models
dp_models = {}
dp_models["model"] = ConditionalUnet1D(
    a_dim=a_dim, o_dim=o_dim, config=dp_config
).to(device)
ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
dp_models["ema_model"] = ConditionalUnet1D(
    a_dim=a_dim, o_dim=o_dim, config=dp_config
).to(device)

# Create agent
agent = DiffusionPolicy(
    a_dim=a_dim, o_dim=o_dim, models=dp_models, ema=ema, config=dp_config
)

agent.load("testing/experiments/trainer/.runs/diffusion_policy_trajectory_2/checkpoints/agent_90.pt")


def generate_evaluation_video_2(env, agent, media_path, epoch_label, obs_horizon):
    """Generate evaluation video for the trained policy."""
    # This function depends on your specific environment
    # You'll need to adapt this based on how you want to evaluate your policy

    # Example structure:
    # 1. Create test environment
    # 2. Run policy for several episodes
    # 3. Record frames
    # 4. Save as video
    logger.info(f"Generating evaluation video for epoch {epoch_label}...")

    print(f"{env.action_space.shape=}")
    print(f"{env.observation_space.shape=}")

    # Placeholder - implement based on your environment
    video_path = media_path / f"evaluation_epoch_{epoch_label}.mp4"
    exp_utils.rollout_history(
        file_name=f"media/dp_pushert_{epoch_label}.mp4", env=env, agent=agent, obs_horizon=obs_horizon
    )
    torch.cuda.empty_cache()

generate_evaluation_video_2(env=env, agent=agent, media_path=media_path, epoch_label="1",obs_horizon=agent.cfg["obs_horizon"],
)