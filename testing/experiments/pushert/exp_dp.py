"""Example of training diffusion policy with Markov dataset and generating videos."""

import collections
import os
from pathlib import Path
from typing import List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.markov import MarkovDataset
from datasets.pushert import (
    PushTStateDataset,
    download_dataset,
    normalize_data,
    unnormalize_data,
)
from skrl import logger
from skrl.envs.wrappers.torch import wrap_env
from testing.envs.pushert.pushert import PushTEnv
from testing.experiments.trainer.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
    generate_evaluation_video,
)
from testing.shen.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    DiffusionPolicy,
)
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

from .demo import rollout, save_video

# ---- Rollout helpers ----------------------------------------------------------


def rollout_policy(
    policy: DiffusionPolicy,
    stats: dict,
    max_steps: int = 200,
    seed: int = 100000,
) -> tuple[float, list[np.ndarray]]:
    """Deterministic rollout using the policy's EMA model and PushTEnv."""
    obs_horizon = policy.config["obs_horizon"]
    action_horizon = policy.config["action_horizon"]

    env = PushTEnv()
    env.obs_type = "state"
    env._np_random_seed = seed
    env.render_mode = "rgb_array"

    obs, info = env.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    imgs: list[np.ndarray] = [env.render()]
    rewards: list[float] = []
    done = False
    step_idx = 0

    policy.set_mode("eval")
    policy.ema.copy_to(policy.ema_model.parameters())

    while not done and step_idx < max_steps:
        obs_seq = np.stack(obs_deque)
        norm_obs = normalize_data(obs_seq, stats["obs"])
        obs_tensor = (
            torch.from_numpy(norm_obs)
            .to(policy.device, dtype=torch.float32)
            .unsqueeze(0)
        )

        with torch.no_grad():
            actions_pred, _, _ = policy.act(states=obs_tensor)

        start = obs_horizon - 1
        end = start + action_horizon
        actions = actions_pred[0, start:end, :].detach().cpu().numpy()
        actions = unnormalize_data(actions, stats["action"])

        for act in actions:
            if done or step_idx >= max_steps:
                break

            obs, reward, done, _, info = env.step(act)
            obs_deque.append(obs)

            rewards.append(float(reward))
            imgs.append(env.render())
            step_idx += 1

    env.close()
    return (max(rewards) if rewards else 0.0), imgs


def rollout_policy_with_env(
    policy: DiffusionPolicy,
    env,
    stats: dict,
    max_steps: int = 200,
    seed: int = 100000,
) -> tuple[float, list[np.ndarray]]:
    """Rollout using a provided environment (same logic as rollout_policy)."""
    obs_horizon = policy.config["obs_horizon"]
    action_horizon = policy.config["action_horizon"]

    # Set seed when possible
    if hasattr(env, "_np_random_seed"):
        env._np_random_seed = seed
    try:
        env.render_mode = "rgb_array"
    except Exception:
        pass

    obs, info = env.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    imgs: list[np.ndarray] = [env.render()]
    rewards: list[float] = []
    done = False
    step_idx = 0

    policy.set_mode("eval")
    policy.ema.copy_to(policy.ema_model.parameters())

    while not done and step_idx < max_steps:
        obs_seq = np.stack(obs_deque)
        norm_obs = normalize_data(obs_seq, stats["obs"])
        obs_tensor = (
            torch.from_numpy(norm_obs)
            .to(policy.device, dtype=torch.float32)
            .unsqueeze(0)
        )

        with torch.no_grad():
            actions_pred, _, _ = policy.act(states=obs_tensor)

        start = obs_horizon - 1
        end = start + action_horizon
        actions = actions_pred[0, start:end, :].detach().cpu().numpy()
        actions = unnormalize_data(actions, stats["action"])

        for act in actions:
            if done or step_idx >= max_steps:
                break

            obs, reward, done, _, info = env.step(act)
            obs_deque.append(obs)

            rewards.append(float(reward))
            imgs.append(env.render())
            step_idx += 1

    env.close()
    return (max(rewards) if rewards else 0.0), imgs


# Configuration
dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG
trainer_config = SUPERVISED_TRAINER_DEFAULT_CONFIG.copy()

# Setup paths
model_path = Path(__file__).parent / ".runs"
model_path.mkdir(parents=True, exist_ok=True)
media_path = model_path / "media"
media_path.mkdir(exist_ok=True)
models_path = model_path / "models"
models_path.mkdir(exist_ok=True)

dp_config["experiment"]["directory"] = model_path.as_posix()
dp_config["experiment"]["experiment_name"] = "pushert"
dp_config["experiment"]["wandb"] = False
dp_config["experiment"]["wandb_kwargs"] = {"group": "pushert"}
dp_config["num_workers"] = 0
trainer_config["write_interval"] = 1
trainer_config["checkpoint_interval"] = 10
trainer_config["batch_size"] = 256
trainer_config["epochs"] = 100
trainer_config["shuffle"] = True

device = "cuda" if torch.cuda.is_available() else "cpu"
# Be robust when CUDA is present in the build but not usable in the sandbox
if device == "cuda":
    try:
        _ = torch.cuda.current_device()
    except Exception:
        logger.warning("CUDA unavailable, falling back to CPU")
        device = "cpu"

logger.info("1. Downloading dataset...")
dataset_path = download_dataset()

env = PushTEnv()

env = wrap_env(env, wrapper="gym")

a_dim: int = env.action_space.shape[0]
o_dim: int = env.observation_space.shape[0]

dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=dp_config["pred_horizon"],
    obs_horizon=dp_config["obs_horizon"],
    action_horizon=dp_config["action_horizon"],
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=dp_config["batch_size"],
    num_workers=dp_config["num_workers"],
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=dp_config["num_workers"] > 0,
)

# Build models
dp_models = {}
dp_models["model"] = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(
    device
)
ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
dp_models["ema_model"] = ConditionalUnet1D(
    a_dim=a_dim, o_dim=o_dim, config=dp_config
).to(device)

# Create agent
agent = DiffusionPolicy(
    a_dim=a_dim,
    o_dim=o_dim,
    models=dp_models,
    ema=ema,
    device=device,
    config=dp_config,
)

# Track training progress
epochs_history = []
train_losses = []
val_losses = []


def training_callback(epoch, train_loss, val_loss=None):
    """Callback to track training progress, save checkpoints, and run eval rollouts."""
    epochs_history.append(epoch)
    train_losses.append(train_loss)

    if val_loss is not None:
        val_losses.append(val_loss)

    if epoch % trainer_config["eval_frequency"] != 0:
        return

    # Save training plot
    plt.plot(epochs_history, train_losses, label="Training Loss")
    if val_losses:
        plt.plot(epochs_history, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = media_path / "training_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Save model checkpoints (epoched and latest)
    epoch_model_path = models_path / f"model_epoch_{epoch}.pth"
    agent.save(epoch_model_path.as_posix())
    latest_model_path = models_path / "latest_model.pth"
    agent.save(latest_model_path.as_posix())

    # Eval rollout using EMA weights
    agent.is_trained = True
    agent.ema.copy_to(agent.ema_model.parameters())

    max_reward, frames = rollout(agent, dataset.stats, max_steps=dp_config["max_steps"])
    # video_path = _SAVE_PATH_MEDIA / f"epoch_{epoch:02d}_reward_{max_reward:04f}.mp4"
    video_path = media_path / f"epoch_{epoch:02d}_reward_{max_reward:04f}.mp4"
    save_video(frames, video_path.as_posix(), verbose=False)

    # max_reward, frames = rollout_policy(
    #     policy=agent, stats=dataset.stats, max_steps=dp_config["max_steps"]
    # )
    # video_path = media_path / f"epoch_{epoch:02d}_reward_{max_reward:04f}.mp4"
    # save_frames_as_video(
    #     frames=frames,
    #     output_path=video_path,
    #     fps=30,
    #     codec="mp4v",
    #     is_bgr=False,
    # )
    agent.is_trained = False


def save_frames_as_video(
    frames: List[np.ndarray],
    output_path: Union[str, os.PathLike],
    fps: int = 30,
    codec: str = "mp4v",
    # codec: str = "mp4v",
    frame_size: Optional[tuple] = None,
    is_bgr: bool = False,
) -> None:
    """
    Save a list of frames as a video file.

    Args:
        frames: List of numpy arrays representing frames (H, W, 3)
        output_path: Path where to save the video file
        fps: Frames per second for the output video
        codec: Video codec to use (common: 'mp4v', 'avc1', 'X264', 'MJPG')
        frame_size: Optional tuple (width, height) for video dimensions.
                    If None, uses the size of the first frame.
        is_bgr: Whether frames are in BGR format (OpenCV default).
                If False, assumes RGB and converts to BGR.

    Returns:
        None
    """
    if not frames:
        raise ValueError("No frames provided")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get frame dimensions from first frame if not specified
    if frame_size is None:
        height, width = frames[0].shape[:2]
        frame_size = (width, height)
    else:
        # Resize frames if needed
        if frames[0].shape[:2] != (frame_size[1], frame_size[0]):
            frames = [cv2.resize(frame, frame_size) for frame in frames]

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    try:
        for frame in frames:
            # Convert RGB to BGR if needed (OpenCV uses BGR)
            if not is_bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Ensure frame has correct type (uint8)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Write frame to video
            video_writer.write(frame)

    finally:
        # Release the video writer
        video_writer.release()

    print(f"Video saved to {output_path}")


# Create supervised trainer
trainer = SupervisedTrainer(
    agent=agent,
    trainer_config=trainer_config,
    train_loader=dataloader,
    # valid_loader=valid_loader,
    callback_fn=lambda epoch, loss: training_callback(epoch, loss),
)


trainer.train()
