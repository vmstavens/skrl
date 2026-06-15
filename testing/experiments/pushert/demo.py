import collections

# Assuming these are available from your environment
# from your_diffusion_policy_module import DiffusionPolicy, ConfigDict, DIFFUSION_POLICY_DEFAULT_CONFIG
# from your_env_module import PushTEnv
import logging
import os
import zipfile
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import gdown
import gym
import numpy as np
import torch
import zarr
from gym_pusht.envs import PushTEnv
from tqdm import tqdm

from testing.shen.diffusion_policy_state import DiffusionPolicy

logging.basicConfig(level=logging.WARN)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)


def download_dataset(verbose: bool = False) -> str:
    """Download PushT dataset to cache directory."""

    if verbose:
        logger.setLevel(logging.INFO)

    cache_dir = Path.home() / ".cache" / "pusht_train_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = cache_dir / "pusht_cchi_v7_replay.zarr.zip"

    if not dataset_path.exists():
        logger.info("Downloading PushT dataset...")
        file_id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq"
        gdown.download(id=file_id, output=str(dataset_path), quiet=False)
        logger.info(f"Dataset downloaded to: {dataset_path}")
    else:
        logger.info(f"Dataset already exists at: {dataset_path}")

    logger.setLevel(logging.WARN)

    return str(dataset_path)


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data: dict,
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> dict:
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data: np.ndarray) -> dict:
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def save_video(
    frames: List[np.ndarray], output_path: str = "evaluation.mp4", verbose: bool = False
) -> None:
    """Save frames as a video file."""
    if verbose:
        logger.setLevel(logging.INFO)
    try:
        import cv2

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        logging.info(f"Video saved to: {output_path}")
    except ImportError:
        logging.info("OpenCV not available. Skipping video saving.")
    logger.setLevel(logging.WARN)


def rollout(
    policy: DiffusionPolicy, stats: dict, max_steps: int = 200, seed: int = 100000
) -> Tuple[float, List[np.ndarray]]:
    """
    Evaluate the diffusion policy in the PushT environment using policy.predict().
    """
    # Extract config values
    obs_horizon = policy.config["obs_horizon"]
    action_horizon = policy.config["action_horizon"]

    # Initialize environment
    env = PushTEnv()
    env.obs_type = "state"

    # env.seed(seed)
    env._np_random_seed = seed
    env.render_mode = "rgb_array"

    # Get first observation
    obs, info = env.reset()

    # Keep a queue of last observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    # Save visualization and rewards
    imgs = [env.render()]
    rewards = list()
    done = False
    step_idx = 0

    # with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done and step_idx < max_steps:
        # Prepare observation sequence
        obs_seq = np.stack(obs_deque)
        nobs = normalize_data(obs_seq, stats["obs"])

        # Convert to tensor and move to device
        nobs_tensor = (
            torch.from_numpy(nobs).to(policy.device, dtype=torch.float32).unsqueeze(0)
        )

        # Predict action using policy (this should internally do the diffusion)
        with torch.no_grad():
            action_pred, _, _ = policy.act(nobs_tensor)
            # action_pred = policy.predict(nobs_tensor)

        # Only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[0, start:end, :].detach().cpu().numpy()

        # Unnormalize action
        action = unnormalize_data(action, stats["action"])

        # Execute action_horizon steps without replanning
        for i in range(len(action)):
            if done or step_idx >= max_steps:
                break

            # Step environment
            obs, reward, done, _, info = env.step(action[i])

            # Update observation deque
            obs_deque.append(obs)

            # Track rewards and frames
            rewards.append(reward)
            imgs.append(env.render())

            # Update progress
            step_idx += 1
            # pbar.update(1)
            # pbar.set_postfix(reward=reward)

    env.close()

    max_reward = max(rewards) if rewards else 0.0

    return max_reward, imgs


def rollout_vision(
    policy, stats: Dict, max_steps: int = 200, seed: int = 100000
) -> Tuple[float, List]:
    """
    Vision-specific rollout function for vision-based diffusion policies.

    Args:
        policy: The vision-based diffusion policy
        stats: Dataset statistics for normalization
        max_steps: Maximum number of environment steps
        seed: Environment seed

    Returns:
        max_reward: Maximum reward achieved during rollout
        frames: List of rendered frames
    """
    # Import environment here to avoid circular imports
    # from env.push_t_image_env import PushTImageEnv

    # Initialize environment
    env = PushTEnv()
    env.obs_type = "pixels_agent_pos"

    env._np_random_seed = seed
    env.render_mode = "rgb_array"

    # Get parameters from policy config
    obs_horizon = policy.config.obs_horizon
    action_horizon = policy.config.action_horizon
    device = policy.device

    # Get first observation
    obs, info = env.reset()

    # Keep a queue of last obs_horizon steps of observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    # Save visualization and rewards
    frames = [env.render()]
    rewards = []
    done = False
    step_idx = 0

    # with tqdm(total=max_steps, desc="Vision Policy Rollout") as pbar:
    # while not done:
    while not done and step_idx < max_steps:
        # Stack the last obs_horizon number of observations
        images = np.stack([x["pixels"] for x in obs_deque])
        agent_poses = np.stack([x["agent_pos"] for x in obs_deque])

        # Normalize observations
        nagent_poses = normalize_data(agent_poses, stats=stats["agent_pos"])
        nimages = images  # Images are already normalized to [0,1]

        # Prepare batch for vision policy
        batch = {
            "pixels": torch.from_numpy(nimages)
            .to(device, dtype=torch.float32)
            .unsqueeze(0),
            "agent_pos": torch.from_numpy(nagent_poses)
            .to(device, dtype=torch.float32)
            .unsqueeze(0),
        }

        # Infer action using the policy
        with torch.no_grad():
            action_pred = policy.predict(batch)

            # Unnormalize action
            action_pred = action_pred.detach().to("cpu").numpy()
            action_pred = action_pred[0]  # Remove batch dimension
            # unnormalize
            action_pred = unnormalize_data(action_pred, stats=stats["action"])

            # For vision policies, the predicted action is for the entire horizon
            # We need to take action_horizon steps
            action = action_pred[:action_horizon, :]

        # Execute action_horizon number of steps without replanning
        for i in range(len(action)):
            # Stepping env
            obs, reward, done, _, info = env.step(action[i])

            # Save observations
            obs_deque.append(obs)

            # Save reward and visualization
            rewards.append(reward)
            frames.append(env.render())

            # Update progress bar
            step_idx += 1
            # pbar.update(1)
            # pbar.set_postfix(reward=reward)

            if step_idx >= max_steps:
                done = True
            if done:
                break
        if done:
            break
    # quit()
    # Clean up
    env.close()

    max_reward = max(rewards) if rewards else 0.0
    return max_reward, frames


def normalize_data(data: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Normalize data using dataset statistics.

    Args:
        data: Data to normalize
        stats: Dictionary with 'min' and 'max' keys

    Returns:
        Normalized data
    """
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    min_val = stats["min"]
    max_val = stats["max"]

    # Avoid division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0

    normalized = 2.0 * (data - min_val) / range_val - 1.0
    return normalized.squeeze()


def unnormalize_data(data: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Unnormalize data using dataset statistics.

    Args:
        data: Data to unnormalize
        stats: Dictionary with 'min' and 'max' keys

    Returns:
        Unnormalized data
    """
    min_val = stats["min"]
    max_val = stats["max"]

    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    range_val = max_val - min_val
    unnormalized = (data + 1.0) / 2.0 * range_val + min_val
    return unnormalized.squeeze()
