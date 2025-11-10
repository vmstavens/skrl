import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class DemonstrationDataset(Dataset):
    def __init__(
        self,
        json_paths: List[Union[str, Path]],
        sequence_length: int,
        obs_horizon: int,
        action_horizon: int,
        state_key: str = "states",
        action_key: str = "actions",
        reward_key: Optional[str] = "rewards",
        next_state_key: Optional[str] = "next_states",
        done_key: Optional[str] = "dones",
        pad_before: int = 0,
        pad_after: int = 0,
        normalize: bool = False,
        normalize_rewards: bool = False,
    ):
        """
        A general dataset for loading demonstrations from JSON files and sampling trajectory segments.

        Args:
            json_paths: List of paths to JSON files containing demonstrations
            sequence_length: Total length of sequences to sample (prediction horizon)
            obs_horizon: Number of observation steps to use
            action_horizon: Number of action steps to use
            state_key: Key for states in JSON files
            action_key: Key for actions in JSON files
            reward_key: Key for rewards in JSON files (optional)
            next_state_key: Key for next states in JSON files (optional)
            done_key: Key for done flags in JSON files (optional)
            pad_before: Padding before episodes (usually obs_horizon - 1)
            pad_after: Padding after episodes (usually action_horizon - 1)
            normalize: Whether to normalize states and actions to [-1, 1]
            normalize_rewards: Whether to normalize rewards (only if reward_key is provided)
        """
        self.sequence_length = sequence_length
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.state_key = state_key
        self.action_key = action_key
        self.reward_key = reward_key
        self.next_state_key = next_state_key
        self.done_key = done_key
        self.normalize = normalize
        self.normalize_rewards = normalize_rewards

        # Track which optional keys are available
        self.has_rewards = False
        self.has_next_states = False
        self.has_dones = False

        # Load all demonstrations
        self.episodes = self._load_demonstrations(json_paths)

        # Create episode boundaries
        episode_ends = self._compute_episode_ends()

        # Create sample indices that respect episode boundaries
        self.indices = self._create_sample_indices(
            episode_ends, sequence_length, pad_before, pad_after
        )

        # Compute statistics for normalization
        if normalize or normalize_rewards:
            self.stats = self._compute_stats()
            self.normalized_data = self._normalize_data()
        else:
            self.stats = None
            self.normalized_data = None

    def _load_demonstrations(self, json_paths: List[Union[str, Path]]) -> List[Dict]:
        """Load all demonstration files and concatenate them."""
        episodes = []
        total_steps = 0

        for json_path in json_paths:
            path = Path(json_path)
            if not path.exists():
                logger.warning(f"File {path} does not exist, skipping")
                continue

            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    # Validate required keys
                    if self.state_key not in data or self.action_key not in data:
                        logger.warning(f"File {path} missing required keys, skipping")
                        continue

                    # Convert to numpy arrays
                    states = np.array(data[self.state_key], dtype=np.float32)
                    actions = np.array(data[self.action_key], dtype=np.float32)

                    # Validate shapes
                    if len(states) != len(actions):
                        logger.warning(
                            f"File {path} has mismatched state/action lengths, skipping"
                        )
                        continue

                    episode_data = {
                        "states": states,
                        "actions": actions,
                        "length": len(states),
                    }

                    # Load optional keys if they exist
                    if self.reward_key and self.reward_key in data:
                        rewards = np.array(data[self.reward_key], dtype=np.float32)
                        if len(rewards) == len(states):
                            episode_data["rewards"] = rewards
                            self.has_rewards = True
                        else:
                            logger.warning(
                                f"File {path} has mismatched reward length, skipping rewards"
                            )

                    if self.next_state_key and self.next_state_key in data:
                        next_states = np.array(
                            data[self.next_state_key], dtype=np.float32
                        )
                        if len(next_states) == len(states):
                            episode_data["next_states"] = next_states
                            self.has_next_states = True
                        else:
                            logger.warning(
                                f"File {path} has mismatched next_state length, skipping next_states"
                            )

                    if self.done_key and self.done_key in data:
                        dones = np.array(data[self.done_key], dtype=bool)
                        if len(dones) == len(states):
                            episode_data["dones"] = dones
                            self.has_dones = True
                        else:
                            logger.warning(
                                f"File {path} has mismatched done length, skipping dones"
                            )

                    episodes.append(episode_data)
                    total_steps += len(states)

                    logger.info(f"Loaded {path} with {len(states)} steps")

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error loading {path}: {e}, skipping")
                    continue

        if not episodes:
            raise ValueError("No valid demonstration files found")

        logger.info(f"Loaded {len(episodes)} episodes with {total_steps} total steps")
        logger.info(
            f"Optional keys available - rewards: {self.has_rewards}, next_states: {self.has_next_states}, dones: {self.has_dones}"
        )
        return episodes

    def _compute_episode_ends(self) -> np.ndarray:
        """Compute cumulative episode end indices."""
        episode_ends = []
        current_end = 0
        for episode in self.episodes:
            current_end += episode["length"]
            episode_ends.append(current_end)
        return np.array(episode_ends, dtype=np.int64)

    def _create_sample_indices(
        self,
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int,
        pad_after: int,
    ) -> List[Tuple]:
        """
        Create sample indices that respect episode boundaries.

        Returns list of tuples: (episode_idx, start_idx, end_idx, sample_start, sample_end)
        """
        indices = []
        total_length = episode_ends[-1] if len(episode_ends) > 0 else 0

        for episode_idx, episode in enumerate(self.episodes):
            episode_start = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
            episode_end = episode_ends[episode_idx]
            episode_length = episode["length"]

            # Calculate valid start positions within this episode
            for start_idx in range(
                pad_before, episode_length - sequence_length + 1 - pad_after
            ):
                buffer_start = episode_start + max(0, start_idx - pad_before)
                buffer_end = episode_start + min(
                    episode_length, start_idx + sequence_length + pad_after
                )

                sample_start = (
                    pad_before if start_idx >= pad_before else pad_before - start_idx
                )
                sample_end = sample_start + sequence_length

                indices.append(
                    (
                        episode_idx,  # Which episode
                        buffer_start,  # Start in concatenated buffer
                        buffer_end,  # End in concatenated buffer
                        sample_start,  # Start within sample
                        sample_end,  # End within sample
                    )
                )

        logger.info(f"Created {len(indices)} sample indices")
        return indices

    def _compute_stats(self) -> Dict:
        """Compute statistics for normalization."""
        all_states = np.concatenate([ep["states"] for ep in self.episodes])
        all_actions = np.concatenate([ep["actions"] for ep in self.episodes])

        stats = {
            "states": {
                "mean": np.mean(all_states, axis=0),
                "std": np.std(all_states, axis=0),
            },
            "actions": {
                "mean": np.mean(all_actions, axis=0),
                "std": np.std(all_actions, axis=0),
            },
        }

        # Compute stats for optional keys if they exist
        if self.has_rewards and self.normalize_rewards:
            all_rewards = np.concatenate([ep["rewards"] for ep in self.episodes])
            stats["rewards"] = {
                "mean": np.mean(all_rewards),
                "std": np.std(all_rewards),
            }

        if self.has_next_states and self.normalize:
            all_next_states = np.concatenate(
                [ep["next_states"] for ep in self.episodes]
            )
            stats["next_states"] = {
                "mean": np.mean(all_next_states, axis=0),
                "std": np.std(all_next_states, axis=0),
            }

        # Avoid division by zero
        for key in stats:
            if key in ["states", "actions", "next_states"]:
                stats[key]["std"] = np.where(
                    stats[key]["std"] < 1e-8, 1.0, stats[key]["std"]
                )
            elif key == "rewards":
                if stats[key]["std"] < 1e-8:
                    stats[key]["std"] = 1.0

        return stats

    def _normalize_data(self) -> List[Dict]:
        """Normalize data to [-1, 1] range."""
        normalized_episodes = []
        for episode in self.episodes:
            states = (episode["states"] - self.stats["states"]["mean"]) / self.stats[
                "states"
            ]["std"]
            actions = (episode["actions"] - self.stats["actions"]["mean"]) / self.stats[
                "actions"
            ]["std"]

            normalized_episode = {
                "states": states,
                "actions": actions,
                "length": episode["length"],
            }

            # Normalize optional keys if they exist
            if self.has_rewards and self.normalize_rewards:
                normalized_episode["rewards"] = (
                    episode["rewards"] - self.stats["rewards"]["mean"]
                ) / self.stats["rewards"]["std"]

            if self.has_next_states and self.normalize:
                normalized_episode["next_states"] = (
                    episode["next_states"] - self.stats["next_states"]["mean"]
                ) / self.stats["next_states"]["std"]

            # Don't normalize dones (they're boolean)
            if self.has_dones:
                normalized_episode["dones"] = episode["dones"]

            normalized_episodes.append(normalized_episode)

        return normalized_episodes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample as a dictionary of tensors."""
        episode_idx, buffer_start, buffer_end, sample_start, sample_end = self.indices[
            idx
        ]

        # Get the data (normalized or raw)
        if self.normalized_data is not None:
            episode = self.normalized_data[episode_idx]
        else:
            episode = self.episodes[episode_idx]

        # Extract the sequence for required keys
        states_seq = episode["states"][sample_start:sample_end]
        actions_seq = episode["actions"][sample_start:sample_end]

        # Convert to torch tensors
        result = {
            "states": torch.from_numpy(states_seq).float(),
            "actions": torch.from_numpy(actions_seq).float(),
        }

        # Extract optional keys if they exist
        if self.has_rewards:
            rewards_seq = episode["rewards"][sample_start:sample_end]
            result["rewards"] = torch.from_numpy(rewards_seq).float()

        if self.has_next_states:
            next_states_seq = episode["next_states"][sample_start:sample_end]
            result["next_states"] = torch.from_numpy(next_states_seq).float()

        if self.has_dones:
            dones_seq = episode["dones"][sample_start:sample_end]
            result["dones"] = torch.from_numpy(dones_seq).bool()

        # Add observation horizon information
        result["observations"] = result["states"][: self.obs_horizon]

        print(result.keys())

        return result

    def get_data_stats(self) -> Dict:
        """Get dataset statistics."""
        return self.stats

    def get_optional_keys_available(self) -> Dict[str, bool]:
        """Get which optional keys are available in the dataset."""
        return {
            "rewards": self.has_rewards,
            "next_states": self.has_next_states,
            "dones": self.has_dones,
        }


# Example usage
if __name__ == "__main__":
    # Find all JSON files in a directory
    data_dir = Path("data/robotB_data_trimmed/trimmed/valid")
    json_files = list(data_dir.glob("*.json"))

    # Create dataset with optional keys
    dataset = DemonstrationDataset(
        json_paths=json_files,
        sequence_length=16,  # Total sequence length
        obs_horizon=2,  # Use first 2 steps as observations
        action_horizon=8,  # Action horizon
        state_key="states",
        action_key="actions",
        reward_key="rewards",  # Optional
        next_state_key="next_states",  # Optional
        done_key="dones",  # Optional
        pad_before=1,  # obs_horizon - 1
        pad_after=7,  # action_horizon - 1
        normalize=True,
        normalize_rewards=True,  # Normalize rewards if they exist
    )

    # Check which optional keys are available
    print("Available optional keys:", dataset.get_optional_keys_available())

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Test the dataloader
    for batch in dataloader:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Observations: {batch['observations'].shape}")
        print(f"Actions: {batch['actions'].shape}")

        if "rewards" in batch:
            print(f"Rewards: {batch['rewards'].shape}")
        if "next_states" in batch:
            print(f"Next states: {batch['next_states'].shape}")
        if "dones" in batch:
            print(f"Dones: {batch['dones'].shape}")
        break
