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


class MarkovDataset(Dataset):
    def __init__(
        self,
        json_paths: List[Union[str, Path]],
        state_key: str = "states",
        action_key: str = "actions",
        reward_key: Optional[str] = "rewards",
        next_state_key: Optional[str] = "next_states",
        done_key: Optional[str] = "dones",
        normalize: bool = False,
        normalize_rewards: bool = False,
    ):
        """
        A Markov process dataset that returns (state, action, next_state, reward, done) transitions.

        Args:
            json_paths: List of paths to JSON files containing demonstrations
            state_key: Key for states in JSON files
            action_key: Key for actions in JSON files
            reward_key: Key for rewards in JSON files (optional)
            next_state_key: Key for next states in JSON files (optional)
            done_key: Key for done flags in JSON files (optional)
            normalize: Whether to normalize states and actions to [-1, 1]
            normalize_rewards: Whether to normalize rewards (only if reward_key is provided)
        """
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

        # Create transition indices (episode_idx, step_idx)
        self.indices = self._create_transition_indices()

        # Compute statistics for normalization
        if normalize or normalize_rewards:
            self.stats = self._compute_stats()
            self.normalized_data = self._normalize_data()
        else:
            self.stats = None
            self.normalized_data = None

    def _load_demonstrations(self, json_paths: List[Union[str, Path]]) -> List[Dict]:
        """Load all demonstration files."""
        episodes = []
        total_transitions = 0

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
                    total_transitions += (
                        len(states) - 1
                    )  # -1 because we need next state

                    logger.info(f"Loaded {path} with {len(states)} steps")

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error loading {path}: {e}, skipping")
                    continue

        if not episodes:
            raise ValueError("No valid demonstration files found")

        logger.info(
            f"Loaded {len(episodes)} episodes with {total_transitions} total transitions"
        )
        logger.info(
            f"Optional keys available - rewards: {self.has_rewards}, next_states: {self.has_next_states}, dones: {self.has_dones}"
        )
        return episodes

    def _create_transition_indices(self) -> List[Tuple[int, int]]:
        """Create indices for all valid transitions (episode_idx, step_idx)."""
        indices = []
        for episode_idx, episode in enumerate(self.episodes):
            # For each episode, we have transitions from step 0 to length-2
            # because we need next_state at step+1
            for step_idx in range(episode["length"] - 1):
                indices.append((episode_idx, step_idx))

        logger.info(f"Created {len(indices)} transition indices")
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
        """
        Returns one Markov transition: (state, action, next_state, reward, done).
        """
        episode_idx, step_idx = self.indices[idx]

        # Get the data (normalized or raw)
        if self.normalized_data is not None:
            episode = self.normalized_data[episode_idx]
        else:
            episode = self.episodes[episode_idx]

        # Extract current state and action
        state = episode["states"][step_idx]
        action = episode["actions"][step_idx]

        # Extract next state (always step_idx + 1)
        if self.has_next_states:
            next_state = episode["next_states"][step_idx]
        else:
            next_state = episode["states"][step_idx + 1]

        # Extract optional components
        reward = 0.0
        if self.has_rewards:
            reward = episode["rewards"][step_idx]

        done = False
        if self.has_dones:
            done = episode["dones"][step_idx]

        # Convert to torch tensors
        transition = {
            "state": torch.from_numpy(state).float(),
            "action": torch.from_numpy(action).float(),
            "next_state": torch.from_numpy(next_state).float(),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.bool),
        }

        return transition

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

    # Create Markov dataset
    dataset = MarkovDataset(
        json_paths=json_files,
        state_key="states",
        action_key="actions",
        reward_key="rewards",  # Optional
        next_state_key="next_states",  # Optional
        done_key="dones",  # Optional
        normalize=True,
        normalize_rewards=True,
    )

    # Check which optional keys are available
    print("Available optional keys:", dataset.get_optional_keys_available())

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Test the dataloader
    for batch in dataloader:
        print(f"State: {batch['state'].shape}")
        print(f"Action: {batch['action'].shape}")
        print(f"Next state: {batch['next_state'].shape}")
        print(f"Reward: {batch['reward'].shape}")
        print(f"Done: {batch['done'].shape}")
        break

    # You can also get individual transitions
    single_transition = dataset[0]
    print("\nSingle transition:")
    for key, value in single_transition.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")
