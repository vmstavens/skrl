import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


class TransitionDataset(Dataset):
    def __init__(
        self,
        json_paths: List[Union[str, Path]],
        state_key: str = "states",
        action_key: str = "actions",
        reward_key: Optional[str] = "rewards",
        next_state_key: Optional[str] = "next_states",
        done_key: Optional[str] = "dones",
        reward_fn: Optional[
            Callable[[np.ndarray, np.ndarray, np.ndarray], float]
        ] = None,
        termination_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
        normalize: bool = False,
        normalize_rewards: bool = False,
    ):
        """
        A dataset for loading single transitions from JSON files with optional reward
        and termination functions.

        Args:
            json_paths: List of paths to JSON files containing demonstrations
            state_key: Key for states in JSON files
            action_key: Key for actions in JSON files
            reward_key: Key for rewards in JSON files (optional)
            next_state_key: Key for next states in JSON files (optional)
            done_key: Key for done flags in JSON files (optional)
            reward_fn: Function to compute rewards: (state, action, next_state) -> reward
            termination_fn: Function to compute termination: (state, next_state) -> done
            normalize: Whether to normalize states and actions to [-1, 1]
            normalize_rewards: Whether to normalize rewards (only if reward_key is provided)
        """
        self.state_key = state_key
        self.action_key = action_key
        self.reward_key = reward_key
        self.next_state_key = next_state_key
        self.done_key = done_key
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn
        self.normalize = normalize
        self.normalize_rewards = normalize_rewards

        # Track which data sources are available
        self.has_rewards = False
        self.has_next_states = False
        self.has_dones = False
        self.using_reward_fn = reward_fn is not None
        self.using_termination_fn = termination_fn is not None
        self.next_state_source = "unknown"  # Track where next_states came from

        # Load all demonstrations
        self.transitions = self._load_transitions(json_paths)

        # Compute statistics for normalization
        if normalize or normalize_rewards:
            self.stats = self._compute_stats()
            self.normalized_data = self._normalize_data()
        else:
            self.stats = None
            self.normalized_data = None

    def _load_transitions(self, json_paths: List[Union[str, Path]]) -> List[Dict]:
        """Load all demonstration files and create individual transitions."""
        transitions = []
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

                    episode_length = len(states)

                    # 1. Check if next_states exist in data
                    next_states_from_data = None
                    if self.next_state_key and self.next_state_key in data:
                        next_states_from_data = np.array(
                            data[self.next_state_key], dtype=np.float32
                        )
                        if len(next_states_from_data) == episode_length:
                            self.has_next_states = True
                            self.next_state_source = "data"
                            logger.info(f"Using next_states from data for {path}")
                        else:
                            logger.warning(
                                f"File {path} has mismatched next_state length, will generate from states"
                            )
                            next_states_from_data = None

                    # 2. If no next_states in data, generate from state array
                    if next_states_from_data is None:
                        if episode_length > 1:
                            next_states_from_data = np.zeros_like(states)
                            next_states_from_data[:-1] = states[1:]
                            next_states_from_data[-1] = states[
                                -1
                            ]  # Terminal state repeats
                            self.has_next_states = True
                            self.next_state_source = "generated"
                            logger.info(
                                f"Generated next_states from state sequence for {path}"
                            )
                        else:
                            logger.warning(
                                f"Cannot generate next_states for single-step episode in {path}"
                            )
                            self.has_next_states = False

                    # Load rewards from data if available
                    rewards_data = None
                    if self.reward_key and self.reward_key in data:
                        rewards_data = np.array(data[self.reward_key], dtype=np.float32)
                        if len(rewards_data) == episode_length:
                            self.has_rewards = True
                        else:
                            logger.warning(
                                f"File {path} has mismatched reward length, skipping rewards from data"
                            )
                            rewards_data = None

                    # Load dones from data if available
                    dones_data = None
                    if self.done_key and self.done_key in data:
                        dones_data = np.array(data[self.done_key], dtype=bool)
                        if len(dones_data) == episode_length:
                            self.has_dones = True
                        else:
                            logger.warning(
                                f"File {path} has mismatched done length, skipping dones from data"
                            )
                            dones_data = None

                    # Create transition dictionaries
                    for i in range(episode_length):
                        transition = {
                            "state": states[i],
                            "action": actions[i],
                        }

                        # Add next_state if available
                        if self.has_next_states and next_states_from_data is not None:
                            transition["next_state"] = next_states_from_data[i]

                        # Determine reward source
                        if self.using_reward_fn and self.has_next_states:
                            # Use reward function
                            try:
                                transition["reward"] = self.reward_fn(
                                    states[i], actions[i], next_states_from_data[i]
                                )
                                self.has_rewards = True
                            except Exception as e:
                                logger.warning(f"Error in reward_fn at step {i}: {e}")
                                if rewards_data is not None:
                                    transition["reward"] = rewards_data[i]
                                    self.has_rewards = True
                        elif rewards_data is not None:
                            # Use reward from data
                            transition["reward"] = rewards_data[i]
                            self.has_rewards = True

                        # Determine termination source
                        if self.using_termination_fn and self.has_next_states:
                            # Use termination function
                            try:
                                transition["done"] = self.termination_fn(
                                    states[i], next_states_from_data[i]
                                )
                                self.has_dones = True
                            except Exception as e:
                                logger.warning(
                                    f"Error in termination_fn at step {i}: {e}"
                                )
                                if dones_data is not None:
                                    transition["done"] = dones_data[i]
                                    self.has_dones = True
                                else:
                                    transition["done"] = i == episode_length - 1
                                    self.has_dones = True
                        elif dones_data is not None:
                            # Use done from data
                            transition["done"] = dones_data[i]
                            self.has_dones = True
                        else:
                            # Default: last step is terminal
                            transition["done"] = i == episode_length - 1
                            self.has_dones = True

                        transitions.append(transition)

                    total_transitions += episode_length
                    logger.info(f"Loaded {path} with {episode_length} transitions")

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error loading {path}: {e}, skipping")
                    continue

        if not transitions:
            raise ValueError("No valid demonstration files found")

        logger.info(f"Loaded {len(transitions)} total transitions")
        logger.info(
            f"Data sources - rewards: {self.has_rewards} (function: {self.using_reward_fn}), "
            f"next_states: {self.has_next_states} (source: {self.next_state_source}), "
            f"dones: {self.has_dones} (function: {self.using_termination_fn})"
        )
        return transitions

    def _compute_stats(self) -> Dict:
        """Compute statistics for normalization."""
        all_states = np.array([t["state"] for t in self.transitions])
        all_actions = np.array([t["action"] for t in self.transitions])

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
            all_rewards = np.array([t["reward"] for t in self.transitions])
            stats["rewards"] = {
                "mean": np.mean(all_rewards),
                "std": np.std(all_rewards),
            }

        if self.has_next_states and self.normalize:
            all_next_states = np.array([t["next_state"] for t in self.transitions])
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
        normalized_transitions = []
        for transition in self.transitions:
            state = (transition["state"] - self.stats["states"]["mean"]) / self.stats[
                "states"
            ]["std"]
            action = (
                transition["action"] - self.stats["actions"]["mean"]
            ) / self.stats["actions"]["std"]

            normalized_transition = {
                "state": state,
                "action": action,
            }

            # Normalize optional keys if they exist
            if self.has_rewards and self.normalize_rewards:
                normalized_transition["reward"] = (
                    transition["reward"] - self.stats["rewards"]["mean"]
                ) / self.stats["rewards"]["std"]

            if self.has_next_states and self.normalize:
                normalized_transition["next_state"] = (
                    transition["next_state"] - self.stats["next_states"]["mean"]
                ) / self.stats["next_states"]["std"]

            # Don't normalize dones (they're boolean)
            if self.has_dones:
                normalized_transition["done"] = transition["done"]

            normalized_transitions.append(normalized_transition)

        return normalized_transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single transition as a dictionary of tensors."""
        # Get the data (normalized or raw)
        if self.normalized_data is not None:
            transition = self.normalized_data[idx]
        else:
            transition = self.transitions[idx]

        # Convert to torch tensors
        result = {
            "state": torch.from_numpy(transition["state"]).float(),
            "action": torch.from_numpy(transition["action"]).float(),
        }

        # Extract optional keys if they exist
        if self.has_rewards:
            result["reward"] = torch.tensor(transition["reward"]).float()

        if self.has_next_states:
            result["next_state"] = torch.from_numpy(transition["next_state"]).float()

        if self.has_dones:
            result["done"] = torch.tensor(transition["done"]).bool()

        return result

    def get_data_stats(self) -> Dict:
        """Get dataset statistics."""
        return self.stats

    def get_data_sources(self) -> Dict[str, str]:
        """Get which data sources are available and how they were computed."""
        return {
            "rewards": f"{self.has_rewards} (function: {self.using_reward_fn})",
            "next_states": f"{self.has_next_states} (source: {self.next_state_source})",
            "dones": f"{self.has_dones} (function: {self.using_termination_fn})",
        }


# Example usage with different scenarios
if __name__ == "__main__":
    # Define custom reward and termination functions
    def custom_reward_fn(state, action, next_state):
        # Example: negative distance from origin
        distance = np.linalg.norm(next_state)
        return -distance

    def custom_termination_fn(state, next_state):
        # Example: terminate if too far from origin
        distance = np.linalg.norm(next_state)
        return distance > 10.0

    # Find all JSON files in a directory
    data_dir = Path("data/robotB_data_trimmed/trimmed/valid")
    json_files = list(data_dir.glob("*.json"))

    print("=== Scenario 1: Data has next_states ===")
    # Create dataset assuming data has next_states
    try:
        dataset1 = TransitionDataset(
            json_paths=json_files[:1],  # Use just one file for testing
            state_key="states",
            action_key="actions",
            next_state_key="next_states",  # Looking for this in data
            reward_fn=custom_reward_fn,
            termination_fn=custom_termination_fn,
        )
        print("Data sources:", dataset1.get_data_sources())
    except Exception as e:
        print(f"Scenario 1 failed: {e}")

    print("\n=== Scenario 2: Data has NO next_states ===")
    # Create dataset when data doesn't have next_states
    try:
        dataset2 = TransitionDataset(
            json_paths=json_files[:1],
            state_key="states",
            action_key="actions",
            next_state_key="nonexistent_key",  # Key that doesn't exist
            reward_fn=custom_reward_fn,
            termination_fn=custom_termination_fn,
        )
        print("Data sources:", dataset2.get_data_sources())
    except Exception as e:
        print(f"Scenario 2 failed: {e}")

    print("\n=== Scenario 3: No next_state key specified ===")
    # Create dataset without specifying next_state_key
    try:
        dataset3 = TransitionDataset(
            json_paths=json_files[:1],
            state_key="states",
            action_key="actions",
            # No next_state_key specified
            reward_fn=custom_reward_fn,
            termination_fn=custom_termination_fn,
        )
        print("Data sources:", dataset3.get_data_sources())
    except Exception as e:
        print(f"Scenario 3 failed: {e}")
