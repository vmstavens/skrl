import json
import logging
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.datasets import load_data_files


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


# from utils.dpy.utils.demo import (
#     create_sample_indices,
#     get_data_stats,
#     normalize_data,
#     sample_sequence,
# )

logging.basicConfig(level=logging.WARN)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)


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


class StateDataset(Dataset):
    """Dataset for states only."""

    def __init__(
        self,
        dataset_dir: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        fn_data_extractor: Optional[
            Callable[[str], tuple[dict[str, np.ndarray], np.ndarray]]
        ] = None,
    ) -> None:
        if fn_data_extractor is not None:
            self.data, self.episode_indices = fn_data_extractor(dataset_dir)
        else:
            self.data, self.episode_indices = load_data_files(dataset_dir)

        self.obs_dim = self.data["states"].shape[1]
        self.act_dim = self.data["actions"].shape[1]

        # Normalize
        self.stats: Dict[str, Dict[str, np.ndarray]] = {
            k: get_data_stats(v) for k, v in self.data.items()
        }
        self.normalized_train_data: Dict[str, np.ndarray] = {
            k: normalize_data(v, self.stats[k]) for k, v in self.data.items()
        }

        # Build indices
        self.indices: List[Tuple[int, int, int, int]] = create_sample_indices(
            episode_ends=self.episode_indices,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        self.pred_horizon: int = pred_horizon
        self.obs_horizon: int = obs_horizon
        self.action_horizon: int = action_horizon

        # Create unique stats filename based on dataset directory
        self.STATS_PATH = Path(__file__).parent / ".stats"
        self.STATS_PATH.mkdir(parents=True, exist_ok=True)

        # Extract dataset name from path and create safe filename
        dataset_name = Path(dataset_dir).name
        # Remove any characters that might be problematic in filenames
        safe_dataset_name = "".join(
            c for c in dataset_name if c.isalnum() or c in ("-", "_")
        ).rstrip()
        # Include parameters in filename for additional uniqueness
        self.STATS_FILE_PATH = (
            self.STATS_PATH
            / f"state_stats_{safe_dataset_name}_ph{pred_horizon}_oh{obs_horizon}_ah{action_horizon}.pkl"
        )

        with open(self.STATS_FILE_PATH.as_posix(), "wb") as f:
            pickle.dump(self.stats, f)
        logger.info(f"Saving StateDataset stats to {self.STATS_FILE_PATH.as_posix()}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )
        nsample: Dict[str, np.ndarray] = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        nsample["states"] = nsample["states"][: self.obs_horizon]
        # nsample["actions"] = nsample["actions"][: self.action_horizon]
        return (
            torch.FloatTensor(nsample["states"]),
            torch.FloatTensor(nsample["actions"]),
        )


if __name__ == "__main__":
    # State-only dataset
    state_dataset = StateDataset(
        dataset_dir="data/sim/trajs/",
        pred_horizon=16,
        obs_horizon=8,
        action_horizon=8,
    )
