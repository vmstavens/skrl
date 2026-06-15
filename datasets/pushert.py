import collections
import json
import logging
import os
import pickle
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import gdown
import numpy as np
import pandas as pd
import torch
from numcodecs import get_codec
from PIL import Image
from torch.utils.data import Dataset

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


class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        # read from zarr dataset
        # dataset_root = zarr.open("/content/pusht_cchi_v7_replay.zarr.zip")
        # dataset_root = zarr.open(dataset_path, 'r')
        dataset_dir = Path(dataset_path).parent
        extract_path = dataset_dir / "extracted_zarr"

        # Reuse existing extraction if present to avoid repeated writes (which can fail in sandboxes)
        if not extract_path.exists() or not (extract_path / ".zattrs").exists():
            extract_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dataset_path, "r") as zf:
                zf.extractall(extract_path)

        dataset_root = Path(extract_path)

        def load_zarr_array(array_dir: Path) -> np.ndarray:
            """Lightweight Zarr v2 reader to avoid heavy open() async path."""
            metadata = json.loads((array_dir / ".zarray").read_text())
            shape = tuple(metadata["shape"])
            chunks = tuple(metadata["chunks"])
            dtype = np.dtype(metadata["dtype"])
            compressor = get_codec(metadata["compressor"])

            arr = np.empty(shape, dtype=dtype)
            # iterate over chunk grid and stitch into output array
            grid = [range((s + c - 1) // c) for s, c in zip(shape, chunks)]

            for idx in np.ndindex(*[len(g) for g in grid]):
                chunk_indices = [grid[d][idx[d]] for d in range(len(grid))]
                # chunk file names are joined by '.' for >1 dims, plain for 1D
                chunk_name = (
                    ".".join(str(i) for i in chunk_indices)
                    if len(chunk_indices) > 1
                    else str(chunk_indices[0])
                )
                chunk_path = array_dir / chunk_name
                with open(chunk_path, "rb") as f:
                    chunk_raw = f.read()
                chunk_data = compressor.decode(chunk_raw)
                chunk_array = np.frombuffer(chunk_data, dtype=dtype).reshape(chunks)

                slices = []
                for dim, c in enumerate(chunks):
                    start = chunk_indices[dim] * c
                    end = min(start + c, shape[dim])
                    slices.append(slice(start, end))
                arr[tuple(slices)] = chunk_array[
                    tuple(slice(0, s.stop - s.start) for s in slices)
                ]

            return arr

        # All demonstration episodes are concatinated in the first dimension N
        self.train_data = {
            # (N, action_dim)
            "action": load_zarr_array(dataset_root / "data" / "action"),
            # (N, obs_dim)
            "obs": load_zarr_array(dataset_root / "data" / "state"),
        }
        # Marks one-past the last index for each episode
        episode_ends = load_zarr_array(dataset_root / "meta" / "episode_ends")

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in self.train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        return nsample["obs"], nsample["action"]
        # return nsample


if __name__ == "__main__":
    pass
