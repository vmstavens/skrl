import json
import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.datasets import load_data_files
from utils.dpy.utils.demo import (
    create_sample_indices,
    get_data_stats,
    normalize_data,
    sample_sequence,
)


class ImageStateDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        transform=None,
        eval_mode: bool = False,
    ):
        print("dataset_path: ", dataset_path)
        self.data, self.episode_indices = load_data_files(dataset_path)
        print("self.data: ", len(self.data))
        self.transform = transform

        # load images (index 0 assumed to be path)
        images = [self._load_image(row[0]) for row in self.data]
        images = np.stack([img.numpy() for img in images], axis=0)

        def to_numpy(idx):
            return np.array([row[idx] for row in self.data], dtype=np.float32)

        train_data = {
            "agent_pos": to_numpy(1),
            "action": to_numpy(2),
            "image": images,
        }
        episode_ends = self.episode_indices

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        stats, normalized_train_data = {}, {}
        for key, data in train_data.items():
            if key == "image":  # skip normalization for images
                normalized_train_data[key] = data
            else:
                stats[key] = get_data_stats(data)
                normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        with open("stats.pkl", "wb") as f:
            pickle.dump(stats, f)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )
        nsample["image"] = nsample["image"][: self.obs_horizon, :]
        nsample["agent_pos"] = nsample["agent_pos"][: self.obs_horizon, :]
        return nsample

    def _load_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
