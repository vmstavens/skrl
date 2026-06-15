import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def generate_attractor_step(state, attractor, decay_factor=2.0):
    """
    Given a single state, compute the corresponding attractor action.
    """
    to_attractor = attractor - state
    distance = torch.norm(to_attractor, dim=-1, keepdim=True)
    scale = 1.0 - torch.exp(-decay_factor * distance)
    direction = to_attractor / (distance + 1e-8)
    action = direction * scale
    return action


class AttractorTrajectoryDataset(Dataset):
    def __init__(
        self,
        pred_horizon,
        obs_horizon,  # included for API parity
        action_horizon,  # included for API parity
        state_dim,
        box_size: float = 2.0,
        decay_factor: float = 2.0,
        num_trajectories: int = 10000,
    ):
        """
        Creates a dataset of random trajectories. Each trajectory has length pred_horizon.

        :param num_trajectories: Number of trajectories in the dataset
        :param pred_horizon: Length of each trajectory (states/actions)
        :param obs_horizon: Unused but present for compatibility with PushT
        :param action_horizon: Unused but present for compatibility
        :param box_size: Range for uniformly sampling initial states
        :param decay_factor: For computing attractor action magnitude
        :param state_dim: Dimensionality of states and actions
        """
        self.num_trajectories = num_trajectories
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.box_size = box_size
        self.decay_factor = decay_factor
        self.state_dim = state_dim

        # We’ll fix the attractor to the origin by default,
        # but you can generalize it if needed
        self.attractor = torch.zeros(state_dim)

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        """
        Returns one trajectory with 'obs' and 'action' each of shape
        (pred_horizon, state_dim).
        """
        # Random initial state
        current_state = (torch.rand(self.state_dim) - 0.5) * self.box_size

        obs_traj = []
        action_traj = []

        for _ in range(self.pred_horizon):
            # Record the state
            obs_traj.append(current_state.clone())

            # Compute the action for the current state
            action = generate_attractor_step(
                current_state, self.attractor, decay_factor=self.decay_factor
            )
            action_traj.append(action.clone())

            # "Next state" could be the result of taking that action
            current_state = current_state + action

        # Stack into tensors of shape (pred_horizon, state_dim)
        obs_traj = torch.stack(obs_traj, dim=0)
        action_traj = torch.stack(action_traj, dim=0)

        # Mimic the return format of your existing dataset
        return obs_traj, action_traj
