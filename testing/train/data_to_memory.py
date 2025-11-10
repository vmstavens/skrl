import numpy as np
import torch
from torch import utils
from torch.utils.data import Dataset

from skrl.memories.torch import RandomMemory


def data_to_memory(dataset: Dataset) -> RandomMemory:
    # Get all transitions from the dataset
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []

    for i in range(len(dataset)):
        transition = dataset[i]
        states.append(transition["state"].numpy())
        actions.append(transition["action"].numpy())

        if "next_state" in transition:
            next_states.append(transition["next_state"].numpy())
        else:
            # Handle case where next_state might not be available
            next_states.append(transition["state"].numpy())  # Fallback

        if "reward" in transition:
            rewards.append(transition["reward"].numpy())
        else:
            rewards.append(0.0)  # Default reward

        if "done" in transition:
            dones.append(transition["done"].numpy())
        else:
            dones.append(False)  # Default not done

    # Convert to arrays
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    terminated = torch.tensor(dones, dtype=torch.float32)

    memory_size = len(states)

    a_dim = actions.shape[1]  # Get action dimension from data
    o_dim = states.shape[1]  # Get observation dimension from data

    # Create expert memory
    expert_memory = RandomMemory(memory_size=memory_size)
    expert_memory.create_tensor(name="states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="actions", size=a_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="next_states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
    expert_memory.create_tensor(name="terminated", size=1, dtype=torch.float32)

    # Add samples to memory
    expert_memory.add_samples(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards.unsqueeze(-1),  # Add dimension for reward size
        terminated=terminated.unsqueeze(-1),  # Add dimension for terminated size
    )
    return expert_memory
