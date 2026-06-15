"""Interactive Diffusion Policy rollout for the pipe_insert experiment.

Loads a trained diffusion policy checkpoint and steps the environment whenever
the user presses a key. Useful for quick sanity-checks of a saved model.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable

import numpy as np
import torch

from testing.experiments.pipe_insert.exp_utils import setup_environment
from testing.shen.diffusion_policy_state import DiffusionPolicy


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / ".runs/data_100/models/model_epoch_40.pth"
)
DEFAULT_STATS_PATH = (
    Path(__file__).resolve().parent.parent / ".stats/pipe_insert_stats.pkl"
)


def load_stats(stats_path: Path) -> Dict:
    """Load normalization stats from pickle or JSON."""
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    if stats_path.suffix == ".pkl":
        with open(stats_path, "rb") as f:
            return pickle.load(f)

    with open(stats_path) as f:
        return json.load(f)


def reset_history(env, obs_horizon: int) -> Deque[torch.Tensor]:
    """Reset environment and seed the observation history deque."""
    obs, _ = env.reset()
    # Remove batch dimension for stacking; keep on CPU for convenience
    obs_cpu = obs.detach().cpu()
    obs_array = obs_cpu.squeeze(0)
    return deque([obs_array] * obs_horizon, maxlen=obs_horizon)


def infer_actions(
    policy: DiffusionPolicy,
    obs_history: Iterable[torch.Tensor],
) -> np.ndarray:
    """Generate an action sequence from the current observation history."""
    obs_horizon = policy.config["obs_horizon"]
    action_horizon = policy.config.get(
        "action_horizon", policy.config.get("pred_horizon")
    )

    # Stack to shape (batch=1, obs_horizon, obs_dim)
    obs_seq = torch.stack(list(obs_history)).permute(1, 0, 2).to(
        policy.device, dtype=torch.float32
    )

    with torch.no_grad():
        actions, _, _ = policy.act(
            states=obs_seq, timestep=0, timesteps=action_horizon
        )

    # actions: (batch, pred_horizon, action_dim)
    actions_np = actions[0].detach().cpu().numpy()
    start = obs_horizon - 1
    end = start + action_horizon
    return actions_np[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive DP rollout for pipe_insert"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to diffusion policy checkpoint (.pth)",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=DEFAULT_STATS_PATH,
        help="Path to normalization stats (pickle/json)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum environment steps before exiting",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    env = setup_environment(batch_size=1)
    device = env.device

    print(f"Loading model from {args.model}")
    policy = DiffusionPolicy.load(
        path=args.model, device=device, a_dim=None, o_dim=None
    )

    # Attach stats for normalization if they are not baked into the checkpoint
    if policy.stats is None:
        print(f"Loading stats from {args.stats}")
        policy.stats = load_stats(args.stats)

    obs_history = reset_history(env, policy.config["obs_horizon"])

    step_count = 0
    print(
        "Interactive control ready: [Enter]=step policy | 'r' reset | 'q' quit"
    )

    while step_count < args.max_steps:
        command = input("> ").strip().lower()
        if command in {"q", "quit"}:
            break
        if command in {"r", "reset"}:
            obs_history = reset_history(env, policy.config["obs_horizon"])
            print("Environment reset.")
            continue

        actions = infer_actions(policy, obs_history)
        for action in actions:
            act_tensor = (
                torch.as_tensor(action, device=device, dtype=torch.float32)
                .unsqueeze(0)
            )
            next_obs, reward, terminated, truncated, _ = env.step(actions=act_tensor)
            env.render()

            obs_history.append(next_obs.detach().cpu().squeeze(0))
            step_count += 1

            rew_value = float(reward[0].item()) if hasattr(reward, "shape") else float(
                reward
            )
            print(
                f"step {step_count}: action={np.array2string(action, precision=3)} reward={rew_value:.3f}"
            )

            terminated_flag = (
                bool(terminated[0]) if hasattr(terminated, "__len__") else bool(terminated)
            )
            truncated_flag = (
                bool(truncated[0]) if hasattr(truncated, "__len__") else bool(truncated)
            )

            if terminated_flag or truncated_flag:
                print("Episode ended. Resetting environment.")
                obs_history = reset_history(env, policy.config["obs_horizon"])
                break

            if step_count >= args.max_steps:
                break

    env.close()
    print("Simulation finished.")


if __name__ == "__main__":
    main()
