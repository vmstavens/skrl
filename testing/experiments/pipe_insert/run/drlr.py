import json
from pathlib import Path

import torch

from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_bc_models,
    get_drlr_config,
    get_expert_memory2,
    get_memory,
    get_td3_models,
    rollout_markov,
    setup_environment,
)
from testing.shen.drlr import DRLR


def load_trained_agent(env, checkpoint_path=None):
    """Load the trained DRLR agent"""
    # 1) Setup models
    rl_models = get_td3_models(env)
    il_models = get_bc_models(env)

    # Load the pre-trained BC policy
    saved_state = torch.load(
        "testing/experiments/pipe_insert/results/models/train_exp_bc_20251125_10_31_42/checkpoints/best_agent.pt"
    )
    policy_state = saved_state["policy"]
    il_models["policy"].load_state_dict(policy_state)
    il_models["policy"].eval()

    # 2) Get config
    drlr_config = get_drlr_config("rollout_eval", env, wandb=False)

    # 3) Get memory (minimal for rollout)
    expert_memory = get_expert_memory2(expert_data_dir="data/pipe_insert/smooth")
    memory = get_memory(env, capacity=10_000)  # Smaller capacity for rollout

    # 4) Create agent
    agent = DRLR(
        models=rl_models,
        models_il=il_models,
        memory=memory,
        expert_memory=expert_memory,
        cfg=drlr_config,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # 5) Load trained RL weights if available
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading trained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Load RL policy weights
        if "policy" in checkpoint:
            rl_models["policy"].load_state_dict(checkpoint["policy"])
            rl_models["target_policy"].load_state_dict(
                checkpoint["policy"]
            )  # Usually same for target
            print("Loaded RL policy weights")

        # Load critic weights if available
        if "critic_1" in checkpoint:
            rl_models["critic_1"].load_state_dict(checkpoint["critic_1"])
            rl_models["critic_2"].load_state_dict(checkpoint["critic_2"])
            rl_models["target_critic_1"].load_state_dict(
                checkpoint["critic_1"]
            )  # Usually same for target
            rl_models["target_critic_2"].load_state_dict(checkpoint["critic_2"])
            print("Loaded critic weights")

    return agent


def main():
    # Set seed for reproducibility
    exp_set_seed()

    # Setup environment
    print("Setting up environment...")
    env = setup_environment()

    # Find the latest checkpoint (modify this path as needed)
    checkpoint_dir = Path("testing/experiments/pipe_insert/results/models")

    # Look for the most recent DRLR training checkpoint
    drlr_checkpoints = list(
        checkpoint_dir.glob("train_exp_drlr_*/checkpoints/best_agent.pt")
    )

    checkpoint_path = None
    if drlr_checkpoints:
        # Get the most recent checkpoint by modification time
        checkpoint_path = max(drlr_checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        print("No DRLR checkpoint found. Using BC policy only.")
        # Alternative: look for any recent training checkpoint
        all_checkpoints = list(checkpoint_dir.glob("*/checkpoints/best_agent.pt"))
        if all_checkpoints:
            checkpoint_path = max(all_checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"Using alternative checkpoint: {checkpoint_path}")

    # Load trained agent
    print("Loading trained agent...")
    agent = load_trained_agent(env, checkpoint_path)

    # Perform rollout
    print("Starting rollout...")
    output_video_path = "data/rollouts/drlr_rollout.mp4"

    # Create output directory
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    rollout_markov(
        file_name=output_video_path,
        env=env,
        agent=agent,
        num_timesteps=1000,  # Adjust as needed
        end_on_terminate=True,  # Stop when task is completed
    )

    print(f"Rollout completed! Video saved to: {output_video_path}")

    # Also save the rollout data
    rollout_data_path = "data/rollouts/rollout_data.json"
    print(f"Rollout data saved to: {rollout_data_path}")


if __name__ == "__main__":
    main()
