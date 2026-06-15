from datetime import datetime
from pathlib import Path

import torch

from performance import save_timings, timer
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_dp_config,
    get_drlr_config,
    get_expert_memory_2,
    get_memory,
    get_td3_models,
    setup_environment,
)
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel
from testing.shen.ibrl_sac_o_o2_clean_test import IBRL

expert_data_dir = "testing/experiments/pipe_insert/run/data_collection_2"
dp_checkpoint = (
    "testing/experiments/pipe_insert/.runs/data_100_2_1_action_horizon/models/latest_model.pth"
)


def load_trained_dp_policy(
    checkpoint_path: Path, device: torch.device, config: dict, a_dim: int, o_dim: int
) -> DiffusionPolicy:
    """Load a trained diffusion policy for IL guidance."""
    dp_models: dict = {}
    dp_models["model"] = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=config)
    ema = EMAModel(dp_models["model"].parameters(), power=config["ema_power"])
    dp_models["ema_model"] = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=config)

    agent = DiffusionPolicy(
        a_dim=a_dim,
        o_dim=o_dim,
        models=dp_models,
        ema=ema,
        device=device,
        config=config,
    )
    agent = agent.load(checkpoint_path, device=device)
    return agent


def main() -> None:
    date_time = datetime.now()
    time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")

    actor = "il"
    num_steps = 1000

    exp_name = Path(__file__).stem + f"_actor_{actor}_" + time_stamp
    exp_dir = Path(__file__).parent

    exp_set_seed()

    env = setup_environment(batch_size=1, version=2)

    drlr_config = get_drlr_config(exp_name, env, wandb=False)
    drlr_config["actor"] = actor

    dp_config = get_dp_config(exp_name=exp_name, env=env)

    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]

    dp_policy = load_trained_dp_policy(
        checkpoint_path=dp_checkpoint,
        device=env.device,
        config=dp_config,
        a_dim=a_dim,
        o_dim=o_dim,
    )
    dp_policy._num_diffusion_iters = 10
    dp_policy._act_horizon = 1

    il_models = {"policy": dp_policy}
    rl_models = get_td3_models(env)

    expert_memory = get_expert_memory_2(
        expert_data_dir=expert_data_dir,
        states_label="states_dense",
        actions_label="actions",
        rewards_label="rewards_dense",
        next_states_label="next_states_dense",
        dones_label="dones",
    )

    memory = get_memory(env, capacity=100_000)

    device = env.device
    agent = IBRL(
        models=rl_models,
        models_il=il_models,
        memory=memory,
        expert_memory=expert_memory,
        cfg=drlr_config,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    obs = env.reset()

    with timer("rollout"):
        for t in range(num_steps):
            states = torch.as_tensor(obs, device=device, dtype=torch.float32)
            agent.pre_interaction(states, t, num_steps)
            actions, _, _ = agent.act(states, t, num_steps)
            obs, reward, done, info = env.step(actions)
            next_states = torch.as_tensor(obs, device=device, dtype=torch.float32)
            agent.post_interaction(next_states, t, num_steps)
            if (t + 1) % 100 == 0:
                reward_val = float(torch.as_tensor(reward).mean())
                print(f"step={t+1} reward={reward_val:.4f}")

    save_timings(exp_dir / "results/models" / exp_name / "performance.json")


if __name__ == "__main__":
    main()
