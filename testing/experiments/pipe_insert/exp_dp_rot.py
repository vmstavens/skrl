"""Train diffusion policy on pipe_insert data and generate rollouts."""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.pushert import normalize_data
from skrl import logger
from testing.envs.pipe_insert import PipeInsert
from testing.experiments.pipe_insert.exp_utils import get_dp_config, setup_environment
from testing.experiments.trainer.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
)
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


class PipeInsertDataset(Dataset):
    """Create horizon-aligned state/action sequences from JSON demos."""

    def __init__(
        self,
        data_dir: Path,
        pred_horizon: int,
        obs_horizon: int,
        pct: float = 1.0,
        subsample_every: int = 1,
        states_label: str = "states",
        actions_label: str = "actions",
    ):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.subsample_every = max(1, int(subsample_every))

        self.state_sequences: List[np.ndarray] = []
        self.action_sequences: List[np.ndarray] = []

        all_states: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []

        data_files = data_dir.glob("*.json")
        data_files = list(data_files)
        data_files = data_files[: int(len(data_files) * pct)]

        def _subsample(
            states: np.ndarray, actions: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if self.subsample_every <= 1:
                return states, actions
            step = self.subsample_every
            # Subsample states
            states_sub = states[::step]
            # Accumulate actions between sampled states
            accum_actions = []
            for i in range(0, len(actions), step):
                accum_actions.append(actions[i : i + step].sum(axis=0))
            actions_sub = np.stack(accum_actions, axis=0)
            min_len = min(len(states_sub), len(actions_sub))
            return states_sub[:min_len], actions_sub[:min_len]

        if len(data_files) == 0:
            raise ValueError((f"No files found in {data_dir}"))

        for path in sorted(data_files):
            # for path in sorted(data_dir.glob("*.json")):
            with open(path, "r") as f:
                data: dict = json.load(f)

            if states_label not in data.keys():
                raise ValueError(
                    f"Key '{states_label}' not in {path} (keys: {data.keys()}) "
                )

            states = np.asarray(data[states_label], dtype=np.float32)
            # states = np.asarray(data["states"], dtype=np.float32)
            actions = np.asarray(data[actions_label], dtype=np.float32)

            if len(states) == 0:
                raise ValueError(f"Obs states is empty with label {states_label=}")
            if len(actions) == 0:
                raise ValueError(f"Obs actions is empty with label {actions_label=}")

            # actions = np.asarray(data["actions"], dtype=np.float32)
            states, actions = _subsample(states, actions)
            if len(states) == 0 or len(actions) == 0:
                continue
            if len(states) < pred_horizon or len(actions) < pred_horizon:
                continue
            for start in range(0, len(states) - pred_horizon + 1):
                s_seq = states[start : start + pred_horizon]
                a_seq = actions[start : start + pred_horizon]
                self.state_sequences.append(s_seq)
                self.action_sequences.append(a_seq)
            all_states.append(states)
            all_actions.append(actions)

        with open("testing/experiments/pipe_insert/tmp/test.json", "w") as f:
            json.dump(
                {"states": states.tolist(), "actions": actions.tolist()}, f, indent=4
            )

        # Stats for normalization
        stacked_states = np.concatenate(all_states, axis=0)
        stacked_actions = np.concatenate(all_actions, axis=0)
        self.stats = {
            "obs": {
                "min": stacked_states.min(axis=0),
                "max": stacked_states.max(axis=0),
            },
            "action": {
                "min": stacked_actions.min(axis=0),
                "max": stacked_actions.max(axis=0),
            },
        }

        if not self.state_sequences or not self.action_sequences:
            raise ValueError(f"No usable sequences were found in {data_dir}")

        self.obs_dim = self.state_sequences[0].shape[-1]
        self.action_dim = self.action_sequences[0].shape[-1]

    def __len__(self) -> int:
        return len(self.state_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self.state_sequences[idx]
        actions = self.action_sequences[idx]
        norm_states = normalize_data(states, self.stats["obs"])
        norm_actions = normalize_data(actions, self.stats["action"])
        obs_seq = norm_states[: self.obs_horizon]
        act_seq = norm_actions
        return torch.from_numpy(obs_seq), torch.from_numpy(act_seq)


def rollout_policy_with_env(
    policy: DiffusionPolicy,
    env,
    max_steps: int = 200,
) -> tuple[float, list[np.ndarray]]:
    """Rollout using a provided environment."""
    obs_horizon = policy.config["obs_horizon"]
    action_horizon = policy.config["action_horizon"]

    obs, _ = env.reset()
    obs_cpu = obs.detach().cpu().squeeze(0)  # torch tensor
    obs_deque = deque([obs_cpu] * obs_horizon, maxlen=obs_horizon)

    frames: list[np.ndarray] = [env.render()]
    rewards: list[float] = []
    done = False
    step_idx = 0
    was_training = policy.model.training
    policy.set_mode("eval")
    policy.ema.copy_to(policy.ema_model.parameters())

    while not done and step_idx < max_steps:
        obs_seq = (
            torch.stack(list(obs_deque))
            .unsqueeze(0)
            .to(policy.device, dtype=torch.float32)
        )

        with torch.no_grad():
            actions_pred, _, _ = policy.act(states=obs_seq)

        start = obs_horizon - 1
        act = actions_pred[0, start, :].detach().cpu().numpy()

        # env expects batch dimension
        act_tensor = torch.from_numpy(act).to(policy.device).unsqueeze(0)

        # scale down action for slowing
        act_tensor *= 0.1

        next_obs, reward, terminated, truncated, _ = env.step(actions=act_tensor)
        next_obs_cpu = next_obs.detach().cpu().squeeze(0)
        obs_deque.append(next_obs_cpu)

        rewards.append(float(reward[0].item() if hasattr(reward, "shape") else reward))
        frames.append(env.render())
        if step_idx == 0:
            print(f"{act=}")
        step_idx += 1

        done = bool(terminated[0] or truncated[0])

    env.close()
    if was_training:
        policy.train()
    return (max(rewards) if rewards else 0.0), frames


def main():
    data_dir = Path("testing/experiments/pipe_insert/run/data_collection_2_rot")
    # data_dir = Path("data/mj_data/0_degrees_filt_sm")

    # Environment (batch_size 1 for rollouts)
    env = setup_environment(batch_size=1)

    # Configs
    dp_config = get_dp_config(exp_name="pipe_insert_2_rot", env=env, wandb=False)
    trainer_config = SUPERVISED_TRAINER_DEFAULT_CONFIG.copy()
    trainer_config["batch_size"] = dp_config["batch_size"]
    trainer_config["num_epochs"] = 100
    # trainer_config["num_epochs"] = dp_config["num_epochs"]
    trainer_config["eval_frequency"] = dp_config["eval_frequency"]
    trainer_config["num_workers"] = dp_config["num_workers"]
    dp_config.setdefault("subsample_every", 1)
    # dp_config.setdefault("subsample_every", 30)

    dp_config["batch_size"] = 256
    trainer_config["batch_size"] = dp_config["batch_size"]

    dp_config["pred_horizon"] = 8
    dp_config["action_horizon"] = 1

    device = env.device

    pct = 1.0

    # Dataset + loaders
    full_dataset = PipeInsertDataset(
        data_dir=data_dir,
        pred_horizon=dp_config["pred_horizon"],
        obs_horizon=dp_config["obs_horizon"],
        pct=pct,
        subsample_every=dp_config.get("subsample_every", 1),
        states_label="states_dense",
    )
    val_ratio = 0.1
    val_size = max(1, int(len(full_dataset) * val_ratio))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=dp_config["batch_size"],
        shuffle=True,
        num_workers=dp_config["num_workers"],
        pin_memory=True,
        persistent_workers=dp_config["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dp_config["batch_size"],
        shuffle=False,
        num_workers=dp_config["num_workers"],
        pin_memory=True,
        persistent_workers=dp_config["num_workers"] > 0,
    )

    # Align model dimensions with dataset (env definition may differ from data)
    a_dim = full_dataset.action_dim
    o_dim = full_dataset.obs_dim
    if a_dim != env.action_space.shape[0]:
        logger.warning(
            f"Dataset action dim ({a_dim}) != env action dim ({env.action_space.shape[0]}), using dataset dim"
        )
    if o_dim != env.observation_space.shape[0]:
        logger.warning(
            f"Dataset obs dim ({o_dim}) != env obs dim ({env.observation_space.shape[0]}), using dataset dim"
        )

    dp_config["obs_dim"] = o_dim
    dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]

    # Models
    dp_models: dict = {}
    dp_models["model"] = ConditionalUnet1D(
        a_dim=a_dim, o_dim=o_dim, config=dp_config
    ).to(device)
    ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
    dp_models["ema_model"] = ConditionalUnet1D(
        a_dim=a_dim, o_dim=o_dim, config=dp_config
    ).to(device)

    agent = DiffusionPolicy(
        a_dim=a_dim,
        o_dim=o_dim,
        models=dp_models,
        ema=ema,
        device=device,
        config=dp_config,
    )
    # Attach stats so the agent can handle normalization internally
    agent.stats = full_dataset.stats

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(__file__).parent
        / f".runs/data_{int(pct * 100)}_2_1_action_horizon_rot_{timestamp}"
    )
    # run_dir = Path(__file__).parent / f".runs/data_{int(pct * 100)}_2"
    run_dir.mkdir(parents=True, exist_ok=True)
    media_dir = run_dir / "media"
    media_dir.mkdir(exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Tracking
    epochs_history: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    def training_callback(epoch: int, train_loss: float, val_loss=None):
        epochs_history.append(epoch)
        train_losses.append(train_loss)
        if val_loss is not None:
            val_losses.append(val_loss)
        if epoch % trainer_config["eval_frequency"] != 0:
            return

        # Plot losses
        plt.plot(epochs_history, train_losses, label="Training Loss")
        if val_losses:
            plt.plot(epochs_history, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = media_dir / "training_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Sync EMA model before saving checkpoints
        agent.ema.copy_to(agent.ema_model.parameters())

        agent.set_mode("eval")

        # Save checkpoints
        epoch_model_path = models_dir / f"model_epoch_{epoch}.pth"
        agent.save(epoch_model_path.as_posix())
        latest_model_path = models_dir / "latest_model.pth"
        agent.save(latest_model_path.as_posix())

        agent.set_mode("train")
        # quit()

    agent.set_mode("train")

    # Trainer
    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config=trainer_config,
        train_loader=dataloader,
        valid_loader=val_loader,
        callback_fn=lambda epoch, loss, val_loss=None: training_callback(
            epoch, loss, val_loss
        ),
    )

    print("start training")
    trainer.train()


def save_frames_as_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    codec: str = "mp4v",
) -> None:
    if not frames:
        return
    import cv2

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    try:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
    finally:
        writer.release()
    logger.info(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
