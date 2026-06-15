"""Minimal supervised trainer for diffusion policy with PushT rollouts."""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from datasets.pushert import PushTStateDataset, download_dataset
from skrl.agents.torch import Agent
from testing.envs.pushert.pushert import PushTEnv
from testing.shen.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    DiffusionPolicy,
)
from testing.shen.dp_models import ConditionalUnet1D, EMAModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SUPERVISED_TRAINER_DEFAULT_CONFIG = {
    "epochs": 10,
    "validation_split": 0.2,
    "shuffle": True,
    "eval_frequency": 10,
    "num_workers": 0,
    "num_epochs": 100,
}


class SupervisedTrainer:
    """Lightweight trainer for offline supervised updates."""

    def __init__(
        self,
        agent: Agent,
        trainer_config: Optional[dict] = None,
        train_loader: Optional[DataLoader] = None,
        valid_loader: Optional[DataLoader] = None,
        callback_fn: Optional[Callable] = None,
    ):
        self.config = {**SUPERVISED_TRAINER_DEFAULT_CONFIG, **(trainer_config or {})}
        self.agent = agent
        self.epochs = self.config["num_epochs"]
        self.eval_frequency = self.config["eval_frequency"]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self._callback_fn = callback_fn

        # pass trainer config to agent (if agent uses it)
        self.agent.init(trainer_cfg=self.config)

    def _validate(self) -> float:
        self.agent.eval()
        total_loss, batch_count = 0.0, 0
        with torch.no_grad():
            for inputs, targets in self.valid_loader:
                inputs, targets = (
                    inputs.to(self.agent.device),
                    targets.to(self.agent.device),
                )
                loss = self.agent._update(inputs, targets)
                total_loss += loss.item()
                batch_count += 1
        return total_loss / max(batch_count, 1)

    def train(self):
        assert self.train_loader is not None, "Set train_loader first"

        self.agent.set_running_mode("train")
        self.agent.set_mode("train")

        for epoch in range(self.epochs):
            epoch_loss, batch_count = 0.0, 0

            for inputs, targets in tqdm.tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                file=sys.stdout,
            ):
                inputs, targets = (
                    inputs.to(self.agent.device),
                    targets.to(self.agent.device),
                )

                loss = self.agent._update(inputs, targets)

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            self.agent.track_data("Training/Loss", avg_loss)

            if epoch % self.eval_frequency == 0 and self._callback_fn:
                val_loss = None
                if self.valid_loader is not None:
                    val_loss = self._validate()
                    self.agent.track_data("Validation/Loss", val_loss)
                    self.agent.set_mode("train")
                self._callback_fn(epoch, avg_loss, val_loss)
                self.agent.set_mode("train")

            self.agent.write_tracking_data(epoch, self.epochs)


def rollout_pusht(
    policy: DiffusionPolicy, env: PushTEnv, max_steps: int = 200
) -> float:
    """Rollout the policy on PushT using first-action-only loop."""
    obs_horizon = policy.config["obs_horizon"]
    rewards: list[float] = []

    obs, _ = env.reset()
    obs = torch.as_tensor(obs, device=policy.device, dtype=torch.float32)
    obs_deque = deque([obs] * obs_horizon, maxlen=obs_horizon)

    was_training = policy.model.training
    policy.set_mode("eval")
    policy.ema.copy_to(policy.ema_model.parameters())

    for step in range(max_steps):
        obs_seq = torch.stack(list(obs_deque)).unsqueeze(0)
        with torch.no_grad():
            actions_pred, _, _ = policy.act(states=obs_seq)
        act = actions_pred[0, obs_horizon - 1, :].detach().cpu().numpy()

        obs, reward, done, truncated, info = env.step(act)
        obs_t = torch.as_tensor(obs, device=policy.device, dtype=torch.float32)
        obs_deque.append(obs_t)

        rewards.append(float(reward))
        if done:
            break

    if was_training:
        policy.train()
    return max(rewards) if rewards else 0.0


def train_diffusion_policy_pushert(
    epochs: int = 50, batch_size: int = 256, eval_frequency: int = 10
) -> DiffusionPolicy:
    """Train diffusion policy on PushT state dataset and perform rollouts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PushTEnv()
    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]

    # Dataset
    dataset_path = download_dataset()
    dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG.copy()
    pred_horizon = dp_config["pred_horizon"]
    act_horizon = dp_config["action_horizon"]
    obs_horizon = dp_config["obs_horizon"]

    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=dp_config.get("action_horizon", act_horizon),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dp_config.get("num_workers", 0),
        pin_memory=True,
        persistent_workers=dp_config.get("num_workers", 0) > 0,
    )

    dp_models = {
        "model": ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(
            device
        ),
        "ema_model": ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(
            device
        ),
    }
    ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])

    agent = DiffusionPolicy(
        a_dim=a_dim,
        o_dim=o_dim,
        models=dp_models,
        ema=ema,
        device=device,
        config=dp_config,
    )
    agent.stats = dataset.stats

    trainer_cfg = SUPERVISED_TRAINER_DEFAULT_CONFIG.copy()
    trainer_cfg["num_epochs"] = epochs
    trainer_cfg["eval_frequency"] = eval_frequency

    run_dir = Path(__file__).parent / ".runs" / "pusht_dp"
    run_dir.mkdir(parents=True, exist_ok=True)
    media_dir = run_dir / "media"
    media_dir.mkdir(exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    loss_history: list[float] = []

    def callback(epoch: int, train_loss: float, val_loss: float | None = None):
        if val_loss is None:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        else:
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )
        env = PushTEnv()
        reward = rollout_pusht(agent, env, max_steps=200)
        logger.info(f"Rollout reward: {reward:.3f}")

        loss_history.append(train_loss)
        # Save loss plot
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(loss_history, label="train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(media_dir / "loss.png", dpi=150)
        plt.close()

        # Save model checkpoints
        agent.save((models_dir / f"model_epoch_{epoch}.pth").as_posix())
        agent.save((models_dir / "latest_model.pth").as_posix())

        # Render rollout to video
        frames = []
        env_video = PushTEnv()
        obs, _ = env_video.reset()
        obs = torch.as_tensor(obs, device=agent.device, dtype=torch.float32)
        obs_deque = deque(
            [obs] * agent.config["obs_horizon"],
            maxlen=agent.config["obs_horizon"],
        )

        was_training = agent.model.training
        agent.set_mode("eval")
        agent.ema.copy_to(agent.ema_model.parameters())
        for _ in range(200):
            obs_seq = torch.stack(list(obs_deque)).unsqueeze(0)
            with torch.no_grad():
                actions_pred, _, _ = agent.act(states=obs_seq)
            act = (
                actions_pred[0, agent.config["obs_horizon"] - 1, :]
                .detach()
                .cpu()
                .numpy()
            )
            obs, reward, done, truncated, info = env_video.step(act)
            frames.append(env_video.render())

            obs_t = torch.as_tensor(obs, device=agent.device, dtype=torch.float32)
            obs_deque.append(obs_t)
            if done:
                break

        if was_training:
            agent.train()

        import cv2

        if frames:
            height, width, _ = frames[0].shape
            out_path = media_dir / f"rollout_epoch_{epoch}.mp4"
            writer = cv2.VideoWriter(
                out_path.as_posix(),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (width, height),
            )
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            logger.info(f"Saved rollout video to {out_path}")

    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config=trainer_cfg,
        train_loader=loader,
        callback_fn=callback,
    )
    trainer.train()
    return agent


if __name__ == "__main__":
    train_diffusion_policy_pushert(epochs=100)
