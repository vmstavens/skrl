"""Train diffusion policy on the mocap pipe_insert dataset (rotation)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from skrl import logger
from testing.experiments.pipe_insert.exp_dp_rot import PipeInsertDataset
from testing.experiments.pipe_insert.exp_utils import get_dp_config, setup_environment
from testing.experiments.trainer.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
)
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


def _latest_checkpoint(runs_dir: Path) -> Path:
    pattern = "data_*_2_1_action_horizon_rot_*/models/latest_model.pth"
    candidates = list(runs_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {runs_dir.as_posix()} matching {pattern}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_config(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("config", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DP policy (rotation) on the mocap dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to the new dataset folder (json demos).",
        default="testing/experiments/pipe_insert/run/data_collection_2_mocap",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path or 'latest' to auto-pick.",
    )
    parser.add_argument("--obs-horizon", type=int, default=None)
    parser.add_argument("--pred-horizon", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--pct", type=float, default=1.0)
    parser.add_argument("--subsample-every", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    runs_dir = Path(__file__).parent / ".runs"
    resume_path = None
    if args.resume:
        if args.resume == "latest":
            resume_path = _latest_checkpoint(runs_dir)
        else:
            resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path.as_posix()}")

    # Environment (batch_size 1 for device reference)
    env = setup_environment(batch_size=1)
    device = env.device

    dp_config = get_dp_config(exp_name="pipe_insert_2_rot_mocap", env=env)

    if resume_path is not None:
        ckpt_cfg = _load_config(resume_path)
        dp_config.update(ckpt_cfg)
        obs_horizon = ckpt_cfg.get("obs_horizon")
        pred_horizon = ckpt_cfg.get("pred_horizon")
        if obs_horizon is None or pred_horizon is None:
            raise ValueError("Checkpoint missing obs_horizon/pred_horizon config.")
    else:
        dp_config["pred_horizon"] = 8
        dp_config["action_horizon"] = 1
        if args.obs_horizon is not None:
            dp_config["obs_horizon"] = args.obs_horizon
        if args.pred_horizon is not None:
            dp_config["pred_horizon"] = args.pred_horizon
        if args.action_horizon is not None:
            dp_config["action_horizon"] = args.action_horizon

    data_dir = args.data_dir
    full_dataset = PipeInsertDataset(
        data_dir=data_dir,
        pred_horizon=int(dp_config["pred_horizon"]),
        obs_horizon=int(dp_config["obs_horizon"]),
        pct=args.pct,
        subsample_every=args.subsample_every,
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

    if args.batch_size is not None:
        dp_config["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        dp_config["num_epochs"] = args.num_epochs

    dataloader = DataLoader(
        train_dataset,
        batch_size=dp_config["batch_size"],
        shuffle=True,
        num_workers=dp_config.get("num_workers", 0),
        pin_memory=True,
        persistent_workers=dp_config.get("num_workers", 0) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dp_config["batch_size"],
        shuffle=False,
        num_workers=dp_config.get("num_workers", 0),
        pin_memory=True,
        persistent_workers=dp_config.get("num_workers", 0) > 0,
    )

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

    if resume_path is not None:
        agent = DiffusionPolicy.load(
            resume_path.as_posix(), a_dim=a_dim, o_dim=o_dim, device=device
        )
        agent.to(device)
        agent.device = device
        agent.model.device = device
        agent.ema_model.device = device
        agent.model._unwrapped_module.to(device)
        agent.ema_model._unwrapped_module.to(device)
        agent.ema.shadow_params = [
            p.to(device) if torch.is_tensor(p) else p for p in agent.ema.shadow_params
        ]

        if agent.optimizer is not None:
            for state in agent.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
    else:
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
    agent.stats = full_dataset.stats

    trainer_config = SUPERVISED_TRAINER_DEFAULT_CONFIG.copy()
    trainer_config["batch_size"] = dp_config["batch_size"]
    trainer_config["num_epochs"] = dp_config["num_epochs"]
    trainer_config["eval_frequency"] = dp_config["eval_frequency"]
    trainer_config["num_workers"] = dp_config.get("num_workers", 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = "finetune_mocap" if resume_path is not None else "data_mocap"
    run_dir = Path(__file__).parent / f".runs/{run_tag}_{int(args.pct * 100)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    media_dir = run_dir / "media"
    media_dir.mkdir(exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

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

        plt.plot(epochs_history, train_losses, label="Training Loss")
        if val_losses:
            plt.plot(epochs_history, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = media_dir / "training_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        history_path = media_dir / "training_history.json"
        history_payload = {
            "epochs": epochs_history,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        history_path.write_text(json.dumps(history_payload, indent=2))

        agent.ema.copy_to(agent.ema_model.parameters())
        agent.set_mode("eval")

        epoch_model_path = models_dir / f"model_epoch_{epoch}.pth"
        agent.save(epoch_model_path.as_posix())
        latest_model_path = models_dir / "latest_model.pth"
        agent.save(latest_model_path.as_posix())

        agent.set_mode("train")

    trainer = SupervisedTrainer(
        agent=agent,
        trainer_config=trainer_config,
        train_loader=dataloader,
        valid_loader=val_loader,
        callback_fn=lambda epoch, loss, val_loss=None: training_callback(
            epoch, loss, val_loss
        ),
    )

    if resume_path is not None:
        logger.info(f"Resuming from {resume_path}")
    logger.info(f"Training data: {data_dir}")
    trainer.train()


if __name__ == "__main__":
    main()
