"""Train diffusion policy on the align-z-and-position pipe_insert dataset."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import tyro
from torch.utils.data import DataLoader

from skrl import logger
from testing.experiments.pipe_insert.exp_dp_rot import PipeInsertDataset
from testing.experiments.trainer.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
)
from testing.shen.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    DiffusionPolicy,
)
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


@dataclass
class Args:
    """Train DP policy on the align-z-and-position mocap dataset."""

    data_dir: Path = Path(
        "testing/experiments/pipe_insert/V3/datasets/subsample_x3_reduced_se3"
    )
    obs_horizon: Optional[int] = None
    pred_horizon: Optional[int] = None
    action_horizon: Optional[int] = None
    pct: float = 1.0
    subsample_every: int = 1
    num_epochs: int = 100
    batch_size: int = 256
    device: Optional[str] = None


def main(args: Optional[Args] = None) -> None:
    if args is None:
        args = tyro.cli(Args)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dp_config = copy.deepcopy(DIFFUSION_POLICY_STATE_DEFAULT_CONFIG)
    dp_config["experiment"]["experiment_name"] = "pipe_insert_reduced_se3"
    dp_config["experiment"]["wandb"] = False

    dp_config["pred_horizon"] = 16
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
        states_label="states",
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
        pin_memory=device.type == "cuda",
        # persistent_workers=dp_config.get("num_workers", 0) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dp_config["batch_size"],
        shuffle=False,
        num_workers=dp_config.get("num_workers", 0),
        pin_memory=device.type == "cuda",
        # persistent_workers=dp_config.get("num_workers", 0) > 0,
    )

    a_dim = full_dataset.action_dim
    o_dim = full_dataset.obs_dim

    dp_config["obs_dim"] = o_dim
    dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]

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
    run_tag = "align_z_and_pos_x3_16"
    run_dir = (
        Path(__file__).parent / f".runs/{run_tag}"
        # Path(__file__).parent / f".runs/{run_tag}_{int(args.pct * 100)}_{timestamp}"
    )
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

    logger.info(f"Training data: {data_dir}")
    logger.info(f"Training device: {device}")
    trainer.train()


if __name__ == "__main__":
    main()
