from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import torch

from fcn_architectures import normalize_architecture_name
from fcn_checkpoint_contract import CHECKPOINT_FORMAT, CHECKPOINT_VERSION

from .tasks import get_training_task


def build_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    val_loss,
    alphabet,
    config,
    train_losses,
    val_losses,
    scheduler=None,
):
    task = get_training_task(config["task"])
    architecture = normalize_architecture_name(str(config["architecture"]))
    architecture_params = dict(config["architecture_params"])
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "loss": loss,
        "val_loss": val_loss,
        "alphabet": alphabet,
        "config": config,
        "model_config": {
            "architecture": architecture,
            "architecture_params": architecture_params,
            "in_channels": config["channels"],
            "num_classes": task.num_outputs(alphabet),
            "task": task.name,
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    val_loss,
    alphabet,
    config,
    train_losses,
    val_losses,
    checkpoint_dir="checkpoints",
    scheduler=None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_epoch_{epoch}_{timestamp}.pth"
    )
    checkpoint = build_checkpoint(
        model,
        optimizer,
        epoch,
        loss,
        val_loss,
        alphabet,
        config,
        train_losses,
        val_losses,
        scheduler=scheduler,
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved to {latest_path}")
    return checkpoint_path


def save_named_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch,
    loss,
    val_loss,
    alphabet,
    config,
    train_losses,
    val_losses,
    scheduler=None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_checkpoint(
            model,
            optimizer,
            epoch,
            loss,
            val_loss,
            alphabet,
            config,
            train_losses,
            val_losses,
            scheduler=scheduler,
        ),
        path,
    )
    return path


__all__ = ["build_checkpoint", "save_checkpoint", "save_named_checkpoint"]
