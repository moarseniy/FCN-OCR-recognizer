from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from fcn_synth_generator.dataset import SingleLineDatasetConfig

from .config import TrainingConfig
from .tasks import TrainingTask


def prepare_batch(imgs, targets, device):
    imgs = imgs.to(device, non_blocking=True)
    if imgs.dtype == torch.uint8:
        imgs = imgs.float().div_(255.0)
    else:
        imgs = imgs.float()
    targets = targets.to(device=device, non_blocking=True)
    return imgs, targets


def validate(
    model,
    loader,
    device,
    max_batches=50,
    preview_saver=None,
    log_every=0,
    augmenter=None,
    *,
    task: TrainingTask,
    task_config: TrainingConfig,
    dataset_config: SingleLineDatasetConfig,
):
    model.eval()
    total_loss = 0.0
    batches = 0
    samples = 0
    started_at = time.perf_counter()

    with torch.no_grad():
        total_batches = (
            min(max_batches, len(loader)) if max_batches is not None else len(loader)
        )
        for batch_idx, (imgs, targets) in enumerate(loader, start=1):
            if max_batches is not None and batches >= max_batches:
                break

            imgs, targets = prepare_batch(imgs, targets, device)
            if augmenter is not None:
                imgs, targets = augmenter.augment_batch(imgs, targets, task.name)
            if preview_saver is not None:
                preview_saver.save_batch(imgs, targets)

            loss = task.compute_loss(
                model(imgs), targets, task_config, dataset_config
            )
            total_loss += loss.item()
            batches += 1
            samples += imgs.size(0)

            if log_every and batch_idx % log_every == 0:
                print(
                    f"  val   batch {batch_idx:04d}/{total_batches:04d} "
                    f"loss={loss.item():.6f} avg={total_loss / batches:.6f} "
                    f"samples={samples}"
                )

    if batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {
        "loss": total_loss / batches,
        "batches": batches,
        "samples": samples,
        "seconds": time.perf_counter() - started_at,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    max_batches=None,
    preview_saver=None,
    log_every=0,
    augmenter=None,
    *,
    task: TrainingTask,
    task_config: TrainingConfig,
    dataset_config: SingleLineDatasetConfig,
):
    model.train()
    total_loss = 0.0
    batches = 0
    samples = 0
    started_at = time.perf_counter()
    total_batches = (
        min(max_batches, len(loader)) if max_batches is not None else len(loader)
    )

    for batch_idx, (imgs, targets) in enumerate(loader, start=1):
        if max_batches is not None and batches >= max_batches:
            break

        imgs, targets = prepare_batch(imgs, targets, device)
        if augmenter is not None:
            imgs, targets = augmenter.augment_batch(imgs, targets, task.name)
        if preview_saver is not None:
            preview_saver.save_batch(imgs, targets)

        loss = task.compute_loss(model(imgs), targets, task_config, dataset_config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1
        samples += imgs.size(0)
        if log_every and batch_idx % log_every == 0:
            print(
                f"  train batch {batch_idx:04d}/{total_batches:04d} "
                f"loss={loss.item():.6f} avg={total_loss / batches:.6f} "
                f"samples={samples}"
            )

    if batches == 0:
        raise RuntimeError("Training loader produced no batches")
    return {
        "loss": total_loss / batches,
        "batches": batches,
        "samples": samples,
        "seconds": time.perf_counter() - started_at,
    }


def tensor_to_pil(image_tensor):
    image = image_tensor.detach().cpu().float().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image[0]
    if image.shape[0] == 1:
        array = (image[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def describe_target_for_preview(target):
    if torch.is_floating_point(target):
        if target.dim() == 3 and target.size(0) == 2:
            top_y = int(target[0].amax(dim=1).argmax().item()) if target.numel() else -1
            bottom_y = int(target[1].amax(dim=1).argmax().item()) if target.numel() else -1
            max_value = float(target.max().item()) if target.numel() else 0.0
            return (
                "<baseline_detection "
                f"top_y={top_y} bottom_y={bottom_y} max={max_value:.3f}>"
            )
        peak_count = int((target > 0.5).sum().item())
        max_value = float(target.max().item()) if target.numel() else 0.0
        return (
            "<vertical_segmentation "
            f"peaks={peak_count}/{target.numel()} max={max_value:.3f}>"
        )
    return "<fcn_ocr>"


class InputPreviewSaver:
    def __init__(self, output_dir, count):
        self.output_path = Path(output_dir)
        self.count = count
        self.saved = 0
        self.labels_file = None
        if count > 0:
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.labels_file = (self.output_path / "labels.tsv").open("w")
            self.labels_file.write("file\ttarget\n")

    def save_batch(self, images, targets):
        if self.count <= 0 or self.saved >= self.count:
            return
        for image, target in zip(images, targets):
            if self.saved >= self.count:
                return
            filename = f"{self.saved:04d}.png"
            tensor_to_pil(image).save(self.output_path / filename)
            self.labels_file.write(
                f"{filename}\t{describe_target_for_preview(target)}\n"
            )
            self.labels_file.flush()
            self.saved += 1

    def close(self):
        if self.labels_file is not None:
            self.labels_file.close()
            self.labels_file = None
            print(f"Saved {self.saved} input previews to {self.output_path}")


def append_training_log(log_path: Path, row) -> None:
    is_new_file = not log_path.exists()
    with log_path.open("a") as file:
        if is_new_file:
            file.write(
                "epoch\ttrain_loss\tval_loss\ttrain_batches\tval_batches\t"
                "train_samples\tval_samples\tlr\tepoch_seconds\tis_best\n"
            )
        file.write(
            f"{row['epoch']}\t{row['train_loss']:.8f}\t{row['val_loss']:.8f}\t"
            f"{row['train_batches']}\t{row['val_batches']}\t"
            f"{row['train_samples']}\t{row['val_samples']}\t"
            f"{row['lr']:.8g}\t{row['epoch_seconds']:.3f}\t{int(row['is_best'])}\n"
        )


__all__ = [
    "InputPreviewSaver",
    "append_training_log",
    "describe_target_for_preview",
    "prepare_batch",
    "tensor_to_pil",
    "train_one_epoch",
    "validate",
]
