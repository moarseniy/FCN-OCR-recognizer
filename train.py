# train.py
from fcn_synth_generator.chunk_dataset import (
    ChunkedLineDataset,
    load_chunk_metadata,
)
from fcn_synth_generator.chunk_metadata import (
    CHUNK_METADATA_FILENAME,
    GENERATION_CONFIG_FILENAME,
    ChunkMetadata,
)
from fcn_synth_generator.dataset import (
    SUPPORTED_AUGMENTATIONS,
    SingleLineDatasetConfig,
)
from fcn_synth_generator.gpu_augmentations import GpuTextAugmenter
from fcn_synth_generator.run_directories import (
    is_timestamped_directory,
    latest_timestamped_directory,
    timestamped_directory,
)
import argparse
import math
import shutil
import time
import yaml
from typing import Any, Callable
from torch.utils.data import DataLoader, Sampler, Subset, random_split

import torch
from fcn_architectures import (
    available_architectures,
    create_model,
    normalize_architecture_name,
)
from fcn_training import (
    TrainingTask,
    all_training_task_config_fields,
    available_training_tasks,
    get_training_task,
)
from fcn_checkpoint_contract import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    validate_checkpoint_contract,
)

from datetime import datetime
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SUPPORTED_SCHEDULERS = ("none", "reduce_on_plateau", "cosine", "step")
SUPPORTED_OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")
TRAINING_CONFIG_FILENAME = "training_config.yaml"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str
    architecture_params: dict[str, Any] = Field(default_factory=dict)

    chunks_dir: str = Field(min_length=1)

    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=128, ge=1)
    batch_count: int | None = Field(default=None, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    optimizer: str = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    adam_eps: float = Field(default=1e-8, gt=0.0)
    sgd_momentum: float = Field(default=0.9, ge=0.0)
    sgd_nesterov: bool = False
    rmsprop_alpha: float = Field(default=0.99, gt=0.0, lt=1.0)
    rmsprop_momentum: float = Field(default=0.0, ge=0.0)
    rmsprop_eps: float = Field(default=1e-8, gt=0.0)
    task: str
    fcn_ocr_crop_left: int = Field(default=6, ge=0)
    fcn_ocr_crop_right: int = Field(default=5, ge=0)
    fcn_ocr_strict_width: bool = False
    fcn_ocr_target_min_majority: float = Field(default=0.6, ge=0.0, le=1.0)
    fcn_ocr_space_weight: float = Field(default=1.0, gt=0.0)
    vertical_segmentation_crop_left: int = Field(default=0, ge=0)
    vertical_segmentation_crop_right: int = Field(default=0, ge=0)
    vertical_segmentation_strict_width: bool = True
    vertical_segmentation_loss: str = "mse"
    vertical_segmentation_positive_weight: float = Field(default=1.0, ge=1.0)
    baseline_detection_strict_size: bool = True
    baseline_detection_loss: str = "bce"
    baseline_detection_positive_weight: float = Field(default=4.0, ge=1.0)
    scheduler: str = "reduce_on_plateau"
    scheduler_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    scheduler_patience: int = Field(default=3, ge=0)
    scheduler_min_lr: float = Field(default=1e-6, ge=0.0)
    scheduler_threshold: float = Field(default=1e-4, ge=0.0)
    scheduler_cooldown: int = Field(default=0, ge=0)
    scheduler_t_max: int | None = Field(default=None, ge=1)
    scheduler_eta_min: float = Field(default=1e-6, ge=0.0)
    scheduler_step_size: int = Field(default=10, ge=1)
    scheduler_gamma: float = Field(default=0.5, gt=0.0)
    checkpoint_dir: str = "checkpoints"
    max_train_batches: int | None = None
    max_val_batches: int | None = 50
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = 0
    resume: bool = False

    num_workers: int = Field(default=0, ge=0)
    drop_last: bool = False
    prefetch_factor: int = Field(default=2, ge=1)
    persistent_workers: bool = True
    chunk_cache_size: int = Field(default=2, ge=1)
    chunk_aware_batches: bool = True

    log_every: int = Field(default=1, ge=0)
    preview_samples: int = Field(default=0, ge=0)
    preview_dir: str = "input_previews"

    gpu_augmentations: bool = True
    gpu_augment_val: bool = False
    augmentation_probabilities: dict[str, float] = Field(default_factory=dict)
    augmentations: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def model_validate_with_paths(
        cls, data: Any, config_path: str | Path
    ) -> "TrainingConfig":
        data = dict(data)
        config_dir = Path(config_path).resolve().parent
        for key in ("chunks_dir", "checkpoint_dir", "preview_dir"):
            value = data.get(key)
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = config_dir / path
                data[key] = str(path.resolve())
        return cls.model_validate(data)

    @field_validator("scheduler")
    @classmethod
    def scheduler_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        if value not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SUPPORTED_SCHEDULERS}")
        return value

    @field_validator("optimizer")
    @classmethod
    def optimizer_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        if value not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {SUPPORTED_OPTIMIZERS}")
        return value

    @field_validator("architecture")
    @classmethod
    def architecture_must_be_supported(cls, value: str) -> str:
        value = normalize_architecture_name(value)
        if value not in available_architectures():
            raise ValueError(f"architecture must be one of {available_architectures()}")
        return value

    @field_validator("task")
    @classmethod
    def task_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        supported = available_training_tasks()
        if value not in supported:
            raise ValueError(f"task must be one of {supported}")
        return value

    @field_validator("vertical_segmentation_loss", "baseline_detection_loss")
    @classmethod
    def task_loss_name_must_be_normalized(cls, value: str) -> str:
        return value.lower()

    @model_validator(mode="after")
    def task_fields_must_match_selected_task(self) -> "TrainingConfig":
        task = get_training_task(self.task)
        explicit_task_fields = self.model_fields_set & all_training_task_config_fields()
        unexpected = sorted(explicit_task_fields - task.config_fields)
        if unexpected:
            raise ValueError(
                f"training task {task.name!r} does not accept config fields: {unexpected}"
            )
        task.validate_config(self)
        return self

    @field_validator("augmentation_probabilities")
    @classmethod
    def augmentation_probabilities_must_be_valid(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        unknown = sorted(set(value) - set(SUPPORTED_AUGMENTATIONS))
        if unknown:
            raise ValueError(f"unknown augmentations: {unknown}")
        for name, probability in value.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"probability for {name} must be between 0 and 1")
        return value

    @field_validator("augmentations")
    @classmethod
    def augmentations_must_be_known(
        cls, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        unknown = sorted(set(value) - set(SUPPORTED_AUGMENTATIONS))
        if unknown:
            raise ValueError(f"unknown augmentation configs: {unknown}")
        return value

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
    """Сохраняет чекпоинт модели"""
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

    # Сохраняем также последнюю модель
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved to {latest_path}")

    return checkpoint_path


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
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
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


def create_optimizer(model, config: TrainingConfig):
    parameters = model.parameters()
    if config.optimizer == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        if config.sgd_nesterov and config.sgd_momentum <= 0.0:
            raise ValueError("sgd_nesterov requires sgd_momentum > 0")
        return torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            nesterov=config.sgd_nesterov,
        )
    if config.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=config.lr,
            alpha=config.rmsprop_alpha,
            eps=config.rmsprop_eps,
            weight_decay=config.weight_decay,
            momentum=config.rmsprop_momentum,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def print_optimizer_summary(config: TrainingConfig) -> None:
    print("Optimizer: ", config.optimizer)
    print(f"  lr={config.lr:g} weight_decay={config.weight_decay:g}")
    if config.optimizer in {"adam", "adamw"}:
        print(
            f"  betas=({config.adam_beta1:g}, {config.adam_beta2:g}) "
            f"eps={config.adam_eps:g}"
        )
    elif config.optimizer == "sgd":
        print(f"  momentum={config.sgd_momentum:g} nesterov={config.sgd_nesterov}")
    elif config.optimizer == "rmsprop":
        print(
            f"  alpha={config.rmsprop_alpha:g} momentum={config.rmsprop_momentum:g} "
            f"eps={config.rmsprop_eps:g}"
        )


def create_scheduler(optimizer, config: TrainingConfig):
    if config.scheduler == "none":
        return None
    if config.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            threshold=config.scheduler_threshold,
            cooldown=config.scheduler_cooldown,
            min_lr=config.scheduler_min_lr,
        )
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_t_max or config.epochs,
            eta_min=config.scheduler_eta_min,
        )
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def step_scheduler(
    scheduler, config: TrainingConfig, val_loss: float, optimizer
) -> tuple[float, float]:
    old_lr = current_lr(optimizer)
    if scheduler is None:
        return old_lr, old_lr
    if config.scheduler == "reduce_on_plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()
    return old_lr, current_lr(optimizer)


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
    """Валидация модели"""
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
                imgs, targets = augmenter.augment_batch(
                    imgs, targets, task.name
                )

            if preview_saver is not None:
                preview_saver.save_batch(imgs, targets)

            logits = model(imgs)

            loss = task.compute_loss(
                logits, targets, task_config, dataset_config
            )
            total_loss += loss.item()
            batches += 1
            samples += imgs.size(0)

            if log_every and (batch_idx % log_every == 0):
                running_loss = total_loss / batches
                print(
                    f"  val   batch {batch_idx:04d}/{total_batches:04d} "
                    f"loss={loss.item():.6f} avg={running_loss:.6f} samples={samples}"
                )

            # print(torch.isnan(loss), torch.isinf(loss))

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
            imgs, targets = augmenter.augment_batch(
                imgs, targets, task.name
            )

        if preview_saver is not None:
            preview_saver.save_batch(imgs, targets)

        logits = model(imgs)

        loss = task.compute_loss(
            logits, targets, task_config, dataset_config
        )

        # print(torch.isnan(loss), torch.isinf(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1
        samples += imgs.size(0)

        if log_every and (batch_idx % log_every == 0):
            running_loss = total_loss / batches
            print(
                f"  train batch {batch_idx:04d}/{total_batches:04d} "
                f"loss={loss.item():.6f} avg={running_loss:.6f} samples={samples}"
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
            bottom_y = (
                int(target[1].amax(dim=1).argmax().item()) if target.numel() else -1
            )
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
            description = describe_target_for_preview(target)
            tensor_to_pil(image).save(self.output_path / filename)
            self.labels_file.write(f"{filename}\t{description}\n")
            self.labels_file.flush()
            self.saved += 1

    def close(self):
        if self.labels_file is not None:
            self.labels_file.close()
            self.labels_file = None
            print(f"Saved {self.saved} input previews to {self.output_path}")


def append_training_log(log_path, row):
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


class RandomFixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, batch_count, seed=0):
        if len(dataset) <= 0:
            raise ValueError("dataset must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_count < 1:
            raise ValueError("batch_count must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        dataset_size = len(self.dataset)
        for _ in range(self.batch_count):
            yield torch.randint(
                dataset_size,
                (self.batch_size,),
                generator=generator,
                dtype=torch.long,
            ).tolist()

    def __len__(self):
        return self.batch_count


class ChunkBatchSampler(Sampler):
    def __init__(
        self,
        subset,
        base_dataset,
        batch_size,
        drop_last,
        shuffle,
        seed=0,
        batch_count=None,
    ):
        self.subset = subset
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.batch_count = batch_count
        self.epoch = 0
        self.groups = self._group_subset_positions_by_chunk()
        self.chunk_ids = list(self.groups)
        self.chunk_weights = torch.tensor(
            [len(self.groups[chunk_id]) for chunk_id in self.chunk_ids],
            dtype=torch.double,
        )
        if self.batch_count is not None and self.batch_count < 1:
            raise ValueError("batch_count must be >= 1")
        if not self.chunk_ids:
            raise ValueError("chunk batch sampler got an empty subset")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        if self.batch_count is not None:
            yield from self._iter_sampled_batches(generator)
            return

        chunk_ids = list(self.groups)
        if self.shuffle:
            permutation = torch.randperm(len(chunk_ids), generator=generator).tolist()
            chunk_ids = [chunk_ids[index] for index in permutation]

        for chunk_id in chunk_ids:
            positions = list(self.groups[chunk_id])
            if self.shuffle:
                permutation = torch.randperm(
                    len(positions), generator=generator
                ).tolist()
                positions = [positions[index] for index in permutation]

            for start in range(0, len(positions), self.batch_size):
                batch = positions[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self):
        if self.batch_count is not None:
            return self.batch_count

        total = 0
        for positions in self.groups.values():
            if self.drop_last:
                total += len(positions) // self.batch_size
            else:
                total += math.ceil(len(positions) / self.batch_size)
        return total

    def _group_subset_positions_by_chunk(self):
        groups = {}
        for subset_position in range(len(self.subset)):
            sample_index = self._sample_index(subset_position)
            chunk_id = self.base_dataset.chunk_index_for_sample(sample_index)
            groups.setdefault(chunk_id, []).append(subset_position)
        return groups

    def _iter_sampled_batches(self, generator):
        sampled_group_indices = torch.multinomial(
            self.chunk_weights,
            num_samples=self.batch_count,
            replacement=True,
            generator=generator,
        ).tolist()

        for group_index in sampled_group_indices:
            chunk_id = self.chunk_ids[group_index]
            positions = self.groups[chunk_id]
            if len(positions) >= self.batch_size:
                sampled_position_indices = torch.randperm(
                    len(positions),
                    generator=generator,
                )[: self.batch_size].tolist()
            else:
                sampled_position_indices = torch.randint(
                    len(positions),
                    (self.batch_size,),
                    generator=generator,
                    dtype=torch.long,
                ).tolist()
            yield [
                positions[position_index] for position_index in sampled_position_indices
            ]

    def _sample_index(self, subset_position):
        if isinstance(self.subset, Subset):
            return int(self.subset.indices[subset_position])
        return subset_position


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the FCN OCR recognizer on synthetic lines."
    )
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    return parser.parse_args()


def load_training_config(config_path: str | Path) -> tuple[TrainingConfig, dict]:
    with Path(config_path).open("r") as file:
        config_data = yaml.safe_load(file)
    return TrainingConfig.model_validate_with_paths(
        config_data, config_path
    ), config_data


def dataset_config_from_chunk_metadata(
    config: TrainingConfig,
    metadata: ChunkMetadata,
) -> SingleLineDatasetConfig:
    data = metadata.dataset_config_data()
    data.update(
        {
            "seed": config.seed,
            "augmentation_probabilities": config.augmentation_probabilities,
            "augmentations": config.augmentations,
        }
    )

    return SingleLineDatasetConfig.model_validate(data)


def effective_training_config_data(
    config: TrainingConfig, dataset_config: SingleLineDatasetConfig
) -> dict:
    task = get_training_task(config.task)
    foreign_task_fields = all_training_task_config_fields() - task.config_fields
    data = config.model_dump(exclude=foreign_task_fields)
    data.update(
        {
            "alphabet": dataset_config.alphabet,
            "space_char": dataset_config.space_char,
            "max_text_length": dataset_config.max_text_length,
            "channels": dataset_config.channels,
            "image_height": dataset_config.image_height,
            "image_width": dataset_config.image_width,
            "background": dataset_config.background,
        }
    )
    return data


def load_dataset_from_config(
    config: TrainingConfig,
) -> tuple[torch.utils.data.Dataset, SingleLineDatasetConfig]:
    task = get_training_task(config.task)

    chunks_dir = resolve_chunks_dir(config.chunks_dir)
    config.chunks_dir = str(chunks_dir)
    metadata = load_chunk_metadata(chunks_dir)
    metadata.require_task(task.name)
    dataset_config = dataset_config_from_chunk_metadata(config, metadata)
    dataset = ChunkedLineDataset(
        chunks_dir,
        cache_size=config.chunk_cache_size,
        config=dataset_config,
        task=task.name,
    )
    print(f"Dataset source: chunks ({chunks_dir})")
    print(
        f"Dataset metadata: {chunks_dir / CHUNK_METADATA_FILENAME} "
        f"(format={metadata.format}, version={metadata.version})"
    )
    return dataset, dataset_config


def make_data_loader(dataset, split_dataset, args, shuffle, seed, batch_count=None):
    common_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        common_kwargs["prefetch_factor"] = args.prefetch_factor
        common_kwargs["persistent_workers"] = args.persistent_workers

    if isinstance(dataset, ChunkedLineDataset) and args.chunk_aware_batches:
        return DataLoader(
            split_dataset,
            batch_sampler=ChunkBatchSampler(
                split_dataset,
                dataset,
                args.batch_size,
                args.drop_last,
                shuffle,
                seed,
                batch_count=batch_count,
            ),
            **common_kwargs,
        )

    if batch_count is not None:
        return DataLoader(
            split_dataset,
            batch_sampler=RandomFixedBatchSampler(
                split_dataset,
                args.batch_size,
                batch_count,
                seed,
            ),
            **common_kwargs,
        )

    return DataLoader(
        split_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=args.drop_last,
        **common_kwargs,
    )


def printable_char(char: str) -> str:
    if char == " ":
        return "<space>"
    if char == "\t":
        return "<tab>"
    if char == "\n":
        return "<newline>"
    return char


def validate_and_log_alphabet(
    dataset: ChunkedLineDataset,
    alphabet: str,
    max_text_length: int,
    checkpoint_dir: str | Path,
) -> None:
    metadata = dataset.metadata
    if alphabet != metadata.alphabet:
        raise ValueError(
            "Training alphabet order differs from chunk metadata; class indices would be corrupted"
        )
    text_counts = metadata.text_char_counts
    if text_counts is None or metadata.max_observed_text_length is None:
        raise ValueError("Current metadata must contain text statistics")
    ocr_counts = metadata.fcn_ocr_class_counts
    unused_chars = [char for char in alphabet if text_counts.get(char, 0) == 0]

    stats_path = Path(checkpoint_dir) / "alphabet_stats.tsv"
    with stats_path.open("w") as file:
        file.write("class_index\tchar\ttext_count\tocr_target_count\n")
        for class_index, char in enumerate(alphabet):
            ocr_count = "" if ocr_counts is None else str(ocr_counts[class_index])
            file.write(
                f"{class_index}\t{printable_char(char)}\t{text_counts.get(char, 0)}\t{ocr_count}\n"
            )

    print("\nAlphabet/data check:")
    print(f"  Metadata samples:       {metadata.samples}")
    print("  Alphabet order:         exact metadata order")
    print(
        f"  Unique chars in text:   {sum(count > 0 for count in text_counts.values())}"
    )
    print(f"  Max text length:        {metadata.max_observed_text_length}")
    print(f"  Stats file:             {stats_path}")
    print("  Per-char counts:")
    for class_index, char in enumerate(alphabet):
        ocr_suffix = (
            "" if ocr_counts is None else f", fcn_ocr_targets={ocr_counts[class_index]}"
        )
        print(
            f"    [{class_index:>3}] {printable_char(char):>9}: "
            f"text={text_counts.get(char, 0)}{ocr_suffix}"
        )

    if unused_chars:
        printable = ", ".join(printable_char(char) for char in unused_chars)
        print(f"  Alphabet chars absent in data: {printable}")

    if metadata.max_observed_text_length > max_text_length:
        raise ValueError(
            f"Data contains text length {metadata.max_observed_text_length}, "
            f"but training max_text_length is {max_text_length}"
        )


EpochCallback = Callable[[dict[str, Any]], None]


def resolve_checkpoint_dir(
    configured_dir: str | Path,
    resume: bool = False,
) -> Path:
    base_dir = Path(configured_dir)
    if not resume:
        return timestamped_directory(base_dir)

    if is_timestamped_directory(base_dir):
        return base_dir

    latest_dir = latest_timestamped_directory(
        base_dir,
        required_file="latest_checkpoint.pth",
    )
    if latest_dir is not None:
        return latest_dir
    if (base_dir / "latest_checkpoint.pth").is_file():
        return base_dir
    return timestamped_directory(base_dir)


def resolve_chunks_dir(configured_dir: str | Path) -> Path:
    base_dir = Path(configured_dir)
    if is_timestamped_directory(base_dir):
        return base_dir

    latest_dir = latest_timestamped_directory(
        base_dir,
        required_file=CHUNK_METADATA_FILENAME,
    )
    if latest_dir is not None:
        return latest_dir
    return base_dir


def save_experiment_config_snapshots(
    training_config_path: str | Path,
    chunks_dir: str | Path,
    checkpoint_dir: str | Path,
) -> tuple[Path, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    training_config_path = Path(training_config_path).expanduser().resolve()

    training_snapshot = checkpoint_dir / TRAINING_CONFIG_FILENAME
    if training_config_path != training_snapshot.resolve():
        shutil.copy2(training_config_path, training_snapshot)
    print(f"Training config saved to {training_snapshot}")

    chunks_dir = Path(chunks_dir)
    generation_source = chunks_dir / GENERATION_CONFIG_FILENAME
    if not generation_source.is_file():
        raise FileNotFoundError(
            "Dataset directory must contain its generation config: "
            f"{generation_source}. Regenerate the dataset with the current generator."
        )
    generation_snapshot = checkpoint_dir / GENERATION_CONFIG_FILENAME
    if generation_source.resolve() != generation_snapshot.resolve():
        shutil.copy2(generation_source, generation_snapshot)
    print(f"Generation config saved to {generation_snapshot}")

    return training_snapshot, generation_snapshot


def run_training(
    config_path: str | Path,
    after_epoch: EpochCallback | None = None,
    checkpoint_every: int | None = 5,
    banner: str = "Starting training...",
    completion_title: str = "Training completed!",
    checkpoint_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    args, _ = load_training_config(config_path)
    task = get_training_task(args.task)
    print("START!")

    checkpoint_dir = (
        Path(checkpoint_dir_override)
        if checkpoint_dir_override is not None
        else resolve_checkpoint_dir(args.checkpoint_dir, resume=args.resume)
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    args.chunks_dir = str(resolve_chunks_dir(args.chunks_dir))
    training_config_snapshot, generation_config_snapshot = (
        save_experiment_config_snapshots(
            config_path,
            args.chunks_dir,
            checkpoint_dir,
        )
    )

    dataset, dataset_config = load_dataset_from_config(args)
    config_data = effective_training_config_data(args, dataset_config)
    print(f"Dataset ready! Total samples: {len(dataset)}")

    log_path = checkpoint_dir / "training_log.tsv"
    validate_and_log_alphabet(
        dataset, dataset_config.alphabet, dataset_config.max_text_length, checkpoint_dir
    )

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split")

    split_generator = torch.Generator().manual_seed(dataset_config.seed or 0)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    alphabet = dataset_config.alphabet
    num_classes = task.num_outputs(alphabet)
    print("Alphabet: ", alphabet)
    print("Alphabet length: ", len(alphabet))
    print("Training task: ", task.name)
    print("Output classes: ", num_classes)
    print("Architecture: ", args.architecture)
    if args.architecture_params:
        print("Architecture params: ", args.architecture_params)
    for line in task.summary_lines(args, dataset_config):
        print(line)

    train_loader = make_data_loader(
        dataset,
        train_dataset,
        args,
        shuffle=True,
        seed=dataset_config.seed or 0,
        batch_count=args.batch_count,
    )
    val_loader = make_data_loader(
        dataset,
        val_dataset,
        args,
        shuffle=False,
        seed=(dataset_config.seed or 0) + 100_000,
    )

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    if train_batches == 0 or val_batches == 0:
        raise ValueError("Batch configuration leaves train or validation loader empty")

    print("\nData loaders:")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Drop last:       {args.drop_last}")
    print(f"  Num workers:     {args.num_workers}")
    if args.batch_count is not None:
        print(f"  Batch count:     {args.batch_count} sampled train batches/epoch")
    if isinstance(dataset, ChunkedLineDataset):
        print(f"  Chunk batching:  {args.chunk_aware_batches}")
        print(f"  Chunk cache:     {args.chunk_cache_size} files/worker")
    if args.num_workers > 0:
        print(f"  Prefetch factor: {args.prefetch_factor}")
        print(f"  Persistent:      {args.persistent_workers}")
    print(f"  Train batches:   {train_batches}")
    print(f"  Val batches:     {val_batches}")
    if args.max_train_batches is not None:
        print(
            f"  Train limit:     {min(args.max_train_batches, train_batches)} batches/epoch"
        )
    if args.max_val_batches is not None:
        print(
            f"  Val limit:       {min(args.max_val_batches, val_batches)} batches/epoch"
        )

    train_preview_saver = None
    val_preview_saver = None
    if args.preview_samples > 0:
        train_preview_saver = InputPreviewSaver(
            Path(args.preview_dir) / "train",
            args.preview_samples,
        )
        val_preview_saver = InputPreviewSaver(
            Path(args.preview_dir) / "val",
            args.preview_samples,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device ", device)
    train_augmenter = (
        GpuTextAugmenter(dataset_config) if args.gpu_augmentations else None
    )
    val_augmenter = GpuTextAugmenter(dataset_config) if args.gpu_augment_val else None
    print("GPU augmentations: ", "train" if train_augmenter is not None else "off")
    if val_augmenter is not None:
        print("GPU validation augmentations: on")

    model = create_model(
        args.architecture,
        in_channels=dataset_config.channels,
        num_classes=num_classes,
        **args.architecture_params,
    ).to(device)
    task.validate_model(model, args, dataset_config)

    optimizer = create_optimizer(model, args)
    print_optimizer_summary(args)
    scheduler = create_scheduler(optimizer, args)
    print("LR scheduler: ", args.scheduler)
    if args.scheduler == "reduce_on_plateau":
        print(
            f"  factor={args.scheduler_factor} patience={args.scheduler_patience} "
            f"min_lr={args.scheduler_min_lr:g}"
        )

    train_losses = []
    val_losses = []
    start_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")

    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
    if args.resume and latest_checkpoint.exists():
        print("Found latest checkpoint, loading...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        checkpoint_model_config = validate_checkpoint_contract(checkpoint)
        if checkpoint_model_config.task != task.name:
            raise ValueError(
                "Resume checkpoint task mismatch: "
                f"checkpoint={checkpoint_model_config.task}, config={task.name}"
            )
        if checkpoint["alphabet"] != alphabet:
            raise ValueError("Resume checkpoint alphabet differs from dataset alphabet")
        checkpoint_architecture = normalize_architecture_name(
            checkpoint_model_config.architecture
        )
        if checkpoint_architecture != args.architecture:
            raise ValueError(
                "Resume checkpoint architecture mismatch: "
                f"checkpoint={checkpoint_architecture}, config={args.architecture}"
            )
        if checkpoint_model_config.architecture_params != args.architecture_params:
            raise ValueError(
                "Resume checkpoint architecture_params differ from training config"
            )
        if checkpoint_model_config.in_channels != dataset_config.channels:
            raise ValueError(
                "Resume checkpoint input channels differ from dataset channels"
            )
        if checkpoint_model_config.num_classes != num_classes:
            raise ValueError(
                "Resume checkpoint output classes differ from the selected task"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        best_val_loss = min(val_losses) if val_losses else float("inf")
        best_train_loss = min(train_losses) if train_losses else float("inf")
        print(f"Resuming from epoch {start_epoch}")

    print("\n" + "=" * 60)
    print(banner)
    print("=" * 60 + "\n")

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_started_at = time.perf_counter()
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            train_stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.max_train_batches,
                train_preview_saver,
                args.log_every,
                train_augmenter,
                task=task,
                task_config=args,
                dataset_config=dataset_config,
            )
            train_loss = train_stats["loss"]
            train_losses.append(train_loss)

            val_stats = validate(
                model,
                val_loader,
                device,
                args.max_val_batches,
                val_preview_saver,
                args.log_every,
                val_augmenter,
                task=task,
                task_config=args,
                dataset_config=dataset_config,
            )
            val_loss = val_stats["loss"]
            val_losses.append(val_loss)

            epoch_seconds = time.perf_counter() - epoch_started_at
            is_best_val = val_loss < best_val_loss
            is_best_train = train_loss < best_train_loss
            old_lr, lr = step_scheduler(scheduler, args, val_loss, optimizer)

            append_training_log(
                log_path,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_batches": train_stats["batches"],
                    "val_batches": val_stats["batches"],
                    "train_samples": train_stats["samples"],
                    "val_samples": val_stats["samples"],
                    "lr": lr,
                    "epoch_seconds": epoch_seconds,
                    "is_best": is_best_val,
                },
            )

            print(
                f"  train loss={train_loss:.6f} "
                f"({train_stats['batches']} batches, {train_stats['samples']} samples, {train_stats['seconds']:.1f}s)"
            )
            print(
                f"  val   loss={val_loss:.6f} "
                f"({val_stats['batches']} batches, {val_stats['samples']} samples, {val_stats['seconds']:.1f}s)"
            )
            print(
                f"  diff={abs(train_loss - val_loss):.6f} lr={lr:.3g} epoch_time={epoch_seconds:.1f}s"
            )
            if lr != old_lr:
                print(f"  scheduler changed lr: {old_lr:.3g} -> {lr:.3g}")

            if train_loss < val_loss * 0.7:
                print("  warning: possible overfitting")

            checkpoint_path: Path | None = None
            if checkpoint_every is not None and epoch % checkpoint_every == 0:
                checkpoint_path = Path(
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        train_loss,
                        val_loss,
                        alphabet,
                        config_data,
                        train_losses,
                        val_losses,
                        checkpoint_dir,
                        scheduler=scheduler,
                    )
                )

            if is_best_val:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_dir / "best_model.pth"
                if checkpoint_path is not None:
                    shutil.copy2(checkpoint_path, best_checkpoint_path)
                else:
                    save_named_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        epoch,
                        train_loss,
                        val_loss,
                        alphabet,
                        config_data,
                        train_losses,
                        val_losses,
                        scheduler=scheduler,
                    )
                print(f"  best model saved: {best_checkpoint_path}")

            if is_best_train:
                best_train_loss = train_loss
                best_train_checkpoint_path = checkpoint_dir / "best_train_model.pth"
                if checkpoint_path is not None:
                    shutil.copy2(checkpoint_path, best_train_checkpoint_path)
                else:
                    save_named_checkpoint(
                        best_train_checkpoint_path,
                        model,
                        optimizer,
                        epoch,
                        train_loss,
                        val_loss,
                        alphabet,
                        config_data,
                        train_losses,
                        val_losses,
                        scheduler=scheduler,
                    )

            if after_epoch is not None:
                if checkpoint_path is None:
                    checkpoint_path = Path(
                        save_checkpoint(
                            model,
                            optimizer,
                            epoch,
                            train_loss,
                            val_loss,
                            alphabet,
                            config_data,
                            train_losses,
                            val_losses,
                            checkpoint_dir,
                            scheduler=scheduler,
                        )
                    )
                after_epoch(
                    {
                        "epoch": epoch,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_dir": checkpoint_dir,
                        "training_log_path": log_path,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_stats": train_stats,
                        "val_stats": val_stats,
                        "lr": lr,
                        "epoch_seconds": epoch_seconds,
                        "is_best_val": is_best_val,
                        "is_best_train": is_best_train,
                    }
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("-" * 60)
    finally:
        if train_preview_saver is not None:
            train_preview_saver.close()
        if val_preview_saver is not None:
            val_preview_saver.close()

    print("\n" + "=" * 60)
    print(completion_title)
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Best training loss:   {best_train_loss:.8f}")
    print(f"Training log: {log_path}")
    print("=" * 60)

    return {
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "training_log_path": log_path,
        "checkpoint_dir": checkpoint_dir,
        "training_config_snapshot": training_config_snapshot,
        "generation_config_snapshot": generation_config_snapshot,
    }


if __name__ == "__main__":
    cli_args = parse_args()
    run_training(cli_args.config)
