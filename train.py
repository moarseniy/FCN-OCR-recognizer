
# train.py
from synth_generators.line_generator.chunk_dataset import ChunkedLineDataset, load_chunk_metadata
from synth_generators.line_generator.dataset import SUPPORTED_AUGMENTATIONS, SingleLineDatasetConfig, SingleLineDataset
from synth_generators.line_generator.gpu_augmentations import GpuTextAugmenter
import argparse
from collections import Counter
import math
import time
import yaml
from typing import Any
from torch.utils.data import DataLoader, Sampler, Subset, random_split

import torch
from model import FullyConvTextRecognizer
from loss import ctc_loss

from datetime import datetime
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


SUPPORTED_SCHEDULERS = ("none", "reduce_on_plateau", "cosine", "step")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alphabet: str | None = None
    space_char: str | None = None
    max_text_length: int | None = Field(default=None, ge=1)
    channels: int | None = Field(default=None, ge=1, le=3)
    image_height: int | None = Field(default=None, ge=16)
    image_width: int | None = Field(default=None, ge=32)
    background: int | None = Field(default=None, ge=0, le=255)

    chunks_dir: str | None = None
    generator_config: str | None = None

    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=128, ge=1)
    batch_count: int | None = Field(default=None, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
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
    def model_validate_with_paths(cls, data: Any, config_path: str | Path) -> "TrainingConfig":
        data = dict(data)
        config_dir = Path(config_path).resolve().parent
        for key in ("chunks_dir", "generator_config", "checkpoint_dir", "preview_dir"):
            value = data.get(key)
            if value:
                path = Path(value)
                if not path.is_absolute():
                    data[key] = str(config_dir / path)
        return cls.model_validate(data)

    @field_validator("alphabet")
    @classmethod
    def alphabet_must_be_unique(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value:
            raise ValueError("alphabet must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("alphabet must contain unique characters")
        return value

    @field_validator("scheduler")
    @classmethod
    def scheduler_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        if value not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SUPPORTED_SCHEDULERS}")
        return value

    @field_validator("augmentation_probabilities")
    @classmethod
    def augmentation_probabilities_must_be_valid(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - set(SUPPORTED_AUGMENTATIONS))
        if unknown:
            raise ValueError(f"unknown augmentations: {unknown}")
        for name, probability in value.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"probability for {name} must be between 0 and 1")
        return value

    @field_validator("augmentations")
    @classmethod
    def augmentations_must_be_known(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
        'val_loss': val_loss,
        'alphabet': alphabet,
        'config': config,
        'model_config': {
            'in_channels': config.get('channels', 3),
            'num_classes': len(alphabet) + 1,
            'blank_idx': len(alphabet),
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Сохраняем также последнюю модель
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved to {latest_path}")

    return checkpoint_path


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


def step_scheduler(scheduler, config: TrainingConfig, val_loss: float, optimizer) -> tuple[float, float]:
    old_lr = current_lr(optimizer)
    if scheduler is None:
        return old_lr, old_lr
    if config.scheduler == "reduce_on_plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()
    return old_lr, current_lr(optimizer)

def compute_loss(logits, targets, lengths, blank_idx):
    return ctc_loss(logits, targets, lengths, blank_idx)


def prepare_batch(imgs, targets, lengths, device):
    imgs = imgs.to(device, non_blocking=True)
    if imgs.dtype == torch.uint8:
        imgs = imgs.float().div_(255.0)
    else:
        imgs = imgs.float()
    targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
    lengths = lengths.to(device=device, dtype=torch.long, non_blocking=True)
    return imgs, targets, lengths


def validate(model, loader, device, blank_idx, max_batches=50, preview_saver=None, log_every=0, augmenter=None):
    """Валидация модели"""
    model.eval()
    total_loss = 0.0
    batches = 0
    samples = 0
    started_at = time.perf_counter()

    with torch.no_grad():
        total_batches = min(max_batches, len(loader)) if max_batches is not None else len(loader)
        for batch_idx, (imgs, targets, lengths) in enumerate(loader, start=1):
            if max_batches is not None and batches >= max_batches:
                break

            imgs, targets, lengths = prepare_batch(imgs, targets, lengths, device)
            if augmenter is not None:
                imgs = augmenter(imgs)

            if preview_saver is not None:
                preview_saver.save_batch(imgs, targets, lengths)

            logits = model(imgs)

            loss = compute_loss(logits, targets, lengths, blank_idx)
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
    blank_idx,
    max_batches=None,
    preview_saver=None,
    log_every=0,
    augmenter=None,
):
    model.train()
    total_loss = 0.0
    batches = 0
    samples = 0
    started_at = time.perf_counter()

    total_batches = min(max_batches, len(loader)) if max_batches is not None else len(loader)
    for batch_idx, (imgs, targets, lengths) in enumerate(loader, start=1):
        if max_batches is not None and batches >= max_batches:
            break

        imgs, targets, lengths = prepare_batch(imgs, targets, lengths, device)
        if augmenter is not None:
            imgs = augmenter(imgs)

        if preview_saver is not None:
            preview_saver.save_batch(imgs, targets, lengths)

        logits = model(imgs)

        loss = compute_loss(logits, targets, lengths, blank_idx)

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

def decode_target_for_preview(target, length, alphabet):
    return "".join(alphabet[idx] for idx in target[:length].tolist())

class InputPreviewSaver:
    def __init__(self, output_dir, count, alphabet):
        self.output_path = Path(output_dir)
        self.count = count
        self.alphabet = alphabet
        self.saved = 0
        self.labels_file = None

        if count > 0:
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.labels_file = (self.output_path / "labels.tsv").open("w")
            self.labels_file.write("file\ttext\tlength\n")

    def save_batch(self, images, targets, lengths):
        if self.count <= 0 or self.saved >= self.count:
            return

        for image, target, length in zip(images, targets, lengths):
            if self.saved >= self.count:
                return

            filename = f"{self.saved:04d}.png"
            text = decode_target_for_preview(
                target.long(),
                int(length),
                self.alphabet,
            )
            tensor_to_pil(image).save(self.output_path / filename)
            self.labels_file.write(f"{filename}\t{text}\t{int(length)}\n")
            self.labels_file.flush()
            self.saved += 1

    def close(self):
        if self.labels_file is not None:
            self.labels_file.close()
            self.labels_file = None
            print(f"Saved {self.saved} input previews to {self.output_path}")

def batch_count(sample_count, batch_size, drop_last):
    if drop_last:
        return sample_count // batch_size
    return math.ceil(sample_count / batch_size)

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
    def __init__(self, subset, base_dataset, batch_size, drop_last, shuffle, seed=0, batch_count=None):
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
                permutation = torch.randperm(len(positions), generator=generator).tolist()
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
            yield [positions[position_index] for position_index in sampled_position_indices]

    def _sample_index(self, subset_position):
        if isinstance(self.subset, Subset):
            return int(self.subset.indices[subset_position])
        return subset_position


def parse_args():
    parser = argparse.ArgumentParser(description="Train the FCN OCR recognizer on synthetic lines.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    return parser.parse_args()


def load_training_config(config_path: str | Path) -> tuple[TrainingConfig, dict]:
    with Path(config_path).open("r") as file:
        config_data = yaml.safe_load(file)
    return TrainingConfig.model_validate_with_paths(config_data, config_path), config_data


DATASET_CONFIG_OVERRIDE_FIELDS = (
    "alphabet",
    "space_char",
    "max_text_length",
    "channels",
    "image_height",
    "image_width",
    "background",
)


def dataset_config_from_training_config(
    config: TrainingConfig,
    base_data: dict[str, Any] | None = None,
) -> SingleLineDatasetConfig:
    data = dict(base_data or {})

    for field_name in DATASET_CONFIG_OVERRIDE_FIELDS:
        value = getattr(config, field_name)
        if field_name in config.model_fields_set and value is not None:
            data[field_name] = value

    data.update(
        {
            "seed": config.seed,
            "augmentation_probabilities": config.augmentation_probabilities,
            "augmentations": config.augmentations,
        }
    )

    dataset_config = SingleLineDatasetConfig.model_validate(data)
    if dataset_config.alphabet is None:
        dataset_config = dataset_config.model_copy(update={"alphabet": dataset_config.sample_alphabet})
    return dataset_config


def effective_training_config_data(config: TrainingConfig, dataset_config: SingleLineDatasetConfig) -> dict:
    data = config.model_dump()
    data.update(
        {
            "alphabet": dataset_config.alphabet,
            "sample_alphabet": dataset_config.sample_alphabet,
            "space_char": dataset_config.space_char,
            "max_text_length": dataset_config.max_text_length,
            "channels": dataset_config.channels,
            "image_height": dataset_config.image_height,
            "image_width": dataset_config.image_width,
            "background": dataset_config.background,
        }
    )
    return data


def load_dataset_from_config(config: TrainingConfig) -> tuple[torch.utils.data.Dataset, SingleLineDatasetConfig]:
    if config.chunks_dir:
        metadata = load_chunk_metadata(config.chunks_dir)
        dataset_config = dataset_config_from_training_config(config, metadata)
        dataset = ChunkedLineDataset(config.chunks_dir, cache_size=config.chunk_cache_size, config=dataset_config)
        print(f"Dataset source: chunks ({config.chunks_dir})")
        if metadata:
            print(f"Dataset metadata: {Path(config.chunks_dir) / 'metadata.yaml'}")
        else:
            print("Dataset metadata: not found; using training config/defaults")
        return dataset, dataset_config

    if not config.generator_config:
        raise ValueError("Training config must contain either chunks_dir or generator_config")

    with Path(config.generator_config).open("r") as file:
        generator_data = yaml.safe_load(file)
    generator_config = SingleLineDatasetConfig.model_validate_with_paths(generator_data, config.generator_config)
    generator_config = dataset_config_from_training_config(config, generator_config.model_dump())

    dataset = SingleLineDataset(generator_config)
    print(f"Dataset source: online generator ({config.generator_config})")
    return dataset, generator_config


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


def iter_dataset_texts(dataset):
    if hasattr(dataset, "iter_texts"):
        yield from dataset.iter_texts()
        return

    if isinstance(dataset, SingleLineDataset):
        for index in range(len(dataset)):
            yield dataset.generate_sample_from_index(index).text
        return

    raise TypeError("Dataset does not expose texts for alphabet validation")


def validate_and_log_alphabet(dataset, alphabet: str, max_text_length: int, checkpoint_dir: str | Path) -> None:
    counts = Counter()
    sample_count = 0
    max_observed_length = 0

    for text in iter_dataset_texts(dataset):
        sample_count += 1
        max_observed_length = max(max_observed_length, len(text))
        counts.update(text)

    alphabet_set = set(alphabet)
    data_chars = set(counts)
    missing_chars = sorted(data_chars - alphabet_set)
    unused_chars = [char for char in alphabet if char not in data_chars]

    stats_path = Path(checkpoint_dir) / "alphabet_stats.tsv"
    with stats_path.open("w") as file:
        file.write("char\tcount\tin_training_alphabet\n")
        for char in alphabet:
            file.write(f"{printable_char(char)}\t{counts.get(char, 0)}\t1\n")
        for char in missing_chars:
            file.write(f"{printable_char(char)}\t{counts[char]}\t0\n")

    print("\nAlphabet/data check:")
    print(f"  Samples scanned:        {sample_count}")
    print(f"  Unique chars in data:   {len(data_chars)}")
    print(f"  Max text length:        {max_observed_length}")
    print(f"  Stats file:             {stats_path}")
    print("  Per-char counts:")
    for char in alphabet:
        print(f"    {printable_char(char):>9}: {counts.get(char, 0)}")

    if unused_chars:
        printable = ", ".join(printable_char(char) for char in unused_chars)
        print(f"  Alphabet chars absent in data: {printable}")

    if max_observed_length > max_text_length:
        raise ValueError(
            f"Data contains text length {max_observed_length}, "
            f"but training max_text_length is {max_text_length}"
        )
    if missing_chars:
        printable = ", ".join(printable_char(char) for char in missing_chars)
        raise ValueError(f"Training alphabet is missing data chars: {printable}")


if __name__ == "__main__":
    cli_args = parse_args()
    args, _ = load_training_config(cli_args.config)
    print("START!")
    dataset, dataset_config = load_dataset_from_config(args)
    config_data = effective_training_config_data(args, dataset_config)
    print(f"Dataset ready! Total samples: {len(dataset)}")

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = Path(checkpoint_dir) / "training_log.tsv"
    validate_and_log_alphabet(dataset, dataset_config.alphabet, dataset_config.max_text_length, checkpoint_dir)

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

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
    blank_idx = len(alphabet)
    print("Alphabet: ", alphabet)
    print("Alphabet length: ", len(alphabet))
    print("Blank index: ", blank_idx)

    train_loader = make_data_loader(
        dataset,
        train_dataset,
        args,
        shuffle=True,
        seed=dataset_config.seed or 0,
        batch_count=args.batch_count,
    )
    val_loader = make_data_loader(dataset, val_dataset, args, shuffle=False, seed=(dataset_config.seed or 0) + 100_000)

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
        print(f"  Train limit:     {min(args.max_train_batches, train_batches)} batches/epoch")
    if args.max_val_batches is not None:
        print(f"  Val limit:       {min(args.max_val_batches, val_batches)} batches/epoch")

    train_preview_saver = None
    val_preview_saver = None
    if args.preview_samples > 0:
        train_preview_saver = InputPreviewSaver(
            Path(args.preview_dir) / "train",
            args.preview_samples,
            alphabet,
        )
        val_preview_saver = InputPreviewSaver(
            Path(args.preview_dir) / "val",
            args.preview_samples,
            alphabet,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device ", device)
    train_augmenter = GpuTextAugmenter(dataset_config) if args.gpu_augmentations else None
    val_augmenter = GpuTextAugmenter(dataset_config) if args.gpu_augment_val else None
    print("GPU augmentations: ", "train" if train_augmenter is not None else "off")
    if val_augmenter is not None:
        print("GPU validation augmentations: on")

    model = FullyConvTextRecognizer(
        in_channels=dataset_config.channels,
        num_classes=len(alphabet) + 1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = create_scheduler(optimizer, args)
    print("LR scheduler: ", args.scheduler)
    if args.scheduler == "reduce_on_plateau":
        print(
            f"  factor={args.scheduler_factor} patience={args.scheduler_patience} "
            f"min_lr={args.scheduler_min_lr:g}"
        )

    # Списки для хранения истории лоссов
    train_losses = []
    val_losses = []

    start_epoch = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')

    # Можно загрузить последний чекпоинт если нужно продолжить обучение
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if args.resume and os.path.exists(latest_checkpoint):
        print("Found latest checkpoint, loading...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint['epoch'] + 1

        # Загружаем историю лоссов если она есть
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            best_val_loss = min(val_losses) if val_losses else float('inf')
            best_train_loss = min(train_losses) if train_losses else float('inf')

        print(f"Resuming from epoch {start_epoch}")

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = time.perf_counter()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            blank_idx,
            args.max_train_batches,
            train_preview_saver,
            args.log_every,
            train_augmenter,
        )
        train_loss = train_stats["loss"]
        train_losses.append(train_loss)

        val_stats = validate(
            model,
            val_loader,
            device,
            blank_idx,
            args.max_val_batches,
            val_preview_saver,
            args.log_every,
            val_augmenter,
        )
        val_loss = val_stats["loss"]
        val_losses.append(val_loss)

        epoch_seconds = time.perf_counter() - epoch_started_at
        is_best_val = val_loss < best_val_loss
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
        print(f"  diff={abs(train_loss - val_loss):.6f} lr={lr:.3g} epoch_time={epoch_seconds:.1f}s")
        if lr != old_lr:
            print(f"  scheduler changed lr: {old_lr:.3g} -> {lr:.3g}")

        if train_loss < val_loss * 0.7:
            print("  warning: possible overfitting")

        # Сохраняем чекпоинт каждые 5 эпох
        if epoch % 5 == 0:
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

            # Сохраняем лучшую модель по валидационному лоссу
        if is_best_val:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'loss': train_loss,
                'val_loss': val_loss,
                'alphabet': alphabet,
                'config': config_data,
                'model_config': {
                    'in_channels': dataset_config.channels,
                    'num_classes': len(alphabet) + 1,
                    'blank_idx': blank_idx,
                },
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  best model saved: {best_checkpoint_path}")

        # Сохраняем лучшую модель по тренировочному лоссу (для сравнения)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_checkpoint_path = os.path.join(checkpoint_dir, 'best_train_model.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'loss': train_loss,
                'val_loss': val_loss,
                'alphabet': alphabet,
                'config': config_data,
                'model_config': {
                    'in_channels': dataset_config.channels,
                    'num_classes': len(alphabet) + 1,
                    'blank_idx': blank_idx,
                },
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, best_train_checkpoint_path)

        print("-" * 60)

    if train_preview_saver is not None:
        train_preview_saver.close()
    if val_preview_saver is not None:
        val_preview_saver.close()

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Best training loss:   {best_train_loss:.8f}")
    print(f"Training log: {log_path}")
    print("="*60)
