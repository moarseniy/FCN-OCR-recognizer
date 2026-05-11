
# train.py
from synth_generators.line_generator.chunk_dataset import ChunkedLineDataset
from synth_generators.line_generator.dataset import SUPPORTED_AUGMENTATIONS, SingleLineDatasetConfig, SingleLineDataset
from synth_generators.line_generator.gpu_augmentations import GpuTextAugmenter
import argparse
import math
import time
import yaml
from torch.utils.data import DataLoader, Sampler, Subset, random_split

import torch
from model import FullyConvTextRecognizer, transform_back
from loss import ctc_loss, logreg_loss

from datetime import datetime
import os
from pathlib import Path

import numpy as np
from PIL import Image

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
):
    """Сохраняет чекпоинт модели"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pth')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': val_loss,
        'alphabet': alphabet,
        'config': config,
        'model_config': {
            'in_channels': config.get('channels', 3),
            'num_classes': len(alphabet) + (1 if config.get('target_mode', 'ctc') == 'ctc' else 0),
            'blank_idx': len(alphabet) if config.get('target_mode', 'ctc') == 'ctc' else None,
            'target_mode': config.get('target_mode', 'ctc'),
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

def compute_loss(logits, targets, lengths, blank_idx, target_mode):
    if target_mode == "ctc":
        return ctc_loss(logits, targets, lengths, blank_idx)
    return logreg_loss(logits, targets)


def prepare_batch(imgs, targets, lengths, device):
    imgs = imgs.to(device, non_blocking=True)
    if imgs.dtype == torch.uint8:
        imgs = imgs.float().div_(255.0)
    else:
        imgs = imgs.float()
    targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
    lengths = lengths.to(device=device, dtype=torch.long, non_blocking=True)
    return imgs, targets, lengths


def validate(model, loader, device, blank_idx, target_mode, max_batches=50, preview_saver=None, log_every=0, augmenter=None):
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
            if target_mode == "column":
                logits = transform_back(logits, imgs.shape[3])

            loss = compute_loss(logits, targets, lengths, blank_idx, target_mode)
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
    target_mode,
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
        if target_mode == "column":
            logits = transform_back(logits, imgs.shape[3])

        loss = compute_loss(logits, targets, lengths, blank_idx, target_mode)

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

def decode_target_for_preview(target, length, alphabet, target_mode, space_char):
    if target_mode == "ctc":
        return "".join(alphabet[idx] for idx in target[:length].tolist())

    chars = []
    previous_idx = None
    for idx in target.tolist():
        if idx != previous_idx:
            chars.append(alphabet[idx])
            previous_idx = idx
    return "".join(chars).strip(space_char)

class InputPreviewSaver:
    def __init__(self, output_dir, count, alphabet, target_mode, space_char):
        self.output_path = Path(output_dir)
        self.count = count
        self.alphabet = alphabet
        self.target_mode = target_mode
        self.space_char = space_char
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
                self.target_mode,
                self.space_char,
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


class ChunkBatchSampler(Sampler):
    def __init__(self, subset, base_dataset, batch_size, drop_last, shuffle, seed=0):
        self.subset = subset
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.groups = self._group_subset_positions_by_chunk()

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

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

    def _sample_index(self, subset_position):
        if isinstance(self.subset, Subset):
            return int(self.subset.indices[subset_position])
        return subset_position


def parse_args():
    parser = argparse.ArgumentParser(description="Train the FCN OCR recognizer on synthetic lines.")
    parser.add_argument(
        "--config",
        default="synth_generators/line_generator/example_config.yaml",
        help="Path to a SingleLineDataset YAML config.",
    )
    parser.add_argument(
        "--chunks-dir",
        default=None,
        help="Directory with pre-rendered uint8 torch chunks produced by materialize.py.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional train batches per epoch limit.")
    parser.add_argument("--max-val-batches", type=int, default=50, help="Validation batches per epoch limit.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--drop-last", action="store_true", help="Drop incomplete train/val batches.")
    parser.add_argument("--log-every", type=int, default=1, help="Print every N batch losses. 0 disables batch logs.")
    parser.add_argument("--chunk-cache-size", type=int, default=2, help="Loaded chunk files cached per DataLoader worker.")
    parser.add_argument(
        "--chunk-aware-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group batches by chunk file for offline datasets. Enabled by default.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when num_workers > 0.")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive between epochs when num_workers > 0.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest_checkpoint.pth.")
    parser.add_argument("--target-mode", choices=["ctc", "column"], default=None, help="Override config target_mode.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of samples used for validation.")
    parser.add_argument("--preview-samples", type=int, default=0, help="Save N actual input images seen by train and val loops.")
    parser.add_argument("--preview-dir", default="input_previews", help="Directory for saved input previews.")
    parser.add_argument(
        "--gpu-augmentations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply configured augmentations on the training device. Enabled by default.",
    )
    parser.add_argument(
        "--gpu-augment-val",
        action="store_true",
        help="Also apply GPU augmentations during validation.",
    )
    return parser.parse_args()


def dataset_render_config(dataset_config):
    config_data = dataset_config.model_dump()
    config_data["noise_std"] = 0.0
    config_data["blur_radius"] = 0.0
    config_data["max_rotation_degrees"] = 0.0
    config_data["augmentation_probabilities"] = {name: 0.0 for name in SUPPORTED_AUGMENTATIONS}
    return SingleLineDatasetConfig.model_validate(config_data)


def load_dataset_from_args(args):
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
    if args.target_mode:
        config_data["target_mode"] = args.target_mode
    dataset_config = SingleLineDatasetConfig.model_validate_with_paths(config_data, args.config)

    if args.chunks_dir:
        dataset = ChunkedLineDataset(args.chunks_dir, cache_size=args.chunk_cache_size)
        print(f"Dataset source: chunks ({args.chunks_dir})")
        print(f"Dataset config: {args.config}")
        return dataset, dataset_config, config_data

    render_config = dataset_render_config(dataset_config) if args.gpu_augmentations else dataset_config
    dataset = SingleLineDataset(render_config)
    print(f"Dataset source: online generator ({args.config})")
    return dataset, dataset_config, config_data


def make_data_loader(dataset, split_dataset, args, shuffle, seed):
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


if __name__ == "__main__":
    args = parse_args()
    print("START!")
    dataset, dataset_config, config_data = load_dataset_from_args(args)
    print(f"Dataset ready! Total samples: {len(dataset)}")

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
    target_mode = dataset_config.target_mode
    blank_idx = len(alphabet) if target_mode == "ctc" else None
    print("Alphabet: ", alphabet)
    print("Alphabet length: ", len(alphabet))
    print("Target mode: ", target_mode)
    if blank_idx is not None:
        print("Blank index: ", blank_idx)

    train_loader = make_data_loader(dataset, train_dataset, args, shuffle=True, seed=dataset_config.seed or 0)
    val_loader = make_data_loader(dataset, val_dataset, args, shuffle=False, seed=(dataset_config.seed or 0) + 100_000)

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    if train_batches == 0 or val_batches == 0:
        raise ValueError("Batch configuration leaves train or validation loader empty")

    print("\nData loaders:")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Drop last:       {args.drop_last}")
    print(f"  Num workers:     {args.num_workers}")
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
            target_mode,
            dataset_config.space_char,
        )
        val_preview_saver = InputPreviewSaver(
            Path(args.preview_dir) / "val",
            args.preview_samples,
            alphabet,
            target_mode,
            dataset_config.space_char,
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
        num_classes=len(alphabet) + (1 if target_mode == "ctc" else 0)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Списки для хранения истории лоссов
    train_losses = []
    val_losses = []

    # Создаем директорию для чекпоинтов и графиков
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = Path(checkpoint_dir) / "training_log.tsv"

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
            target_mode,
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
            target_mode,
            args.max_val_batches,
            val_preview_saver,
            args.log_every,
            val_augmenter,
        )
        val_loss = val_stats["loss"]
        val_losses.append(val_loss)

        epoch_seconds = time.perf_counter() - epoch_started_at
        is_best_val = val_loss < best_val_loss
        lr = optimizer.param_groups[0]["lr"]

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
            )

            # Сохраняем лучшую модель по валидационному лоссу
        if is_best_val:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
                'alphabet': alphabet,
                'config': config_data,
                'model_config': {
                    'in_channels': dataset_config.channels,
                    'num_classes': len(alphabet) + (1 if target_mode == "ctc" else 0),
                    'blank_idx': blank_idx,
                    'target_mode': target_mode,
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
                'loss': train_loss,
                'val_loss': val_loss,
                'alphabet': alphabet,
                'config': config_data,
                'model_config': {
                    'in_channels': dataset_config.channels,
                    'num_classes': len(alphabet) + (1 if target_mode == "ctc" else 0),
                    'blank_idx': blank_idx,
                    'target_mode': target_mode,
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
