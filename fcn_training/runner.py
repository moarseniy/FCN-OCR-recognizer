from __future__ import annotations

from fcn_synth_generator.chunk_dataset import (
    ChunkedLineDataset,
)
from fcn_synth_generator.gpu_augmentations import GpuTextAugmenter
import shutil
import time
from typing import Any
from torch.utils.data import random_split

import torch
from fcn_architectures import (
    create_model,
    normalize_architecture_name,
)
from .tasks import get_training_task
from .config import load_training_config
from .checkpoints import save_checkpoint, save_named_checkpoint
from .optimization import (
    create_optimizer,
    create_scheduler,
    print_optimizer_summary,
    step_scheduler,
)
from .data import make_data_loader
from .dataset import (
    effective_training_config_data,
    load_dataset_from_config,
    resolve_chunks_dir,
    validate_and_log_alphabet,
)
from .engine import (
    InputPreviewSaver,
    append_training_log,
    train_one_epoch,
    validate,
)
from .experiment import (
    EpochCallback,
    resolve_checkpoint_dir,
    save_experiment_config_snapshots,
)
from fcn_checkpoint_contract import validate_checkpoint_contract

from pathlib import Path

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


__all__ = ["run_training"]
