from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import random_split

from evaluate_ocr import evaluate, optimize_preprocess
from model import FullyConvTextRecognizer
from synth_generators.line_generator.chunk_dataset import ChunkedLineDataset
from synth_generators.line_generator.gpu_augmentations import GpuTextAugmenter
from train import (
    InputPreviewSaver,
    append_training_log,
    create_scheduler,
    effective_training_config_data,
    load_dataset_from_config,
    load_training_config,
    make_data_loader,
    save_checkpoint,
    step_scheduler,
    train_one_epoch,
    validate,
    validate_and_log_alphabet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCR and run evaluate_ocr after every epoch.")
    parser.add_argument("--train-config", required=True, help="Path to training YAML config.")
    parser.add_argument("--eval-json", required=True, help="Path to Label Studio export JSON.")
    parser.add_argument("--eval-images", required=True, help="Folder with evaluation images.")
    parser.add_argument(
        "--eval-out-dir",
        default=None,
        help="Directory for per-epoch evaluation CSV/TSV files. Defaults to checkpoint_dir/evaluate_ocr.",
    )
    parser.add_argument("--eval-device", default=None, help="Evaluation device: cuda, cpu, or empty for auto.")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-log-every", type=int, default=0)
    parser.add_argument("--scale-x", type=float, default=0.0)
    parser.add_argument("--y-pad", type=float, default=0.0)
    parser.add_argument("--baseline-crop", action="store_true")
    parser.add_argument("--baseline-top-pad", type=float, default=0.12)
    parser.add_argument("--baseline-bottom-pad", type=float, default=0.18)
    parser.add_argument("--no-baseline-deskew", action="store_true")
    parser.add_argument("--baseline-max-angle", type=float, default=12.0)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-scale-x-min", type=float, default=-0.25)
    parser.add_argument("--optuna-scale-x-max", type=float, default=0.25)
    parser.add_argument("--optuna-y-pad-min", type=float, default=-0.25)
    parser.add_argument("--optuna-y-pad-max", type=float, default=0.25)
    parser.add_argument(
        "--optuna-metric",
        default="global_char_accuracy",
        choices=[
            "line_accuracy",
            "average_char_accuracy",
            "global_char_accuracy",
            "average_levenshtein",
            "total_levenshtein",
        ],
    )
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument("--optuna-storage", default=None)
    return parser.parse_args()


def append_eval_summary(log_path: Path, row: dict) -> None:
    is_new_file = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        if is_new_file:
            file.write(
                "epoch\tcheckpoint\tcsv\tscale_x\ty_pad\tbaseline_crop\tline_accuracy\t"
                "average_char_accuracy\tglobal_char_accuracy\taverage_levenshtein\t"
                "total_levenshtein\trecognized_samples\ttotal_samples\tspeed\t"
                "optuna_trials\toptuna_metric\n"
            )
        file.write(
            f"{row['epoch']}\t{row['checkpoint']}\t{row['csv']}\t"
            f"{row['scale_x']:.8f}\t{row['y_pad']:.8f}\t"
            f"{row.get('baseline_crop', False)}\t"
            f"{row['line_accuracy']:.8f}\t{row['average_char_accuracy']:.8f}\t"
            f"{row['global_char_accuracy']:.8f}\t{row['average_levenshtein']:.8f}\t"
            f"{row['total_levenshtein']}\t{row['recognized_samples']}\t"
            f"{row['total_samples']}\t{row['speed']:.6f}\t"
            f"{row.get('optuna_trials', 0)}\t{row.get('optuna_metric', '')}\n"
        )


def evaluate_epoch(cli_args: argparse.Namespace, checkpoint_path: Path, epoch: int, eval_dir: Path) -> dict:
    epoch_number = epoch + 1
    output_csv = eval_dir / f"epoch_{epoch_number:04d}.csv"
    trials_output = eval_dir / f"epoch_{epoch_number:04d}_optuna_trials.tsv"

    if cli_args.optuna_trials > 0:
        metrics = optimize_preprocess(
            json_path=Path(cli_args.eval_json),
            images_dir=Path(cli_args.eval_images),
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
            device=cli_args.eval_device,
            batch_size=cli_args.eval_batch_size,
            limit=cli_args.eval_limit,
            trials=cli_args.optuna_trials,
            scale_x_min=cli_args.optuna_scale_x_min,
            scale_x_max=cli_args.optuna_scale_x_max,
            y_pad_min=cli_args.optuna_y_pad_min,
            y_pad_max=cli_args.optuna_y_pad_max,
            metric_name=cli_args.optuna_metric,
            log_every=cli_args.eval_log_every,
            trials_output=trials_output,
            study_name=cli_args.optuna_study_name,
            storage=cli_args.optuna_storage,
            baseline_crop=cli_args.baseline_crop,
            baseline_top_pad=cli_args.baseline_top_pad,
            baseline_bottom_pad=cli_args.baseline_bottom_pad,
            baseline_deskew=not cli_args.no_baseline_deskew,
            baseline_max_angle=cli_args.baseline_max_angle,
        )
    else:
        metrics = evaluate(
            json_path=Path(cli_args.eval_json),
            images_dir=Path(cli_args.eval_images),
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
            device=cli_args.eval_device,
            scale_x=cli_args.scale_x,
            y_pad=cli_args.y_pad,
            batch_size=cli_args.eval_batch_size,
            limit=cli_args.eval_limit,
            log_every=cli_args.eval_log_every,
            baseline_crop=cli_args.baseline_crop,
            baseline_top_pad=cli_args.baseline_top_pad,
            baseline_bottom_pad=cli_args.baseline_bottom_pad,
            baseline_deskew=not cli_args.no_baseline_deskew,
            baseline_max_angle=cli_args.baseline_max_angle,
        )

    metrics["csv"] = str(output_csv)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["epoch"] = epoch_number
    return metrics


def main() -> None:
    cli_args = parse_args()
    train_config, _ = load_training_config(cli_args.train_config)

    print("START train_with_eval!")
    dataset, dataset_config = load_dataset_from_config(train_config)
    config_data = effective_training_config_data(train_config, dataset_config)
    print(f"Dataset ready! Total samples: {len(dataset)}")

    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = checkpoint_dir / "training_log.tsv"
    eval_dir = Path(cli_args.eval_out_dir) if cli_args.eval_out_dir else checkpoint_dir / "evaluate_ocr"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_summary_path = eval_dir / "eval_summary.tsv"

    validate_and_log_alphabet(dataset, dataset_config.alphabet, dataset_config.max_text_length, checkpoint_dir)

    val_size = max(1, int(len(dataset) * train_config.val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split")

    split_generator = torch.Generator().manual_seed(dataset_config.seed or 0)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
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
        train_config,
        shuffle=True,
        seed=dataset_config.seed or 0,
        batch_count=train_config.batch_count,
    )
    val_loader = make_data_loader(
        dataset,
        val_dataset,
        train_config,
        shuffle=False,
        seed=(dataset_config.seed or 0) + 100_000,
    )

    print("\nData loaders:")
    print(f"  Batch size:      {train_config.batch_size}")
    print(f"  Drop last:       {train_config.drop_last}")
    print(f"  Num workers:     {train_config.num_workers}")
    if isinstance(dataset, ChunkedLineDataset):
        print(f"  Chunk batching:  {train_config.chunk_aware_batches}")
        print(f"  Chunk cache:     {train_config.chunk_cache_size} files/worker")
    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")
    if train_config.max_val_batches is not None:
        print(f"  Val limit:       {min(train_config.max_val_batches, len(val_loader))} batches/epoch")
    print(f"Evaluation output: {eval_dir}")

    train_preview_saver = None
    val_preview_saver = None
    if train_config.preview_samples > 0:
        train_preview_saver = InputPreviewSaver(Path(train_config.preview_dir) / "train", train_config.preview_samples, alphabet)
        val_preview_saver = InputPreviewSaver(Path(train_config.preview_dir) / "val", train_config.preview_samples, alphabet)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device ", device)
    train_augmenter = GpuTextAugmenter(dataset_config) if train_config.gpu_augmentations else None
    val_augmenter = GpuTextAugmenter(dataset_config) if train_config.gpu_augment_val else None
    print("GPU augmentations: ", "train" if train_augmenter is not None else "off")

    model = FullyConvTextRecognizer(
        in_channels=dataset_config.channels,
        num_classes=len(alphabet) + 1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    scheduler = create_scheduler(optimizer, train_config)
    print("LR scheduler: ", train_config.scheduler)
    if train_config.scheduler == "reduce_on_plateau":
        print(
            f"  factor={train_config.scheduler_factor} patience={train_config.scheduler_patience} "
            f"min_lr={train_config.scheduler_min_lr:g}"
        )

    train_losses = []
    val_losses = []
    start_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")

    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
    if train_config.resume and latest_checkpoint.exists():
        print("Found latest checkpoint, loading...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        best_val_loss = min(val_losses) if val_losses else float("inf")
        best_train_loss = min(train_losses) if train_losses else float("inf")
        print(f"Resuming from epoch {start_epoch}")

    print("\n" + "=" * 60)
    print("Starting training with per-epoch OCR evaluation...")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, train_config.epochs):
        epoch_started_at = time.perf_counter()
        print(f"\nEpoch {epoch + 1}/{train_config.epochs}")

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            blank_idx,
            train_config.max_train_batches,
            train_preview_saver,
            train_config.log_every,
            train_augmenter,
        )
        train_loss = train_stats["loss"]
        train_losses.append(train_loss)

        val_stats = validate(
            model,
            val_loader,
            device,
            blank_idx,
            train_config.max_val_batches,
            val_preview_saver,
            train_config.log_every,
            val_augmenter,
        )
        val_loss = val_stats["loss"]
        val_losses.append(val_loss)

        epoch_seconds = time.perf_counter() - epoch_started_at
        is_best_val = val_loss < best_val_loss
        is_best_train = train_loss < best_train_loss
        old_lr, lr = step_scheduler(scheduler, train_config, val_loss, optimizer)

        append_training_log(
            training_log_path,
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
                str(checkpoint_dir),
                scheduler=scheduler,
            )
        )

        if is_best_val:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / "best_model.pth"
            shutil.copy2(checkpoint_path, best_checkpoint_path)
            print(f"  best model saved: {best_checkpoint_path}")

        if is_best_train:
            best_train_loss = train_loss
            best_train_checkpoint_path = checkpoint_dir / "best_train_model.pth"
            shutil.copy2(checkpoint_path, best_train_checkpoint_path)

        print("\nRunning OCR evaluation for this epoch...")
        eval_metrics = evaluate_epoch(cli_args, checkpoint_path, epoch, eval_dir)
        append_eval_summary(eval_summary_path, eval_metrics)
        print(f"Evaluation summary: {eval_summary_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("-" * 60)

    if train_preview_saver is not None:
        train_preview_saver.close()
    if val_preview_saver is not None:
        val_preview_saver.close()

    print("\n" + "=" * 60)
    print("Training with evaluation completed!")
    print(f"Best validation loss: {best_val_loss:.8f}")
    print(f"Best training loss:   {best_train_loss:.8f}")
    print(f"Training log: {training_log_path}")
    print(f"Evaluation summary: {eval_summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
