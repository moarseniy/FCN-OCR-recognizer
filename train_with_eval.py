from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_ocr import evaluate, optimize_preprocess
from train import load_training_config, run_training


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
    parser.add_argument("--x-pad", type=float, default=0.0)
    parser.add_argument("--baseline-crop", action="store_true")
    parser.add_argument("--baseline-top-pad", type=float, default=0.12)
    parser.add_argument("--baseline-bottom-pad", type=float, default=0.18)
    parser.add_argument("--no-baseline-deskew", action="store_true")
    parser.add_argument("--baseline-max-angle", type=float, default=12.0)
    parser.add_argument("--no-baseline-strict-lines", action="store_true")
    parser.add_argument("--baseline-line-pad", type=float, default=0.08)
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
                "epoch\tcheckpoint\tcsv\tscale_x\ty_pad\tx_pad\tbaseline_crop\tbaseline_line_pad\tline_accuracy\t"
                "average_char_accuracy\tglobal_char_accuracy\taverage_levenshtein\t"
                "total_levenshtein\trecognized_samples\ttotal_samples\tspeed\t"
                "optuna_trials\toptuna_metric\n"
            )
        file.write(
            f"{row['epoch']}\t{row['checkpoint']}\t{row['csv']}\t"
            f"{row['scale_x']:.8f}\t{row['y_pad']:.8f}\t{row.get('x_pad', 0.0):.8f}\t"
            f"{row.get('baseline_crop', False)}\t{row.get('baseline_line_pad', 0.0):.8f}\t"
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
            x_pad=cli_args.x_pad,
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
            baseline_strict_lines=not cli_args.no_baseline_strict_lines,
            baseline_line_pad=cli_args.baseline_line_pad,
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
            x_pad=cli_args.x_pad,
            batch_size=cli_args.eval_batch_size,
            limit=cli_args.eval_limit,
            log_every=cli_args.eval_log_every,
            baseline_crop=cli_args.baseline_crop,
            baseline_top_pad=cli_args.baseline_top_pad,
            baseline_bottom_pad=cli_args.baseline_bottom_pad,
            baseline_deskew=not cli_args.no_baseline_deskew,
            baseline_max_angle=cli_args.baseline_max_angle,
            baseline_strict_lines=not cli_args.no_baseline_strict_lines,
            baseline_line_pad=cli_args.baseline_line_pad,
        )

    metrics["csv"] = str(output_csv)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["epoch"] = epoch_number
    return metrics


def main() -> None:
    cli_args = parse_args()
    train_config, _ = load_training_config(cli_args.train_config)
    checkpoint_dir = Path(train_config.checkpoint_dir)
    eval_dir = Path(cli_args.eval_out_dir) if cli_args.eval_out_dir else checkpoint_dir / "evaluate_ocr"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_summary_path = eval_dir / "eval_summary.tsv"

    print("START train_with_eval!")
    print(f"Evaluation output: {eval_dir}")

    def after_epoch(context: dict) -> None:
        print("\nRunning OCR evaluation for this epoch...")
        eval_metrics = evaluate_epoch(
            cli_args,
            Path(context["checkpoint_path"]),
            int(context["epoch"]),
            eval_dir,
        )
        append_eval_summary(eval_summary_path, eval_metrics)
        print(f"Evaluation summary: {eval_summary_path}")

    result = run_training(
        cli_args.train_config,
        after_epoch=after_epoch,
        checkpoint_every=1,
        banner="Starting training with per-epoch OCR evaluation...",
        completion_title="Training with evaluation completed!",
    )
    print(f"Evaluation summary: {eval_summary_path}")
    result["eval_summary_path"] = eval_summary_path


if __name__ == "__main__":
    main()
