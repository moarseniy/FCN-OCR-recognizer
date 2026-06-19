from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from evaluate_ocr import (
    parse_args as parse_evaluation_args,
    resolve_inference_args,
    run_evaluation,
)
from train import load_training_config, resolve_checkpoint_dir, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCR and run evaluate_ocr after every epoch.")
    parser.add_argument("--train-config", required=True, help="Path to training YAML config.")
    parser.add_argument(
        "--evaluation-config",
        required=True,
        help="Path to the same evaluation YAML accepted by evaluate_ocr.py.",
    )
    parser.add_argument(
        "--eval-out-dir",
        default=None,
        help="Directory for per-epoch evaluation CSV/TSV files. Defaults to checkpoint_dir/evaluate_ocr.",
    )
    return parser.parse_args()


def append_eval_summary(log_path: Path, row: dict) -> None:
    is_new_file = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        if is_new_file:
            file.write(
                "epoch\tcheckpoint\tcsv\tscale_x\ty_pad\tx_pad\tbaseline_crop\tbaseline_line_pad\tbaseline_line_pad_px\tline_accuracy\t"
                "average_char_accuracy\tglobal_char_accuracy\taverage_levenshtein\t"
                "total_levenshtein\trecognized_samples\ttotal_samples\tspeed\t"
                "optuna_trials\toptuna_metric\n"
            )
        file.write(
            f"{row['epoch']}\t{row['checkpoint']}\t{row['csv']}\t"
            f"{row['scale_x']:.8f}\t{row['y_pad']:.8f}\t{row.get('x_pad', 0.0):.8f}\t"
            f"{row.get('baseline_crop', False)}\t{row.get('baseline_line_pad', 0.0):.8f}\t"
            f"{row.get('baseline_line_pad_px', 0.0):.8f}\t"
            f"{row['line_accuracy']:.8f}\t{row['average_char_accuracy']:.8f}\t"
            f"{row['global_char_accuracy']:.8f}\t{row['average_levenshtein']:.8f}\t"
            f"{row['total_levenshtein']}\t{row['recognized_samples']}\t"
            f"{row['total_samples']}\t{row['speed']:.6f}\t"
            f"{row.get('optuna_trials', 0)}\t{row.get('optuna_metric', '')}\n"
        )


def evaluate_epoch(
    evaluation_args: argparse.Namespace,
    checkpoint_path: Path,
    epoch: int,
    eval_dir: Path,
) -> dict:
    epoch_number = epoch + 1
    output_csv = eval_dir / f"epoch_{epoch_number:04d}.csv"
    trials_output = eval_dir / f"epoch_{epoch_number:04d}_optuna_trials.tsv"

    metrics = run_evaluation(
        evaluation_args,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        trials_output=trials_output,
        print_inference_command=False,
    )

    metrics["csv"] = str(output_csv)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["epoch"] = epoch_number
    metrics["optuna_trials"] = evaluation_args.optuna_trials
    metrics["optuna_metric"] = evaluation_args.optuna_metric
    return metrics


def main() -> None:
    cli_args = parse_args()
    evaluation_args = resolve_inference_args(
        parse_evaluation_args(["--config", cli_args.evaluation_config])
    )
    train_config, _ = load_training_config(cli_args.train_config)
    checkpoint_dir = resolve_checkpoint_dir(
        train_config.checkpoint_dir,
        resume=train_config.resume,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_config_path = Path(cli_args.evaluation_config).expanduser().resolve()
    evaluation_snapshot = checkpoint_dir / "evaluation_config.yaml"
    if evaluation_config_path != evaluation_snapshot.resolve():
        shutil.copy2(evaluation_config_path, evaluation_snapshot)
    eval_dir = Path(cli_args.eval_out_dir) if cli_args.eval_out_dir else checkpoint_dir / "evaluate_ocr"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_summary_path = eval_dir / "eval_summary.tsv"

    print("START train_with_eval!")
    print(f"Evaluation config: {evaluation_config_path}")
    print(f"Evaluation output: {eval_dir}")

    def after_epoch(context: dict) -> None:
        print("\nRunning OCR evaluation for this epoch...")
        eval_metrics = evaluate_epoch(
            evaluation_args,
            Path(context["checkpoint_path"]),
            int(context["epoch"]),
            eval_dir,
        )
        append_eval_summary(eval_summary_path, eval_metrics)
        print(f"Evaluation summary: {eval_summary_path}")
        print(
            f"{float(eval_metrics[evaluation_args.optuna_metric]):.12g}",
            file=sys.stderr,
            flush=True,
        )

    result = run_training(
        cli_args.train_config,
        after_epoch=after_epoch,
        checkpoint_every=1,
        banner="Starting training with per-epoch OCR evaluation...",
        completion_title="Training with evaluation completed!",
        checkpoint_dir_override=checkpoint_dir,
    )
    print(f"Evaluation summary: {eval_summary_path}")
    result["eval_summary_path"] = eval_summary_path


if __name__ == "__main__":
    main()
