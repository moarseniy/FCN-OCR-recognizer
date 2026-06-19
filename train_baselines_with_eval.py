from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
import sys
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluate_baselines import build_jobs, evaluate_detector, optimize
from fcn_ocr import BaselineDetector
from fcn_ocr.evaluation_config import expand_optuna_ranges
from train import load_training_config, resolve_checkpoint_dir, run_training


MINIMIZE_METRICS = {
    "failure_penalized_normalized_mae",
    "normalized_mae",
    "combined_mae_px",
    "top_mae_px",
    "bottom_mae_px",
}
MAXIMIZE_METRICS = {"success_rate", "mean_coverage"}
SUPPORTED_BEST_METRICS = MINIMIZE_METRICS | MAXIMIZE_METRICS


class BaselineTrainEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_config: str
    markup_json: str
    images_dir: str | None = None
    output_dir: str | None = None
    device: str | None = None
    limit: int | None = Field(default=None, ge=1)
    evaluate_every: int = Field(default=1, ge=1)

    threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    failure_penalty: float = Field(default=1.0, ge=0.0)
    best_metric: str = "failure_penalized_normalized_mae"
    best_checkpoint_name: str = "best_manual_baselines_model.pth"

    optuna_trials: int = Field(default=0, ge=0)
    optuna_threshold_min: float = Field(default=0.10, gt=0.0, lt=1.0)
    optuna_threshold_max: float = Field(default=0.90, gt=0.0, lt=1.0)
    optuna_trials_output: bool = True
    optuna_study_name: str | None = None
    optuna_storage: str | None = None

    @field_validator("best_metric")
    @classmethod
    def best_metric_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        if value not in SUPPORTED_BEST_METRICS:
            raise ValueError(f"best_metric must be one of {sorted(SUPPORTED_BEST_METRICS)}")
        return value

    @field_validator("best_checkpoint_name")
    @classmethod
    def best_checkpoint_name_must_be_a_file(cls, value: str) -> str:
        if not value or Path(value).name != value:
            raise ValueError("best_checkpoint_name must be a plain file name")
        return value

    @model_validator(mode="after")
    def threshold_range_must_be_valid(self) -> "BaselineTrainEvalConfig":
        if self.optuna_threshold_max <= self.optuna_threshold_min:
            raise ValueError("optuna_threshold_max must be greater than optuna_threshold_min")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "BaselineTrainEvalConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        raw = expand_optuna_ranges(raw, valid_fields=set(cls.model_fields))
        config_dir = config_path.parent
        for key in ("train_config", "markup_json", "images_dir", "output_dir"):
            value = raw.get(key)
            if value:
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    raw[key] = str((config_dir / candidate).resolve())
        storage = raw.get("optuna_storage")
        if isinstance(storage, str) and storage.startswith("sqlite:///"):
            database = Path(storage.removeprefix("sqlite:///")).expanduser()
            if not database.is_absolute():
                raw["optuna_storage"] = f"sqlite:///{(config_dir / database).resolve()}"
        return cls.model_validate(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural baseline detector and evaluate manual baselines after each epoch."
    )
    parser.add_argument("--config", required=True, help="Path to baseline train/evaluation YAML config.")
    return parser.parse_args()


def metric_direction(metric: str) -> Literal["minimize", "maximize"]:
    return "minimize" if metric in MINIMIZE_METRICS else "maximize"


def is_better(value: float, best_value: float | None, direction: str) -> bool:
    if not math.isfinite(value):
        return False
    if best_value is None:
        return True
    if direction == "minimize":
        return value < best_value
    return value > best_value


def load_previous_best(summary_path: Path, metric: str, direction: str) -> tuple[float | None, int | None]:
    if not summary_path.exists():
        return None, None

    best_value: float | None = None
    best_epoch: int | None = None
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            raw_value = row.get(metric)
            if raw_value in {None, ""}:
                continue
            try:
                value = float(raw_value)
                epoch = int(row["epoch"])
            except (TypeError, ValueError, KeyError):
                continue
            if is_better(value, best_value, direction):
                best_value = value
                best_epoch = epoch
    return best_value, best_epoch


def append_summary(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "epoch",
        "checkpoint",
        "csv",
        "train_loss",
        "val_loss",
        "lr",
        "threshold",
        "total_samples",
        "successful_samples",
        "failed_samples",
        "success_rate",
        "top_mae_px",
        "bottom_mae_px",
        "combined_mae_px",
        "normalized_mae",
        "failure_penalized_normalized_mae",
        "mean_coverage",
        "elapsed",
        "is_best_manual",
        "best_metric",
        "optuna_trials",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def save_best_metadata(
    path: Path,
    metrics: dict[str, Any],
    best_metric: str,
    best_checkpoint_path: Path,
) -> None:
    payload = {
        "epoch": metrics["epoch"],
        "metric": best_metric,
        "value": metrics[best_metric],
        "threshold": metrics["threshold"],
        "source_checkpoint": metrics["checkpoint"],
        "best_checkpoint": str(best_checkpoint_path),
        "metrics": {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    cli_args = parse_args()
    config = BaselineTrainEvalConfig.load(cli_args.config)
    markup_path = Path(config.markup_json)
    if not markup_path.is_file():
        raise FileNotFoundError(f"Manual baseline markup not found: {markup_path}")
    if config.images_dir is not None and not Path(config.images_dir).is_dir():
        raise NotADirectoryError(f"Evaluation images directory not found: {config.images_dir}")

    train_config, _ = load_training_config(config.train_config)
    if train_config.loss_mode != "baseline_heatmap":
        raise ValueError(
            "Baseline train/evaluation requires loss_mode=baseline_heatmap; "
            f"got {train_config.loss_mode!r}"
        )

    checkpoint_dir = resolve_checkpoint_dir(
        train_config.checkpoint_dir,
        resume=train_config.resume,
    )
    eval_dir = Path(config.output_dir) if config.output_dir else checkpoint_dir / "evaluate_baselines"
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "eval_summary.tsv"
    best_checkpoint_path = checkpoint_dir / config.best_checkpoint_name
    best_metadata_path = eval_dir / "best_manual_baselines.json"

    _, jobs = build_jobs(
        markup_path,
        Path(config.images_dir) if config.images_dir else None,
        config.limit,
    )
    if not jobs:
        raise ValueError("No saved samples with both top and bottom baseline markup")

    direction = metric_direction(config.best_metric)
    best_value, best_epoch = load_previous_best(summary_path, config.best_metric, direction)
    if best_value is not None and not best_checkpoint_path.exists():
        print(
            f"Previous best metric was found at epoch {best_epoch}, but {best_checkpoint_path} is missing; "
            "the best checkpoint will be recreated from new evaluations."
        )
        best_value = None
        best_epoch = None

    print("START baseline training with manual evaluation!")
    print(f"Training config:       {config.train_config}")
    print(f"Manual markup:         {config.markup_json}")
    print(f"Evaluation samples:    {len(jobs)}")
    print(f"Evaluation output:     {eval_dir}")
    print(f"Best manual metric:    {config.best_metric} ({direction})")
    print(f"Best manual checkpoint: {best_checkpoint_path}")
    if best_value is not None:
        print(f"Previous best:         epoch {best_epoch}, value={best_value:.8f}")

    def after_epoch(context: dict[str, Any]) -> None:
        nonlocal best_value, best_epoch

        epoch_number = int(context["epoch"]) + 1
        if epoch_number % config.evaluate_every != 0:
            return

        checkpoint_path = Path(context["checkpoint_path"])
        output_csv = eval_dir / f"epoch_{epoch_number:04d}.csv"
        detector = BaselineDetector(
            checkpoint_path,
            device=config.device,
            threshold=config.threshold,
        )

        print(f"\nRunning manual baseline evaluation for epoch {epoch_number}...")
        if config.optuna_trials > 0:
            trial_output = (
                eval_dir / f"epoch_{epoch_number:04d}_optuna_trials.tsv"
                if config.optuna_trials_output
                else None
            )
            study_name = (
                f"{config.optuna_study_name}_epoch_{epoch_number:04d}"
                if config.optuna_study_name
                else None
            )
            metrics = optimize(
                detector,
                jobs,
                output_csv=output_csv,
                trials=config.optuna_trials,
                failure_penalty=config.failure_penalty,
                threshold_min=config.optuna_threshold_min,
                threshold_max=config.optuna_threshold_max,
                trials_output=trial_output,
                study_name=study_name,
                storage=config.optuna_storage,
            )
        else:
            metrics = evaluate_detector(
                detector,
                jobs,
                output_csv=output_csv,
                failure_penalty=config.failure_penalty,
                verbose=True,
            )

        metrics.update(
            {
                "epoch": epoch_number,
                "checkpoint": str(checkpoint_path),
                "csv": str(output_csv),
                "train_loss": float(context["train_loss"]),
                "val_loss": float(context["val_loss"]),
                "lr": float(context["lr"]),
                "best_metric": config.best_metric,
                "optuna_trials": config.optuna_trials,
            }
        )

        metric_value = float(metrics[config.best_metric])
        is_best_manual = is_better(metric_value, best_value, direction)
        metrics["is_best_manual"] = is_best_manual
        if is_best_manual:
            best_value = metric_value
            best_epoch = epoch_number
            shutil.copy2(checkpoint_path, best_checkpoint_path)
            save_best_metadata(
                best_metadata_path,
                metrics,
                config.best_metric,
                best_checkpoint_path,
            )
            print(
                f"Best manual baseline model updated: epoch={epoch_number}, "
                f"{config.best_metric}={metric_value:.8f}"
            )
            print(f"  checkpoint: {best_checkpoint_path}")

        append_summary(summary_path, metrics)
        print(f"Evaluation summary: {summary_path}")
        print(f"{metric_value:.12g}", file=sys.stderr, flush=True)

        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_training(
        config.train_config,
        after_epoch=after_epoch,
        checkpoint_every=1,
        banner="Starting baseline training with per-epoch manual evaluation...",
        completion_title="Baseline training with manual evaluation completed!",
        checkpoint_dir_override=checkpoint_dir,
    )

    print(f"Evaluation summary:      {summary_path}")
    print(f"Best manual checkpoint: {best_checkpoint_path}")
    if best_value is not None:
        print(f"Best manual result:     epoch {best_epoch}, {config.best_metric}={best_value:.8f}")


if __name__ == "__main__":
    main()
