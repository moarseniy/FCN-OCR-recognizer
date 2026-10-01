from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fcn_ocr.evaluation.images import RGBImageCache
from fcn_ocr.evaluation.vertical_segmentation_runner import (
    build_rows_and_jobs,
    evaluate_prepared,
)
from fcn_synth_generator.run_directories import latest_timestamped_directory
from fcn_training import load_training_config, resolve_checkpoint_dir
from fcn_training.runner import run_training


MAXIMIZE_METRICS = {
    "cut_f1",
    "cut_precision",
    "cut_recall",
    "length_accuracy",
}
MINIMIZE_METRICS = {
    "cut_mae_px",
    "average_abs_length_error",
    "total_abs_length_error",
    "normalized_length_error",
}
SUPPORTED_METRICS = MAXIMIZE_METRICS | MINIMIZE_METRICS


class VerticalInferenceParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cut_threshold: float | None = Field(default=0.5, gt=0.0, lt=1.0)
    cut_min_width: int | None = Field(default=1, ge=1)
    cut_max_width: int | None = Field(default=0, ge=0)
    cut_smooth_radius: int | None = Field(default=0, ge=0)
    scale_x: float = Field(default=0.0, gt=-0.95)
    y_pad: float = Field(default=0.0, gt=-0.95)
    x_pad: float = Field(default=0.03, ge=0.0)
    baseline_crop: bool = True
    baseline_line_pad: float = Field(default=0.08, ge=0.0)
    baseline_line_pad_px: float = Field(default=0.0, ge=0.0)
    baseline_deskew: bool = True
    baseline_max_angle: float = Field(default=12.0, gt=0.0)
    baseline_detector_checkpoint: str | None = None
    baseline_detector_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)


class VerticalTrainEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_config: str
    markup_json: str
    images_dir: str | None = None
    output_dir: str | None = None
    device: str | None = None
    limit: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=64, ge=1)
    log_every: int = Field(default=100, ge=0)
    evaluate_every: int = Field(default=1, ge=1)
    cut_tolerance_px: float = Field(default=3.0, ge=0.0)
    best_metric: str = "cut_f1"
    best_checkpoint_name: str = "best_manual_vertical_segmentation_model.pth"
    parameters: VerticalInferenceParameters = Field(
        default_factory=VerticalInferenceParameters
    )

    @field_validator("best_metric")
    @classmethod
    def best_metric_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        if value not in SUPPORTED_METRICS:
            raise ValueError(f"best_metric must be one of {sorted(SUPPORTED_METRICS)}")
        return value

    @field_validator("best_checkpoint_name")
    @classmethod
    def checkpoint_name_must_be_a_file(cls, value: str) -> str:
        if not value or Path(value).name != value:
            raise ValueError("best_checkpoint_name must be a plain file name")
        return value

    @model_validator(mode="after")
    def baseline_crop_requires_detector(self) -> "VerticalTrainEvalConfig":
        if self.parameters.baseline_crop and not self.parameters.baseline_detector_checkpoint:
            raise ValueError(
                "parameters.baseline_detector_checkpoint is required when baseline_crop is true"
            )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "VerticalTrainEvalConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as file:
            import yaml

            raw = yaml.safe_load(file) or {}
        if not isinstance(raw, dict):
            raise ValueError("vertical train/evaluation config must be a YAML mapping")

        config_dir = config_path.parent
        for key in ("train_config", "markup_json", "images_dir", "output_dir"):
            value = raw.get(key)
            if value:
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    raw[key] = str((config_dir / candidate).resolve())

        parameters = raw.get("parameters") or {}
        detector_path = parameters.get("baseline_detector_checkpoint")
        if detector_path:
            candidate = Path(detector_path).expanduser()
            if not candidate.is_absolute():
                candidate = config_dir / candidate
            candidate = candidate.resolve()
            if candidate.is_dir():
                run_dir = latest_timestamped_directory(candidate)
                if run_dir is None:
                    raise FileNotFoundError(
                        f"No timestamped baseline detector run found under {candidate}"
                    )
                candidate = run_dir / "best_manual_baseline_detection_model.pth"
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"Manual-best baseline detector checkpoint not found: {candidate}"
                )
            parameters["baseline_detector_checkpoint"] = str(candidate)
            raw["parameters"] = parameters

        return cls.model_validate(raw)


def metric_direction(metric: str) -> Literal["minimize", "maximize"]:
    return "maximize" if metric in MAXIMIZE_METRICS else "minimize"


def is_better(value: float, best_value: float | None, direction: str) -> bool:
    if not math.isfinite(value):
        return False
    if best_value is None:
        return True
    return value > best_value if direction == "maximize" else value < best_value


def append_summary(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "epoch",
        "checkpoint",
        "csv",
        "train_loss",
        "val_loss",
        "lr",
        "best_metric",
        "metric_value",
        "is_best_manual",
        "total_samples",
        "evaluated_samples",
        "manual_cut_samples",
        "expected_cuts",
        "predicted_cuts",
        "matched_cuts",
        "cut_precision",
        "cut_recall",
        "cut_f1",
        "cut_mae_px",
        "length_accuracy",
        "average_abs_length_error",
        "normalized_length_error",
        "cut_threshold",
        "cut_min_width",
        "cut_max_width",
        "cut_smooth_radius",
        "scale_x",
        "y_pad",
        "x_pad",
        "baseline_crop",
        "baseline_detector_checkpoint",
        "baseline_detector_threshold",
        "elapsed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
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
        "source_checkpoint": metrics["checkpoint"],
        "best_checkpoint": str(best_checkpoint_path),
        "metrics": {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train vertical segmentation and evaluate manual cut annotations "
            "after each epoch."
        )
    )
    parser.add_argument("--config", required=True, help="Train/evaluation YAML config.")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    config = VerticalTrainEvalConfig.load(cli_args.config)
    markup_path = Path(config.markup_json)
    if not markup_path.is_file():
        raise FileNotFoundError(f"Manual vertical segmentation markup not found: {markup_path}")
    if config.images_dir is not None and not Path(config.images_dir).is_dir():
        raise NotADirectoryError(f"Evaluation images directory not found: {config.images_dir}")

    training_config, _ = load_training_config(config.train_config)
    if training_config.task != "vertical_segmentation":
        raise ValueError(
            "Vertical train/evaluation requires task=vertical_segmentation; "
            f"got {training_config.task!r}"
        )

    if config.parameters.baseline_detector_checkpoint:
        detector_path = Path(config.parameters.baseline_detector_checkpoint)
        if not detector_path.is_file():
            raise FileNotFoundError(f"Baseline detector checkpoint not found: {detector_path}")

    base_rows, jobs = build_rows_and_jobs(
        markup_path,
        Path(config.images_dir) if config.images_dir else None,
        config.limit,
    )
    manual_cut_samples = sum(bool(row.get("gt_cuts")) for row in base_rows)
    if not jobs or manual_cut_samples == 0:
        raise ValueError(
            "No usable images with manual vertical cuts; annotate cuts and verify images_root"
        )

    checkpoint_dir = resolve_checkpoint_dir(
        training_config.checkpoint_dir,
        resume=training_config.resume,
    )
    eval_dir = (
        Path(config.output_dir)
        if config.output_dir
        else checkpoint_dir / "evaluate_vertical_segmentation"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "eval_summary.tsv"
    best_checkpoint_path = checkpoint_dir / config.best_checkpoint_name
    best_metadata_path = eval_dir / "best_manual_vertical_segmentation.json"
    direction = metric_direction(config.best_metric)
    best_value: float | None = None
    best_epoch: int | None = None

    config_path = Path(cli_args.config).expanduser().resolve()
    config_snapshot = checkpoint_dir / "evaluation_config.yaml"
    if config_path != config_snapshot.resolve():
        shutil.copy2(config_path, config_snapshot)

    image_cache = RGBImageCache(max_megabytes=512.0)
    print("START vertical segmentation training with manual evaluation!")
    print(f"Training config:        {config.train_config}")
    print(f"Manual markup:          {config.markup_json}")
    print(f"Manual cut samples:     {manual_cut_samples}")
    print(f"Evaluation output:      {eval_dir}")
    print(f"Best manual metric:     {config.best_metric} ({direction})")
    print(f"Best manual checkpoint: {best_checkpoint_path}")
    print(f"Inference parameters:   {config.parameters.model_dump()}")

    def after_epoch(context: dict[str, Any]) -> None:
        nonlocal best_value, best_epoch
        epoch_number = int(context["epoch"]) + 1
        if epoch_number % config.evaluate_every != 0:
            return

        checkpoint_path = Path(context["checkpoint_path"])
        output_csv = eval_dir / f"epoch_{epoch_number:04d}.csv"
        print(f"\nRunning manual vertical segmentation evaluation for epoch {epoch_number}...")
        metrics = evaluate_prepared(
            base_rows=base_rows,
            jobs=jobs,
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
            device=config.device,
            batch_size=config.batch_size,
            log_every=config.log_every,
            verbose=True,
            cut_tolerance_px=config.cut_tolerance_px,
            image_loader=image_cache.load,
            **config.parameters.model_dump(),
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
            }
        )

        metric_value = float(metrics[config.best_metric])
        is_best = is_better(metric_value, best_value, direction)
        metrics["is_best_manual"] = is_best
        metrics["metric_value"] = metric_value
        if is_best:
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
                f"Best manual vertical segmenter updated: epoch={epoch_number}, "
                f"{config.best_metric}={metric_value:.8f}"
            )
            print(f"  checkpoint: {best_checkpoint_path}")

        append_summary(summary_path, metrics)
        print(f"Evaluation summary: {summary_path}")
        print(f"{metric_value:.12g}", file=sys.stderr, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_training(
        config.train_config,
        after_epoch=after_epoch,
        checkpoint_every=1,
        banner="Starting vertical segmentation training with per-epoch manual evaluation...",
        completion_title="Vertical segmentation training with manual evaluation completed!",
        checkpoint_dir_override=checkpoint_dir,
    )
    print(f"Evaluation summary:      {summary_path}")
    print(f"Best manual checkpoint: {best_checkpoint_path}")
    if best_value is not None:
        print(
            f"Best manual result:      epoch {best_epoch}, "
            f"{config.best_metric}={best_value:.8f}"
        )


if __name__ == "__main__":
    main()
