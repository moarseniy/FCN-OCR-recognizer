from __future__ import annotations

import argparse
import csv
import json
import shlex
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fcn_ocr import BaselineDetector, InferenceConfig
from tool.evaluation import interpolate_polyline, polyline_x_bounds
from tool.markup import annotated_items, load_document, safe_image_path


def build_jobs(
    markup_path: Path,
    images_dir: Path | None,
    limit: int | None,
) -> tuple[Path, list[tuple[dict[str, Any], Path]]]:
    document = load_document(markup_path)
    root = (
        images_dir.expanduser().resolve()
        if images_dir is not None
        else Path(document["images_root"]).expanduser().resolve()
    )
    jobs: list[tuple[dict[str, Any], Path]] = []
    for item in annotated_items(document):
        baselines = item.get("baselines") or {}
        if len(baselines.get("top", [])) < 2 or len(baselines.get("bottom", [])) < 2:
            continue
        try:
            image_path = safe_image_path(root, str(item["image"]))
        except FileNotFoundError:
            continue
        jobs.append((item, image_path))
        if limit is not None and len(jobs) >= limit:
            break
    return root, jobs


def prediction_lines(
    detection: dict[str, Any],
    xs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    top_y = float(detection["topline_slope"]) * xs + float(detection["topline_intercept"])
    bottom_y = float(detection["bottom_slope"]) * xs + float(detection["bottom_intercept"])
    return top_y, bottom_y


def prediction_x_bounds(
    detection: dict[str, Any],
    image_width: int,
) -> tuple[float, float]:
    top_x = np.asarray(detection.get("topline_profile_x", []), dtype=np.float64)
    bottom_x = np.asarray(detection.get("profile_x", []), dtype=np.float64)
    if top_x.size and bottom_x.size:
        return max(float(top_x.min()), float(bottom_x.min())), min(
            float(top_x.max()),
            float(bottom_x.max()),
        )
    return 0.0, float(max(0, image_width - 1))


def evaluate_detector(
    detector: BaselineDetector,
    jobs: list[tuple[dict[str, Any], Path]],
    output_csv: Path | None,
    failure_penalty: float,
    verbose: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top_error_sum = 0.0
    bottom_error_sum = 0.0
    point_count = 0
    normalized_error_sum = 0.0
    normalized_point_count = 0
    successful = 0
    failed = 0
    coverage_sum = 0.0
    started_at = time.perf_counter()

    for item, image_path in jobs:
        row = {
            "image": item["image"],
            "status": "",
            "top_mae_px": "",
            "bottom_mae_px": "",
            "combined_mae_px": "",
            "normalized_mae": "",
            "coverage": 0.0,
            "error": "",
        }
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
            detection = detector.detect(image)
            row["status"] = str(detection.get("status", "unknown"))
            if not detection.get("ok"):
                failed += 1
                row["error"] = row["status"]
                rows.append(row)
                continue

            baselines = item["baselines"]
            gt_left = max(
                polyline_x_bounds(baselines["top"])[0],
                polyline_x_bounds(baselines["bottom"])[0],
            )
            gt_right = min(
                polyline_x_bounds(baselines["top"])[1],
                polyline_x_bounds(baselines["bottom"])[1],
            )
            pred_left, pred_right = prediction_x_bounds(detection, image.width)
            left = max(gt_left, pred_left, 0.0)
            right = min(gt_right, pred_right, float(image.width - 1))
            if right < left:
                raise ValueError("Predicted and annotated baselines do not overlap")

            xs = np.arange(int(np.ceil(left)), int(np.floor(right)) + 1, dtype=np.float64)
            if xs.size < 2:
                raise ValueError("Baseline overlap is too narrow")
            gt_top = interpolate_polyline(baselines["top"], xs)
            gt_bottom = interpolate_polyline(baselines["bottom"], xs)
            pred_top, pred_bottom = prediction_lines(detection, xs)
            valid = gt_bottom > gt_top
            if int(np.count_nonzero(valid)) < 2:
                raise ValueError("Annotated top/bottom baselines are reversed")

            top_errors = np.abs(pred_top[valid] - gt_top[valid])
            bottom_errors = np.abs(pred_bottom[valid] - gt_bottom[valid])
            line_heights = np.maximum(gt_bottom[valid] - gt_top[valid], 1.0)
            normalized = (top_errors + bottom_errors) * 0.5 / line_heights
            top_mae = float(np.mean(top_errors))
            bottom_mae = float(np.mean(bottom_errors))
            normalized_mae = float(np.mean(normalized))
            coverage = float(xs.size) / max(1.0, gt_right - gt_left + 1.0)

            row.update(
                {
                    "top_mae_px": top_mae,
                    "bottom_mae_px": bottom_mae,
                    "combined_mae_px": (top_mae + bottom_mae) * 0.5,
                    "normalized_mae": normalized_mae,
                    "coverage": coverage,
                }
            )
            successful += 1
            top_error_sum += float(np.sum(top_errors))
            bottom_error_sum += float(np.sum(bottom_errors))
            point_count += int(top_errors.size)
            normalized_error_sum += float(np.sum(normalized))
            normalized_point_count += int(normalized.size)
            coverage_sum += coverage
        except Exception as exc:
            failed += 1
            row["status"] = row["status"] or "error"
            row["error"] = repr(exc)
        rows.append(row)

    elapsed = time.perf_counter() - started_at
    combined_mae = (
        (top_error_sum + bottom_error_sum) / (2.0 * point_count)
        if point_count
        else 0.0
    )
    normalized_mae = (
        normalized_error_sum / normalized_point_count
        if normalized_point_count
        else 0.0
    )
    total = len(jobs)
    metrics = {
        "total_samples": total,
        "successful_samples": successful,
        "failed_samples": failed,
        "success_rate": successful / total if total else 0.0,
        "top_mae_px": top_error_sum / point_count if point_count else 0.0,
        "bottom_mae_px": bottom_error_sum / point_count if point_count else 0.0,
        "combined_mae_px": combined_mae,
        "normalized_mae": normalized_mae,
        "mean_coverage": coverage_sum / successful if successful else 0.0,
        "failure_penalized_normalized_mae": (
            (normalized_mae * successful + failure_penalty * failed) / total
            if total
            else float(failure_penalty)
        ),
        "threshold": detector.baseline_detector_threshold,
        "elapsed": elapsed,
    }
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["image"])
            writer.writeheader()
            writer.writerows(rows)
    if verbose:
        print_metrics(metrics, output_csv)
    return metrics


def print_metrics(metrics: dict[str, Any], output_csv: Path | None) -> None:
    print("=== Baseline evaluation ===")
    print(f"Total samples:              {metrics['total_samples']}")
    print(f"Successful samples:         {metrics['successful_samples']}")
    print(f"Failed samples:             {metrics['failed_samples']}")
    print(f"Success rate:               {metrics['success_rate']:.4f}")
    print(f"Top baseline MAE:           {metrics['top_mae_px']:.3f}px")
    print(f"Bottom baseline MAE:        {metrics['bottom_mae_px']:.3f}px")
    print(f"Combined MAE:               {metrics['combined_mae_px']:.3f}px")
    print(f"Normalized MAE:             {metrics['normalized_mae']:.5f}")
    print(f"Failure-penalized MAE:      {metrics['failure_penalized_normalized_mae']:.5f}")
    print(f"Mean X coverage:            {metrics['mean_coverage']:.4f}")
    print(f"Threshold:                  {metrics['threshold']:.5f}")
    print(f"Elapsed:                    {metrics['elapsed']:.2f}s")
    if output_csv is not None:
        print(f"CSV saved to:               {output_csv}")


def append_trial(path: Path, trial_number: int, metrics: dict[str, Any]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        if is_new:
            file.write(
                "trial\tthreshold\tsuccess_rate\tcombined_mae_px\t"
                "normalized_mae\tfailure_penalized_normalized_mae\n"
            )
        file.write(
            f"{trial_number}\t{metrics['threshold']:.8f}\t{metrics['success_rate']:.8f}\t"
            f"{metrics['combined_mae_px']:.8f}\t{metrics['normalized_mae']:.8f}\t"
            f"{metrics['failure_penalized_normalized_mae']:.8f}\n"
        )


def optimize(
    detector: BaselineDetector,
    jobs: list[tuple[dict[str, Any], Path]],
    output_csv: Path,
    trials: int,
    failure_penalty: float,
    threshold_min: float,
    threshold_max: float,
    trials_output: Path | None,
    study_name: str | None,
    storage: str | None,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna") from exc
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )

    def objective(trial) -> float:
        detector.baseline_detector_threshold = trial.suggest_float("threshold", threshold_min, threshold_max)
        metrics = evaluate_detector(
            detector,
            jobs,
            output_csv=None,
            failure_penalty=failure_penalty,
            verbose=False,
        )
        if trials_output is not None:
            append_trial(trials_output, trial.number, metrics)
        return float(metrics["failure_penalized_normalized_mae"])

    study.optimize(objective, n_trials=trials)
    detector.baseline_detector_threshold = float(study.best_params["threshold"])
    print(f"Best Optuna params: {json.dumps(study.best_params, sort_keys=True)}")
    print(f"Best failure-penalized normalized MAE: {study.best_value:.8f}")
    metrics = evaluate_detector(
        detector,
        jobs,
        output_csv=output_csv,
        failure_penalty=failure_penalty,
        verbose=True,
    )
    metrics["optuna_best_params"] = dict(study.best_params)
    return metrics


def print_inference_command(args: argparse.Namespace, metrics: dict[str, Any], image_path: Path | None) -> None:
    config = InferenceConfig.model_validate(
        {
            "device": args.device,
            "baseline": {
                "enabled": True,
                "detector_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "detector_threshold": metrics["threshold"],
            },
            "ocr": {
                "checkpoint": args.inference_ocr_checkpoint or "<OCR_CHECKPOINT>",
            },
            "decode": {"enabled": False},
        }
    )
    config_path = Path(args.out).expanduser().resolve().with_suffix(".inference.yaml")
    config.save(config_path)
    command = [
        "python",
        "inference.py",
        "--config",
        str(config_path),
        "--image",
        str(image_path) if image_path is not None else "<IMAGE_PATH>",
    ]
    print(f"Inference config saved to:  {config_path}")
    print("\n=== Inference command ===")
    if args.inference_ocr_checkpoint is None:
        print(f"Set ocr.checkpoint in {config_path} before running:")
    print(shlex.join(command))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate neural top/bottom baseline detection.")
    parser.add_argument("--json", required=True, help="Manual markup JSON created by tool.annotation_server.")
    parser.add_argument("--images", default=None, help="Override images directory stored in the markup JSON.")
    parser.add_argument("--checkpoint", required=True, help="Baseline detector checkpoint.")
    parser.add_argument("--out", default="output/baseline_metrics.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--failure-penalty", type=float, default=1.0)
    parser.add_argument("--inference-ocr-checkpoint", default=None)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-threshold-min", type=float, default=0.10)
    parser.add_argument("--optuna-threshold-max", type=float, default=0.90)
    parser.add_argument("--optuna-trials-out", default=None)
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument("--optuna-storage", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, jobs = build_jobs(
        Path(args.json),
        Path(args.images) if args.images else None,
        args.limit,
    )
    if not jobs:
        raise ValueError("No saved samples with both top and bottom baseline markup")
    detector = BaselineDetector(
        args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )
    detector.print_summary()
    if args.optuna_trials > 0:
        metrics = optimize(
            detector,
            jobs,
            output_csv=Path(args.out),
            trials=args.optuna_trials,
            failure_penalty=args.failure_penalty,
            threshold_min=args.optuna_threshold_min,
            threshold_max=args.optuna_threshold_max,
            trials_output=Path(args.optuna_trials_out) if args.optuna_trials_out else None,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
        )
    else:
        metrics = evaluate_detector(
            detector,
            jobs,
            output_csv=Path(args.out),
            failure_penalty=args.failure_penalty,
            verbose=True,
        )
    print_inference_command(args, metrics, jobs[0][1] if jobs else None)


if __name__ == "__main__":
    main()
