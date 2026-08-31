from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from fcn_ocr import BaselineDetector
from fcn_ocr.evaluation import interpolate_polyline, polyline_x_bounds
from fcn_ocr.evaluation.config import (
    evaluation_parameter_range,
    parse_args_with_evaluation_config,
)
from fcn_ocr.evaluation.images import RGBImageCache
from fcn_ocr.evaluation.optuna import (
    create_study,
    file_contract,
    optimize_with_progress,
    validate_study_contract,
)
from fcn_ocr.evaluation.reporting import (
    append_tsv_row,
    save_and_print_inference_command,
    write_csv_rows,
)
from tools.annotation.markup import annotated_items, load_document, safe_image_path


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
    image_loader: Callable[[Path], Image.Image] | None = None,
    heatmap_cache: dict[
        Path,
        tuple[np.ndarray, float, float],
    ]
    | None = None,
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
            if image_loader is None:
                with Image.open(image_path) as image_file:
                    image = image_file.convert("RGB")
            else:
                image = image_loader(image_path)
            if heatmap_cache is None:
                detection = detector.detect(image)
            else:
                heatmap_data = heatmap_cache.get(image_path)
                if heatmap_data is None:
                    heatmap_data = detector.heatmaps(image)
                    heatmap_cache[image_path] = heatmap_data
                detection = detector.detect_from_heatmaps(image, heatmap_data)
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
        write_csv_rows(rows, output_csv)
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
    append_tsv_row(
        path,
        (
            "trial",
            "threshold",
            "success_rate",
            "combined_mae_px",
            "normalized_mae",
            "failure_penalized_normalized_mae",
        ),
        (
            trial_number,
            f"{metrics['threshold']:.8f}",
            f"{metrics['success_rate']:.8f}",
            f"{metrics['combined_mae_px']:.8f}",
            f"{metrics['normalized_mae']:.8f}",
            f"{metrics['failure_penalized_normalized_mae']:.8f}",
        ),
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
    progress: bool = False,
    optuna_seed: int = 0,
    image_cache_mb: float = 512.0,
    cache_neural_outputs: bool = True,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if not 0.0 < threshold_min <= threshold_max < 1.0:
        raise ValueError("threshold range must satisfy 0 < min <= max < 1")
    study = create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        seed=optuna_seed,
    )
    validate_study_contract(
        study,
        {
            "evaluator": "baseline_detection",
            "checkpoint": file_contract(detector.checkpoint_path),
            "samples": [
                {
                    "image": str(path.expanduser().resolve()),
                    "baselines": item.get("baselines"),
                }
                for item, path in jobs
            ],
            "failure_penalty": failure_penalty,
            "parameters": {"threshold": [threshold_min, threshold_max]},
        },
    )
    image_cache = RGBImageCache(image_cache_mb)
    heatmap_cache = {} if cache_neural_outputs else None

    if heatmap_cache is not None:
        print(f"Caching baseline heatmaps for {len(jobs)} images...")
        for _, image_path in jobs:
            image = image_cache.load(image_path)
            heatmap_cache[image_path] = detector.heatmaps(image)

    def objective(trial) -> float:
        detector.baseline_detector_threshold = trial.suggest_float("threshold", threshold_min, threshold_max)
        metrics = evaluate_detector(
            detector,
            jobs,
            output_csv=None,
            failure_penalty=failure_penalty,
            verbose=False,
            image_loader=image_cache.load,
            heatmap_cache=heatmap_cache,
        )
        if trials_output is not None:
            append_trial(trials_output, trial.number, metrics)
        return float(metrics["failure_penalized_normalized_mae"])

    optimize_with_progress(
        study,
        objective,
        n_trials=trials,
        metric_name="baseline MAE",
        enabled=progress,
    )
    detector.baseline_detector_threshold = float(study.best_params["threshold"])
    print(f"Best Optuna params: {json.dumps(study.best_params, sort_keys=True)}")
    print(f"Best failure-penalized normalized MAE: {study.best_value:.8f}")
    metrics = evaluate_detector(
        detector,
        jobs,
        output_csv=output_csv,
        failure_penalty=failure_penalty,
        verbose=True,
        image_loader=image_cache.load,
        heatmap_cache=heatmap_cache,
    )
    metrics["optuna_best_params"] = dict(study.best_params)
    metrics["optuna_image_cache"] = image_cache.stats()
    metrics["optuna_neural_output_cache"] = bool(heatmap_cache is not None)
    cache_stats = image_cache.stats()
    print(
        "Optuna runtime reuse: detector_loads=1, "
        f"neural_forward_passes={len(jobs) if heatmap_cache is not None else 'per trial'}, "
        f"image_cache_hits={cache_stats['hits']}, "
        f"misses={cache_stats['misses']}"
    )
    return metrics


def print_inference_command(args: argparse.Namespace, metrics: dict[str, Any], image_path: Path | None) -> None:
    config_data: dict[str, Any] = {
        "device": args.device,
        "baseline_detection": {
            "enabled": True,
            "detector_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "detector_threshold": metrics["threshold"],
        },
    }
    save_and_print_inference_command(config_data, args.out, image_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate neural top/bottom baseline detection.")
    parser.add_argument("--config", default=None, help="Evaluation YAML config.")
    parser.add_argument("--json", default=None, help="Manual markup JSON created by tools.annotation.server.")
    parser.add_argument("--images", default=None, help="Override images directory stored in the markup JSON.")
    parser.add_argument("--checkpoint", default=None, help="Baseline detector checkpoint.")
    parser.add_argument("--out", default="output/baseline_metrics.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--failure-penalty", type=float, default=1.0)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-trials-out", default=None)
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument("--optuna-storage", default=None)
    parser.add_argument("--optuna-seed", type=int, default=0)
    parser.add_argument("--optuna-image-cache-mb", type=float, default=512.0)
    parser.add_argument(
        "--optuna-cache-neural-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache baseline heatmaps so threshold trials do not rerun the FCN.",
    )
    parser.add_argument(
        "--optuna-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the interactive Optuna progress bar.",
    )
    return parse_args_with_evaluation_config(
        parser,
        path_fields=(
            "json",
            "images",
            "checkpoint",
            "out",
            "optuna_trials_out",
        ),
        required_fields=("json", "checkpoint"),
        parameter_ranges={"threshold": (None, None)},
        argv=argv,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
        threshold_min, threshold_max = evaluation_parameter_range(args, "threshold")
        if threshold_min is None or threshold_max is None:
            raise ValueError(
                "Optuna has no tunable baseline parameters; set "
                "parameters.threshold: [min, max] or use optuna_trials: 0"
            )
        metrics = optimize(
            detector,
            jobs,
            output_csv=Path(args.out),
            trials=args.optuna_trials,
            failure_penalty=args.failure_penalty,
            threshold_min=float(threshold_min),
            threshold_max=float(threshold_max),
            trials_output=Path(args.optuna_trials_out) if args.optuna_trials_out else None,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            progress=args.optuna_progress,
            optuna_seed=args.optuna_seed,
            image_cache_mb=args.optuna_image_cache_mb,
            cache_neural_outputs=args.optuna_cache_neural_outputs,
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
