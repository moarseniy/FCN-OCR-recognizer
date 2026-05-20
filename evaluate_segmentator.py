from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from fcn_ocr import VerticalSegmentator


def get_gt_text(task: dict[str, Any]) -> str:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            text_items = result.get("value", {}).get("text", [])
            if text_items:
                return str(text_items[0]).strip()
    return ""


def get_image_name(task: dict[str, Any]) -> str:
    image_path = task.get("data", {}).get("image", "")
    return Path(image_path).name


def build_rows_and_jobs(
    json_path: Path,
    images_dir: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[int, Path]]]:
    with json_path.open("r", encoding="utf-8") as file:
        tasks = json.load(file)

    if limit is not None:
        tasks = tasks[:limit]

    rows: list[dict[str, Any]] = []
    jobs: list[tuple[int, Path]] = []

    for task in tasks:
        image_name = get_image_name(task)
        image_path = images_dir / image_name
        gt = get_gt_text(task)
        row = {
            "task_id": task.get("id"),
            "image": image_name,
            "gt": gt,
            "gt_len": len(gt),
            "pred_len": 0,
            "gap_count": 0,
            "length_error": 0,
            "abs_length_error": 0,
            "gap_runs": "",
            "error": "",
        }
        if image_path.exists():
            jobs.append((len(rows), image_path))
        else:
            row["error"] = f"image_not_found: {image_path}"
        rows.append(row)

    return rows, jobs


def gap_runs_text(result) -> str:
    if result.mode == "cut_projection":
        return " ".join(f"{run.start}:{run.gap_probability:.3f}" for run in result.runs if run.label == 1)
    return " ".join(f"{run.start}-{run.end}:{run.gap_probability:.3f}" for run in result.runs if run.label == 1)


def segment_count(result) -> int:
    if result.mode == "cut_projection":
        return len(result.cut_positions or [])
    return sum(1 for run in result.runs if run.label == 1)


def segment_images(
    segmentator: VerticalSegmentator,
    jobs: list[tuple[int, Path]],
    batch_size: int,
    log_every: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    predictions: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}
    processed = 0
    started_at = time.perf_counter()

    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        try:
            batch_predictions = segment_batch(segmentator, batch_jobs)
            predictions.update(batch_predictions)
            for row_index, _ in batch_jobs:
                errors[row_index] = ""
        except Exception as batch_error:
            for row_index, path in batch_jobs:
                try:
                    prediction = segment_batch(segmentator, [(row_index, path)])[row_index]
                    predictions[row_index] = prediction
                    errors[row_index] = ""
                except Exception as image_error:
                    errors[row_index] = f"batch_error={batch_error!r}; image_error={image_error!r}"

        processed += len(batch_jobs)
        if log_every > 0 and (processed == len(jobs) or processed % log_every == 0):
            elapsed = max(1e-9, time.perf_counter() - started_at)
            print(f"Segmented {processed}/{len(jobs)} images ({processed / elapsed:.2f} img/s)")

    return predictions, errors


@torch.no_grad()
def segment_batch(
    segmentator: VerticalSegmentator,
    batch_jobs: list[tuple[int, Path]],
) -> dict[int, dict[str, Any]]:
    tensors: list[torch.Tensor] = []
    output_lengths: list[int] = []
    max_width = 0

    for _, path in batch_jobs:
        with Image.open(path) as image:
            tensor = segmentator._preprocess_pil_3d(image)
        tensors.append(tensor)
        max_width = max(max_width, tensor.size(2))
        output_lengths.append(segmentator.output_width_for_input_width(tensor.size(2)))

    if not tensors:
        return {}

    batch = torch.ones(
        (len(tensors), segmentator.in_channels, segmentator.image_height, max_width),
        dtype=tensors[0].dtype,
        device=segmentator.device,
    )
    for batch_index, tensor in enumerate(tensors):
        batch[batch_index, :, :, : tensor.size(2)] = tensor

    logits = segmentator.model(batch)

    predictions: dict[int, dict[str, Any]] = {}
    for batch_index, ((row_index, _), output_length) in enumerate(zip(batch_jobs, output_lengths)):
        sample_logits = logits[batch_index : batch_index + 1, :, :output_length]
        result = segmentator.analyze_segmentation_logits(
            sample_logits,
            input_shape=(1, segmentator.in_channels, segmentator.image_height, tensors[batch_index].size(2)),
        )
        gap_count = segment_count(result)
        pred_len = gap_count + 1 if result.raw_indices else 0
        predictions[row_index] = {
            "pred_len": pred_len,
            "gap_count": gap_count,
            "gap_runs": gap_runs_text(result),
        }

    return predictions


def compute_metrics(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    total = len(rows)
    evaluated = sum(1 for row in rows if not row["error"])
    exact = 0
    total_abs_error = 0
    total_signed_error = 0
    total_gt_len = 0

    for row in rows:
        if row["error"]:
            row["length_error"] = -row["gt_len"]
            row["abs_length_error"] = abs(row["length_error"])
            continue
        else:
            row["length_error"] = row["pred_len"] - row["gt_len"]
            row["abs_length_error"] = abs(row["length_error"])
            exact += int(row["length_error"] == 0)

        total_abs_error += row["abs_length_error"]
        total_signed_error += row["length_error"]
        total_gt_len += row["gt_len"]

    return {
        "total_samples": total,
        "evaluated_samples": evaluated,
        "exact_length_matches": exact,
        "length_accuracy": exact / evaluated if evaluated else 0.0,
        "average_abs_length_error": total_abs_error / evaluated if evaluated else 0.0,
        "total_abs_length_error": total_abs_error,
        "average_signed_length_error": total_signed_error / evaluated if evaluated else 0.0,
        "normalized_length_error": total_abs_error / total_gt_len if total_gt_len else 0.0,
        "elapsed": elapsed,
        "speed": evaluated / elapsed if elapsed > 0 else 0.0,
    }


def write_rows_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "image",
                "gt",
                "gt_len",
                "pred_len",
                "gap_count",
                "length_error",
                "abs_length_error",
                "gap_runs",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(metrics: dict[str, Any], output_csv: Path | None = None) -> None:
    print("=== Segmentator length evaluation ===")
    print(f"Total samples:              {metrics['total_samples']}")
    print(f"Evaluated samples:          {metrics['evaluated_samples']}")
    print(f"Exact length matches:       {metrics['exact_length_matches']}")
    print(f"Length accuracy:            {metrics['length_accuracy']:.4f}")
    print(f"Avg abs length error:       {metrics['average_abs_length_error']:.4f}")
    print(f"Total abs length error:     {metrics['total_abs_length_error']}")
    print(f"Avg signed length error:    {metrics['average_signed_length_error']:.4f}")
    print(f"Normalized length error:    {metrics['normalized_length_error']:.4f}")
    print(f"Elapsed:                    {metrics['elapsed']:.2f}s")
    print(f"Speed:                      {metrics['speed']:.2f} img/s")
    print(f"segmentator_mode:           {metrics.get('segmentator_mode', 'binary_gaps')}")
    print(f"gap_threshold:              {metrics['gap_threshold']:.5f}")
    print(f"min_gap_width:              {metrics['min_gap_width']}")
    print(f"merge_gap_width:            {metrics['merge_gap_width']}")
    if metrics.get("segmentator_mode") == "cut_projection":
        print(f"peak_min_distance:          {metrics['peak_min_distance']}")
    print(f"scale_x:                    {metrics['scale_x']:+.5f}")
    print(f"y_pad:                      {metrics['y_pad']:+.5f}")
    print(f"baseline_crop:              {metrics['baseline_crop']}")
    print(f"baseline_top_pad:           {metrics['baseline_top_pad']:.5f}")
    print(f"baseline_bottom_pad:        {metrics['baseline_bottom_pad']:.5f}")
    print(f"baseline_deskew:            {metrics['baseline_deskew']}")
    print(f"baseline_max_angle:         {metrics['baseline_max_angle']:.5f}")
    if output_csv is not None:
        print(f"CSV saved to:               {output_csv}")


def configure_segmentator(
    segmentator: VerticalSegmentator,
    gap_threshold: float | None,
    min_gap_width: int | None,
    merge_gap_width: int | None,
    scale_x: float,
    y_pad: float,
    baseline_crop: bool,
    baseline_top_pad: float,
    baseline_bottom_pad: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
) -> None:
    if scale_x <= -0.95:
        raise ValueError("scale_x must be > -0.95")
    if y_pad <= -0.95:
        raise ValueError("y_pad must be > -0.95")
    if baseline_top_pad < 0.0:
        raise ValueError("baseline_top_pad must be >= 0")
    if baseline_bottom_pad < 0.0:
        raise ValueError("baseline_bottom_pad must be >= 0")
    if baseline_max_angle <= 0.0:
        raise ValueError("baseline_max_angle must be > 0")

    segmentator.gap_threshold = segmentator._resolve_gap_threshold(
        gap_threshold,
        {"segmentator_gap_threshold": segmentator.gap_threshold},
    )
    segmentator.min_gap_width = segmentator._resolve_non_negative_int(
        min_gap_width,
        {"segmentator_min_gap_width": segmentator.min_gap_width},
        "segmentator_min_gap_width",
        default=1,
        min_value=1,
    )
    segmentator.merge_gap_width = segmentator._resolve_non_negative_int(
        merge_gap_width,
        {"segmentator_merge_gap_width": segmentator.merge_gap_width},
        "segmentator_merge_gap_width",
        default=0,
        min_value=0,
    )
    if getattr(segmentator, "target_format", "") == "cut_projection":
        segmentator.peak_min_distance = segmentator.min_gap_width
    segmentator.scale_x = float(scale_x)
    segmentator.y_pad = float(y_pad)
    segmentator.baseline_crop = bool(baseline_crop)
    segmentator.baseline_top_pad = float(baseline_top_pad)
    segmentator.baseline_bottom_pad = float(baseline_bottom_pad)
    segmentator.baseline_deskew = bool(baseline_deskew)
    segmentator.baseline_max_angle = float(baseline_max_angle)


def evaluate_with_segmentator(
    base_rows: list[dict[str, Any]],
    jobs: list[tuple[int, Path]],
    segmentator: VerticalSegmentator,
    output_csv: Path | None,
    batch_size: int,
    log_every: int,
    verbose: bool,
) -> dict[str, Any]:
    rows = deepcopy(base_rows)
    started_at = time.perf_counter()
    predictions, errors = segment_images(segmentator, jobs, batch_size=batch_size, log_every=log_every)
    elapsed = time.perf_counter() - started_at

    for row_index, prediction in predictions.items():
        rows[row_index].update(prediction)
    for row_index, error in errors.items():
        rows[row_index]["error"] = error

    metrics = compute_metrics(rows, elapsed)
    metrics["segmentator_mode"] = getattr(segmentator, "target_format", "binary_gaps")
    metrics["gap_threshold"] = float(segmentator.gap_threshold)
    metrics["min_gap_width"] = int(segmentator.min_gap_width)
    metrics["merge_gap_width"] = int(segmentator.merge_gap_width)
    metrics["peak_min_distance"] = int(getattr(segmentator, "peak_min_distance", segmentator.min_gap_width))
    metrics["scale_x"] = float(segmentator.scale_x)
    metrics["y_pad"] = float(segmentator.y_pad)
    metrics["baseline_crop"] = bool(segmentator.baseline_crop)
    metrics["baseline_top_pad"] = float(segmentator.baseline_top_pad)
    metrics["baseline_bottom_pad"] = float(segmentator.baseline_bottom_pad)
    metrics["baseline_deskew"] = bool(segmentator.baseline_deskew)
    metrics["baseline_max_angle"] = float(segmentator.baseline_max_angle)

    if output_csv is not None:
        write_rows_csv(rows, output_csv)
    if verbose:
        print_metrics(metrics, output_csv)
    return metrics


def evaluate_prepared(
    base_rows: list[dict[str, Any]],
    jobs: list[tuple[int, Path]],
    checkpoint_path: Path,
    output_csv: Path | None,
    device: str | None,
    batch_size: int,
    log_every: int,
    verbose: bool,
    gap_threshold: float | None,
    min_gap_width: int | None,
    merge_gap_width: int | None,
    scale_x: float,
    y_pad: float,
    baseline_crop: bool,
    baseline_top_pad: float,
    baseline_bottom_pad: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
) -> dict[str, Any]:
    segmentator = VerticalSegmentator(checkpoint_path, device=device, verbose=False)
    configure_segmentator(
        segmentator,
        gap_threshold=gap_threshold,
        min_gap_width=min_gap_width,
        merge_gap_width=merge_gap_width,
        scale_x=scale_x,
        y_pad=y_pad,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
    )
    if verbose:
        segmentator.print_summary()
    return evaluate_with_segmentator(
        base_rows=base_rows,
        jobs=jobs,
        segmentator=segmentator,
        output_csv=output_csv,
        batch_size=batch_size,
        log_every=log_every,
        verbose=verbose,
    )


def append_trial_log(path: Path, trial_number: int, metrics: dict[str, Any], metric_name: str) -> None:
    is_new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        if is_new_file:
            file.write(
                "trial\tgap_threshold\tmin_gap_width\tmerge_gap_width\tscale_x\ty_pad\t"
                "baseline_crop\tbaseline_top_pad\tbaseline_bottom_pad\tbaseline_deskew\tbaseline_max_angle\t"
                "metric\tlength_accuracy\taverage_abs_length_error\ttotal_abs_length_error\t"
                "average_signed_length_error\tnormalized_length_error\tspeed\n"
            )
        file.write(
            f"{trial_number}\t{metrics['gap_threshold']:.8f}\t{metrics['min_gap_width']}\t"
            f"{metrics['merge_gap_width']}\t{metrics['scale_x']:.8f}\t{metrics['y_pad']:.8f}\t"
            f"{int(metrics['baseline_crop'])}\t{metrics['baseline_top_pad']:.8f}\t"
            f"{metrics['baseline_bottom_pad']:.8f}\t{int(metrics['baseline_deskew'])}\t"
            f"{metrics['baseline_max_angle']:.8f}\t{metrics[metric_name]:.8f}\t"
            f"{metrics['length_accuracy']:.8f}\t{metrics['average_abs_length_error']:.8f}\t"
            f"{metrics['total_abs_length_error']}\t{metrics['average_signed_length_error']:.8f}\t"
            f"{metrics['normalized_length_error']:.8f}\t{metrics['speed']:.6f}\n"
        )


def optimize(
    json_path: Path,
    images_dir: Path,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    batch_size: int,
    limit: int | None,
    trials: int,
    metric_name: str,
    log_every: int,
    trials_output: Path | None,
    gap_threshold_min: float,
    gap_threshold_max: float,
    min_gap_width_min: int,
    min_gap_width_max: int,
    merge_gap_width_min: int,
    merge_gap_width_max: int,
    scale_x_min: float,
    scale_x_max: float,
    y_pad_min: float,
    y_pad_max: float,
    tune_baseline_crop: bool,
    tune_baseline_params: bool,
    tune_baseline_deskew: bool,
    baseline_crop: bool,
    baseline_top_pad: float,
    baseline_bottom_pad: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
    baseline_top_pad_min: float,
    baseline_top_pad_max: float,
    baseline_bottom_pad_min: float,
    baseline_bottom_pad_max: float,
    baseline_max_angle_min: float,
    baseline_max_angle_max: float,
    study_name: str | None = None,
    storage: str | None = None,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna") from exc
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    if trials < 1:
        raise ValueError("trials must be >= 1")

    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    segmentator = VerticalSegmentator(checkpoint_path, device=device, verbose=True)
    direction = "maximize" if metric_name == "length_accuracy" else "minimize"
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )

    def objective(trial) -> float:
        trial_baseline_crop = (
            trial.suggest_categorical("baseline_crop", [False, True])
            if tune_baseline_crop
            else baseline_crop
        )
        trial_baseline_top_pad = baseline_top_pad
        trial_baseline_bottom_pad = baseline_bottom_pad
        trial_baseline_max_angle = baseline_max_angle
        if bool(trial_baseline_crop) and tune_baseline_params:
            trial_baseline_top_pad = trial.suggest_float(
                "baseline_top_pad",
                baseline_top_pad_min,
                baseline_top_pad_max,
            )
            trial_baseline_bottom_pad = trial.suggest_float(
                "baseline_bottom_pad",
                baseline_bottom_pad_min,
                baseline_bottom_pad_max,
            )
            trial_baseline_max_angle = trial.suggest_float(
                "baseline_max_angle",
                baseline_max_angle_min,
                baseline_max_angle_max,
            )
        trial_baseline_deskew = (
            trial.suggest_categorical("baseline_deskew", [False, True])
            if bool(trial_baseline_crop) and tune_baseline_deskew
            else baseline_deskew
        )
        configure_segmentator(
            segmentator,
            gap_threshold=trial.suggest_float("gap_threshold", gap_threshold_min, gap_threshold_max),
            min_gap_width=trial.suggest_int("min_gap_width", min_gap_width_min, min_gap_width_max),
            merge_gap_width=trial.suggest_int("merge_gap_width", merge_gap_width_min, merge_gap_width_max),
            scale_x=trial.suggest_float("scale_x", scale_x_min, scale_x_max),
            y_pad=trial.suggest_float("y_pad", y_pad_min, y_pad_max),
            baseline_crop=bool(trial_baseline_crop),
            baseline_top_pad=trial_baseline_top_pad,
            baseline_bottom_pad=trial_baseline_bottom_pad,
            baseline_deskew=bool(trial_baseline_deskew),
            baseline_max_angle=trial_baseline_max_angle,
        )
        metrics = evaluate_with_segmentator(
            base_rows=base_rows,
            jobs=jobs,
            segmentator=segmentator,
            output_csv=None,
            batch_size=batch_size,
            log_every=0,
            verbose=False,
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                trial.set_user_attr(key, value)
        if trials_output is not None:
            append_trial_log(trials_output, trial.number, metrics, metric_name)
        return float(metrics[metric_name])

    print(
        "Optuna segmentator search: "
        f"trials={trials}, metric={metric_name}, "
        f"gap_threshold=[{gap_threshold_min}, {gap_threshold_max}], "
        f"min_gap_width=[{min_gap_width_min}, {min_gap_width_max}], "
        f"merge_gap_width=[{merge_gap_width_min}, {merge_gap_width_max}], "
        f"scale_x=[{scale_x_min}, {scale_x_max}], y_pad=[{y_pad_min}, {y_pad_max}], "
        f"tune_baseline_crop={tune_baseline_crop}, "
        f"tune_baseline_params={tune_baseline_params}, "
        f"tune_baseline_deskew={tune_baseline_deskew}"
    )
    study.optimize(objective, n_trials=trials)

    best_params = dict(study.best_params)
    if "baseline_crop" not in best_params:
        best_params["baseline_crop"] = baseline_crop
    if "baseline_top_pad" not in best_params:
        best_params["baseline_top_pad"] = baseline_top_pad
    if "baseline_bottom_pad" not in best_params:
        best_params["baseline_bottom_pad"] = baseline_bottom_pad
    if "baseline_deskew" not in best_params:
        best_params["baseline_deskew"] = baseline_deskew
    if "baseline_max_angle" not in best_params:
        best_params["baseline_max_angle"] = baseline_max_angle
    print(f"Best params: {best_params}, {metric_name}={study.best_value:.8f}")

    configure_segmentator(
        segmentator,
        gap_threshold=float(best_params["gap_threshold"]),
        min_gap_width=int(best_params["min_gap_width"]),
        merge_gap_width=int(best_params["merge_gap_width"]),
        scale_x=float(best_params["scale_x"]),
        y_pad=float(best_params["y_pad"]),
        baseline_crop=bool(best_params["baseline_crop"]),
        baseline_top_pad=float(best_params["baseline_top_pad"]),
        baseline_bottom_pad=float(best_params["baseline_bottom_pad"]),
        baseline_deskew=bool(best_params["baseline_deskew"]),
        baseline_max_angle=float(best_params["baseline_max_angle"]),
    )
    final_metrics = evaluate_with_segmentator(
        base_rows=base_rows,
        jobs=jobs,
        segmentator=segmentator,
        output_csv=output_csv,
        batch_size=batch_size,
        log_every=log_every,
        verbose=True,
    )
    final_metrics["optuna_trials"] = trials
    final_metrics["optuna_metric"] = metric_name
    final_metrics["optuna_best_value"] = float(study.best_value)
    return final_metrics


def evaluate(
    json_path: Path,
    images_dir: Path,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    batch_size: int,
    limit: int | None,
    log_every: int,
    gap_threshold: float | None,
    min_gap_width: int | None,
    merge_gap_width: int | None,
    scale_x: float,
    y_pad: float,
    baseline_crop: bool,
    baseline_top_pad: float,
    baseline_bottom_pad: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
) -> dict[str, Any]:
    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    return evaluate_prepared(
        base_rows=base_rows,
        jobs=jobs,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        device=device,
        batch_size=batch_size,
        log_every=log_every,
        verbose=True,
        gap_threshold=gap_threshold,
        min_gap_width=min_gap_width,
        merge_gap_width=merge_gap_width,
        scale_x=scale_x,
        y_pad=y_pad,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune/evaluate vertical segmentator using Label Studio text lengths.")
    parser.add_argument("--json", required=True, help="Path to Label Studio export JSON.")
    parser.add_argument("--images", required=True, help="Folder with images.")
    parser.add_argument("--checkpoint", required=True, help="Path to binary vertical segmentator checkpoint.")
    parser.add_argument("--out", default="segmentator_length_metrics.csv", help="Output CSV path.")
    parser.add_argument("--device", default=None, help="Device to use: cuda, cpu, or empty for auto.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--gap-threshold", type=float, default=None)
    parser.add_argument("--min-gap-width", type=int, default=None)
    parser.add_argument("--merge-gap-width", type=int, default=None)
    parser.add_argument("--scale-x", type=float, default=0.0)
    parser.add_argument("--y-pad", type=float, default=0.0)
    parser.add_argument("--baseline-crop", action="store_true")
    parser.add_argument("--baseline-top-pad", type=float, default=0.12)
    parser.add_argument("--baseline-bottom-pad", type=float, default=0.18)
    parser.add_argument("--no-baseline-deskew", action="store_true")
    parser.add_argument("--baseline-max-angle", type=float, default=12.0)

    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument(
        "--optuna-metric",
        default="average_abs_length_error",
        choices=[
            "length_accuracy",
            "average_abs_length_error",
            "total_abs_length_error",
            "normalized_length_error",
        ],
    )
    parser.add_argument("--optuna-gap-threshold-min", type=float, default=0.25)
    parser.add_argument("--optuna-gap-threshold-max", type=float, default=0.85)
    parser.add_argument("--optuna-min-gap-width-min", type=int, default=1)
    parser.add_argument("--optuna-min-gap-width-max", type=int, default=4)
    parser.add_argument("--optuna-merge-gap-width-min", type=int, default=0)
    parser.add_argument("--optuna-merge-gap-width-max", type=int, default=3)
    parser.add_argument("--optuna-scale-x-min", type=float, default=-0.25)
    parser.add_argument("--optuna-scale-x-max", type=float, default=0.25)
    parser.add_argument("--optuna-y-pad-min", type=float, default=-0.25)
    parser.add_argument("--optuna-y-pad-max", type=float, default=0.25)
    parser.add_argument("--optuna-tune-baseline-crop", action="store_true")
    parser.add_argument(
        "--optuna-tune-baseline-params",
        action="store_true",
        help="Tune baseline crop paddings and max angle when baseline crop is enabled in a trial.",
    )
    parser.add_argument(
        "--optuna-tune-baseline-deskew",
        action="store_true",
        help="Tune baseline deskew on/off when baseline crop is enabled in a trial.",
    )
    parser.add_argument("--optuna-baseline-top-pad-min", type=float, default=0.02)
    parser.add_argument("--optuna-baseline-top-pad-max", type=float, default=0.30)
    parser.add_argument("--optuna-baseline-bottom-pad-min", type=float, default=0.02)
    parser.add_argument("--optuna-baseline-bottom-pad-max", type=float, default=0.45)
    parser.add_argument("--optuna-baseline-max-angle-min", type=float, default=4.0)
    parser.add_argument("--optuna-baseline-max-angle-max", type=float, default=18.0)
    parser.add_argument("--optuna-trials-out", default=None)
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument("--optuna-storage", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.optuna_trials > 0:
        optimize(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            trials=args.optuna_trials,
            metric_name=args.optuna_metric,
            log_every=args.log_every,
            trials_output=Path(args.optuna_trials_out) if args.optuna_trials_out else None,
            gap_threshold_min=args.optuna_gap_threshold_min,
            gap_threshold_max=args.optuna_gap_threshold_max,
            min_gap_width_min=args.optuna_min_gap_width_min,
            min_gap_width_max=args.optuna_min_gap_width_max,
            merge_gap_width_min=args.optuna_merge_gap_width_min,
            merge_gap_width_max=args.optuna_merge_gap_width_max,
            scale_x_min=args.optuna_scale_x_min,
            scale_x_max=args.optuna_scale_x_max,
            y_pad_min=args.optuna_y_pad_min,
            y_pad_max=args.optuna_y_pad_max,
            tune_baseline_crop=args.optuna_tune_baseline_crop,
            tune_baseline_params=args.optuna_tune_baseline_params,
            tune_baseline_deskew=args.optuna_tune_baseline_deskew,
            baseline_crop=args.baseline_crop,
            baseline_top_pad=args.baseline_top_pad,
            baseline_bottom_pad=args.baseline_bottom_pad,
            baseline_deskew=not args.no_baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
            baseline_top_pad_min=args.optuna_baseline_top_pad_min,
            baseline_top_pad_max=args.optuna_baseline_top_pad_max,
            baseline_bottom_pad_min=args.optuna_baseline_bottom_pad_min,
            baseline_bottom_pad_max=args.optuna_baseline_bottom_pad_max,
            baseline_max_angle_min=args.optuna_baseline_max_angle_min,
            baseline_max_angle_max=args.optuna_baseline_max_angle_max,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
        )
    else:
        evaluate(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            log_every=args.log_every,
            gap_threshold=args.gap_threshold,
            min_gap_width=args.min_gap_width,
            merge_gap_width=args.merge_gap_width,
            scale_x=args.scale_x,
            y_pad=args.y_pad,
            baseline_crop=args.baseline_crop,
            baseline_top_pad=args.baseline_top_pad,
            baseline_bottom_pad=args.baseline_bottom_pad,
            baseline_deskew=not args.no_baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
        )


if __name__ == "__main__":
    main()
