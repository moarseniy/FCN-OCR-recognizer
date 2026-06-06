from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import shlex
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from fcn_ocr import VerticalSegmentator
from tool.evaluation import match_sorted_points
from tool.markup import annotated_items, is_manual_markup, safe_image_path


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

    if is_manual_markup(tasks):
        return build_manual_rows_and_jobs(tasks, images_dir, limit)

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
            "cut_count": 0,
            "length_error": 0,
            "abs_length_error": 0,
            "cuts": "",
            "gt_cuts": [],
            "pred_cuts": [],
            "matched_cuts": 0,
            "false_positive_cuts": 0,
            "false_negative_cuts": 0,
            "cut_mae_px": 0.0,
            "error": "",
        }
        if not image_path.exists():
            continue
        jobs.append((len(rows), image_path))
        rows.append(row)

    return rows, jobs


def build_manual_rows_and_jobs(
    document: dict[str, Any],
    images_dir: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[int, Path]]]:
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[int, Path]] = []
    items = annotated_items(document)
    if limit is not None:
        items = items[:limit]

    for item in items:
        cuts = sorted(float(value) for value in item.get("cuts", []))
        if len(cuts) < 2:
            continue
        try:
            image_path = safe_image_path(images_dir, str(item["image"]))
        except FileNotFoundError:
            continue
        row = {
            "task_id": "",
            "image": str(item["image"]),
            "gt": "",
            "gt_len": max(0, len(cuts) - 1),
            "pred_len": 0,
            "cut_count": 0,
            "length_error": 0,
            "abs_length_error": 0,
            "cuts": "",
            "gt_cuts": cuts,
            "pred_cuts": [],
            "matched_cuts": 0,
            "false_positive_cuts": 0,
            "false_negative_cuts": 0,
            "cut_mae_px": 0.0,
            "error": "",
        }
        jobs.append((len(rows), image_path))
        rows.append(row)
    return rows, jobs


def cuts_text(result) -> str:
    return " ".join(f"{run.start}:{run.score:.3f}" for run in result.runs if run.label == 1)


def segment_count(result) -> int:
    return len(result.cut_positions or [])


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
    source_x_maps: list[np.ndarray] = []
    max_width = 0

    for _, path in batch_jobs:
        with Image.open(path) as image:
            tensor, source_x_map = segmentator.preprocess_pil_with_source_x(image)
        tensors.append(tensor)
        source_x_maps.append(source_x_map)
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
        cut_count = segment_count(result)
        pred_len = max(0, cut_count - 1) if result.raw_indices else 0
        predicted_source_cuts = cut_positions_to_source(
            result.cut_positions or [],
            output_width=output_length,
            source_x_map=source_x_maps[batch_index],
        )
        predictions[row_index] = {
            "pred_len": pred_len,
            "cut_count": cut_count,
            "cuts": cuts_text(result),
            "pred_cuts": predicted_source_cuts,
        }

    return predictions


def cut_positions_to_source(
    cut_positions: list[int],
    output_width: int,
    source_x_map: np.ndarray,
) -> list[float]:
    if output_width <= 0 or source_x_map.size == 0:
        return []
    input_width = int(source_x_map.shape[1])
    column_source_x = np.full(input_width, np.nan, dtype=np.float64)
    for column in range(input_width):
        values = source_x_map[:, column]
        valid = values[values >= 0.0]
        if valid.size:
            column_source_x[column] = float(np.median(valid))
    valid_columns = np.flatnonzero(np.isfinite(column_source_x))
    if valid_columns.size == 0:
        return []
    if valid_columns.size == 1:
        column_source_x[:] = column_source_x[valid_columns[0]]
    else:
        missing = ~np.isfinite(column_source_x)
        column_source_x[missing] = np.interp(
            np.flatnonzero(missing),
            valid_columns,
            column_source_x[valid_columns],
        )

    mapped: list[float] = []
    for position in cut_positions:
        if output_width <= 1:
            input_position = 0.0
        else:
            input_position = float(position) * float(input_width - 1) / float(output_width - 1)
        left = int(np.floor(input_position))
        right = min(input_width - 1, left + 1)
        fraction = input_position - left
        mapped.append(
            float(column_source_x[left] * (1.0 - fraction) + column_source_x[right] * fraction)
        )
    return mapped


def compute_metrics(
    rows: list[dict[str, Any]],
    elapsed: float,
    cut_tolerance_px: float,
) -> dict[str, Any]:
    total = len(rows)
    evaluated = sum(1 for row in rows if not row["error"])
    exact = 0
    total_abs_error = 0
    total_signed_error = 0
    total_gt_len = 0
    expected_cuts = 0
    predicted_cuts = 0
    matched_cuts = 0
    total_cut_error = 0.0
    manual_samples = 0

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

        gt_cuts = [float(value) for value in row.get("gt_cuts", [])]
        if gt_cuts and not row["error"]:
            manual_samples += 1
            pred_cuts = [float(value) for value in row.get("pred_cuts", [])]
            matches = match_sorted_points(gt_cuts, pred_cuts, cut_tolerance_px)
            row["matched_cuts"] = len(matches)
            row["false_positive_cuts"] = len(pred_cuts) - len(matches)
            row["false_negative_cuts"] = len(gt_cuts) - len(matches)
            row["cut_mae_px"] = (
                sum(match.error for match in matches) / len(matches)
                if matches
                else 0.0
            )
            expected_cuts += len(gt_cuts)
            predicted_cuts += len(pred_cuts)
            matched_cuts += len(matches)
            total_cut_error += sum(match.error for match in matches)

    cut_precision = matched_cuts / predicted_cuts if predicted_cuts else 0.0
    cut_recall = matched_cuts / expected_cuts if expected_cuts else 0.0
    cut_f1 = (
        2.0 * cut_precision * cut_recall / (cut_precision + cut_recall)
        if cut_precision + cut_recall > 0.0
        else 0.0
    )

    return {
        "total_samples": total,
        "evaluated_samples": evaluated,
        "exact_length_matches": exact,
        "length_accuracy": exact / evaluated if evaluated else 0.0,
        "average_abs_length_error": total_abs_error / evaluated if evaluated else 0.0,
        "total_abs_length_error": total_abs_error,
        "average_signed_length_error": total_signed_error / evaluated if evaluated else 0.0,
        "normalized_length_error": total_abs_error / total_gt_len if total_gt_len else 0.0,
        "manual_cut_samples": manual_samples,
        "expected_cuts": expected_cuts,
        "predicted_cuts": predicted_cuts,
        "matched_cuts": matched_cuts,
        "cut_precision": cut_precision,
        "cut_recall": cut_recall,
        "cut_f1": cut_f1,
        "cut_mae_px": total_cut_error / matched_cuts if matched_cuts else 0.0,
        "cut_tolerance_px": float(cut_tolerance_px),
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
                "cut_count",
                "length_error",
                "abs_length_error",
                "cuts",
                "gt_cuts",
                "pred_cuts",
                "matched_cuts",
                "false_positive_cuts",
                "false_negative_cuts",
                "cut_mae_px",
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
    if metrics["manual_cut_samples"]:
        print(f"Manual cut samples:         {metrics['manual_cut_samples']}")
        print(f"Cut precision:              {metrics['cut_precision']:.4f}")
        print(f"Cut recall:                 {metrics['cut_recall']:.4f}")
        print(f"Cut F1:                     {metrics['cut_f1']:.4f}")
        print(f"Cut MAE:                    {metrics['cut_mae_px']:.3f}px")
        print(f"Cut tolerance:              {metrics['cut_tolerance_px']:.2f}px")
    print(f"Elapsed:                    {metrics['elapsed']:.2f}s")
    print(f"Speed:                      {metrics['speed']:.2f} img/s")
    print(f"segmentator_mode:           {metrics.get('segmentator_mode', 'cut_projection')}")
    print(f"cut_threshold:              {metrics['cut_threshold']:.5f}")
    print(f"peak_min_distance:          {metrics['peak_min_distance']}")
    print(f"scale_x:                    {metrics['scale_x']:+.5f}")
    print(f"y_pad:                      {metrics['y_pad']:+.5f}")
    print(f"x_pad:                      {metrics['x_pad']:.5f}")
    print(f"baseline_crop:              {metrics['baseline_crop']}")
    print(f"baseline_strict_lines:      {metrics['baseline_strict_lines']}")
    print(f"baseline_top_pad:           {metrics['baseline_top_pad']:.5f}")
    print(f"baseline_bottom_pad:        {metrics['baseline_bottom_pad']:.5f}")
    print(f"baseline_line_pad:          {metrics['baseline_line_pad']:.5f}")
    print(f"baseline_line_pad_px:       {metrics['baseline_line_pad_px']:.2f}")
    print(f"baseline_deskew:            {metrics['baseline_deskew']}")
    print(f"baseline_max_angle:         {metrics['baseline_max_angle']:.5f}")
    if metrics.get("baseline_detector_checkpoint"):
        print(f"baseline_detector:          {metrics['baseline_detector_checkpoint']}")
        print(f"baseline_detector_thr:      {metrics['baseline_detector_threshold']:.5f}")
    if output_csv is not None:
        print(f"CSV saved to:               {output_csv}")


def configure_segmentator(
    segmentator: VerticalSegmentator,
    cut_threshold: float | None,
    peak_min_distance: int | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    baseline_crop: bool,
    baseline_top_pad: float,
    baseline_bottom_pad: float,
    baseline_strict_lines: bool,
    baseline_line_pad: float,
    baseline_line_pad_px: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
    baseline_detector_threshold: float,
) -> None:
    if scale_x <= -0.95:
        raise ValueError("scale_x must be > -0.95")
    if y_pad <= -0.95:
        raise ValueError("y_pad must be > -0.95")
    if x_pad < 0.0:
        raise ValueError("x_pad must be >= 0")
    if baseline_top_pad < 0.0:
        raise ValueError("baseline_top_pad must be >= 0")
    if baseline_bottom_pad < 0.0:
        raise ValueError("baseline_bottom_pad must be >= 0")
    if baseline_line_pad < 0.0:
        raise ValueError("baseline_line_pad must be >= 0")
    if baseline_line_pad_px < 0.0:
        raise ValueError("baseline_line_pad_px must be >= 0")
    if baseline_max_angle <= 0.0:
        raise ValueError("baseline_max_angle must be > 0")
    if not 0.0 < baseline_detector_threshold < 1.0:
        raise ValueError("baseline_detector_threshold must be between 0 and 1")

    segmentator.cut_threshold = segmentator._resolve_cut_threshold(
        cut_threshold,
        {"segmentator_cut_threshold": segmentator.cut_threshold},
    )
    segmentator.peak_min_distance = segmentator._resolve_non_negative_int(
        peak_min_distance,
        {"segmentator_peak_min_distance": segmentator.peak_min_distance},
        "segmentator_peak_min_distance",
        default=segmentator.peak_min_distance,
        min_value=1,
    )
    segmentator.scale_x = float(scale_x)
    segmentator.y_pad = float(y_pad)
    segmentator.x_pad = float(x_pad)
    segmentator.baseline_crop = bool(baseline_crop)
    segmentator.baseline_top_pad = float(baseline_top_pad)
    segmentator.baseline_bottom_pad = float(baseline_bottom_pad)
    segmentator.baseline_strict_lines = bool(baseline_strict_lines)
    segmentator.baseline_line_pad = float(baseline_line_pad)
    segmentator.baseline_line_pad_px = float(baseline_line_pad_px)
    segmentator.baseline_deskew = bool(baseline_deskew)
    segmentator.baseline_max_angle = float(baseline_max_angle)
    segmentator.baseline_detector_threshold = float(baseline_detector_threshold)


def evaluate_with_segmentator(
    base_rows: list[dict[str, Any]],
    jobs: list[tuple[int, Path]],
    segmentator: VerticalSegmentator,
    output_csv: Path | None,
    batch_size: int,
    log_every: int,
    verbose: bool,
    cut_tolerance_px: float,
) -> dict[str, Any]:
    rows = deepcopy(base_rows)
    started_at = time.perf_counter()
    predictions, errors = segment_images(segmentator, jobs, batch_size=batch_size, log_every=log_every)
    elapsed = time.perf_counter() - started_at

    for row_index, prediction in predictions.items():
        rows[row_index].update(prediction)
    for row_index, error in errors.items():
        rows[row_index]["error"] = error

    metrics = compute_metrics(rows, elapsed, cut_tolerance_px=cut_tolerance_px)
    metrics["segmentator_mode"] = getattr(segmentator, "target_format", "cut_projection")
    metrics["cut_threshold"] = float(segmentator.cut_threshold)
    metrics["peak_min_distance"] = int(segmentator.peak_min_distance)
    metrics["cut_postprocess"] = str(segmentator.cut_postprocess)
    metrics["cut_min_width"] = int(segmentator.cut_min_width)
    metrics["cut_max_width"] = int(segmentator.cut_max_width)
    metrics["cut_candidate_threshold"] = float(segmentator.cut_candidate_threshold)
    metrics["cut_smooth_radius"] = int(segmentator.cut_smooth_radius)
    metrics["scale_x"] = float(segmentator.scale_x)
    metrics["y_pad"] = float(segmentator.y_pad)
    metrics["x_pad"] = float(segmentator.x_pad)
    metrics["baseline_crop"] = bool(segmentator.baseline_crop)
    metrics["baseline_top_pad"] = float(segmentator.baseline_top_pad)
    metrics["baseline_bottom_pad"] = float(segmentator.baseline_bottom_pad)
    metrics["baseline_strict_lines"] = bool(segmentator.baseline_strict_lines)
    metrics["baseline_line_pad"] = float(segmentator.baseline_line_pad)
    metrics["baseline_line_pad_px"] = float(segmentator.baseline_line_pad_px)
    metrics["baseline_deskew"] = bool(segmentator.baseline_deskew)
    metrics["baseline_max_angle"] = float(segmentator.baseline_max_angle)
    metrics["baseline_detector_checkpoint"] = (
        str(segmentator.baseline_detector_checkpoint) if segmentator.baseline_detector_checkpoint else ""
    )
    metrics["baseline_detector_threshold"] = float(segmentator.baseline_detector_threshold)

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
    cut_threshold: float | None,
    peak_min_distance: int | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    baseline_crop: bool,
    baseline_top_pad: float,
    baseline_bottom_pad: float,
    baseline_strict_lines: bool,
    baseline_line_pad: float,
    baseline_line_pad_px: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
    baseline_detector_checkpoint: Path | None,
    baseline_detector_threshold: float,
    cut_tolerance_px: float,
) -> dict[str, Any]:
    segmentator = VerticalSegmentator(
        checkpoint_path,
        device=device,
        verbose=False,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_strict_lines=baseline_strict_lines,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=baseline_detector_threshold,
        cut_threshold=cut_threshold,
        peak_min_distance=peak_min_distance,
    )
    configure_segmentator(
        segmentator,
        cut_threshold=cut_threshold,
        peak_min_distance=peak_min_distance,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_strict_lines=baseline_strict_lines,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_detector_threshold=baseline_detector_threshold,
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
        cut_tolerance_px=cut_tolerance_px,
    )


def append_trial_log(path: Path, trial_number: int, metrics: dict[str, Any], metric_name: str) -> None:
    is_new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        if is_new_file:
            file.write(
                "trial\tcut_threshold\tpeak_min_distance\tscale_x\ty_pad\tx_pad\t"
                "baseline_crop\tbaseline_strict_lines\tbaseline_top_pad\tbaseline_bottom_pad\t"
                "baseline_line_pad\tbaseline_line_pad_px\tbaseline_deskew\tbaseline_max_angle\t"
                "baseline_detector_threshold\t"
                "metric\tlength_accuracy\taverage_abs_length_error\ttotal_abs_length_error\t"
                "average_signed_length_error\tnormalized_length_error\tcut_precision\tcut_recall\t"
                "cut_f1\tcut_mae_px\tspeed\n"
            )
        file.write(
            f"{trial_number}\t{metrics['cut_threshold']:.8f}\t{metrics['peak_min_distance']}\t"
            f"{metrics['scale_x']:.8f}\t{metrics['y_pad']:.8f}\t{metrics['x_pad']:.8f}\t"
            f"{int(metrics['baseline_crop'])}\t{int(metrics['baseline_strict_lines'])}\t"
            f"{metrics['baseline_top_pad']:.8f}\t"
            f"{metrics['baseline_bottom_pad']:.8f}\t{metrics['baseline_line_pad']:.8f}\t"
            f"{metrics['baseline_line_pad_px']:.8f}\t"
            f"{int(metrics['baseline_deskew'])}\t"
            f"{metrics['baseline_max_angle']:.8f}\t{metrics['baseline_detector_threshold']:.8f}\t"
            f"{metrics[metric_name]:.8f}\t"
            f"{metrics['length_accuracy']:.8f}\t{metrics['average_abs_length_error']:.8f}\t"
            f"{metrics['total_abs_length_error']}\t{metrics['average_signed_length_error']:.8f}\t"
            f"{metrics['normalized_length_error']:.8f}\t{metrics['cut_precision']:.8f}\t"
            f"{metrics['cut_recall']:.8f}\t{metrics['cut_f1']:.8f}\t"
            f"{metrics['cut_mae_px']:.8f}\t{metrics['speed']:.6f}\n"
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
    cut_threshold_min: float,
    cut_threshold_max: float,
    peak_min_distance_min: int,
    peak_min_distance_max: int,
    scale_x_min: float,
    scale_x_max: float,
    y_pad_min: float,
    y_pad_max: float,
    x_pad: float,
    tune_baseline_crop: bool,
    tune_baseline_params: bool,
    tune_baseline_bbox_pads: bool,
    tune_baseline_line_pad: bool,
    tune_baseline_line_pad_px: bool,
    tune_baseline_max_angle: bool,
    tune_baseline_deskew: bool,
    baseline_crop: bool,
    baseline_top_pad: float,
    baseline_bottom_pad: float,
    baseline_strict_lines: bool,
    baseline_line_pad: float,
    baseline_line_pad_px: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
    baseline_detector_checkpoint: Path | None,
    baseline_detector_threshold: float,
    baseline_top_pad_min: float,
    baseline_top_pad_max: float,
    baseline_bottom_pad_min: float,
    baseline_bottom_pad_max: float,
    baseline_line_pad_min: float,
    baseline_line_pad_max: float,
    baseline_line_pad_px_min: float,
    baseline_line_pad_px_max: float,
    baseline_max_angle_min: float,
    baseline_max_angle_max: float,
    baseline_detector_threshold_min: float | None,
    baseline_detector_threshold_max: float | None,
    study_name: str | None = None,
    storage: str | None = None,
    cut_tolerance_px: float = 3.0,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna") from exc
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    if trials < 1:
        raise ValueError("trials must be >= 1")

    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    has_manual_cuts = any(bool(row.get("gt_cuts")) for row in base_rows)
    if metric_name == "auto":
        metric_name = "cut_f1" if has_manual_cuts else "average_abs_length_error"
    if metric_name.startswith("cut_") and not has_manual_cuts:
        raise ValueError(
            f"--optuna-metric {metric_name} requires manual markup created by tool.annotation_server"
        )
    segmentator = VerticalSegmentator(
        checkpoint_path,
        device=device,
        verbose=True,
        scale_x=0.0,
        y_pad=0.0,
        x_pad=x_pad,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_strict_lines=baseline_strict_lines,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=baseline_detector_threshold,
    )
    direction = "maximize" if metric_name in {"length_accuracy", "cut_precision", "cut_recall", "cut_f1"} else "minimize"
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
        tune_active_line_pad = bool(trial_baseline_crop) and (
            tune_baseline_line_pad
            or (tune_baseline_params and baseline_strict_lines)
        )
        tune_active_bbox_pads = bool(trial_baseline_crop) and (
            tune_baseline_bbox_pads
            or (tune_baseline_params and not baseline_strict_lines)
        )
        tune_active_line_pad_px = bool(trial_baseline_crop) and tune_baseline_line_pad_px
        tune_active_max_angle = bool(trial_baseline_crop) and tune_baseline_max_angle
        tune_active_detector = bool(trial_baseline_crop) and baseline_detector_checkpoint is not None

        trial_baseline_top_pad = baseline_top_pad
        trial_baseline_bottom_pad = baseline_bottom_pad
        trial_baseline_line_pad = baseline_line_pad
        trial_baseline_line_pad_px = baseline_line_pad_px
        trial_baseline_max_angle = baseline_max_angle
        trial_baseline_detector_threshold = baseline_detector_threshold
        if tune_active_bbox_pads:
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
        if tune_active_line_pad:
            trial_baseline_line_pad = trial.suggest_float(
                "baseline_line_pad",
                baseline_line_pad_min,
                baseline_line_pad_max,
            )
        if tune_active_line_pad_px:
            trial_baseline_line_pad_px = trial.suggest_float(
                "baseline_line_pad_px",
                baseline_line_pad_px_min,
                baseline_line_pad_px_max,
            )
        if tune_active_max_angle:
            trial_baseline_max_angle = trial.suggest_float(
                "baseline_max_angle",
                baseline_max_angle_min,
                baseline_max_angle_max,
            )
        if tune_active_detector:
            if baseline_detector_threshold_min is not None or baseline_detector_threshold_max is not None:
                if baseline_detector_threshold_min is None or baseline_detector_threshold_max is None:
                    raise ValueError("baseline detector threshold tuning requires both min and max")
                trial_baseline_detector_threshold = trial.suggest_float(
                    "baseline_detector_threshold",
                    baseline_detector_threshold_min,
                    baseline_detector_threshold_max,
                )
        trial_baseline_deskew = (
            trial.suggest_categorical("baseline_deskew", [False, True])
            if bool(trial_baseline_crop) and tune_baseline_deskew
            else baseline_deskew
        )
        configure_segmentator(
            segmentator,
            cut_threshold=trial.suggest_float("cut_threshold", cut_threshold_min, cut_threshold_max),
            peak_min_distance=trial.suggest_int("peak_min_distance", peak_min_distance_min, peak_min_distance_max),
            scale_x=trial.suggest_float("scale_x", scale_x_min, scale_x_max),
            y_pad=trial.suggest_float("y_pad", y_pad_min, y_pad_max),
            x_pad=x_pad,
            baseline_crop=bool(trial_baseline_crop),
            baseline_top_pad=trial_baseline_top_pad,
            baseline_bottom_pad=trial_baseline_bottom_pad,
            baseline_strict_lines=baseline_strict_lines,
            baseline_line_pad=trial_baseline_line_pad,
            baseline_line_pad_px=trial_baseline_line_pad_px,
            baseline_deskew=bool(trial_baseline_deskew),
            baseline_max_angle=trial_baseline_max_angle,
            baseline_detector_threshold=trial_baseline_detector_threshold,
        )
        metrics = evaluate_with_segmentator(
            base_rows=base_rows,
            jobs=jobs,
            segmentator=segmentator,
            output_csv=None,
            batch_size=batch_size,
            log_every=0,
            verbose=False,
            cut_tolerance_px=cut_tolerance_px,
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
        f"cut_threshold=[{cut_threshold_min}, {cut_threshold_max}], "
        f"peak_min_distance=[{peak_min_distance_min}, {peak_min_distance_max}], "
        f"scale_x=[{scale_x_min}, {scale_x_max}], y_pad=[{y_pad_min}, {y_pad_max}], "
        f"x_pad={x_pad}, baseline_detector={baseline_detector_checkpoint}, "
        f"tune_baseline_crop={tune_baseline_crop}, "
        f"tune_baseline_params={tune_baseline_params}, "
        f"tune_bbox_pads={tune_baseline_bbox_pads}, "
        f"tune_line_pad={tune_baseline_line_pad}, "
        f"tune_line_pad_px={tune_baseline_line_pad_px}, "
        f"tune_max_angle={tune_baseline_max_angle}, "
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
    if "baseline_strict_lines" not in best_params:
        best_params["baseline_strict_lines"] = baseline_strict_lines
    if "baseline_line_pad" not in best_params:
        best_params["baseline_line_pad"] = baseline_line_pad
    if "baseline_line_pad_px" not in best_params:
        best_params["baseline_line_pad_px"] = baseline_line_pad_px
    if "baseline_deskew" not in best_params:
        best_params["baseline_deskew"] = baseline_deskew
    if "baseline_max_angle" not in best_params:
        best_params["baseline_max_angle"] = baseline_max_angle
    if "baseline_detector_threshold" not in best_params:
        best_params["baseline_detector_threshold"] = baseline_detector_threshold
    print(f"Best params: {best_params}, {metric_name}={study.best_value:.8f}")

    configure_segmentator(
        segmentator,
        cut_threshold=float(best_params["cut_threshold"]),
        peak_min_distance=int(best_params["peak_min_distance"]),
        scale_x=float(best_params["scale_x"]),
        y_pad=float(best_params["y_pad"]),
        x_pad=x_pad,
        baseline_crop=bool(best_params["baseline_crop"]),
        baseline_top_pad=float(best_params["baseline_top_pad"]),
        baseline_bottom_pad=float(best_params["baseline_bottom_pad"]),
        baseline_strict_lines=bool(best_params["baseline_strict_lines"]),
        baseline_line_pad=float(best_params["baseline_line_pad"]),
        baseline_line_pad_px=float(best_params["baseline_line_pad_px"]),
        baseline_deskew=bool(best_params["baseline_deskew"]),
        baseline_max_angle=float(best_params["baseline_max_angle"]),
        baseline_detector_threshold=float(best_params["baseline_detector_threshold"]),
    )
    final_metrics = evaluate_with_segmentator(
        base_rows=base_rows,
        jobs=jobs,
        segmentator=segmentator,
        output_csv=output_csv,
        batch_size=batch_size,
        log_every=log_every,
        verbose=True,
        cut_tolerance_px=cut_tolerance_px,
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
    cut_threshold: float | None,
    peak_min_distance: int | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    baseline_crop: bool,
    baseline_top_pad: float,
    baseline_bottom_pad: float,
    baseline_strict_lines: bool,
    baseline_line_pad: float,
    baseline_line_pad_px: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
    baseline_detector_checkpoint: Path | None,
    baseline_detector_threshold: float,
    cut_tolerance_px: float,
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
        cut_threshold=cut_threshold,
        peak_min_distance=peak_min_distance,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_strict_lines=baseline_strict_lines,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=baseline_detector_threshold,
        cut_tolerance_px=cut_tolerance_px,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune/evaluate vertical cut segmentation.")
    parser.add_argument(
        "--json",
        required=True,
        help="Label Studio export JSON or manual markup JSON created by tool.annotation_server.",
    )
    parser.add_argument("--images", required=True, help="Folder with images.")
    parser.add_argument("--checkpoint", required=True, help="Path to vertical cut segmentator checkpoint.")
    parser.add_argument(
        "--inference-ocr-checkpoint",
        default=None,
        help="OCR checkpoint to place into the printed inference.py command. It is not loaded or executed.",
    )
    parser.add_argument("--out", default="segmentator_length_metrics.csv", help="Output CSV path.")
    parser.add_argument("--device", default=None, help="Device to use: cuda, cpu, or empty for auto.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--cut-tolerance-px",
        type=float,
        default=3.0,
        help="Maximum source-image X error for matching a predicted cut to manual markup.",
    )

    parser.add_argument("--cut-threshold", type=float, default=None)
    parser.add_argument("--peak-min-distance", type=int, default=None)
    parser.add_argument("--scale-x", type=float, default=0.0)
    parser.add_argument("--y-pad", type=float, default=0.0)
    parser.add_argument("--x-pad", type=float, default=0.0)
    parser.add_argument("--baseline-crop", action="store_true")
    parser.add_argument("--baseline-top-pad", type=float, default=0.12)
    parser.add_argument("--baseline-bottom-pad", type=float, default=0.18)
    parser.add_argument("--no-baseline-strict-lines", action="store_true")
    parser.add_argument("--baseline-line-pad", type=float, default=0.08)
    parser.add_argument("--baseline-line-pad-px", type=float, default=0.0)
    parser.add_argument("--no-baseline-deskew", action="store_true")
    parser.add_argument("--baseline-max-angle", type=float, default=12.0)
    parser.add_argument("--baseline-detector-checkpoint", default=None)
    parser.add_argument("--baseline-detector-threshold", type=float, default=0.35)

    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument(
        "--optuna-metric",
        default="auto",
        choices=[
            "auto",
            "length_accuracy",
            "average_abs_length_error",
            "total_abs_length_error",
            "normalized_length_error",
            "cut_precision",
            "cut_recall",
            "cut_f1",
        ],
    )
    parser.add_argument("--optuna-cut-threshold-min", type=float, default=0.25)
    parser.add_argument("--optuna-cut-threshold-max", type=float, default=0.85)
    parser.add_argument("--optuna-peak-min-distance-min", type=int, default=1)
    parser.add_argument("--optuna-peak-min-distance-max", type=int, default=4)
    parser.add_argument("--optuna-scale-x-min", type=float, default=-0.25)
    parser.add_argument("--optuna-scale-x-max", type=float, default=0.25)
    parser.add_argument("--optuna-y-pad-min", type=float, default=-0.25)
    parser.add_argument("--optuna-y-pad-max", type=float, default=0.25)
    parser.add_argument("--optuna-tune-baseline-crop", action="store_true")
    parser.add_argument(
        "--optuna-tune-baseline-params",
        action="store_true",
        help=(
            "Tune active baseline crop params. With strict lines this tunes only baseline_line_pad; "
            "with --no-baseline-strict-lines it tunes old bbox top/bottom pads."
        ),
    )
    parser.add_argument(
        "--optuna-tune-baseline-bbox-pads",
        action="store_true",
        help="Explicitly tune old bbox-assisted baseline_top_pad/baseline_bottom_pad.",
    )
    parser.add_argument(
        "--optuna-tune-baseline-line-pad",
        action="store_true",
        help="Explicitly tune strict-lines baseline_line_pad.",
    )
    parser.add_argument(
        "--optuna-tune-baseline-line-pad-px",
        action="store_true",
        help="Explicitly tune absolute strict-lines baseline_line_pad_px.",
    )
    parser.add_argument(
        "--optuna-tune-baseline-max-angle",
        action="store_true",
        help="Explicitly tune baseline_max_angle.",
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
    parser.add_argument("--optuna-baseline-line-pad-min", type=float, default=0.0)
    parser.add_argument("--optuna-baseline-line-pad-max", type=float, default=0.16)
    parser.add_argument("--optuna-baseline-line-pad-px-min", type=float, default=0.0)
    parser.add_argument("--optuna-baseline-line-pad-px-max", type=float, default=6.0)
    parser.add_argument("--optuna-baseline-max-angle-min", type=float, default=4.0)
    parser.add_argument("--optuna-baseline-max-angle-max", type=float, default=18.0)
    parser.add_argument("--optuna-baseline-detector-threshold-min", type=float, default=None)
    parser.add_argument("--optuna-baseline-detector-threshold-max", type=float, default=None)
    parser.add_argument("--optuna-trials-out", default=None)
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument("--optuna-storage", default=None)
    return parser.parse_args()


def _print_inference_command(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    _, jobs = build_rows_and_jobs(Path(args.json), Path(args.images), args.limit)
    image_path = str(jobs[0][1]) if jobs else "<IMAGE_PATH>"
    ocr_checkpoint = args.inference_ocr_checkpoint or "<OCR_CHECKPOINT>"
    command = [
        "python",
        "inference.py",
        "--checkpoint",
        str(ocr_checkpoint),
        "--image",
        image_path,
        "--segmentator-checkpoint",
        str(args.checkpoint),
        "--decode-with-segmentator",
        "--segmentator-cut-threshold",
        str(metrics["cut_threshold"]),
        "--segmentator-peak-min-distance",
        str(metrics["peak_min_distance"]),
        "--segmentator-cut-postprocess",
        str(metrics["cut_postprocess"]),
        "--segmentator-cut-min-width",
        str(metrics["cut_min_width"]),
        "--segmentator-cut-max-width",
        str(metrics["cut_max_width"]),
        "--segmentator-cut-candidate-threshold",
        str(metrics["cut_candidate_threshold"]),
        "--segmentator-cut-smooth-radius",
        str(metrics["cut_smooth_radius"]),
        "--scale-x",
        str(metrics["scale_x"]),
        "--y-pad",
        str(metrics["y_pad"]),
        "--x-pad",
        str(metrics["x_pad"]),
    ]
    if args.device:
        command.extend(["--device", str(args.device)])

    if metrics["baseline_crop"]:
        command.append("--baseline-crop")
        command.extend(
            [
                "--baseline-top-pad",
                str(metrics["baseline_top_pad"]),
                "--baseline-bottom-pad",
                str(metrics["baseline_bottom_pad"]),
                "--baseline-line-pad",
                str(metrics["baseline_line_pad"]),
                "--baseline-line-pad-px",
                str(metrics["baseline_line_pad_px"]),
                "--baseline-max-angle",
                str(metrics["baseline_max_angle"]),
                "--baseline-detector-threshold",
                str(metrics["baseline_detector_threshold"]),
            ]
        )
        if not metrics["baseline_deskew"]:
            command.append("--no-baseline-deskew")
        if not metrics["baseline_strict_lines"]:
            command.append("--no-baseline-strict-lines")
        if metrics.get("baseline_detector_checkpoint"):
            command.extend(
                [
                    "--baseline-detector-checkpoint",
                    str(metrics["baseline_detector_checkpoint"]),
                ]
            )

    print("\n=== Inference command ===")
    if args.inference_ocr_checkpoint is None:
        print("Replace <OCR_CHECKPOINT> with the OCR model path:")
    print(shlex.join(command))


def main() -> None:
    args = parse_args()
    if args.optuna_trials > 0:
        metrics = optimize(
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
            cut_threshold_min=args.optuna_cut_threshold_min,
            cut_threshold_max=args.optuna_cut_threshold_max,
            peak_min_distance_min=args.optuna_peak_min_distance_min,
            peak_min_distance_max=args.optuna_peak_min_distance_max,
            scale_x_min=args.optuna_scale_x_min,
            scale_x_max=args.optuna_scale_x_max,
            y_pad_min=args.optuna_y_pad_min,
            y_pad_max=args.optuna_y_pad_max,
            x_pad=args.x_pad,
            tune_baseline_crop=args.optuna_tune_baseline_crop,
            tune_baseline_params=args.optuna_tune_baseline_params,
            tune_baseline_bbox_pads=args.optuna_tune_baseline_bbox_pads,
            tune_baseline_line_pad=args.optuna_tune_baseline_line_pad,
            tune_baseline_line_pad_px=args.optuna_tune_baseline_line_pad_px,
            tune_baseline_max_angle=args.optuna_tune_baseline_max_angle,
            tune_baseline_deskew=args.optuna_tune_baseline_deskew,
            baseline_crop=args.baseline_crop,
            baseline_top_pad=args.baseline_top_pad,
            baseline_bottom_pad=args.baseline_bottom_pad,
            baseline_strict_lines=not args.no_baseline_strict_lines,
            baseline_line_pad=args.baseline_line_pad,
            baseline_line_pad_px=args.baseline_line_pad_px,
            baseline_deskew=not args.no_baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
            baseline_detector_checkpoint=Path(args.baseline_detector_checkpoint) if args.baseline_detector_checkpoint else None,
            baseline_detector_threshold=args.baseline_detector_threshold,
            baseline_top_pad_min=args.optuna_baseline_top_pad_min,
            baseline_top_pad_max=args.optuna_baseline_top_pad_max,
            baseline_bottom_pad_min=args.optuna_baseline_bottom_pad_min,
            baseline_bottom_pad_max=args.optuna_baseline_bottom_pad_max,
            baseline_line_pad_min=args.optuna_baseline_line_pad_min,
            baseline_line_pad_max=args.optuna_baseline_line_pad_max,
            baseline_line_pad_px_min=args.optuna_baseline_line_pad_px_min,
            baseline_line_pad_px_max=args.optuna_baseline_line_pad_px_max,
            baseline_max_angle_min=args.optuna_baseline_max_angle_min,
            baseline_max_angle_max=args.optuna_baseline_max_angle_max,
            baseline_detector_threshold_min=args.optuna_baseline_detector_threshold_min,
            baseline_detector_threshold_max=args.optuna_baseline_detector_threshold_max,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            cut_tolerance_px=args.cut_tolerance_px,
        )
    else:
        metrics = evaluate(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            log_every=args.log_every,
            cut_threshold=args.cut_threshold,
            peak_min_distance=args.peak_min_distance,
            scale_x=args.scale_x,
            y_pad=args.y_pad,
            x_pad=args.x_pad,
            baseline_crop=args.baseline_crop,
            baseline_top_pad=args.baseline_top_pad,
            baseline_bottom_pad=args.baseline_bottom_pad,
            baseline_strict_lines=not args.no_baseline_strict_lines,
            baseline_line_pad=args.baseline_line_pad,
            baseline_line_pad_px=args.baseline_line_pad_px,
            baseline_deskew=not args.no_baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
            baseline_detector_checkpoint=Path(args.baseline_detector_checkpoint) if args.baseline_detector_checkpoint else None,
            baseline_detector_threshold=args.baseline_detector_threshold,
            cut_tolerance_px=args.cut_tolerance_px,
        )
    _print_inference_command(args, metrics)


if __name__ == "__main__":
    main()
