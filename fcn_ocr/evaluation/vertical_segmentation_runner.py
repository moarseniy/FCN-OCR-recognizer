from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch

from fcn_ocr import VerticalSegmenter
from fcn_ocr.evaluation import (
    CUT_RESULT_FIELDS,
    compute_cut_metrics,
    label_studio_samples,
    load_json_document,
)
from fcn_ocr.evaluation.reporting import (
    write_csv_rows,
)
from tools.annotation.markup import annotated_items, is_manual_markup, safe_image_path


@dataclass(frozen=True)
class SegmentationInference:
    logits: torch.Tensor
    input_shape: tuple[int, ...]
    output_length: int
    source_x_map: np.ndarray


def build_rows_and_jobs(
    json_path: Path,
    images_dir: Path | None,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[int, Path]]]:
    document = load_json_document(json_path)

    if is_manual_markup(document):
        images_root = (
            images_dir.expanduser().resolve()
            if images_dir is not None
            else Path(document["images_root"]).expanduser().resolve()
        )
        return build_manual_rows_and_jobs(document, images_root, limit)

    if images_dir is None:
        raise ValueError(
            "--images is required for Label Studio JSON; "
            "manual markup JSON can use its stored images_root"
        )
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[int, Path]] = []
    for sample in label_studio_samples(document, images_dir.expanduser().resolve(), limit):
        row = {
            "task_id": sample.task_id,
            "image": sample.image_name,
            "gt": sample.text,
            "gt_len": len(sample.text),
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
        jobs.append((len(rows), sample.image_path))
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
    vertical_segmentation: VerticalSegmenter,
    jobs: list[tuple[int, Path]],
    batch_size: int,
    log_every: int,
    image_loader: Callable[[Path], Image.Image] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    inferences, errors = infer_segment_images(
        vertical_segmentation,
        jobs,
        batch_size=batch_size,
        log_every=log_every,
        image_loader=image_loader,
    )
    predictions, postprocess_errors = postprocess_segment_inferences(
        vertical_segmentation,
        inferences,
    )
    errors.update(postprocess_errors)
    return predictions, errors


def infer_segment_images(
    vertical_segmentation: VerticalSegmenter,
    jobs: list[tuple[int, Path]],
    batch_size: int,
    log_every: int,
    image_loader: Callable[[Path], Image.Image] | None = None,
) -> tuple[dict[int, SegmentationInference], dict[int, str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    inferences: dict[int, SegmentationInference] = {}
    errors: dict[int, str] = {}
    processed = 0
    started_at = time.perf_counter()

    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        try:
            batch_inferences = infer_segment_batch(
                vertical_segmentation,
                batch_jobs,
                image_loader=image_loader,
            )
            inferences.update(batch_inferences)
            for row_index, _ in batch_jobs:
                errors[row_index] = ""
        except Exception as batch_error:
            for row_index, path in batch_jobs:
                try:
                    inference = infer_segment_batch(
                        vertical_segmentation,
                        [(row_index, path)],
                        image_loader=image_loader,
                    )[row_index]
                    inferences[row_index] = inference
                    errors[row_index] = ""
                except Exception as image_error:
                    errors[row_index] = f"batch_error={batch_error!r}; image_error={image_error!r}"

        processed += len(batch_jobs)
        if log_every > 0 and (processed == len(jobs) or processed % log_every == 0):
            elapsed = max(1e-9, time.perf_counter() - started_at)
            print(f"Inferred {processed}/{len(jobs)} images ({processed / elapsed:.2f} img/s)")

    return inferences, errors


def segment_batch(
    vertical_segmentation: VerticalSegmenter,
    batch_jobs: list[tuple[int, Path]],
    image_loader: Callable[[Path], Image.Image] | None = None,
) -> dict[int, dict[str, Any]]:
    inferences = infer_segment_batch(
        vertical_segmentation,
        batch_jobs,
        image_loader=image_loader,
    )
    predictions, errors = postprocess_segment_inferences(
        vertical_segmentation,
        inferences,
    )
    failures = [error for error in errors.values() if error]
    if failures:
        first_error = failures[0]
        raise RuntimeError(first_error)
    return predictions


@torch.no_grad()
def infer_segment_batch(
    vertical_segmentation: VerticalSegmenter,
    batch_jobs: list[tuple[int, Path]],
    image_loader: Callable[[Path], Image.Image] | None = None,
) -> dict[int, SegmentationInference]:
    tensors: list[torch.Tensor] = []
    output_lengths: list[int] = []
    source_x_maps: list[np.ndarray] = []
    max_width = 0

    for _, path in batch_jobs:
        if image_loader is None:
            with Image.open(path) as image:
                tensor, source_x_map = vertical_segmentation.preprocess_pil_with_source_x(
                    image
                )
        else:
            tensor, source_x_map = vertical_segmentation.preprocess_pil_with_source_x(
                image_loader(path)
            )
        tensors.append(tensor)
        source_x_maps.append(source_x_map)
        max_width = max(max_width, tensor.size(2))
        output_lengths.append(vertical_segmentation.output_width_for_input_width(tensor.size(2)))

    if not tensors:
        return {}

    batch = torch.ones(
        (len(tensors), vertical_segmentation.in_channels, vertical_segmentation.image_height, max_width),
        dtype=tensors[0].dtype,
        device=vertical_segmentation.device,
    )
    for batch_index, tensor in enumerate(tensors):
        batch[batch_index, :, :, : tensor.size(2)] = tensor

    logits = vertical_segmentation.model(batch)

    inferences: dict[int, SegmentationInference] = {}
    for batch_index, ((row_index, _), output_length) in enumerate(zip(batch_jobs, output_lengths)):
        inferences[row_index] = SegmentationInference(
            logits=logits[batch_index : batch_index + 1, :, :output_length].detach().cpu(),
            input_shape=(
                1,
                vertical_segmentation.in_channels,
                vertical_segmentation.image_height,
                tensors[batch_index].size(2),
            ),
            output_length=output_length,
            source_x_map=source_x_maps[batch_index],
        )
    return inferences


def postprocess_segment_inferences(
    vertical_segmentation: VerticalSegmenter,
    inferences: dict[int, SegmentationInference],
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    predictions: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}
    for row_index, inference in inferences.items():
        try:
            result = vertical_segmentation.analyze_segmentation_logits(
                inference.logits,
                input_shape=inference.input_shape,
            )
            cut_count = segment_count(result)
            predicted_source_cuts = cut_positions_to_source(
                result.cut_positions or [],
                output_width=inference.output_length,
                source_x_map=inference.source_x_map,
            )
            predictions[row_index] = {
                "pred_len": max(0, cut_count - 1) if result.raw_indices else 0,
                "cut_count": cut_count,
                "cuts": cuts_text(result),
                "pred_cuts": predicted_source_cuts,
            }
            errors[row_index] = ""
        except Exception as error:
            errors[row_index] = repr(error)
    return predictions, errors


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


def print_metrics(metrics: dict[str, Any], output_csv: Path | None = None) -> None:
    print("=== Vertical segmentation length evaluation ===")
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
    print(f"Task:                       {metrics['task']}")
    print(f"cut_threshold:              {metrics['cut_threshold']:.5f}")
    print(f"cut_min_width:              {metrics['cut_min_width']}")
    print(f"cut_max_width:              {metrics['cut_max_width']}")
    print(f"cut_smooth_radius:          {metrics['cut_smooth_radius']}")
    print(f"scale_x:                    {metrics['scale_x']:+.5f}")
    print(f"y_pad:                      {metrics['y_pad']:+.5f}")
    print(f"x_pad:                      {metrics['x_pad']:.5f}")
    print(f"baseline_crop:              {metrics['baseline_crop']}")
    print(f"baseline_line_pad:          {metrics['baseline_line_pad']:.5f}")
    print(f"baseline_line_pad_px:       {metrics['baseline_line_pad_px']:.2f}")
    print(f"baseline_deskew:            {metrics['baseline_deskew']}")
    print(f"baseline_max_angle:         {metrics['baseline_max_angle']:.5f}")
    if metrics.get("baseline_detector_checkpoint"):
        print(f"baseline_detector:          {metrics['baseline_detector_checkpoint']}")
        print(f"baseline_detector_thr:      {metrics['baseline_detector_threshold']:.5f}")
    if output_csv is not None:
        print(f"CSV saved to:               {output_csv}")


def configure_vertical_segmentation(
    vertical_segmentation: VerticalSegmenter,
    cut_threshold: float | None,
    cut_min_width: int | None,
    cut_max_width: int | None,
    cut_smooth_radius: int | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    baseline_crop: bool,
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
    if baseline_line_pad < 0.0:
        raise ValueError("baseline_line_pad must be >= 0")
    if baseline_line_pad_px < 0.0:
        raise ValueError("baseline_line_pad_px must be >= 0")
    if baseline_max_angle <= 0.0:
        raise ValueError("baseline_max_angle must be > 0")
    if not 0.0 < baseline_detector_threshold < 1.0:
        raise ValueError("baseline_detector_threshold must be between 0 and 1")

    vertical_segmentation.cut_threshold = vertical_segmentation._resolve_cut_threshold(cut_threshold)
    vertical_segmentation.cut_min_width = vertical_segmentation._resolve_non_negative_int(
        cut_min_width,
        "cut_min_width",
        default=vertical_segmentation.cut_min_width,
        min_value=1,
    )
    vertical_segmentation.cut_max_width = vertical_segmentation._resolve_non_negative_int(
        cut_max_width,
        "cut_max_width",
        default=vertical_segmentation.cut_max_width,
        min_value=0,
    )
    vertical_segmentation.cut_smooth_radius = vertical_segmentation._resolve_non_negative_int(
        cut_smooth_radius,
        "cut_smooth_radius",
        default=vertical_segmentation.cut_smooth_radius,
        min_value=0,
    )
    vertical_segmentation.scale_x = float(scale_x)
    vertical_segmentation.y_pad = float(y_pad)
    vertical_segmentation.x_pad = float(x_pad)
    vertical_segmentation.baseline_crop = bool(baseline_crop)
    vertical_segmentation.baseline_line_pad = float(baseline_line_pad)
    vertical_segmentation.baseline_line_pad_px = float(baseline_line_pad_px)
    vertical_segmentation.baseline_deskew = bool(baseline_deskew)
    vertical_segmentation.baseline_max_angle = float(baseline_max_angle)
    vertical_segmentation.baseline_detector_threshold = float(baseline_detector_threshold)


def evaluate_with_vertical_segmentation(
    base_rows: list[dict[str, Any]],
    jobs: list[tuple[int, Path]],
    vertical_segmentation: VerticalSegmenter,
    output_csv: Path | None,
    batch_size: int,
    log_every: int,
    verbose: bool,
    cut_tolerance_px: float,
    image_loader: Callable[[Path], Image.Image] | None = None,
    inference_cache: dict[int, SegmentationInference] | None = None,
    inference_errors: dict[int, str] | None = None,
) -> dict[str, Any]:
    rows = deepcopy(base_rows)
    started_at = time.perf_counter()
    if inference_cache is None:
        predictions, errors = segment_images(
            vertical_segmentation,
            jobs,
            batch_size=batch_size,
            log_every=log_every,
            image_loader=image_loader,
        )
    else:
        predictions, errors = postprocess_segment_inferences(
            vertical_segmentation,
            inference_cache,
        )
        for row_index, error in (inference_errors or {}).items():
            if error:
                errors[row_index] = error
    elapsed = time.perf_counter() - started_at

    for row_index, prediction in predictions.items():
        rows[row_index].update(prediction)
    for row_index, error in errors.items():
        rows[row_index]["error"] = error

    metrics = compute_cut_metrics(rows, elapsed, cut_tolerance_px=cut_tolerance_px)
    metrics["task"] = vertical_segmentation.task
    metrics["cut_threshold"] = float(vertical_segmentation.cut_threshold)
    metrics["cut_min_width"] = int(vertical_segmentation.cut_min_width)
    metrics["cut_max_width"] = int(vertical_segmentation.cut_max_width)
    metrics["cut_smooth_radius"] = int(vertical_segmentation.cut_smooth_radius)
    metrics["scale_x"] = float(vertical_segmentation.scale_x)
    metrics["y_pad"] = float(vertical_segmentation.y_pad)
    metrics["x_pad"] = float(vertical_segmentation.x_pad)
    metrics["baseline_crop"] = bool(vertical_segmentation.baseline_crop)
    metrics["baseline_line_pad"] = float(vertical_segmentation.baseline_line_pad)
    metrics["baseline_line_pad_px"] = float(vertical_segmentation.baseline_line_pad_px)
    metrics["baseline_deskew"] = bool(vertical_segmentation.baseline_deskew)
    metrics["baseline_max_angle"] = float(vertical_segmentation.baseline_max_angle)
    metrics["baseline_detector_checkpoint"] = (
        str(vertical_segmentation.baseline_detector_checkpoint) if vertical_segmentation.baseline_detector_checkpoint else ""
    )
    metrics["baseline_detector_threshold"] = float(vertical_segmentation.baseline_detector_threshold)

    if output_csv is not None:
        write_csv_rows(rows, output_csv, CUT_RESULT_FIELDS)
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
    cut_min_width: int | None,
    cut_max_width: int | None,
    cut_smooth_radius: int | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    baseline_crop: bool,
    baseline_line_pad: float,
    baseline_line_pad_px: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
    baseline_detector_checkpoint: Path | None,
    baseline_detector_threshold: float,
    cut_tolerance_px: float,
    image_loader: Callable[[Path], Image.Image] | None = None,
) -> dict[str, Any]:
    vertical_segmentation = VerticalSegmenter(
        checkpoint_path,
        device=device,
        verbose=False,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        baseline_crop=baseline_crop,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=baseline_detector_threshold,
        cut_threshold=cut_threshold,
        cut_min_width=cut_min_width,
        cut_max_width=cut_max_width,
        cut_smooth_radius=cut_smooth_radius,
    )
    configure_vertical_segmentation(
        vertical_segmentation,
        cut_threshold=cut_threshold,
        cut_min_width=cut_min_width,
        cut_max_width=cut_max_width,
        cut_smooth_radius=cut_smooth_radius,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        baseline_crop=baseline_crop,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_detector_threshold=baseline_detector_threshold,
    )
    if verbose:
        vertical_segmentation.print_summary()
    return evaluate_with_vertical_segmentation(
        base_rows=base_rows,
        jobs=jobs,
        vertical_segmentation=vertical_segmentation,
        output_csv=output_csv,
        batch_size=batch_size,
        log_every=log_every,
        verbose=verbose,
        cut_tolerance_px=cut_tolerance_px,
        image_loader=image_loader,
    )


def evaluate(
    json_path: Path,
    images_dir: Path | None,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    batch_size: int,
    limit: int | None,
    log_every: int,
    cut_threshold: float | None,
    cut_min_width: int | None,
    cut_max_width: int | None,
    cut_smooth_radius: int | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    baseline_crop: bool,
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
        cut_min_width=cut_min_width,
        cut_max_width=cut_max_width,
        cut_smooth_radius=cut_smooth_radius,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        baseline_crop=baseline_crop,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=baseline_detector_threshold,
        cut_tolerance_px=cut_tolerance_px,
    )
