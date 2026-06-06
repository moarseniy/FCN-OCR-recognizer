from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import shlex
import time
from pathlib import Path
from typing import Any

from PIL import Image

from fcn_ocr import TextRecognizer, VerticalSegmentator


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n < m:
        return levenshtein(b, a)

    previous = list(range(m + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[m]


def char_accuracy(gt: str, pred: str) -> float:
    if not gt:
        return 1.0 if not pred else 0.0

    distance = levenshtein(gt, pred)
    return max(0.0, 1.0 - distance / len(gt))


def exact_match(gt: str, pred: str) -> bool:
    return gt == pred


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


def recognize_images(
    recognizer: TextRecognizer,
    jobs: list[tuple[int, Path]],
    batch_size: int,
    log_every: int,
) -> tuple[dict[int, str], dict[int, str]]:
    predictions: dict[int, str] = {}
    errors: dict[int, str] = {}
    processed = 0
    started_at = time.perf_counter()

    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        paths = [path for _, path in batch_jobs]

        try:
            batch_results = recognizer.recognize_paths_text(paths, batch_size=len(paths))
            for (row_index, _), (_, text) in zip(batch_jobs, batch_results):
                predictions[row_index] = text.strip()
                errors[row_index] = ""
        except Exception as batch_error:
            for row_index, path in batch_jobs:
                try:
                    text, _ = recognizer.recognize(path)
                    predictions[row_index] = text.strip()
                    errors[row_index] = ""
                except Exception as image_error:
                    predictions[row_index] = ""
                    errors[row_index] = f"batch_error={batch_error!r}; image_error={image_error!r}"

        processed += len(batch_jobs)
        if log_every > 0 and (processed == len(jobs) or processed % log_every == 0):
            elapsed = max(1e-9, time.perf_counter() - started_at)
            speed = processed / elapsed
            print(f"Recognized {processed}/{len(jobs)} images ({speed:.2f} img/s)")

    return predictions, errors


def recognize_images_with_segmentator(
    recognizer: TextRecognizer,
    segmentator: VerticalSegmentator,
    jobs: list[tuple[int, Path]],
    log_every: int,
    segmentator_decode_top_k: int,
    segmentator_edge_trim: bool,
    segmentator_edge_min_ink_ratio: float,
    segmentator_edge_min_pixel_density: float,
    segmentator_edge_min_width: int,
    segmentator_boundary_cuts: str,
    segmentator_boundary_cut_max_edge_ratio: float,
    segmentator_decode_center_fraction: float,
    segmentator_decode_min_score_width: int,
) -> tuple[dict[int, str], dict[int, str]]:
    predictions: dict[int, str] = {}
    errors: dict[int, str] = {}
    processed = 0
    started_at = time.perf_counter()

    for row_index, path in jobs:
        try:
            with Image.open(path) as image_file:
                source_image = image_file.convert("RGB")

            input_tensor = recognizer.preprocess_pil(source_image)
            ocr_logits, _ = recognizer.logits_from_tensor(input_tensor)
            segmentator_input_tensor = segmentator.preprocess_pil(source_image)
            segmentation_result = segmentator.segment_tensor_debug(segmentator_input_tensor)

            text_bounds_info = recognizer.text_x_bounds_from_tensor(input_tensor)
            if text_bounds_info["ok"]:
                text_x_bounds = (int(text_bounds_info["left"]), int(text_bounds_info["right"]))
            else:
                text_x_bounds = None

            cut_decoding_result = recognizer.decode_legacy_with_cuts(
                ocr_logits,
                segmentation_result,
                input_width=int(input_tensor.shape[-1]),
                top_k=segmentator_decode_top_k,
                text_x_bounds=text_x_bounds,
                input_tensor=input_tensor,
                trim_empty_edges=segmentator_edge_trim,
                edge_min_ink_ratio=segmentator_edge_min_ink_ratio,
                edge_min_pixel_density=segmentator_edge_min_pixel_density,
                edge_min_width=segmentator_edge_min_width,
                boundary_cuts=segmentator_boundary_cuts,
                boundary_cut_max_edge_ratio=segmentator_boundary_cut_max_edge_ratio,
                center_fraction=segmentator_decode_center_fraction,
                min_score_width=segmentator_decode_min_score_width,
            )
            predictions[row_index] = cut_decoding_result.text.strip()
            errors[row_index] = ""
        except Exception as image_error:
            predictions[row_index] = ""
            errors[row_index] = repr(image_error)

        processed += 1
        if log_every > 0 and (processed == len(jobs) or processed % log_every == 0):
            elapsed = max(1e-9, time.perf_counter() - started_at)
            speed = processed / elapsed
            print(f"Recognized {processed}/{len(jobs)} images with segmentator ({speed:.2f} img/s)")

    return predictions, errors


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
        row = {
            "task_id": task.get("id"),
            "image": image_name,
            "gt": get_gt_text(task),
            "pred": "",
            "exact_match": 0,
            "char_accuracy": 0.0,
            "levenshtein": 0,
            "gt_len": 0,
            "pred_len": 0,
            "error": "",
        }

        if not image_path.exists():
            continue
        jobs.append((len(rows), image_path))
        rows.append(row)

    return rows, jobs


def compute_metrics(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    total = 0
    exact_ok = 0
    total_lev = 0
    total_gt_chars = 0
    total_char_acc = 0.0

    for row in rows:
        gt = row["gt"]
        pred = row["pred"]
        lev = levenshtein(gt, pred)
        c_acc = char_accuracy(gt, pred)
        is_exact = exact_match(gt, pred)

        row["exact_match"] = int(is_exact)
        row["char_accuracy"] = round(c_acc, 6)
        row["levenshtein"] = lev
        row["gt_len"] = len(gt)
        row["pred_len"] = len(pred)

        total += 1
        exact_ok += int(is_exact)
        total_lev += lev
        total_gt_chars += len(gt)
        total_char_acc += c_acc

    recognized = sum(1 for row in rows if not row["error"])
    return {
        "total_samples": total,
        "recognized_samples": recognized,
        "exact_matches": exact_ok,
        "line_accuracy": exact_ok / total if total else 0.0,
        "average_char_accuracy": total_char_acc / total if total else 0.0,
        "global_char_accuracy": max(0.0, 1.0 - total_lev / total_gt_chars) if total_gt_chars else 0.0,
        "average_levenshtein": total_lev / total if total else 0.0,
        "total_levenshtein": total_lev,
        "elapsed": elapsed,
        "speed": recognized / elapsed if elapsed > 0 else 0.0,
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
                "pred",
                "exact_match",
                "char_accuracy",
                "levenshtein",
                "gt_len",
                "pred_len",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(metrics: dict[str, Any], output_csv: Path | None = None) -> None:
    print("=== OCR evaluation ===")
    print(f"Total samples:              {metrics['total_samples']}")
    print(f"Recognized samples:         {metrics['recognized_samples']}")
    print(f"Exact line matches:         {metrics['exact_matches']}")
    print(f"Line accuracy:              {metrics['line_accuracy']:.4f}")
    print(f"Average char accuracy:      {metrics['average_char_accuracy']:.4f}")
    print(f"Global char accuracy:       {metrics['global_char_accuracy']:.4f}")
    print(f"Average Levenshtein:        {metrics['average_levenshtein']:.4f}")
    print(f"Total Levenshtein:          {metrics['total_levenshtein']}")
    print(f"Elapsed:                    {metrics['elapsed']:.2f}s")
    print(f"Speed:                      {metrics['speed']:.2f} img/s")
    print(f"scale_x:                    {metrics['scale_x']:+.5f}")
    print(f"y_pad:                      {metrics['y_pad']:+.5f}")
    print(f"x_pad:                      {metrics['x_pad']:.5f}")
    print(f"Baseline crop:              {metrics['baseline_crop']}")
    print(f"Baseline strict lines:      {metrics['baseline_strict_lines']}")
    print(f"Baseline line pad:          {metrics['baseline_line_pad']:.5f}")
    print(f"Baseline line pad px:       {metrics['baseline_line_pad_px']:.2f}")
    if metrics.get("baseline_detector_checkpoint"):
        print(f"Baseline detector:          {metrics['baseline_detector_checkpoint']}")
        print(f"Baseline detector thr:      {metrics['baseline_detector_threshold']:.5f}")
    print(f"Decode with segmentator:    {metrics['decode_with_segmentator']}")
    if metrics.get("segmentator_checkpoint"):
        print(f"Segmentator checkpoint:     {metrics['segmentator_checkpoint']}")
        print(f"Segmentator cut threshold:  {metrics['segmentator_cut_threshold']:.5f}")
        print(f"Segmentator postprocess:    {metrics['segmentator_cut_postprocess']}")
        print(f"Segmentator cut min width:  {metrics['segmentator_cut_min_width']}")
        print(f"Segmentator cut max width:  {metrics['segmentator_cut_max_width']}")
        print(f"Segmentator candidate thr:  {metrics['segmentator_cut_candidate_threshold']:.5f}")
        print(f"Segmentator smooth radius:  {metrics['segmentator_cut_smooth_radius']}")
        print(f"Decode center fraction:     {metrics['segmentator_decode_center_fraction']:.5f}")
        print(f"Decode min score width:     {metrics['segmentator_decode_min_score_width']}")
    if output_csv is not None:
        print(f"CSV saved to:               {output_csv}")


def evaluate_prepared(
    base_rows: list[dict[str, Any]],
    jobs: list[tuple[int, Path]],
    checkpoint_path: Path,
    output_csv: Path | None,
    device: str | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    batch_size: int,
    log_every: int,
    verbose: bool,
    baseline_crop: bool = False,
    baseline_top_pad: float = 0.12,
    baseline_bottom_pad: float = 0.18,
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
    baseline_strict_lines: bool = True,
    baseline_line_pad: float = 0.08,
    baseline_line_pad_px: float = 0.0,
    baseline_detector_checkpoint: Path | None = None,
    baseline_detector_threshold: float = 0.35,
    segmentator_checkpoint: Path | None = None,
    decode_with_segmentator: bool = False,
    segmentator_cut_threshold: float | None = None,
    segmentator_peak_min_distance: int | None = None,
    segmentator_cut_postprocess: str | None = None,
    segmentator_cut_min_width: int | None = None,
    segmentator_cut_max_width: int | None = None,
    segmentator_cut_candidate_threshold: float | None = None,
    segmentator_cut_smooth_radius: int | None = None,
    segmentator_decode_top_k: int = 8,
    segmentator_edge_trim: bool = True,
    segmentator_edge_min_ink_ratio: float = 0.035,
    segmentator_edge_min_pixel_density: float = 0.003,
    segmentator_edge_min_width: int = 2,
    segmentator_boundary_cuts: str = "auto",
    segmentator_boundary_cut_max_edge_ratio: float = 0.45,
    segmentator_decode_center_fraction: float = 0.6,
    segmentator_decode_min_score_width: int = 1,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if decode_with_segmentator and segmentator_checkpoint is None:
        raise ValueError("decode_with_segmentator requires segmentator_checkpoint")

    rows = deepcopy(base_rows)
    started_at = time.perf_counter()
    recognizer = TextRecognizer(
        checkpoint_path,
        device=device,
        verbose=verbose,
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
    )

    segmentator = None
    if decode_with_segmentator:
        segmentator = VerticalSegmentator(
            segmentator_checkpoint,
            device=device,
            verbose=verbose,
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
            cut_threshold=segmentator_cut_threshold,
            peak_min_distance=segmentator_peak_min_distance,
            cut_postprocess=segmentator_cut_postprocess,
            cut_min_width=segmentator_cut_min_width,
            cut_max_width=segmentator_cut_max_width,
            cut_candidate_threshold=segmentator_cut_candidate_threshold,
            cut_smooth_radius=segmentator_cut_smooth_radius,
        )
        predictions, errors = recognize_images_with_segmentator(
            recognizer,
            segmentator,
            jobs,
            log_every=log_every,
            segmentator_decode_top_k=segmentator_decode_top_k,
            segmentator_edge_trim=segmentator_edge_trim,
            segmentator_edge_min_ink_ratio=segmentator_edge_min_ink_ratio,
            segmentator_edge_min_pixel_density=segmentator_edge_min_pixel_density,
            segmentator_edge_min_width=segmentator_edge_min_width,
            segmentator_boundary_cuts=segmentator_boundary_cuts,
            segmentator_boundary_cut_max_edge_ratio=segmentator_boundary_cut_max_edge_ratio,
            segmentator_decode_center_fraction=segmentator_decode_center_fraction,
            segmentator_decode_min_score_width=segmentator_decode_min_score_width,
        )
    else:
        predictions, errors = recognize_images(recognizer, jobs, batch_size=batch_size, log_every=log_every)

    elapsed = time.perf_counter() - started_at

    for row_index, prediction in predictions.items():
        rows[row_index]["pred"] = prediction
    for row_index, error in errors.items():
        rows[row_index]["error"] = error

    metrics = compute_metrics(rows, elapsed)
    metrics["scale_x"] = float(scale_x)
    metrics["y_pad"] = float(y_pad)
    metrics["x_pad"] = float(x_pad)
    metrics["baseline_crop"] = bool(baseline_crop)
    metrics["baseline_strict_lines"] = bool(baseline_strict_lines)
    metrics["baseline_line_pad"] = float(baseline_line_pad)
    metrics["baseline_line_pad_px"] = float(baseline_line_pad_px)
    metrics["baseline_detector_checkpoint"] = str(baseline_detector_checkpoint) if baseline_detector_checkpoint else ""
    metrics["baseline_detector_threshold"] = float(baseline_detector_threshold)
    metrics["decode_with_segmentator"] = bool(decode_with_segmentator)
    metrics["segmentator_checkpoint"] = str(segmentator_checkpoint) if segmentator_checkpoint else ""
    metrics["segmentator_decode_top_k"] = int(segmentator_decode_top_k)
    metrics["segmentator_edge_trim"] = bool(segmentator_edge_trim)
    metrics["segmentator_edge_min_ink_ratio"] = float(segmentator_edge_min_ink_ratio)
    metrics["segmentator_edge_min_pixel_density"] = float(segmentator_edge_min_pixel_density)
    metrics["segmentator_edge_min_width"] = int(segmentator_edge_min_width)
    metrics["segmentator_boundary_cuts"] = str(segmentator_boundary_cuts)
    metrics["segmentator_boundary_cut_max_edge_ratio"] = float(segmentator_boundary_cut_max_edge_ratio)
    metrics["segmentator_decode_center_fraction"] = float(segmentator_decode_center_fraction)
    metrics["segmentator_decode_min_score_width"] = int(segmentator_decode_min_score_width)
    if segmentator is not None:
        metrics["segmentator_cut_threshold"] = float(segmentator.cut_threshold)
        metrics["segmentator_peak_min_distance"] = int(segmentator.peak_min_distance)
        metrics["segmentator_cut_postprocess"] = str(segmentator.cut_postprocess)
        metrics["segmentator_cut_min_width"] = int(segmentator.cut_min_width)
        metrics["segmentator_cut_max_width"] = int(segmentator.cut_max_width)
        metrics["segmentator_cut_candidate_threshold"] = float(segmentator.cut_candidate_threshold)
        metrics["segmentator_cut_smooth_radius"] = int(segmentator.cut_smooth_radius)
    else:
        metrics["segmentator_cut_threshold"] = float(segmentator_cut_threshold or 0.0)
        metrics["segmentator_peak_min_distance"] = int(segmentator_peak_min_distance or 0)
        metrics["segmentator_cut_postprocess"] = str(segmentator_cut_postprocess or "")
        metrics["segmentator_cut_min_width"] = int(segmentator_cut_min_width or 0)
        metrics["segmentator_cut_max_width"] = int(segmentator_cut_max_width or 0)
        metrics["segmentator_cut_candidate_threshold"] = float(segmentator_cut_candidate_threshold or 0.0)
        metrics["segmentator_cut_smooth_radius"] = int(segmentator_cut_smooth_radius or 0)

    if output_csv is not None:
        write_rows_csv(rows, output_csv)

    if verbose:
        print_metrics(metrics, output_csv)

    return metrics


def _trial_params_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    names = [
        "scale_x",
        "y_pad",
        "x_pad",
        "baseline_detector_threshold",
        "baseline_line_pad",
        "baseline_line_pad_px",
        "segmentator_cut_threshold",
        "segmentator_peak_min_distance",
        "segmentator_cut_postprocess",
        "segmentator_cut_min_width",
        "segmentator_cut_max_width",
        "segmentator_cut_candidate_threshold",
        "segmentator_cut_smooth_radius",
        "segmentator_boundary_cuts",
        "segmentator_boundary_cut_max_edge_ratio",
        "segmentator_decode_center_fraction",
        "segmentator_decode_min_score_width",
        "segmentator_edge_min_width",
    ]
    return {name: metrics[name] for name in names if name in metrics}


def append_trial_log(
    path: Path,
    trial_number: int,
    metrics: dict[str, Any],
    metric_name: str,
    trial_params: dict[str, Any] | None = None,
) -> None:
    is_new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        if is_new_file:
            file.write(
                "trial\tparams\tmetric\tmetric_value\tline_accuracy\taverage_char_accuracy\t"
                "global_char_accuracy\taverage_levenshtein\ttotal_levenshtein\tspeed\n"
            )
        params = json.dumps(
            trial_params if trial_params is not None else _trial_params_snapshot(metrics),
            ensure_ascii=False,
            sort_keys=True,
        )
        file.write(
            f"{trial_number}\t{params}\t{metric_name}\t{metrics[metric_name]:.8f}\t"
            f"{metrics['line_accuracy']:.8f}\t{metrics['average_char_accuracy']:.8f}\t"
            f"{metrics['global_char_accuracy']:.8f}\t{metrics['average_levenshtein']:.8f}\t"
            f"{metrics['total_levenshtein']}\t{metrics['speed']:.6f}\n"
        )


def _suggest_float_or_fixed(
    trial,
    name: str,
    fixed: float | None,
    min_value: float | None,
    max_value: float | None,
) -> float | None:
    if min_value is None and max_value is None:
        return fixed
    if min_value is None or max_value is None:
        raise ValueError(f"{name} tuning requires both min and max")
    return float(trial.suggest_float(name, float(min_value), float(max_value)))


def _suggest_int_or_fixed(
    trial,
    name: str,
    fixed: int | None,
    min_value: int | None,
    max_value: int | None,
) -> int | None:
    if min_value is None and max_value is None:
        return fixed
    if min_value is None or max_value is None:
        raise ValueError(f"{name} tuning requires both min and max")
    return int(trial.suggest_int(name, int(min_value), int(max_value)))


def _best_or_fixed(best_params: dict[str, Any], name: str, fixed: Any) -> Any:
    return best_params[name] if name in best_params else fixed


def optimize_preprocess(
    json_path: Path,
    images_dir: Path,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    batch_size: int,
    limit: int | None,
    trials: int,
    scale_x_min: float,
    scale_x_max: float,
    y_pad_min: float,
    y_pad_max: float,
    x_pad: float,
    metric_name: str,
    log_every: int,
    trials_output: Path | None,
    study_name: str | None = None,
    storage: str | None = None,
    optuna_tune_baseline_line_pad: bool = False,
    optuna_tune_baseline_line_pad_px: bool = False,
    baseline_crop: bool = False,
    baseline_top_pad: float = 0.12,
    baseline_bottom_pad: float = 0.18,
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
    baseline_strict_lines: bool = True,
    baseline_line_pad: float = 0.08,
    baseline_line_pad_px: float = 0.0,
    baseline_detector_checkpoint: Path | None = None,
    baseline_detector_threshold: float = 0.35,
    segmentator_checkpoint: Path | None = None,
    decode_with_segmentator: bool = False,
    segmentator_cut_threshold: float | None = None,
    segmentator_peak_min_distance: int | None = None,
    segmentator_cut_postprocess: str | None = None,
    segmentator_cut_min_width: int | None = None,
    segmentator_cut_max_width: int | None = None,
    segmentator_cut_candidate_threshold: float | None = None,
    segmentator_cut_smooth_radius: int | None = None,
    segmentator_decode_top_k: int = 8,
    segmentator_edge_trim: bool = True,
    segmentator_edge_min_ink_ratio: float = 0.035,
    segmentator_edge_min_pixel_density: float = 0.003,
    segmentator_edge_min_width: int = 2,
    segmentator_boundary_cuts: str = "auto",
    segmentator_boundary_cut_max_edge_ratio: float = 0.45,
    segmentator_decode_center_fraction: float = 0.6,
    segmentator_decode_min_score_width: int = 1,
    x_pad_min: float | None = None,
    x_pad_max: float | None = None,
    baseline_detector_threshold_min: float | None = None,
    baseline_detector_threshold_max: float | None = None,
    baseline_line_pad_min: float | None = None,
    baseline_line_pad_max: float | None = None,
    baseline_line_pad_px_min: float | None = None,
    baseline_line_pad_px_max: float | None = None,
    segmentator_cut_threshold_min: float | None = None,
    segmentator_cut_threshold_max: float | None = None,
    segmentator_peak_min_distance_min: int | None = None,
    segmentator_peak_min_distance_max: int | None = None,
    segmentator_cut_min_width_min: int | None = None,
    segmentator_cut_min_width_max: int | None = None,
    segmentator_cut_max_width_min: int | None = None,
    segmentator_cut_max_width_max: int | None = None,
    segmentator_cut_candidate_threshold_min: float | None = None,
    segmentator_cut_candidate_threshold_max: float | None = None,
    segmentator_cut_smooth_radius_min: int | None = None,
    segmentator_cut_smooth_radius_max: int | None = None,
    segmentator_boundary_cut_max_edge_ratio_min: float | None = None,
    segmentator_boundary_cut_max_edge_ratio_max: float | None = None,
    segmentator_decode_center_fraction_min: float | None = None,
    segmentator_decode_center_fraction_max: float | None = None,
    segmentator_decode_min_score_width_min: int | None = None,
    segmentator_decode_min_score_width_max: int | None = None,
    segmentator_edge_min_width_min: int | None = None,
    segmentator_edge_min_width_max: int | None = None,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna") from exc
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    if trials < 1:
        raise ValueError("trials must be >= 1")

    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    direction = "minimize" if metric_name in {"average_levenshtein", "total_levenshtein"} else "maximize"
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )

    def objective(trial) -> float:
        tune_active_baseline_line_pad = bool(baseline_crop) and bool(baseline_strict_lines) and (
            bool(optuna_tune_baseline_line_pad)
            or baseline_line_pad_min is not None
            or baseline_line_pad_max is not None
        )
        tune_active_baseline_line_pad_px = (
            bool(baseline_crop)
            and bool(baseline_strict_lines)
            and bool(optuna_tune_baseline_line_pad_px)
        )
        tune_active_baseline_detector = bool(baseline_crop) and baseline_detector_checkpoint is not None
        tune_active_segmentator = bool(decode_with_segmentator) and segmentator_checkpoint is not None

        scale_x = trial.suggest_float("scale_x", scale_x_min, scale_x_max)
        y_pad = trial.suggest_float("y_pad", y_pad_min, y_pad_max)
        current_x_pad = _suggest_float_or_fixed(trial, "x_pad", x_pad, x_pad_min, x_pad_max)
        current_baseline_detector_threshold = (
            _suggest_float_or_fixed(
                trial,
                "baseline_detector_threshold",
                baseline_detector_threshold,
                baseline_detector_threshold_min,
                baseline_detector_threshold_max,
            )
            if tune_active_baseline_detector
            else baseline_detector_threshold
        )
        current_baseline_line_pad = (
            _suggest_float_or_fixed(
                trial,
                "baseline_line_pad",
                baseline_line_pad,
                baseline_line_pad_min,
                baseline_line_pad_max,
            )
            if tune_active_baseline_line_pad
            else baseline_line_pad
        )
        current_baseline_line_pad_px = (
            _suggest_float_or_fixed(
                trial,
                "baseline_line_pad_px",
                baseline_line_pad_px,
                baseline_line_pad_px_min,
                baseline_line_pad_px_max,
            )
            if tune_active_baseline_line_pad_px
            else baseline_line_pad_px
        )
        current_segmentator_cut_threshold = (
            _suggest_float_or_fixed(
                trial,
                "segmentator_cut_threshold",
                segmentator_cut_threshold,
                segmentator_cut_threshold_min,
                segmentator_cut_threshold_max,
            )
            if tune_active_segmentator
            else segmentator_cut_threshold
        )
        current_segmentator_peak_min_distance = (
            _suggest_int_or_fixed(
                trial,
                "segmentator_peak_min_distance",
                segmentator_peak_min_distance,
                segmentator_peak_min_distance_min,
                segmentator_peak_min_distance_max,
            )
            if tune_active_segmentator
            else segmentator_peak_min_distance
        )
        current_segmentator_cut_min_width = (
            _suggest_int_or_fixed(
                trial,
                "segmentator_cut_min_width",
                segmentator_cut_min_width,
                segmentator_cut_min_width_min,
                segmentator_cut_min_width_max,
            )
            if tune_active_segmentator
            else segmentator_cut_min_width
        )
        current_segmentator_cut_max_width = (
            _suggest_int_or_fixed(
                trial,
                "segmentator_cut_max_width",
                segmentator_cut_max_width,
                segmentator_cut_max_width_min,
                segmentator_cut_max_width_max,
            )
            if tune_active_segmentator
            else segmentator_cut_max_width
        )
        current_segmentator_cut_candidate_threshold = (
            _suggest_float_or_fixed(
                trial,
                "segmentator_cut_candidate_threshold",
                segmentator_cut_candidate_threshold,
                segmentator_cut_candidate_threshold_min,
                segmentator_cut_candidate_threshold_max,
            )
            if tune_active_segmentator
            else segmentator_cut_candidate_threshold
        )
        current_segmentator_cut_smooth_radius = (
            _suggest_int_or_fixed(
                trial,
                "segmentator_cut_smooth_radius",
                segmentator_cut_smooth_radius,
                segmentator_cut_smooth_radius_min,
                segmentator_cut_smooth_radius_max,
            )
            if tune_active_segmentator
            else segmentator_cut_smooth_radius
        )
        current_segmentator_boundary_cut_max_edge_ratio = (
            _suggest_float_or_fixed(
                trial,
                "segmentator_boundary_cut_max_edge_ratio",
                segmentator_boundary_cut_max_edge_ratio,
                segmentator_boundary_cut_max_edge_ratio_min,
                segmentator_boundary_cut_max_edge_ratio_max,
            )
            if tune_active_segmentator
            else segmentator_boundary_cut_max_edge_ratio
        )
        current_segmentator_decode_center_fraction = (
            _suggest_float_or_fixed(
                trial,
                "segmentator_decode_center_fraction",
                segmentator_decode_center_fraction,
                segmentator_decode_center_fraction_min,
                segmentator_decode_center_fraction_max,
            )
            if tune_active_segmentator
            else segmentator_decode_center_fraction
        )
        current_segmentator_decode_min_score_width = (
            _suggest_int_or_fixed(
                trial,
                "segmentator_decode_min_score_width",
                segmentator_decode_min_score_width,
                segmentator_decode_min_score_width_min,
                segmentator_decode_min_score_width_max,
            )
            if tune_active_segmentator
            else segmentator_decode_min_score_width
        )
        current_segmentator_edge_min_width = (
            _suggest_int_or_fixed(
                trial,
                "segmentator_edge_min_width",
                segmentator_edge_min_width,
                segmentator_edge_min_width_min,
                segmentator_edge_min_width_max,
            )
            if tune_active_segmentator
            else segmentator_edge_min_width
        )

        metrics = evaluate_prepared(
            base_rows,
            jobs,
            checkpoint_path=checkpoint_path,
            output_csv=None,
            device=device,
            scale_x=scale_x,
            y_pad=y_pad,
            x_pad=float(current_x_pad or 0.0),
            batch_size=batch_size,
            log_every=0,
            verbose=False,
            baseline_crop=baseline_crop,
            baseline_top_pad=baseline_top_pad,
            baseline_bottom_pad=baseline_bottom_pad,
            baseline_deskew=baseline_deskew,
            baseline_max_angle=baseline_max_angle,
            baseline_strict_lines=baseline_strict_lines,
            baseline_line_pad=float(current_baseline_line_pad or 0.0),
            baseline_line_pad_px=float(current_baseline_line_pad_px or 0.0),
            baseline_detector_checkpoint=baseline_detector_checkpoint,
            baseline_detector_threshold=float(current_baseline_detector_threshold or 0.35),
            segmentator_checkpoint=segmentator_checkpoint,
            decode_with_segmentator=decode_with_segmentator,
            segmentator_cut_threshold=current_segmentator_cut_threshold,
            segmentator_peak_min_distance=current_segmentator_peak_min_distance,
            segmentator_cut_postprocess=segmentator_cut_postprocess,
            segmentator_cut_min_width=current_segmentator_cut_min_width,
            segmentator_cut_max_width=current_segmentator_cut_max_width,
            segmentator_cut_candidate_threshold=current_segmentator_cut_candidate_threshold,
            segmentator_cut_smooth_radius=current_segmentator_cut_smooth_radius,
            segmentator_decode_top_k=segmentator_decode_top_k,
            segmentator_edge_trim=segmentator_edge_trim,
            segmentator_edge_min_ink_ratio=segmentator_edge_min_ink_ratio,
            segmentator_edge_min_pixel_density=segmentator_edge_min_pixel_density,
            segmentator_edge_min_width=int(current_segmentator_edge_min_width or 1),
            segmentator_boundary_cuts=segmentator_boundary_cuts,
            segmentator_boundary_cut_max_edge_ratio=float(current_segmentator_boundary_cut_max_edge_ratio or 0.0),
            segmentator_decode_center_fraction=float(current_segmentator_decode_center_fraction or 1.0),
            segmentator_decode_min_score_width=int(current_segmentator_decode_min_score_width or 1),
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool, str)):
                trial.set_user_attr(key, value)
        if trials_output is not None:
            append_trial_log(trials_output, trial.number, metrics, metric_name, dict(trial.params))
        return float(metrics[metric_name])

    print(
        "Optuna OCR search: "
        f"trials={trials}, metric={metric_name}, "
        f"scale_x=[{scale_x_min}, {scale_x_max}], y_pad=[{y_pad_min}, {y_pad_max}], "
        f"decode_with_segmentator={decode_with_segmentator}, baseline_crop={baseline_crop}, "
        f"tune_baseline_line_pad={optuna_tune_baseline_line_pad}, "
        f"tune_baseline_line_pad_px={optuna_tune_baseline_line_pad_px}"
    )
    study.optimize(objective, n_trials=trials)

    best_params = dict(study.best_params)
    print(f"Best Optuna params: {json.dumps(best_params, ensure_ascii=False, sort_keys=True)}")
    print(f"Best {metric_name}: {study.best_value:.8f}")

    final_metrics = evaluate_prepared(
        base_rows,
        jobs,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        device=device,
        scale_x=float(best_params["scale_x"]),
        y_pad=float(best_params["y_pad"]),
        x_pad=float(_best_or_fixed(best_params, "x_pad", x_pad)),
        batch_size=batch_size,
        log_every=log_every,
        verbose=True,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_strict_lines=baseline_strict_lines,
        baseline_line_pad=float(_best_or_fixed(best_params, "baseline_line_pad", baseline_line_pad)),
        baseline_line_pad_px=float(_best_or_fixed(best_params, "baseline_line_pad_px", baseline_line_pad_px)),
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=float(
            _best_or_fixed(best_params, "baseline_detector_threshold", baseline_detector_threshold)
        ),
        segmentator_checkpoint=segmentator_checkpoint,
        decode_with_segmentator=decode_with_segmentator,
        segmentator_cut_threshold=_best_or_fixed(
            best_params,
            "segmentator_cut_threshold",
            segmentator_cut_threshold,
        ),
        segmentator_peak_min_distance=_best_or_fixed(
            best_params,
            "segmentator_peak_min_distance",
            segmentator_peak_min_distance,
        ),
        segmentator_cut_postprocess=segmentator_cut_postprocess,
        segmentator_cut_min_width=_best_or_fixed(
            best_params,
            "segmentator_cut_min_width",
            segmentator_cut_min_width,
        ),
        segmentator_cut_max_width=_best_or_fixed(
            best_params,
            "segmentator_cut_max_width",
            segmentator_cut_max_width,
        ),
        segmentator_cut_candidate_threshold=_best_or_fixed(
            best_params,
            "segmentator_cut_candidate_threshold",
            segmentator_cut_candidate_threshold,
        ),
        segmentator_cut_smooth_radius=_best_or_fixed(
            best_params,
            "segmentator_cut_smooth_radius",
            segmentator_cut_smooth_radius,
        ),
        segmentator_decode_top_k=segmentator_decode_top_k,
        segmentator_edge_trim=segmentator_edge_trim,
        segmentator_edge_min_ink_ratio=segmentator_edge_min_ink_ratio,
        segmentator_edge_min_pixel_density=segmentator_edge_min_pixel_density,
        segmentator_edge_min_width=int(
            _best_or_fixed(best_params, "segmentator_edge_min_width", segmentator_edge_min_width)
        ),
        segmentator_boundary_cuts=segmentator_boundary_cuts,
        segmentator_boundary_cut_max_edge_ratio=float(
            _best_or_fixed(
                best_params,
                "segmentator_boundary_cut_max_edge_ratio",
                segmentator_boundary_cut_max_edge_ratio,
            )
        ),
        segmentator_decode_center_fraction=float(
            _best_or_fixed(
                best_params,
                "segmentator_decode_center_fraction",
                segmentator_decode_center_fraction,
            )
        ),
        segmentator_decode_min_score_width=int(
            _best_or_fixed(
                best_params,
                "segmentator_decode_min_score_width",
                segmentator_decode_min_score_width,
            )
        ),
    )
    final_metrics["optuna_trials"] = trials
    final_metrics["optuna_metric"] = metric_name
    final_metrics["optuna_best_value"] = float(study.best_value)
    final_metrics["optuna_best_params"] = best_params
    return final_metrics


def evaluate(
    json_path: Path,
    images_dir: Path,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    batch_size: int,
    limit: int | None,
    log_every: int,
    verbose: bool = True,
    baseline_crop: bool = False,
    baseline_top_pad: float = 0.12,
    baseline_bottom_pad: float = 0.18,
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
    baseline_strict_lines: bool = True,
    baseline_line_pad: float = 0.08,
    baseline_line_pad_px: float = 0.0,
    baseline_detector_checkpoint: Path | None = None,
    baseline_detector_threshold: float = 0.35,
    segmentator_checkpoint: Path | None = None,
    decode_with_segmentator: bool = False,
    segmentator_cut_threshold: float | None = None,
    segmentator_peak_min_distance: int | None = None,
    segmentator_cut_postprocess: str | None = None,
    segmentator_cut_min_width: int | None = None,
    segmentator_cut_max_width: int | None = None,
    segmentator_cut_candidate_threshold: float | None = None,
    segmentator_cut_smooth_radius: int | None = None,
    segmentator_decode_top_k: int = 8,
    segmentator_edge_trim: bool = True,
    segmentator_edge_min_ink_ratio: float = 0.035,
    segmentator_edge_min_pixel_density: float = 0.003,
    segmentator_edge_min_width: int = 2,
    segmentator_boundary_cuts: str = "auto",
    segmentator_boundary_cut_max_edge_ratio: float = 0.45,
    segmentator_decode_center_fraction: float = 0.6,
    segmentator_decode_min_score_width: int = 1,
) -> dict[str, Any]:
    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    return evaluate_prepared(
        base_rows,
        jobs,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        device=device,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        batch_size=batch_size,
        log_every=log_every,
        verbose=verbose,
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
        segmentator_checkpoint=segmentator_checkpoint,
        decode_with_segmentator=decode_with_segmentator,
        segmentator_cut_threshold=segmentator_cut_threshold,
        segmentator_peak_min_distance=segmentator_peak_min_distance,
        segmentator_cut_postprocess=segmentator_cut_postprocess,
        segmentator_cut_min_width=segmentator_cut_min_width,
        segmentator_cut_max_width=segmentator_cut_max_width,
        segmentator_cut_candidate_threshold=segmentator_cut_candidate_threshold,
        segmentator_cut_smooth_radius=segmentator_cut_smooth_radius,
        segmentator_decode_top_k=segmentator_decode_top_k,
        segmentator_edge_trim=segmentator_edge_trim,
        segmentator_edge_min_ink_ratio=segmentator_edge_min_ink_ratio,
        segmentator_edge_min_pixel_density=segmentator_edge_min_pixel_density,
        segmentator_edge_min_width=segmentator_edge_min_width,
        segmentator_boundary_cuts=segmentator_boundary_cuts,
        segmentator_boundary_cut_max_edge_ratio=segmentator_boundary_cut_max_edge_ratio,
        segmentator_decode_center_fraction=segmentator_decode_center_fraction,
        segmentator_decode_min_score_width=segmentator_decode_min_score_width,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FCN OCR on a Label Studio JSON export.")
    parser.add_argument("--json", required=True, help="Path to Label Studio export JSON.")
    parser.add_argument("--images", required=True, help="Folder with images.")
    parser.add_argument("--checkpoint", required=True, help="Path to OCR checkpoint.")
    parser.add_argument("--out", default="ocr_metrics.csv", help="Output CSV path.")
    parser.add_argument("--device", default=None, help="Device to use: cuda, cpu, or empty for auto.")
    parser.add_argument("--scale-x", type=float, default=0.0, help="Normalized horizontal inference scale.")
    parser.add_argument("--y-pad", type=float, default=0.0, help="Normalized vertical inference padding/crop.")
    parser.add_argument("--x-pad", type=float, default=0.0, help="Normalized symmetric horizontal inference padding.")

    parser.add_argument("--baseline-crop", action="store_true", help="Use baseline detection/crop before y-pad and resize.")
    parser.add_argument("--baseline-top-pad", type=float, default=0.12)
    parser.add_argument("--baseline-bottom-pad", type=float, default=0.18)
    parser.add_argument("--no-baseline-deskew", action="store_true")
    parser.add_argument("--baseline-max-angle", type=float, default=12.0)
    parser.add_argument("--no-baseline-strict-lines", action="store_true")
    parser.add_argument("--baseline-line-pad", type=float, default=0.08)
    parser.add_argument("--baseline-line-pad-px", type=float, default=0.0)
    parser.add_argument("--baseline-detector-checkpoint", default=None)
    parser.add_argument("--baseline-detector-threshold", type=float, default=0.35)

    parser.add_argument("--segmentator-checkpoint", default=None)
    parser.add_argument("--decode-with-segmentator", action="store_true")
    parser.add_argument("--segmentator-cut-threshold", type=float, default=None)
    parser.add_argument("--segmentator-peak-min-distance", type=int, default=None)
    parser.add_argument("--segmentator-cut-postprocess", choices=("peaks", "widths"), default=None)
    parser.add_argument("--segmentator-cut-min-width", type=int, default=None)
    parser.add_argument("--segmentator-cut-max-width", type=int, default=None)
    parser.add_argument("--segmentator-cut-candidate-threshold", type=float, default=None)
    parser.add_argument("--segmentator-cut-smooth-radius", type=int, default=None)
    parser.add_argument("--segmentator-decode-top-k", type=int, default=8)
    parser.add_argument("--segmentator-decode-center-fraction", type=float, default=0.6)
    parser.add_argument("--segmentator-decode-min-score-width", type=int, default=1)
    parser.add_argument("--no-segmentator-edge-trim", action="store_true")
    parser.add_argument("--segmentator-edge-min-ink-ratio", type=float, default=0.035)
    parser.add_argument("--segmentator-edge-min-pixel-density", type=float, default=0.003)
    parser.add_argument("--segmentator-edge-min-width", type=int, default=2)
    parser.add_argument("--segmentator-boundary-cuts", choices=("auto", "off", "on"), default="auto")
    parser.add_argument("--segmentator-boundary-cut-max-edge-ratio", type=float, default=0.45)

    parser.add_argument("--batch-size", type=int, default=32, help="Direct OCR inference batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to evaluate.")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N recognized images; 0 disables.")
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="If > 0, tune enabled parameter ranges with Optuna before final evaluation.",
    )
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
    parser.add_argument("--optuna-trials-out", default=None, help="Optional TSV file with Optuna trial metrics.")
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument("--optuna-storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///study.db.")

    parser.add_argument("--optuna-x-pad-min", type=float, default=None)
    parser.add_argument("--optuna-x-pad-max", type=float, default=None)
    parser.add_argument(
        "--optuna-tune-baseline-line-pad",
        action="store_true",
        help="Explicitly tune strict-lines baseline_line_pad when its min/max range is provided.",
    )
    parser.add_argument(
        "--optuna-tune-baseline-line-pad-px",
        action="store_true",
        help="Explicitly tune absolute strict-lines baseline_line_pad_px.",
    )
    parser.add_argument("--optuna-baseline-detector-threshold-min", type=float, default=None)
    parser.add_argument("--optuna-baseline-detector-threshold-max", type=float, default=None)
    parser.add_argument("--optuna-baseline-line-pad-min", type=float, default=None)
    parser.add_argument("--optuna-baseline-line-pad-max", type=float, default=None)
    parser.add_argument("--optuna-baseline-line-pad-px-min", type=float, default=None)
    parser.add_argument("--optuna-baseline-line-pad-px-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-cut-threshold-min", type=float, default=None)
    parser.add_argument("--optuna-segmentator-cut-threshold-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-peak-min-distance-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-peak-min-distance-max", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-min-width-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-min-width-max", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-max-width-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-max-width-max", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-candidate-threshold-min", type=float, default=None)
    parser.add_argument("--optuna-segmentator-cut-candidate-threshold-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-cut-smooth-radius-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-smooth-radius-max", type=int, default=None)
    parser.add_argument("--optuna-segmentator-boundary-cut-max-edge-ratio-min", type=float, default=None)
    parser.add_argument("--optuna-segmentator-boundary-cut-max-edge-ratio-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-decode-center-fraction-min", type=float, default=None)
    parser.add_argument("--optuna-segmentator-decode-center-fraction-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-decode-min-score-width-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-decode-min-score-width-max", type=int, default=None)
    parser.add_argument("--optuna-segmentator-edge-min-width-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-edge-min-width-max", type=int, default=None)
    return parser.parse_args()


def _common_eval_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "baseline_crop": args.baseline_crop,
        "baseline_top_pad": args.baseline_top_pad,
        "baseline_bottom_pad": args.baseline_bottom_pad,
        "baseline_deskew": not args.no_baseline_deskew,
        "baseline_max_angle": args.baseline_max_angle,
        "baseline_strict_lines": not args.no_baseline_strict_lines,
        "baseline_line_pad": args.baseline_line_pad,
        "baseline_line_pad_px": args.baseline_line_pad_px,
        "baseline_detector_checkpoint": Path(args.baseline_detector_checkpoint) if args.baseline_detector_checkpoint else None,
        "baseline_detector_threshold": args.baseline_detector_threshold,
        "segmentator_checkpoint": Path(args.segmentator_checkpoint) if args.segmentator_checkpoint else None,
        "decode_with_segmentator": args.decode_with_segmentator,
        "segmentator_cut_threshold": args.segmentator_cut_threshold,
        "segmentator_peak_min_distance": args.segmentator_peak_min_distance,
        "segmentator_cut_postprocess": args.segmentator_cut_postprocess,
        "segmentator_cut_min_width": args.segmentator_cut_min_width,
        "segmentator_cut_max_width": args.segmentator_cut_max_width,
        "segmentator_cut_candidate_threshold": args.segmentator_cut_candidate_threshold,
        "segmentator_cut_smooth_radius": args.segmentator_cut_smooth_radius,
        "segmentator_decode_top_k": args.segmentator_decode_top_k,
        "segmentator_edge_trim": not args.no_segmentator_edge_trim,
        "segmentator_edge_min_ink_ratio": args.segmentator_edge_min_ink_ratio,
        "segmentator_edge_min_pixel_density": args.segmentator_edge_min_pixel_density,
        "segmentator_edge_min_width": args.segmentator_edge_min_width,
        "segmentator_boundary_cuts": args.segmentator_boundary_cuts,
        "segmentator_boundary_cut_max_edge_ratio": args.segmentator_boundary_cut_max_edge_ratio,
        "segmentator_decode_center_fraction": args.segmentator_decode_center_fraction,
        "segmentator_decode_min_score_width": args.segmentator_decode_min_score_width,
    }


def _print_inference_command(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    _, jobs = build_rows_and_jobs(Path(args.json), Path(args.images), args.limit)
    image_path = str(jobs[0][1]) if jobs else "<IMAGE_PATH>"
    command = [
        "python",
        "inference.py",
        "--checkpoint",
        str(args.checkpoint),
        "--image",
        image_path,
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
                str(args.baseline_top_pad),
                "--baseline-bottom-pad",
                str(args.baseline_bottom_pad),
                "--baseline-line-pad",
                str(metrics["baseline_line_pad"]),
                "--baseline-line-pad-px",
                str(metrics["baseline_line_pad_px"]),
                "--baseline-max-angle",
                str(args.baseline_max_angle),
                "--baseline-detector-threshold",
                str(metrics["baseline_detector_threshold"]),
            ]
        )
        if args.no_baseline_deskew:
            command.append("--no-baseline-deskew")
        if args.no_baseline_strict_lines:
            command.append("--no-baseline-strict-lines")
        if metrics.get("baseline_detector_checkpoint"):
            command.extend(
                [
                    "--baseline-detector-checkpoint",
                    str(metrics["baseline_detector_checkpoint"]),
                ]
            )

    if metrics.get("segmentator_checkpoint"):
        command.extend(
            [
                "--segmentator-checkpoint",
                str(metrics["segmentator_checkpoint"]),
                "--segmentator-cut-threshold",
                str(metrics["segmentator_cut_threshold"]),
                "--segmentator-peak-min-distance",
                str(metrics["segmentator_peak_min_distance"]),
                "--segmentator-cut-postprocess",
                str(metrics["segmentator_cut_postprocess"]),
                "--segmentator-cut-min-width",
                str(metrics["segmentator_cut_min_width"]),
                "--segmentator-cut-max-width",
                str(metrics["segmentator_cut_max_width"]),
                "--segmentator-cut-candidate-threshold",
                str(metrics["segmentator_cut_candidate_threshold"]),
                "--segmentator-cut-smooth-radius",
                str(metrics["segmentator_cut_smooth_radius"]),
                "--segmentator-decode-top-k",
                str(metrics["segmentator_decode_top_k"]),
                "--segmentator-decode-center-fraction",
                str(metrics["segmentator_decode_center_fraction"]),
                "--segmentator-decode-min-score-width",
                str(metrics["segmentator_decode_min_score_width"]),
                "--segmentator-edge-min-ink-ratio",
                str(metrics["segmentator_edge_min_ink_ratio"]),
                "--segmentator-edge-min-pixel-density",
                str(metrics["segmentator_edge_min_pixel_density"]),
                "--segmentator-edge-min-width",
                str(metrics["segmentator_edge_min_width"]),
                "--segmentator-boundary-cuts",
                str(metrics["segmentator_boundary_cuts"]),
                "--segmentator-boundary-cut-max-edge-ratio",
                str(metrics["segmentator_boundary_cut_max_edge_ratio"]),
            ]
        )
        if metrics["decode_with_segmentator"]:
            command.append("--decode-with-segmentator")
        if not metrics["segmentator_edge_trim"]:
            command.append("--no-segmentator-edge-trim")

    print("\n=== Inference command ===")
    print(shlex.join(command))


def main() -> None:
    args = parse_args()
    common_kwargs = _common_eval_kwargs(args)
    if args.optuna_trials > 0:
        metrics = optimize_preprocess(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            trials=args.optuna_trials,
            scale_x_min=args.optuna_scale_x_min,
            scale_x_max=args.optuna_scale_x_max,
            y_pad_min=args.optuna_y_pad_min,
            y_pad_max=args.optuna_y_pad_max,
            x_pad=args.x_pad,
            metric_name=args.optuna_metric,
            log_every=args.log_every,
            trials_output=Path(args.optuna_trials_out) if args.optuna_trials_out else None,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            optuna_tune_baseline_line_pad=args.optuna_tune_baseline_line_pad,
            optuna_tune_baseline_line_pad_px=args.optuna_tune_baseline_line_pad_px,
            x_pad_min=args.optuna_x_pad_min,
            x_pad_max=args.optuna_x_pad_max,
            baseline_detector_threshold_min=args.optuna_baseline_detector_threshold_min,
            baseline_detector_threshold_max=args.optuna_baseline_detector_threshold_max,
            baseline_line_pad_min=args.optuna_baseline_line_pad_min,
            baseline_line_pad_max=args.optuna_baseline_line_pad_max,
            baseline_line_pad_px_min=args.optuna_baseline_line_pad_px_min,
            baseline_line_pad_px_max=args.optuna_baseline_line_pad_px_max,
            segmentator_cut_threshold_min=args.optuna_segmentator_cut_threshold_min,
            segmentator_cut_threshold_max=args.optuna_segmentator_cut_threshold_max,
            segmentator_peak_min_distance_min=args.optuna_segmentator_peak_min_distance_min,
            segmentator_peak_min_distance_max=args.optuna_segmentator_peak_min_distance_max,
            segmentator_cut_min_width_min=args.optuna_segmentator_cut_min_width_min,
            segmentator_cut_min_width_max=args.optuna_segmentator_cut_min_width_max,
            segmentator_cut_max_width_min=args.optuna_segmentator_cut_max_width_min,
            segmentator_cut_max_width_max=args.optuna_segmentator_cut_max_width_max,
            segmentator_cut_candidate_threshold_min=args.optuna_segmentator_cut_candidate_threshold_min,
            segmentator_cut_candidate_threshold_max=args.optuna_segmentator_cut_candidate_threshold_max,
            segmentator_cut_smooth_radius_min=args.optuna_segmentator_cut_smooth_radius_min,
            segmentator_cut_smooth_radius_max=args.optuna_segmentator_cut_smooth_radius_max,
            segmentator_boundary_cut_max_edge_ratio_min=args.optuna_segmentator_boundary_cut_max_edge_ratio_min,
            segmentator_boundary_cut_max_edge_ratio_max=args.optuna_segmentator_boundary_cut_max_edge_ratio_max,
            segmentator_decode_center_fraction_min=args.optuna_segmentator_decode_center_fraction_min,
            segmentator_decode_center_fraction_max=args.optuna_segmentator_decode_center_fraction_max,
            segmentator_decode_min_score_width_min=args.optuna_segmentator_decode_min_score_width_min,
            segmentator_decode_min_score_width_max=args.optuna_segmentator_decode_min_score_width_max,
            segmentator_edge_min_width_min=args.optuna_segmentator_edge_min_width_min,
            segmentator_edge_min_width_max=args.optuna_segmentator_edge_min_width_max,
            **common_kwargs,
        )
    else:
        metrics = evaluate(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            scale_x=args.scale_x,
            y_pad=args.y_pad,
            x_pad=args.x_pad,
            batch_size=args.batch_size,
            limit=args.limit,
            log_every=args.log_every,
            **common_kwargs,
        )
    _print_inference_command(args, metrics)


if __name__ == "__main__":
    main()
