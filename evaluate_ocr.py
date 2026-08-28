from __future__ import annotations

import argparse
from copy import deepcopy
import json
import time
from pathlib import Path
from typing import Any, Sequence

from fcn_ocr import InferenceConfig, OCRPipeline
from fcn_ocr.evaluation import OCR_RESULT_FIELDS, compute_ocr_metrics, load_label_studio_samples
from fcn_ocr.evaluation.config import parse_args_with_evaluation_config
from fcn_ocr.evaluation.optuna import (
    best_or_fixed,
    create_study,
    optimize_with_progress,
    suggest_float_or_fixed,
    suggest_int_or_fixed,
)
from fcn_ocr.evaluation.reporting import (
    append_tsv_row,
    save_and_print_inference_command,
    write_csv_rows,
)


def build_rows_and_jobs(
    json_path: Path,
    images_dir: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[int, Path]]]:
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[int, Path]] = []
    for sample in load_label_studio_samples(json_path, images_dir, limit):
        row = {
            "task_id": sample.task_id,
            "image": sample.image_name,
            "gt": sample.text,
            "pred": "",
            "exact_match": 0,
            "char_accuracy": 0.0,
            "levenshtein": 0,
            "gt_len": 0,
            "pred_len": 0,
            "error": "",
        }
        jobs.append((len(rows), sample.image_path))
        rows.append(row)

    return rows, jobs


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
    print(f"Requested batch size:       {metrics['batch_size']}")
    if "gpu_batches" in metrics:
        print(f"GPU sub-batches:            {metrics['gpu_batches']}")
        print(
            "Average GPU batch size:     "
            f"{metrics['average_gpu_batch_size']:.2f}"
        )
        print(f"Maximum GPU batch size:     {metrics['max_gpu_batch_size']}")
        print(f"Padding efficiency:         {metrics['padding_efficiency']:.4f}")
    print(f"OCR scale_x:                {metrics['scale_x']:+.5f}")
    print(f"OCR y_pad:                  {metrics['y_pad']:+.5f}")
    print(f"OCR x_pad:                  {metrics['x_pad']:.5f}")
    print(f"Baseline crop:              {metrics['baseline_crop']}")
    print(f"Baseline line pad:          {metrics['baseline_line_pad']:.5f}")
    print(f"Baseline line pad px:       {metrics['baseline_line_pad_px']:.2f}")
    if metrics.get("baseline_detector_checkpoint"):
        print(f"Baseline detector:          {metrics['baseline_detector_checkpoint']}")
        print(f"Baseline detector thr:      {metrics['baseline_detector_threshold']:.5f}")
    print(f"Decode with segmentator:    {metrics['decode_with_segmentator']}")
    if metrics.get("segmentator_checkpoint"):
        print(f"Segmentator checkpoint:     {metrics['segmentator_checkpoint']}")
        print(f"Segmentator scale_x:        {metrics['segmentator_scale_x']:+.5f}")
        print(f"Segmentator y_pad:          {metrics['segmentator_y_pad']:+.5f}")
        print(f"Segmentator x_pad:          {metrics['segmentator_x_pad']:.5f}")
        print(f"Segmentator cut threshold:  {metrics['segmentator_cut_threshold']:.5f}")
        print(f"Segmentator cut min width:  {metrics['segmentator_cut_min_width']}")
        print(f"Segmentator cut max width:  {metrics['segmentator_cut_max_width']}")
        print(f"Segmentator smooth radius:  {metrics['segmentator_cut_smooth_radius']}")
        print(f"Decode method:              {metrics['decode_method']}")
        print(f"Decode center fraction:     {metrics['center_fraction']:.5f}")
        print(f"Decode min score width:     {metrics['min_score_width']}")
        if metrics["decode_method"] == "dp":
            print(f"Decode cut weight:          {metrics['cut_weight']:.5f}")
            print(f"Decode OCR weight:          {metrics['ocr_weight']:.5f}")
            print(f"Decode width weight:        {metrics['width_weight']:.5f}")
            print(f"Decode skip cut penalty:    {metrics['skip_cut_penalty']:.5f}")
            glyph_width_prior = metrics.get("glyph_width_prior") or {}
            print(f"Decode glyph width prior:   {bool(glyph_width_prior.get('enabled'))}")
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
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
    baseline_line_pad: float = 0.08,
    baseline_line_pad_px: float = 0.0,
    baseline_detector_checkpoint: Path | None = None,
    baseline_detector_threshold: float = 0.35,
    segmentator_checkpoint: Path | None = None,
    decode_with_segmentator: bool = False,
    segmentator_scale_x: float = 0.0,
    segmentator_y_pad: float = 0.0,
    segmentator_x_pad: float = 0.0,
    segmentator_cut_threshold: float | None = None,
    segmentator_cut_min_width: int | None = None,
    segmentator_cut_max_width: int | None = None,
    segmentator_cut_smooth_radius: int | None = None,
    decode_method: str = "cells",
    decode_top_k: int = 8,
    center_fraction: float = 0.6,
    min_score_width: int = 1,
    cut_weight: float = 1.0,
    ocr_weight: float = 1.0,
    width_weight: float = 0.05,
    skip_cut_penalty: float = 0.35,
    glyph_width_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if decode_with_segmentator and segmentator_checkpoint is None:
        raise ValueError("decode_with_segmentator requires segmentator_checkpoint")

    rows = deepcopy(base_rows)
    started_at = time.perf_counter()
    config_data: dict[str, Any] = {
        "device": device,
        "fcn_ocr": {
            "checkpoint": checkpoint_path,
            "preprocessing": {
                "scale_x": scale_x,
                "y_pad": y_pad,
                "x_pad": x_pad,
            },
            "decode": {
                "enabled": decode_with_segmentator,
                "method": decode_method,
                "top_k": decode_top_k,
                "center_fraction": center_fraction,
                "min_score_width": min_score_width,
                "cut_weight": cut_weight,
                "ocr_weight": ocr_weight,
                "width_weight": width_weight,
                "skip_cut_penalty": skip_cut_penalty,
                "glyph_width_prior": glyph_width_prior or {},
            },
        },
        "debug": {"top_k": decode_top_k},
    }
    if baseline_crop:
        config_data["baseline_detection"] = {
            "enabled": True,
            "detector_checkpoint": baseline_detector_checkpoint,
            "detector_threshold": baseline_detector_threshold,
            "deskew": baseline_deskew,
            "max_angle": baseline_max_angle,
            "line_pad": baseline_line_pad,
            "line_pad_px": baseline_line_pad_px,
        }
    if decode_with_segmentator:
        config_data["vertical_segmentation"] = {
            "checkpoint": segmentator_checkpoint,
            "preprocessing": {
                "scale_x": segmentator_scale_x,
                "y_pad": segmentator_y_pad,
                "x_pad": segmentator_x_pad,
            },
            "cut_threshold": segmentator_cut_threshold,
            "cut_min_width": segmentator_cut_min_width,
            "cut_max_width": segmentator_cut_max_width,
            "cut_smooth_radius": segmentator_cut_smooth_radius,
        }

    pipeline = OCRPipeline(
        InferenceConfig.model_validate(config_data),
        verbose=verbose,
    )
    path_results, batch_metrics = pipeline.recognize_paths_text(
        [path for _, path in jobs],
        batch_size=batch_size,
        log_every=log_every,
    )
    predictions = {
        row_index: result.text
        for (row_index, _), result in zip(jobs, path_results)
    }
    errors = {
        row_index: result.error
        for (row_index, _), result in zip(jobs, path_results)
    }
    elapsed = time.perf_counter() - started_at

    for row_index, prediction in predictions.items():
        rows[row_index]["pred"] = prediction
    for row_index, error in errors.items():
        rows[row_index]["error"] = error

    metrics = compute_ocr_metrics(rows, elapsed)
    metrics.update(batch_metrics)
    metrics["scale_x"] = float(scale_x)
    metrics["y_pad"] = float(y_pad)
    metrics["x_pad"] = float(x_pad)
    metrics["batch_size"] = int(batch_size)
    metrics["baseline_crop"] = bool(baseline_crop)
    metrics["baseline_line_pad"] = float(baseline_line_pad)
    metrics["baseline_line_pad_px"] = float(baseline_line_pad_px)
    metrics["baseline_detector_checkpoint"] = str(baseline_detector_checkpoint) if baseline_detector_checkpoint else ""
    metrics["baseline_detector_threshold"] = float(baseline_detector_threshold)
    metrics["decode_with_segmentator"] = bool(decode_with_segmentator)
    metrics["segmentator_checkpoint"] = str(segmentator_checkpoint) if segmentator_checkpoint else ""
    metrics["segmentator_scale_x"] = float(segmentator_scale_x)
    metrics["segmentator_y_pad"] = float(segmentator_y_pad)
    metrics["segmentator_x_pad"] = float(segmentator_x_pad)
    metrics["decode_method"] = str(decode_method)
    metrics["decode_top_k"] = int(decode_top_k)
    metrics["center_fraction"] = float(center_fraction)
    metrics["min_score_width"] = int(min_score_width)
    metrics["cut_weight"] = float(cut_weight)
    metrics["ocr_weight"] = float(ocr_weight)
    metrics["width_weight"] = float(width_weight)
    metrics["skip_cut_penalty"] = float(skip_cut_penalty)
    metrics["glyph_width_prior"] = glyph_width_prior or {}
    if pipeline.segmentator is not None:
        metrics["segmentator_cut_threshold"] = float(pipeline.segmentator.cut_threshold)
        metrics["segmentator_cut_min_width"] = int(pipeline.segmentator.cut_min_width)
        metrics["segmentator_cut_max_width"] = int(pipeline.segmentator.cut_max_width)
        metrics["segmentator_cut_smooth_radius"] = int(pipeline.segmentator.cut_smooth_radius)
    else:
        metrics["segmentator_cut_threshold"] = float(segmentator_cut_threshold or 0.0)
        metrics["segmentator_cut_min_width"] = int(segmentator_cut_min_width or 0)
        metrics["segmentator_cut_max_width"] = int(segmentator_cut_max_width or 0)
        metrics["segmentator_cut_smooth_radius"] = int(segmentator_cut_smooth_radius or 0)

    if output_csv is not None:
        write_csv_rows(rows, output_csv, OCR_RESULT_FIELDS)

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
        "segmentator_scale_x",
        "segmentator_y_pad",
        "segmentator_x_pad",
        "segmentator_cut_threshold",
        "segmentator_cut_min_width",
        "segmentator_cut_max_width",
        "segmentator_cut_smooth_radius",
        "decode_method",
        "center_fraction",
        "min_score_width",
        "cut_weight",
        "ocr_weight",
        "width_weight",
        "skip_cut_penalty",
    ]
    return {name: metrics[name] for name in names if name in metrics}


def append_trial_log(
    path: Path,
    trial_number: int,
    metrics: dict[str, Any],
    metric_name: str,
    trial_params: dict[str, Any] | None = None,
) -> None:
    params = json.dumps(
        trial_params if trial_params is not None else _trial_params_snapshot(metrics),
        ensure_ascii=False,
        sort_keys=True,
    )
    append_tsv_row(
        path,
        (
            "trial",
            "params",
            "metric",
            "metric_value",
            "line_accuracy",
            "average_char_accuracy",
            "global_char_accuracy",
            "average_levenshtein",
            "total_levenshtein",
            "speed",
        ),
        (
            trial_number,
            params,
            metric_name,
            f"{metrics[metric_name]:.8f}",
            f"{metrics['line_accuracy']:.8f}",
            f"{metrics['average_char_accuracy']:.8f}",
            f"{metrics['global_char_accuracy']:.8f}",
            f"{metrics['average_levenshtein']:.8f}",
            metrics["total_levenshtein"],
            f"{metrics['speed']:.6f}",
        ),
    )


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
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
    baseline_line_pad: float = 0.08,
    baseline_line_pad_px: float = 0.0,
    baseline_detector_checkpoint: Path | None = None,
    baseline_detector_threshold: float = 0.35,
    segmentator_checkpoint: Path | None = None,
    decode_with_segmentator: bool = False,
    segmentator_scale_x: float = 0.0,
    segmentator_y_pad: float = 0.0,
    segmentator_x_pad: float = 0.0,
    segmentator_cut_threshold: float | None = None,
    segmentator_cut_min_width: int | None = None,
    segmentator_cut_max_width: int | None = None,
    segmentator_cut_smooth_radius: int | None = None,
    decode_method: str = "cells",
    decode_top_k: int = 8,
    center_fraction: float = 0.6,
    min_score_width: int = 1,
    cut_weight: float = 1.0,
    ocr_weight: float = 1.0,
    width_weight: float = 0.05,
    skip_cut_penalty: float = 0.35,
    glyph_width_prior: dict[str, Any] | None = None,
    x_pad_min: float | None = None,
    x_pad_max: float | None = None,
    segmentator_scale_x_min: float | None = None,
    segmentator_scale_x_max: float | None = None,
    segmentator_y_pad_min: float | None = None,
    segmentator_y_pad_max: float | None = None,
    segmentator_x_pad_min: float | None = None,
    segmentator_x_pad_max: float | None = None,
    baseline_detector_threshold_min: float | None = None,
    baseline_detector_threshold_max: float | None = None,
    baseline_line_pad_min: float | None = None,
    baseline_line_pad_max: float | None = None,
    baseline_line_pad_px_min: float | None = None,
    baseline_line_pad_px_max: float | None = None,
    segmentator_cut_threshold_min: float | None = None,
    segmentator_cut_threshold_max: float | None = None,
    segmentator_cut_min_width_min: int | None = None,
    segmentator_cut_min_width_max: int | None = None,
    segmentator_cut_max_width_min: int | None = None,
    segmentator_cut_max_width_max: int | None = None,
    segmentator_cut_smooth_radius_min: int | None = None,
    segmentator_cut_smooth_radius_max: int | None = None,
    center_fraction_min: float | None = None,
    center_fraction_max: float | None = None,
    min_score_width_min: int | None = None,
    min_score_width_max: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be >= 1")

    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    direction = "minimize" if metric_name in {"average_levenshtein", "total_levenshtein"} else "maximize"
    study = create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
    )

    def objective(trial) -> float:
        tune_active_baseline_line_pad = bool(baseline_crop) and (
            bool(optuna_tune_baseline_line_pad)
            or baseline_line_pad_min is not None
            or baseline_line_pad_max is not None
        )
        tune_active_baseline_line_pad_px = (
            bool(baseline_crop)
            and bool(optuna_tune_baseline_line_pad_px)
        )
        tune_active_baseline_detector = bool(baseline_crop) and baseline_detector_checkpoint is not None
        tune_active_segmentator = bool(decode_with_segmentator) and segmentator_checkpoint is not None

        scale_x = trial.suggest_float("scale_x", scale_x_min, scale_x_max)
        y_pad = trial.suggest_float("y_pad", y_pad_min, y_pad_max)
        current_x_pad = suggest_float_or_fixed(trial, "x_pad", x_pad, x_pad_min, x_pad_max)
        current_baseline_detector_threshold = (
            suggest_float_or_fixed(
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
            suggest_float_or_fixed(
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
            suggest_float_or_fixed(
                trial,
                "baseline_line_pad_px",
                baseline_line_pad_px,
                baseline_line_pad_px_min,
                baseline_line_pad_px_max,
            )
            if tune_active_baseline_line_pad_px
            else baseline_line_pad_px
        )
        current_segmentator_scale_x = (
            suggest_float_or_fixed(
                trial,
                "segmentator_scale_x",
                segmentator_scale_x,
                segmentator_scale_x_min,
                segmentator_scale_x_max,
            )
            if tune_active_segmentator
            else segmentator_scale_x
        )
        current_segmentator_y_pad = (
            suggest_float_or_fixed(
                trial,
                "segmentator_y_pad",
                segmentator_y_pad,
                segmentator_y_pad_min,
                segmentator_y_pad_max,
            )
            if tune_active_segmentator
            else segmentator_y_pad
        )
        current_segmentator_x_pad = (
            suggest_float_or_fixed(
                trial,
                "segmentator_x_pad",
                segmentator_x_pad,
                segmentator_x_pad_min,
                segmentator_x_pad_max,
            )
            if tune_active_segmentator
            else segmentator_x_pad
        )
        current_segmentator_cut_threshold = (
            suggest_float_or_fixed(
                trial,
                "segmentator_cut_threshold",
                segmentator_cut_threshold,
                segmentator_cut_threshold_min,
                segmentator_cut_threshold_max,
            )
            if tune_active_segmentator
            else segmentator_cut_threshold
        )
        current_segmentator_cut_min_width = (
            suggest_int_or_fixed(
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
            suggest_int_or_fixed(
                trial,
                "segmentator_cut_max_width",
                segmentator_cut_max_width,
                segmentator_cut_max_width_min,
                segmentator_cut_max_width_max,
            )
            if tune_active_segmentator
            else segmentator_cut_max_width
        )
        current_segmentator_cut_smooth_radius = (
            suggest_int_or_fixed(
                trial,
                "segmentator_cut_smooth_radius",
                segmentator_cut_smooth_radius,
                segmentator_cut_smooth_radius_min,
                segmentator_cut_smooth_radius_max,
            )
            if tune_active_segmentator
            else segmentator_cut_smooth_radius
        )
        current_center_fraction = (
            suggest_float_or_fixed(
                trial,
                "center_fraction",
                center_fraction,
                center_fraction_min,
                center_fraction_max,
            )
            if tune_active_segmentator
            else center_fraction
        )
        current_min_score_width = (
            suggest_int_or_fixed(
                trial,
                "min_score_width",
                min_score_width,
                min_score_width_min,
                min_score_width_max,
            )
            if tune_active_segmentator
            else min_score_width
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
            baseline_deskew=baseline_deskew,
            baseline_max_angle=baseline_max_angle,
            baseline_line_pad=float(current_baseline_line_pad or 0.0),
            baseline_line_pad_px=float(current_baseline_line_pad_px or 0.0),
            baseline_detector_checkpoint=baseline_detector_checkpoint,
            baseline_detector_threshold=float(current_baseline_detector_threshold or 0.35),
            segmentator_checkpoint=segmentator_checkpoint,
            decode_with_segmentator=decode_with_segmentator,
            segmentator_scale_x=float(current_segmentator_scale_x or 0.0),
            segmentator_y_pad=float(current_segmentator_y_pad or 0.0),
            segmentator_x_pad=float(current_segmentator_x_pad or 0.0),
            segmentator_cut_threshold=current_segmentator_cut_threshold,
            segmentator_cut_min_width=current_segmentator_cut_min_width,
            segmentator_cut_max_width=current_segmentator_cut_max_width,
            segmentator_cut_smooth_radius=current_segmentator_cut_smooth_radius,
            decode_method=decode_method,
            decode_top_k=decode_top_k,
            center_fraction=float(current_center_fraction or 1.0),
            min_score_width=int(current_min_score_width or 1),
            cut_weight=cut_weight,
            ocr_weight=ocr_weight,
            width_weight=width_weight,
            skip_cut_penalty=skip_cut_penalty,
            glyph_width_prior=glyph_width_prior,
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
        f"ocr_scale_x=[{scale_x_min}, {scale_x_max}], "
        f"ocr_y_pad=[{y_pad_min}, {y_pad_max}], "
        f"decode_with_segmentator={decode_with_segmentator}, baseline_crop={baseline_crop}, "
        f"segmentator_scale_x=[{segmentator_scale_x_min}, {segmentator_scale_x_max}], "
        f"segmentator_y_pad=[{segmentator_y_pad_min}, {segmentator_y_pad_max}], "
        f"segmentator_x_pad=[{segmentator_x_pad_min}, {segmentator_x_pad_max}], "
        f"center_fraction=[{center_fraction_min}, {center_fraction_max}], "
        f"min_score_width=[{min_score_width_min}, {min_score_width_max}], "
        f"tune_baseline_line_pad={optuna_tune_baseline_line_pad}, "
        f"tune_baseline_line_pad_px={optuna_tune_baseline_line_pad_px}"
    )
    optimize_with_progress(
        study,
        objective,
        n_trials=trials,
        metric_name=metric_name,
        enabled=progress,
    )

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
        x_pad=float(best_or_fixed(best_params, "x_pad", x_pad)),
        batch_size=batch_size,
        log_every=log_every,
        verbose=True,
        baseline_crop=baseline_crop,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_line_pad=float(best_or_fixed(best_params, "baseline_line_pad", baseline_line_pad)),
        baseline_line_pad_px=float(best_or_fixed(best_params, "baseline_line_pad_px", baseline_line_pad_px)),
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=float(
            best_or_fixed(best_params, "baseline_detector_threshold", baseline_detector_threshold)
        ),
        segmentator_checkpoint=segmentator_checkpoint,
        decode_with_segmentator=decode_with_segmentator,
        segmentator_scale_x=float(
            best_or_fixed(best_params, "segmentator_scale_x", segmentator_scale_x)
        ),
        segmentator_y_pad=float(
            best_or_fixed(best_params, "segmentator_y_pad", segmentator_y_pad)
        ),
        segmentator_x_pad=float(
            best_or_fixed(best_params, "segmentator_x_pad", segmentator_x_pad)
        ),
        segmentator_cut_threshold=best_or_fixed(
            best_params,
            "segmentator_cut_threshold",
            segmentator_cut_threshold,
        ),
        segmentator_cut_min_width=best_or_fixed(
            best_params,
            "segmentator_cut_min_width",
            segmentator_cut_min_width,
        ),
        segmentator_cut_max_width=best_or_fixed(
            best_params,
            "segmentator_cut_max_width",
            segmentator_cut_max_width,
        ),
        segmentator_cut_smooth_radius=best_or_fixed(
            best_params,
            "segmentator_cut_smooth_radius",
            segmentator_cut_smooth_radius,
        ),
        decode_method=decode_method,
        decode_top_k=decode_top_k,
        center_fraction=float(
            best_or_fixed(
                best_params,
                "center_fraction",
                center_fraction,
            )
        ),
        min_score_width=int(
            best_or_fixed(
                best_params,
                "min_score_width",
                min_score_width,
            )
        ),
        cut_weight=cut_weight,
        ocr_weight=ocr_weight,
        width_weight=width_weight,
        skip_cut_penalty=skip_cut_penalty,
        glyph_width_prior=glyph_width_prior,
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
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
    baseline_line_pad: float = 0.08,
    baseline_line_pad_px: float = 0.0,
    baseline_detector_checkpoint: Path | None = None,
    baseline_detector_threshold: float = 0.35,
    segmentator_checkpoint: Path | None = None,
    decode_with_segmentator: bool = False,
    segmentator_scale_x: float = 0.0,
    segmentator_y_pad: float = 0.0,
    segmentator_x_pad: float = 0.0,
    segmentator_cut_threshold: float | None = None,
    segmentator_cut_min_width: int | None = None,
    segmentator_cut_max_width: int | None = None,
    segmentator_cut_smooth_radius: int | None = None,
    decode_method: str = "cells",
    decode_top_k: int = 8,
    center_fraction: float = 0.6,
    min_score_width: int = 1,
    cut_weight: float = 1.0,
    ocr_weight: float = 1.0,
    width_weight: float = 0.05,
    skip_cut_penalty: float = 0.35,
    glyph_width_prior: dict[str, Any] | None = None,
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
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=baseline_detector_threshold,
        segmentator_checkpoint=segmentator_checkpoint,
        decode_with_segmentator=decode_with_segmentator,
        segmentator_scale_x=segmentator_scale_x,
        segmentator_y_pad=segmentator_y_pad,
        segmentator_x_pad=segmentator_x_pad,
        segmentator_cut_threshold=segmentator_cut_threshold,
        segmentator_cut_min_width=segmentator_cut_min_width,
        segmentator_cut_max_width=segmentator_cut_max_width,
        segmentator_cut_smooth_radius=segmentator_cut_smooth_radius,
        decode_method=decode_method,
        decode_top_k=decode_top_k,
        center_fraction=center_fraction,
        min_score_width=min_score_width,
        cut_weight=cut_weight,
        ocr_weight=ocr_weight,
        width_weight=width_weight,
        skip_cut_penalty=skip_cut_penalty,
        glyph_width_prior=glyph_width_prior,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FCN OCR on a Label Studio JSON export.")
    parser.add_argument("--config", default=None, help="Evaluation YAML config.")
    parser.add_argument("--json", default=None, help="Path to Label Studio export JSON.")
    parser.add_argument("--images", default=None, help="Folder with images.")
    parser.add_argument(
        "--inference-config",
        default=None,
        help="Optional inference YAML used as the fixed baseline/segmentator/OCR configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to OCR checkpoint. Overrides ocr.checkpoint from --inference-config.",
    )
    parser.add_argument("--out", default="ocr_metrics.csv", help="Output CSV path.")
    parser.add_argument("--device", default=None, help="Device to use: cuda, cpu, or empty for auto.")
    parser.add_argument(
        "--scale-x",
        type=float,
        default=None,
        help="Normalized horizontal scale for OCR.",
    )
    parser.add_argument(
        "--y-pad",
        type=float,
        default=None,
        help="Normalized vertical padding/crop for OCR.",
    )
    parser.add_argument(
        "--x-pad",
        type=float,
        default=None,
        help="Normalized symmetric horizontal padding for OCR.",
    )

    parser.add_argument(
        "--baseline-crop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable shared baseline detection and crop.",
    )
    deskew_group = parser.add_mutually_exclusive_group()
    deskew_group.add_argument(
        "--baseline-deskew",
        dest="baseline_deskew",
        action="store_true",
    )
    deskew_group.add_argument(
        "--no-baseline-deskew",
        dest="baseline_deskew",
        action="store_false",
    )
    parser.set_defaults(baseline_deskew=None)
    parser.add_argument("--baseline-max-angle", type=float, default=None)
    parser.add_argument("--baseline-line-pad", type=float, default=None)
    parser.add_argument("--baseline-line-pad-px", type=float, default=None)
    parser.add_argument("--baseline-detector-checkpoint", default=None)
    parser.add_argument("--baseline-detector-threshold", type=float, default=None)

    parser.add_argument("--segmentator-checkpoint", default=None)
    parser.add_argument(
        "--decode-with-segmentator",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--segmentator-scale-x",
        type=float,
        default=None,
        help="Normalized horizontal scale used only by the vertical segmentator.",
    )
    parser.add_argument(
        "--segmentator-y-pad",
        type=float,
        default=None,
        help="Normalized vertical padding/crop used only by the vertical segmentator.",
    )
    parser.add_argument(
        "--segmentator-x-pad",
        type=float,
        default=None,
        help="Normalized horizontal padding used only by the vertical segmentator.",
    )
    parser.add_argument("--segmentator-cut-threshold", type=float, default=None)
    parser.add_argument("--segmentator-cut-min-width", type=int, default=None)
    parser.add_argument("--segmentator-cut-max-width", type=int, default=None)
    parser.add_argument("--segmentator-cut-smooth-radius", type=int, default=None)
    parser.add_argument("--decode-method", choices=["cells", "dp"], default=None)
    parser.add_argument("--decode-top-k", type=int, default=None)
    parser.add_argument("--center-fraction", type=float, default=None)
    parser.add_argument("--min-score-width", type=int, default=None)
    parser.add_argument("--cut-weight", type=float, default=None)
    parser.add_argument("--ocr-weight", type=float, default=None)
    parser.add_argument("--width-weight", type=float, default=None)
    parser.add_argument("--skip-cut-penalty", type=float, default=None)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for OCR and OCR+segmentator evaluation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to evaluate.")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N recognized images; 0 disables.")
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="If > 0, tune enabled parameter ranges with Optuna before final evaluation.",
    )
    parser.add_argument(
        "--optuna-scale-x-min",
        type=float,
        default=-0.25,
    )
    parser.add_argument(
        "--optuna-scale-x-max",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--optuna-y-pad-min",
        type=float,
        default=-0.25,
    )
    parser.add_argument(
        "--optuna-y-pad-max",
        type=float,
        default=0.25,
    )
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
    parser.add_argument(
        "--optuna-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the interactive Optuna progress bar.",
    )

    parser.add_argument(
        "--optuna-x-pad-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--optuna-x-pad-max",
        type=float,
        default=None,
    )
    parser.add_argument("--optuna-segmentator-scale-x-min", type=float, default=None)
    parser.add_argument("--optuna-segmentator-scale-x-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-y-pad-min", type=float, default=None)
    parser.add_argument("--optuna-segmentator-y-pad-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-x-pad-min", type=float, default=None)
    parser.add_argument("--optuna-segmentator-x-pad-max", type=float, default=None)
    parser.add_argument(
        "--optuna-tune-baseline-line-pad",
        action="store_true",
        help="Explicitly tune baseline_line_pad when its min/max range is provided.",
    )
    parser.add_argument(
        "--optuna-tune-baseline-line-pad-px",
        action="store_true",
        help="Explicitly tune absolute baseline_line_pad_px.",
    )
    parser.add_argument("--optuna-baseline-detector-threshold-min", type=float, default=None)
    parser.add_argument("--optuna-baseline-detector-threshold-max", type=float, default=None)
    parser.add_argument("--optuna-baseline-line-pad-min", type=float, default=None)
    parser.add_argument("--optuna-baseline-line-pad-max", type=float, default=None)
    parser.add_argument("--optuna-baseline-line-pad-px-min", type=float, default=None)
    parser.add_argument("--optuna-baseline-line-pad-px-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-cut-threshold-min", type=float, default=None)
    parser.add_argument("--optuna-segmentator-cut-threshold-max", type=float, default=None)
    parser.add_argument("--optuna-segmentator-cut-min-width-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-min-width-max", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-max-width-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-max-width-max", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-smooth-radius-min", type=int, default=None)
    parser.add_argument("--optuna-segmentator-cut-smooth-radius-max", type=int, default=None)
    parser.add_argument("--optuna-center-fraction-min", type=float, default=None)
    parser.add_argument("--optuna-center-fraction-max", type=float, default=None)
    parser.add_argument("--optuna-min-score-width-min", type=int, default=None)
    parser.add_argument("--optuna-min-score-width-max", type=int, default=None)
    return parse_args_with_evaluation_config(
        parser,
        path_fields=(
            "json",
            "images",
            "inference_config",
            "checkpoint",
            "out",
            "baseline_detector_checkpoint",
            "segmentator_checkpoint",
            "optuna_trials_out",
        ),
        required_fields=("json", "images"),
        argv=argv,
    )


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def resolve_inference_args(args: argparse.Namespace) -> argparse.Namespace:
    config = InferenceConfig.load(args.inference_config) if args.inference_config else None
    baseline = config.baseline_detection if config is not None else None
    ocr = config.fcn_ocr if config is not None else None
    segmentator = config.vertical_segmentation if config is not None else None
    decode = ocr.decode if ocr is not None else None

    args.device = _first_defined(args.device, config.device if config else None)
    args.checkpoint = _first_defined(
        args.checkpoint,
        ocr.checkpoint if ocr is not None else None,
    )
    if args.checkpoint is None:
        raise ValueError(
            "--checkpoint is required unless --inference-config contains "
            "an fcn_ocr section"
        )

    ocr_preprocess = ocr.preprocessing if ocr is not None else None
    args.scale_x = float(
        _first_defined(
            args.scale_x,
            ocr_preprocess.scale_x if ocr_preprocess is not None else None,
            0.0,
        )
    )
    args.y_pad = float(
        _first_defined(
            args.y_pad,
            ocr_preprocess.y_pad if ocr_preprocess is not None else None,
            0.0,
        )
    )
    args.x_pad = float(
        _first_defined(
            args.x_pad,
            ocr_preprocess.x_pad if ocr_preprocess is not None else None,
            0.0,
        )
    )

    args.baseline_crop = bool(
        _first_defined(
            args.baseline_crop,
            baseline.enabled if baseline is not None else None,
            False,
        )
    )
    args.baseline_deskew = bool(
        _first_defined(
            args.baseline_deskew,
            baseline.deskew if baseline is not None else None,
            True,
        )
    )
    args.baseline_max_angle = float(
        _first_defined(
            args.baseline_max_angle,
            baseline.max_angle if baseline is not None else None,
            12.0,
        )
    )
    args.baseline_line_pad = float(
        _first_defined(
            args.baseline_line_pad,
            baseline.line_pad if baseline is not None else None,
            0.08,
        )
    )
    args.baseline_line_pad_px = float(
        _first_defined(
            args.baseline_line_pad_px,
            baseline.line_pad_px if baseline is not None else None,
            0.0,
        )
    )
    args.baseline_detector_checkpoint = _first_defined(
        args.baseline_detector_checkpoint,
        baseline.detector_checkpoint if baseline is not None else None,
    )
    args.baseline_detector_threshold = float(
        _first_defined(
            args.baseline_detector_threshold,
            baseline.detector_threshold if baseline is not None else None,
            0.35,
        )
    )

    args.segmentator_checkpoint = _first_defined(
        args.segmentator_checkpoint,
        segmentator.checkpoint if segmentator is not None else None,
    )
    segmentator_preprocess = (
        segmentator.preprocessing
        if segmentator is not None
        else None
    )
    args.segmentator_scale_x = float(
        _first_defined(
            args.segmentator_scale_x,
            segmentator_preprocess.scale_x if segmentator_preprocess is not None else None,
            0.0,
        )
    )
    args.segmentator_y_pad = float(
        _first_defined(
            args.segmentator_y_pad,
            segmentator_preprocess.y_pad if segmentator_preprocess is not None else None,
            0.0,
        )
    )
    args.segmentator_x_pad = float(
        _first_defined(
            args.segmentator_x_pad,
            segmentator_preprocess.x_pad if segmentator_preprocess is not None else None,
            0.0,
        )
    )
    for name in (
        "cut_threshold",
        "cut_min_width",
        "cut_max_width",
        "cut_smooth_radius",
    ):
        argument_name = f"segmentator_{name}"
        config_value = getattr(segmentator, name) if segmentator is not None else None
        setattr(
            args,
            argument_name,
            _first_defined(getattr(args, argument_name), config_value),
        )

    args.decode_with_segmentator = bool(
        _first_defined(
            args.decode_with_segmentator,
            decode.enabled if decode is not None else None,
            False,
        )
    )
    args.decode_method = str(
        _first_defined(
            args.decode_method,
            decode.method if decode is not None else None,
            "cells",
        )
    )
    args.decode_top_k = int(
        _first_defined(
            args.decode_top_k,
            decode.top_k if decode is not None else None,
            8,
        )
    )
    args.center_fraction = float(
        _first_defined(
            args.center_fraction,
            decode.center_fraction if decode is not None else None,
            0.6,
        )
    )
    args.min_score_width = int(
        _first_defined(
            args.min_score_width,
            decode.min_score_width if decode is not None else None,
            1,
        )
    )
    args.cut_weight = float(
        _first_defined(
            args.cut_weight,
            decode.cut_weight if decode is not None else None,
            1.0,
        )
    )
    args.ocr_weight = float(
        _first_defined(
            args.ocr_weight,
            decode.ocr_weight if decode is not None else None,
            1.0,
        )
    )
    args.width_weight = float(
        _first_defined(
            args.width_weight,
            decode.width_weight if decode is not None else None,
            0.05,
        )
    )
    args.skip_cut_penalty = float(
        _first_defined(
            args.skip_cut_penalty,
            decode.skip_cut_penalty if decode is not None else None,
            0.35,
        )
    )
    args.glyph_width_prior = (
        decode.glyph_width_prior.model_dump() if decode is not None else None
    )
    if args.decode_with_segmentator and args.segmentator_checkpoint is None:
        raise ValueError(
            "Segmentator decoding requires segmentator.checkpoint in --inference-config "
            "or --segmentator-checkpoint"
        )
    args.inference_debug_top_k = config.debug.top_k if config is not None else 8
    return args


def _common_eval_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "baseline_crop": args.baseline_crop,
        "baseline_deskew": args.baseline_deskew,
        "baseline_max_angle": args.baseline_max_angle,
        "baseline_line_pad": args.baseline_line_pad,
        "baseline_line_pad_px": args.baseline_line_pad_px,
        "baseline_detector_checkpoint": Path(args.baseline_detector_checkpoint) if args.baseline_detector_checkpoint else None,
        "baseline_detector_threshold": args.baseline_detector_threshold,
        "segmentator_checkpoint": Path(args.segmentator_checkpoint) if args.segmentator_checkpoint else None,
        "decode_with_segmentator": args.decode_with_segmentator,
        "segmentator_scale_x": args.segmentator_scale_x,
        "segmentator_y_pad": args.segmentator_y_pad,
        "segmentator_x_pad": args.segmentator_x_pad,
        "segmentator_cut_threshold": args.segmentator_cut_threshold,
        "segmentator_cut_min_width": args.segmentator_cut_min_width,
        "segmentator_cut_max_width": args.segmentator_cut_max_width,
        "segmentator_cut_smooth_radius": args.segmentator_cut_smooth_radius,
        "decode_method": args.decode_method,
        "decode_top_k": args.decode_top_k,
        "center_fraction": args.center_fraction,
        "min_score_width": args.min_score_width,
        "cut_weight": args.cut_weight,
        "ocr_weight": args.ocr_weight,
        "width_weight": args.width_weight,
        "skip_cut_penalty": args.skip_cut_penalty,
        "glyph_width_prior": args.glyph_width_prior,
    }


def _print_inference_command(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    _, jobs = build_rows_and_jobs(Path(args.json), Path(args.images), args.limit)
    image_path = str(jobs[0][1]) if jobs else "<IMAGE_PATH>"
    has_segmentator = bool(metrics.get("segmentator_checkpoint"))
    config_data: dict[str, Any] = {
        "device": args.device,
        "baseline_detection": {
            "enabled": metrics["baseline_crop"],
            "detector_checkpoint": (
                str(Path(metrics["baseline_detector_checkpoint"]).expanduser().resolve())
                if metrics.get("baseline_detector_checkpoint")
                else None
            ),
            "detector_threshold": metrics["baseline_detector_threshold"],
            "deskew": args.baseline_deskew,
            "max_angle": args.baseline_max_angle,
            "line_pad": metrics["baseline_line_pad"],
            "line_pad_px": metrics["baseline_line_pad_px"],
        },
        "fcn_ocr": {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "preprocessing": {
                "scale_x": metrics["scale_x"],
                "y_pad": metrics["y_pad"],
                "x_pad": metrics["x_pad"],
            },
            "decode": {
                "enabled": bool(metrics["decode_with_segmentator"] and has_segmentator),
                "method": metrics["decode_method"],
                "top_k": metrics["decode_top_k"],
                "center_fraction": metrics["center_fraction"],
                "min_score_width": metrics["min_score_width"],
                "cut_weight": metrics["cut_weight"],
                "ocr_weight": metrics["ocr_weight"],
                "width_weight": metrics["width_weight"],
                "skip_cut_penalty": metrics["skip_cut_penalty"],
                "glyph_width_prior": metrics.get("glyph_width_prior", {}),
            },
        },
        "debug": {
            "top_k": getattr(args, "inference_debug_top_k", 8),
        },
    }
    if has_segmentator:
        config_data["vertical_segmentation"] = {
            "checkpoint": str(
                Path(metrics["segmentator_checkpoint"]).expanduser().resolve()
            ),
            "preprocessing": {
                "scale_x": metrics["segmentator_scale_x"],
                "y_pad": metrics["segmentator_y_pad"],
                "x_pad": metrics["segmentator_x_pad"],
            },
            "cut_threshold": metrics["segmentator_cut_threshold"],
            "cut_min_width": metrics["segmentator_cut_min_width"],
            "cut_max_width": metrics["segmentator_cut_max_width"],
            "cut_smooth_radius": metrics["segmentator_cut_smooth_radius"],
        }
    save_and_print_inference_command(config_data, args.out, image_path)


def run_evaluation(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path | None = None,
    output_csv: Path | None = None,
    trials_output: Path | None = None,
    print_inference_command: bool = True,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path or Path(args.checkpoint)
    output_csv = output_csv or Path(args.out)
    if trials_output is None and args.optuna_trials_out:
        trials_output = Path(args.optuna_trials_out)
    common_kwargs = _common_eval_kwargs(args)
    if args.optuna_trials > 0:
        metrics = optimize_preprocess(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
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
            trials_output=trials_output,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            progress=args.optuna_progress,
            optuna_tune_baseline_line_pad=args.optuna_tune_baseline_line_pad,
            optuna_tune_baseline_line_pad_px=args.optuna_tune_baseline_line_pad_px,
            x_pad_min=args.optuna_x_pad_min,
            x_pad_max=args.optuna_x_pad_max,
            segmentator_scale_x_min=args.optuna_segmentator_scale_x_min,
            segmentator_scale_x_max=args.optuna_segmentator_scale_x_max,
            segmentator_y_pad_min=args.optuna_segmentator_y_pad_min,
            segmentator_y_pad_max=args.optuna_segmentator_y_pad_max,
            segmentator_x_pad_min=args.optuna_segmentator_x_pad_min,
            segmentator_x_pad_max=args.optuna_segmentator_x_pad_max,
            baseline_detector_threshold_min=args.optuna_baseline_detector_threshold_min,
            baseline_detector_threshold_max=args.optuna_baseline_detector_threshold_max,
            baseline_line_pad_min=args.optuna_baseline_line_pad_min,
            baseline_line_pad_max=args.optuna_baseline_line_pad_max,
            baseline_line_pad_px_min=args.optuna_baseline_line_pad_px_min,
            baseline_line_pad_px_max=args.optuna_baseline_line_pad_px_max,
            segmentator_cut_threshold_min=args.optuna_segmentator_cut_threshold_min,
            segmentator_cut_threshold_max=args.optuna_segmentator_cut_threshold_max,
            segmentator_cut_min_width_min=args.optuna_segmentator_cut_min_width_min,
            segmentator_cut_min_width_max=args.optuna_segmentator_cut_min_width_max,
            segmentator_cut_max_width_min=args.optuna_segmentator_cut_max_width_min,
            segmentator_cut_max_width_max=args.optuna_segmentator_cut_max_width_max,
            segmentator_cut_smooth_radius_min=args.optuna_segmentator_cut_smooth_radius_min,
            segmentator_cut_smooth_radius_max=args.optuna_segmentator_cut_smooth_radius_max,
            center_fraction_min=args.optuna_center_fraction_min,
            center_fraction_max=args.optuna_center_fraction_max,
            min_score_width_min=args.optuna_min_score_width_min,
            min_score_width_max=args.optuna_min_score_width_max,
            **common_kwargs,
        )
    else:
        metrics = evaluate(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
            device=args.device,
            scale_x=args.scale_x,
            y_pad=args.y_pad,
            x_pad=args.x_pad,
            batch_size=args.batch_size,
            limit=args.limit,
            log_every=args.log_every,
            **common_kwargs,
        )
    if print_inference_command:
        _print_inference_command(args, metrics)
    return metrics


def main() -> None:
    args = resolve_inference_args(parse_args())
    run_evaluation(args)


if __name__ == "__main__":
    main()
