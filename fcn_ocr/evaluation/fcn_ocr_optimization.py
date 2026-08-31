from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from fcn_ocr.evaluation.images import RGBImageCache
from fcn_ocr.evaluation.optuna import (
    best_or_fixed,
    create_study,
    file_contract,
    optimize_with_progress,
    suggest_float_or_fixed,
    suggest_int_or_fixed,
    validate_float_range,
    validate_int_range,
    validate_study_contract,
)
from fcn_ocr.evaluation.reporting import (
    append_tsv_row,
)

from .fcn_ocr_runner import OCRPipelinePool, build_rows_and_jobs, evaluate_prepared


def _parameter_value(fixed: Any, minimum: Any, maximum: Any) -> Any:
    return fixed if minimum is None else [minimum, maximum]

def _trial_params_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    names = [
        "scale_x",
        "y_pad",
        "x_pad",
        "baseline_detector_threshold",
        "baseline_line_pad",
        "baseline_line_pad_px",
        "vertical_segmentation_scale_x",
        "vertical_segmentation_y_pad",
        "vertical_segmentation_x_pad",
        "vertical_segmentation_cut_threshold",
        "vertical_segmentation_cut_min_width",
        "vertical_segmentation_cut_max_width",
        "vertical_segmentation_cut_smooth_radius",
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
    scale_x: float,
    y_pad: float,
    scale_x_min: float | None,
    scale_x_max: float | None,
    y_pad_min: float | None,
    y_pad_max: float | None,
    x_pad: float,
    metric_name: str,
    log_every: int,
    trials_output: Path | None,
    study_name: str | None = None,
    storage: str | None = None,
    baseline_crop: bool = False,
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
    baseline_line_pad: float = 0.08,
    baseline_line_pad_px: float = 0.0,
    baseline_detector_checkpoint: Path | None = None,
    baseline_detector_threshold: float = 0.35,
    vertical_segmentation_checkpoint: Path | None = None,
    decode_with_vertical_segmentation: bool = False,
    vertical_segmentation_scale_x: float = 0.0,
    vertical_segmentation_y_pad: float = 0.0,
    vertical_segmentation_x_pad: float = 0.0,
    vertical_segmentation_cut_threshold: float | None = None,
    vertical_segmentation_cut_min_width: int | None = None,
    vertical_segmentation_cut_max_width: int | None = None,
    vertical_segmentation_cut_smooth_radius: int | None = None,
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
    vertical_segmentation_scale_x_min: float | None = None,
    vertical_segmentation_scale_x_max: float | None = None,
    vertical_segmentation_y_pad_min: float | None = None,
    vertical_segmentation_y_pad_max: float | None = None,
    vertical_segmentation_x_pad_min: float | None = None,
    vertical_segmentation_x_pad_max: float | None = None,
    baseline_detector_threshold_min: float | None = None,
    baseline_detector_threshold_max: float | None = None,
    baseline_line_pad_min: float | None = None,
    baseline_line_pad_max: float | None = None,
    baseline_line_pad_px_min: float | None = None,
    baseline_line_pad_px_max: float | None = None,
    vertical_segmentation_cut_threshold_min: float | None = None,
    vertical_segmentation_cut_threshold_max: float | None = None,
    vertical_segmentation_cut_min_width_min: int | None = None,
    vertical_segmentation_cut_min_width_max: int | None = None,
    vertical_segmentation_cut_max_width_min: int | None = None,
    vertical_segmentation_cut_max_width_max: int | None = None,
    vertical_segmentation_cut_smooth_radius_min: int | None = None,
    vertical_segmentation_cut_smooth_radius_max: int | None = None,
    center_fraction_min: float | None = None,
    center_fraction_max: float | None = None,
    min_score_width_min: int | None = None,
    min_score_width_max: int | None = None,
    progress: bool = False,
    optuna_seed: int = 0,
    image_cache_mb: float = 512.0,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    validate_float_range("scale_x", scale_x_min, scale_x_max, lower=-0.95)
    validate_float_range("y_pad", y_pad_min, y_pad_max, lower=-0.95)
    validate_float_range("x_pad", x_pad_min, x_pad_max, lower=0.0)
    validate_float_range(
        "baseline_detector_threshold",
        baseline_detector_threshold_min,
        baseline_detector_threshold_max,
        lower=0.0,
        upper=1.0,
    )
    validate_float_range("baseline_line_pad", baseline_line_pad_min, baseline_line_pad_max, lower=0.0)
    validate_float_range(
        "baseline_line_pad_px",
        baseline_line_pad_px_min,
        baseline_line_pad_px_max,
        lower=0.0,
    )
    validate_float_range(
        "vertical_segmentation_scale_x",
        vertical_segmentation_scale_x_min,
        vertical_segmentation_scale_x_max,
        lower=-0.95,
    )
    validate_float_range(
        "vertical_segmentation_y_pad",
        vertical_segmentation_y_pad_min,
        vertical_segmentation_y_pad_max,
        lower=-0.95,
    )
    validate_float_range(
        "vertical_segmentation_x_pad",
        vertical_segmentation_x_pad_min,
        vertical_segmentation_x_pad_max,
        lower=0.0,
    )
    validate_float_range(
        "vertical_segmentation_cut_threshold",
        vertical_segmentation_cut_threshold_min,
        vertical_segmentation_cut_threshold_max,
        lower=0.0,
        upper=1.0,
    )
    validate_int_range(
        "vertical_segmentation_cut_min_width",
        vertical_segmentation_cut_min_width_min,
        vertical_segmentation_cut_min_width_max,
        lower=1,
    )
    validate_int_range(
        "vertical_segmentation_cut_max_width",
        vertical_segmentation_cut_max_width_min,
        vertical_segmentation_cut_max_width_max,
        lower=0,
    )
    validate_int_range(
        "vertical_segmentation_cut_smooth_radius",
        vertical_segmentation_cut_smooth_radius_min,
        vertical_segmentation_cut_smooth_radius_max,
        lower=0,
    )
    validate_float_range(
        "center_fraction",
        center_fraction_min,
        center_fraction_max,
        lower=0.0,
        upper=1.0,
    )
    validate_int_range(
        "min_score_width",
        min_score_width_min,
        min_score_width_max,
        lower=1,
    )

    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    direction = "minimize" if metric_name in {"average_levenshtein", "total_levenshtein"} else "maximize"
    study = create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        seed=optuna_seed,
    )
    pipeline_pool = OCRPipelinePool(verbose_on_create=True)
    image_cache = RGBImageCache(image_cache_mb)
    tune_active_baseline_line_pad = bool(baseline_crop) and baseline_line_pad_min is not None
    tune_active_baseline_line_pad_px = bool(baseline_crop) and baseline_line_pad_px_min is not None
    tune_active_baseline_detector = (
        bool(baseline_crop) and baseline_detector_checkpoint is not None
    )
    tune_active_vertical_segmentation = (
        bool(decode_with_vertical_segmentation)
        and vertical_segmentation_checkpoint is not None
    )
    has_tunable_parameter = any(
        minimum is not None
        for minimum in (
            scale_x_min,
            y_pad_min,
            x_pad_min,
            baseline_line_pad_min if tune_active_baseline_line_pad else None,
            baseline_line_pad_px_min if tune_active_baseline_line_pad_px else None,
            baseline_detector_threshold_min if tune_active_baseline_detector else None,
            vertical_segmentation_scale_x_min if tune_active_vertical_segmentation else None,
            vertical_segmentation_y_pad_min if tune_active_vertical_segmentation else None,
            vertical_segmentation_x_pad_min if tune_active_vertical_segmentation else None,
            vertical_segmentation_cut_threshold_min if tune_active_vertical_segmentation else None,
            vertical_segmentation_cut_min_width_min if tune_active_vertical_segmentation else None,
            vertical_segmentation_cut_max_width_min if tune_active_vertical_segmentation else None,
            vertical_segmentation_cut_smooth_radius_min if tune_active_vertical_segmentation else None,
            center_fraction_min if tune_active_vertical_segmentation else None,
            min_score_width_min if tune_active_vertical_segmentation else None,
        )
    )
    if not has_tunable_parameter:
        raise ValueError(
            "Optuna has no tunable OCR parameters; set at least one "
            "parameters.<name>: [min, max] range or use optuna_trials: 0"
        )
    validate_study_contract(
        study,
        {
            "evaluator": "fcn_ocr",
            "json": file_contract(json_path),
            "images": str(images_dir.expanduser().resolve()),
            "checkpoint": file_contract(checkpoint_path),
            "metric": metric_name,
            "limit": limit,
            "parameters": {
                "scale_x": _parameter_value(scale_x, scale_x_min, scale_x_max),
                "y_pad": _parameter_value(y_pad, y_pad_min, y_pad_max),
                "x_pad": _parameter_value(x_pad, x_pad_min, x_pad_max),
                "baseline_crop": baseline_crop,
                "baseline_deskew": baseline_deskew,
                "baseline_max_angle": baseline_max_angle,
                "baseline_line_pad": _parameter_value(
                    baseline_line_pad,
                    baseline_line_pad_min if tune_active_baseline_line_pad else None,
                    baseline_line_pad_max if tune_active_baseline_line_pad else None,
                ),
                "baseline_line_pad_px": _parameter_value(
                    baseline_line_pad_px,
                    baseline_line_pad_px_min if tune_active_baseline_line_pad_px else None,
                    baseline_line_pad_px_max if tune_active_baseline_line_pad_px else None,
                ),
                "baseline_detector_checkpoint": (
                    file_contract(baseline_detector_checkpoint)
                    if baseline_detector_checkpoint is not None
                    else None
                ),
                "baseline_detector_threshold": _parameter_value(
                    baseline_detector_threshold,
                    baseline_detector_threshold_min if tune_active_baseline_detector else None,
                    baseline_detector_threshold_max if tune_active_baseline_detector else None,
                ),
                "vertical_segmentation_checkpoint": (
                    file_contract(vertical_segmentation_checkpoint)
                    if vertical_segmentation_checkpoint is not None
                    else None
                ),
                "vertical_segmentation_scale_x": _parameter_value(
                    vertical_segmentation_scale_x,
                    vertical_segmentation_scale_x_min if tune_active_vertical_segmentation else None,
                    vertical_segmentation_scale_x_max if tune_active_vertical_segmentation else None,
                ),
                "vertical_segmentation_y_pad": _parameter_value(
                    vertical_segmentation_y_pad,
                    vertical_segmentation_y_pad_min if tune_active_vertical_segmentation else None,
                    vertical_segmentation_y_pad_max if tune_active_vertical_segmentation else None,
                ),
                "vertical_segmentation_x_pad": _parameter_value(
                    vertical_segmentation_x_pad,
                    vertical_segmentation_x_pad_min if tune_active_vertical_segmentation else None,
                    vertical_segmentation_x_pad_max if tune_active_vertical_segmentation else None,
                ),
                "vertical_segmentation_cut_threshold": _parameter_value(
                    vertical_segmentation_cut_threshold,
                    vertical_segmentation_cut_threshold_min if tune_active_vertical_segmentation else None,
                    vertical_segmentation_cut_threshold_max if tune_active_vertical_segmentation else None,
                ),
                "vertical_segmentation_cut_min_width": _parameter_value(
                    vertical_segmentation_cut_min_width,
                    vertical_segmentation_cut_min_width_min if tune_active_vertical_segmentation else None,
                    vertical_segmentation_cut_min_width_max if tune_active_vertical_segmentation else None,
                ),
                "vertical_segmentation_cut_max_width": _parameter_value(
                    vertical_segmentation_cut_max_width,
                    vertical_segmentation_cut_max_width_min if tune_active_vertical_segmentation else None,
                    vertical_segmentation_cut_max_width_max if tune_active_vertical_segmentation else None,
                ),
                "vertical_segmentation_cut_smooth_radius": _parameter_value(
                    vertical_segmentation_cut_smooth_radius,
                    vertical_segmentation_cut_smooth_radius_min if tune_active_vertical_segmentation else None,
                    vertical_segmentation_cut_smooth_radius_max if tune_active_vertical_segmentation else None,
                ),
                "decode_method": decode_method,
                "decode_top_k": decode_top_k,
                "center_fraction": _parameter_value(
                    center_fraction,
                    center_fraction_min if tune_active_vertical_segmentation else None,
                    center_fraction_max if tune_active_vertical_segmentation else None,
                ),
                "min_score_width": _parameter_value(
                    min_score_width,
                    min_score_width_min if tune_active_vertical_segmentation else None,
                    min_score_width_max if tune_active_vertical_segmentation else None,
                ),
                "cut_weight": cut_weight,
                "ocr_weight": ocr_weight,
                "width_weight": width_weight,
                "skip_cut_penalty": skip_cut_penalty,
                "glyph_width_prior": glyph_width_prior,
            },
        },
    )

    def objective(trial) -> float:
        current_scale_x = suggest_float_or_fixed(
            trial, "scale_x", scale_x, scale_x_min, scale_x_max
        )
        current_y_pad = suggest_float_or_fixed(
            trial, "y_pad", y_pad, y_pad_min, y_pad_max
        )
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
        current_vertical_segmentation_scale_x = (
            suggest_float_or_fixed(
                trial,
                "vertical_segmentation_scale_x",
                vertical_segmentation_scale_x,
                vertical_segmentation_scale_x_min,
                vertical_segmentation_scale_x_max,
            )
            if tune_active_vertical_segmentation
            else vertical_segmentation_scale_x
        )
        current_vertical_segmentation_y_pad = (
            suggest_float_or_fixed(
                trial,
                "vertical_segmentation_y_pad",
                vertical_segmentation_y_pad,
                vertical_segmentation_y_pad_min,
                vertical_segmentation_y_pad_max,
            )
            if tune_active_vertical_segmentation
            else vertical_segmentation_y_pad
        )
        current_vertical_segmentation_x_pad = (
            suggest_float_or_fixed(
                trial,
                "vertical_segmentation_x_pad",
                vertical_segmentation_x_pad,
                vertical_segmentation_x_pad_min,
                vertical_segmentation_x_pad_max,
            )
            if tune_active_vertical_segmentation
            else vertical_segmentation_x_pad
        )
        current_vertical_segmentation_cut_threshold = (
            suggest_float_or_fixed(
                trial,
                "vertical_segmentation_cut_threshold",
                vertical_segmentation_cut_threshold,
                vertical_segmentation_cut_threshold_min,
                vertical_segmentation_cut_threshold_max,
            )
            if tune_active_vertical_segmentation
            else vertical_segmentation_cut_threshold
        )
        current_vertical_segmentation_cut_min_width = (
            suggest_int_or_fixed(
                trial,
                "vertical_segmentation_cut_min_width",
                vertical_segmentation_cut_min_width,
                vertical_segmentation_cut_min_width_min,
                vertical_segmentation_cut_min_width_max,
            )
            if tune_active_vertical_segmentation
            else vertical_segmentation_cut_min_width
        )
        current_vertical_segmentation_cut_max_width = (
            suggest_int_or_fixed(
                trial,
                "vertical_segmentation_cut_max_width",
                vertical_segmentation_cut_max_width,
                vertical_segmentation_cut_max_width_min,
                vertical_segmentation_cut_max_width_max,
            )
            if tune_active_vertical_segmentation
            else vertical_segmentation_cut_max_width
        )
        current_vertical_segmentation_cut_smooth_radius = (
            suggest_int_or_fixed(
                trial,
                "vertical_segmentation_cut_smooth_radius",
                vertical_segmentation_cut_smooth_radius,
                vertical_segmentation_cut_smooth_radius_min,
                vertical_segmentation_cut_smooth_radius_max,
            )
            if tune_active_vertical_segmentation
            else vertical_segmentation_cut_smooth_radius
        )
        current_center_fraction = (
            suggest_float_or_fixed(
                trial,
                "center_fraction",
                center_fraction,
                center_fraction_min,
                center_fraction_max,
            )
            if tune_active_vertical_segmentation
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
            if tune_active_vertical_segmentation
            else min_score_width
        )
        metrics = evaluate_prepared(
            base_rows,
            jobs,
            checkpoint_path=checkpoint_path,
            output_csv=None,
            device=device,
            scale_x=float(current_scale_x or 0.0),
            y_pad=float(current_y_pad or 0.0),
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
            vertical_segmentation_checkpoint=vertical_segmentation_checkpoint,
            decode_with_vertical_segmentation=decode_with_vertical_segmentation,
            vertical_segmentation_scale_x=float(current_vertical_segmentation_scale_x or 0.0),
            vertical_segmentation_y_pad=float(current_vertical_segmentation_y_pad or 0.0),
            vertical_segmentation_x_pad=float(current_vertical_segmentation_x_pad or 0.0),
            vertical_segmentation_cut_threshold=current_vertical_segmentation_cut_threshold,
            vertical_segmentation_cut_min_width=current_vertical_segmentation_cut_min_width,
            vertical_segmentation_cut_max_width=current_vertical_segmentation_cut_max_width,
            vertical_segmentation_cut_smooth_radius=current_vertical_segmentation_cut_smooth_radius,
            decode_method=decode_method,
            decode_top_k=decode_top_k,
            center_fraction=float(
                center_fraction
                if current_center_fraction is None
                else current_center_fraction
            ),
            min_score_width=int(current_min_score_width or 1),
            cut_weight=cut_weight,
            ocr_weight=ocr_weight,
            width_weight=width_weight,
            skip_cut_penalty=skip_cut_penalty,
            glyph_width_prior=glyph_width_prior,
            pipeline_pool=pipeline_pool,
            image_loader=image_cache.load,
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
        f"ocr_scale_x={scale_x if scale_x_min is None else f'[{scale_x_min}, {scale_x_max}]'}, "
        f"ocr_y_pad={y_pad if y_pad_min is None else f'[{y_pad_min}, {y_pad_max}]'}, "
        f"decode_with_vertical_segmentation={decode_with_vertical_segmentation}, baseline_crop={baseline_crop}, "
        f"vertical_segmentation_scale_x=[{vertical_segmentation_scale_x_min}, {vertical_segmentation_scale_x_max}], "
        f"vertical_segmentation_y_pad=[{vertical_segmentation_y_pad_min}, {vertical_segmentation_y_pad_max}], "
        f"vertical_segmentation_x_pad=[{vertical_segmentation_x_pad_min}, {vertical_segmentation_x_pad_max}], "
        f"center_fraction=[{center_fraction_min}, {center_fraction_max}], "
        f"min_score_width=[{min_score_width_min}, {min_score_width_max}], "
        f"seed={optuna_seed}, image_cache={image_cache_mb:g} MB"
    )
    optimize_with_progress(
        study,
        objective,
        n_trials=trials,
        metric_name=metric_name,
        enabled=progress,
    )

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_value = float(best_trial.value)
    print(f"Best Optuna params: {json.dumps(best_params, ensure_ascii=False, sort_keys=True)}")
    print(f"Best {metric_name}: {best_value:.8f}")

    final_metrics = evaluate_prepared(
        base_rows,
        jobs,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        device=device,
        scale_x=float(best_or_fixed(best_params, "scale_x", scale_x)),
        y_pad=float(best_or_fixed(best_params, "y_pad", y_pad)),
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
        vertical_segmentation_checkpoint=vertical_segmentation_checkpoint,
        decode_with_vertical_segmentation=decode_with_vertical_segmentation,
        vertical_segmentation_scale_x=float(
            best_or_fixed(best_params, "vertical_segmentation_scale_x", vertical_segmentation_scale_x)
        ),
        vertical_segmentation_y_pad=float(
            best_or_fixed(best_params, "vertical_segmentation_y_pad", vertical_segmentation_y_pad)
        ),
        vertical_segmentation_x_pad=float(
            best_or_fixed(best_params, "vertical_segmentation_x_pad", vertical_segmentation_x_pad)
        ),
        vertical_segmentation_cut_threshold=best_or_fixed(
            best_params,
            "vertical_segmentation_cut_threshold",
            vertical_segmentation_cut_threshold,
        ),
        vertical_segmentation_cut_min_width=best_or_fixed(
            best_params,
            "vertical_segmentation_cut_min_width",
            vertical_segmentation_cut_min_width,
        ),
        vertical_segmentation_cut_max_width=best_or_fixed(
            best_params,
            "vertical_segmentation_cut_max_width",
            vertical_segmentation_cut_max_width,
        ),
        vertical_segmentation_cut_smooth_radius=best_or_fixed(
            best_params,
            "vertical_segmentation_cut_smooth_radius",
            vertical_segmentation_cut_smooth_radius,
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
        pipeline_pool=pipeline_pool,
        image_loader=image_cache.load,
    )
    final_metrics["optuna_trials"] = trials
    final_metrics["optuna_metric"] = metric_name
    final_metrics["optuna_best_value"] = best_value
    final_metrics["optuna_best_params"] = best_params
    final_metrics["optuna_pipeline_loads"] = pipeline_pool.loads
    final_metrics["optuna_image_cache"] = image_cache.stats()
    cache_stats = image_cache.stats()
    print(
        "Optuna runtime reuse: "
        f"pipeline_loads={pipeline_pool.loads}, "
        f"image_cache_hits={cache_stats['hits']}, "
        f"misses={cache_stats['misses']}, "
        f"memory={cache_stats['megabytes']:.1f} MB"
    )
    return final_metrics
