from __future__ import annotations

from pathlib import Path
from typing import Any


from fcn_ocr import VerticalSegmenter
from fcn_ocr.evaluation.images import RGBImageCache
from fcn_ocr.evaluation.optuna import (
    create_study,
    file_contract,
    optimize_with_progress,
    require_float_parameter,
    require_int_parameter,
    validate_study_contract,
    validate_float_range,
    validate_int_range,
)
from fcn_ocr.evaluation.reporting import (
    append_tsv_row,
)

from .vertical_segmentation_runner import (
    build_rows_and_jobs,
    configure_vertical_segmentation,
    evaluate_with_vertical_segmentation,
    infer_segment_images,
)

def append_trial_log(path: Path, trial_number: int, metrics: dict[str, Any], metric_name: str) -> None:
    columns = (
        "trial",
        "cut_threshold",
        "cut_min_width",
        "cut_max_width",
        "cut_smooth_radius",
        "scale_x",
        "y_pad",
        "x_pad",
        "baseline_crop",
        "baseline_line_pad",
        "baseline_line_pad_px",
        "baseline_deskew",
        "baseline_max_angle",
        "baseline_detector_threshold",
        "metric",
        "length_accuracy",
        "average_abs_length_error",
        "total_abs_length_error",
        "average_signed_length_error",
        "normalized_length_error",
        "cut_precision",
        "cut_recall",
        "cut_f1",
        "cut_mae_px",
        "speed",
    )
    values = (
        trial_number,
        f"{metrics['cut_threshold']:.8f}",
        metrics["cut_min_width"],
        metrics["cut_max_width"],
        metrics["cut_smooth_radius"],
        f"{metrics['scale_x']:.8f}",
        f"{metrics['y_pad']:.8f}",
        f"{metrics['x_pad']:.8f}",
        int(metrics["baseline_crop"]),
        f"{metrics['baseline_line_pad']:.8f}",
        f"{metrics['baseline_line_pad_px']:.8f}",
        int(metrics["baseline_deskew"]),
        f"{metrics['baseline_max_angle']:.8f}",
        f"{metrics['baseline_detector_threshold']:.8f}",
        f"{metrics[metric_name]:.8f}",
        f"{metrics['length_accuracy']:.8f}",
        f"{metrics['average_abs_length_error']:.8f}",
        metrics["total_abs_length_error"],
        f"{metrics['average_signed_length_error']:.8f}",
        f"{metrics['normalized_length_error']:.8f}",
        f"{metrics['cut_precision']:.8f}",
        f"{metrics['cut_recall']:.8f}",
        f"{metrics['cut_f1']:.8f}",
        f"{metrics['cut_mae_px']:.8f}",
        f"{metrics['speed']:.6f}",
    )
    append_tsv_row(path, columns, values)


def optimize(
    json_path: Path,
    images_dir: Path | None,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    batch_size: int,
    limit: int | None,
    trials: int,
    metric_name: str,
    log_every: int,
    trials_output: Path | None,
    cut_threshold: float | None,
    cut_threshold_min: float | None,
    cut_threshold_max: float | None,
    cut_min_width: int | None,
    cut_min_width_min: int | None,
    cut_min_width_max: int | None,
    cut_max_width: int | None,
    cut_max_width_min: int | None,
    cut_max_width_max: int | None,
    cut_smooth_radius: int | None,
    cut_smooth_radius_min: int | None,
    cut_smooth_radius_max: int | None,
    scale_x: float,
    scale_x_min: float | None,
    scale_x_max: float | None,
    y_pad: float,
    y_pad_min: float | None,
    y_pad_max: float | None,
    x_pad: float,
    x_pad_min: float | None,
    x_pad_max: float | None,
    tune_baseline_crop: bool,
    tune_baseline_deskew: bool,
    baseline_crop: bool,
    baseline_line_pad: float,
    baseline_line_pad_px: float,
    baseline_deskew: bool,
    baseline_max_angle: float,
    baseline_detector_checkpoint: Path | None,
    baseline_detector_threshold: float,
    baseline_line_pad_min: float | None,
    baseline_line_pad_max: float | None,
    baseline_line_pad_px_min: float | None,
    baseline_line_pad_px_max: float | None,
    baseline_max_angle_min: float | None,
    baseline_max_angle_max: float | None,
    baseline_detector_threshold_min: float | None,
    baseline_detector_threshold_max: float | None,
    study_name: str | None = None,
    storage: str | None = None,
    cut_tolerance_px: float = 3.0,
    progress: bool = False,
    optuna_seed: int = 0,
    image_cache_mb: float = 512.0,
    cache_neural_outputs: bool = True,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be >= 1")

    validate_float_range("cut_threshold", cut_threshold_min, cut_threshold_max, lower=0.0, upper=1.0)
    validate_int_range("cut_min_width", cut_min_width_min, cut_min_width_max, lower=1)
    validate_int_range("cut_max_width", cut_max_width_min, cut_max_width_max, lower=0)
    validate_int_range("cut_smooth_radius", cut_smooth_radius_min, cut_smooth_radius_max, lower=0)
    validate_float_range("scale_x", scale_x_min, scale_x_max)
    validate_float_range("y_pad", y_pad_min, y_pad_max)
    validate_float_range(
        "baseline_line_pad",
        baseline_line_pad_min,
        baseline_line_pad_max,
        lower=0.0,
    )
    validate_float_range(
        "baseline_line_pad_px",
        baseline_line_pad_px_min,
        baseline_line_pad_px_max,
        lower=0.0,
    )
    validate_float_range(
        "baseline_max_angle",
        baseline_max_angle_min,
        baseline_max_angle_max,
        lower=0.0,
    )
    validate_float_range(
        "baseline_detector_threshold",
        baseline_detector_threshold_min,
        baseline_detector_threshold_max,
        lower=0.0,
        upper=1.0,
    )
    if (x_pad_min is None) != (x_pad_max is None):
        raise ValueError("parameters.x_pad tuning requires a complete [min, max] range")
    if x_pad_min is not None and x_pad_max is not None:
        if x_pad_min < 0.0 or x_pad_max < 0.0:
            raise ValueError("x_pad tuning bounds must be >= 0")
        if x_pad_min > x_pad_max:
            raise ValueError("parameters.x_pad requires min <= max")

    baseline_can_be_active = bool(baseline_crop) or bool(tune_baseline_crop)
    has_tunable_parameter = any(
        minimum is not None
        for minimum in (
            cut_threshold_min,
            cut_min_width_min,
            cut_max_width_min,
            cut_smooth_radius_min,
            scale_x_min,
            y_pad_min,
            x_pad_min,
            baseline_line_pad_min if baseline_can_be_active else None,
            baseline_line_pad_px_min if baseline_can_be_active else None,
            baseline_max_angle_min if baseline_can_be_active else None,
            baseline_detector_threshold_min if baseline_can_be_active else None,
        )
    ) or tune_baseline_crop or (baseline_can_be_active and tune_baseline_deskew)
    if not has_tunable_parameter:
        raise ValueError(
            "Optuna has no tunable vertical segmentation parameters; set at least "
            "one parameters.<name>: [min, max] range or use optuna_trials: 0"
        )

    fixed_params: dict[str, Any] = {}
    if cut_threshold_min is None:
        fixed_params["cut_threshold"] = cut_threshold
    if cut_min_width_min is None:
        fixed_params["cut_min_width"] = cut_min_width
    if cut_max_width_min is None:
        fixed_params["cut_max_width"] = cut_max_width
    if cut_smooth_radius_min is None:
        fixed_params["cut_smooth_radius"] = cut_smooth_radius
    if scale_x_min is None:
        fixed_params["scale_x"] = scale_x
    if y_pad_min is None:
        fixed_params["y_pad"] = y_pad
    if x_pad_min is None:
        fixed_params["x_pad"] = x_pad
    if not tune_baseline_crop:
        fixed_params["baseline_crop"] = baseline_crop
    if baseline_line_pad_min is None:
        fixed_params["baseline_line_pad"] = baseline_line_pad
    if baseline_line_pad_px_min is None:
        fixed_params["baseline_line_pad_px"] = baseline_line_pad_px
    if not tune_baseline_deskew:
        fixed_params["baseline_deskew"] = baseline_deskew
    if baseline_max_angle_min is None:
        fixed_params["baseline_max_angle"] = baseline_max_angle
    if baseline_detector_threshold_min is None:
        fixed_params["baseline_detector_threshold"] = baseline_detector_threshold
    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    has_manual_cuts = any(bool(row.get("gt_cuts")) for row in base_rows)
    if metric_name == "auto":
        metric_name = "cut_f1" if has_manual_cuts else "average_abs_length_error"
    if metric_name.startswith("cut_") and not has_manual_cuts:
        raise ValueError(
            f"--optuna-metric {metric_name} requires manual markup created by tools.annotation.server"
        )
    vertical_segmentation = VerticalSegmenter(
        checkpoint_path,
        device=device,
        verbose=True,
        scale_x=0.0,
        y_pad=0.0,
        x_pad=0.0,
        baseline_crop=baseline_crop,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
        baseline_line_pad=baseline_line_pad,
        baseline_line_pad_px=baseline_line_pad_px,
        baseline_detector_checkpoint=baseline_detector_checkpoint,
        baseline_detector_threshold=baseline_detector_threshold,
    )
    direction = "maximize" if metric_name in {"length_accuracy", "cut_precision", "cut_recall", "cut_f1"} else "minimize"
    study = create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        seed=optuna_seed,
    )
    def parameter_value(fixed: Any, minimum: Any, maximum: Any) -> Any:
        return fixed if minimum is None else [minimum, maximum]

    validate_study_contract(
        study,
        {
            "evaluator": "vertical_segmentation",
            "json": file_contract(json_path),
            "images": str(images_dir.expanduser().resolve()) if images_dir else None,
            "checkpoint": file_contract(checkpoint_path),
            "metric": metric_name,
            "limit": limit,
            "cut_tolerance_px": cut_tolerance_px,
            "parameters": {
                "cut_threshold": parameter_value(cut_threshold, cut_threshold_min, cut_threshold_max),
                "cut_min_width": parameter_value(cut_min_width, cut_min_width_min, cut_min_width_max),
                "cut_max_width": parameter_value(cut_max_width, cut_max_width_min, cut_max_width_max),
                "cut_smooth_radius": parameter_value(cut_smooth_radius, cut_smooth_radius_min, cut_smooth_radius_max),
                "scale_x": parameter_value(scale_x, scale_x_min, scale_x_max),
                "y_pad": parameter_value(y_pad, y_pad_min, y_pad_max),
                "x_pad": parameter_value(x_pad, x_pad_min, x_pad_max),
                "baseline_crop": [False, True] if tune_baseline_crop else baseline_crop,
                "baseline_line_pad": parameter_value(
                    baseline_line_pad,
                    baseline_line_pad_min if baseline_can_be_active else None,
                    baseline_line_pad_max if baseline_can_be_active else None,
                ),
                "baseline_line_pad_px": parameter_value(
                    baseline_line_pad_px,
                    baseline_line_pad_px_min if baseline_can_be_active else None,
                    baseline_line_pad_px_max if baseline_can_be_active else None,
                ),
                "baseline_deskew": [False, True] if tune_baseline_deskew else baseline_deskew,
                "baseline_max_angle": parameter_value(
                    baseline_max_angle,
                    baseline_max_angle_min if baseline_can_be_active else None,
                    baseline_max_angle_max if baseline_can_be_active else None,
                ),
                "baseline_detector_threshold": parameter_value(
                    baseline_detector_threshold,
                    baseline_detector_threshold_min if baseline_can_be_active else None,
                    baseline_detector_threshold_max if baseline_can_be_active else None,
                ),
                "baseline_detector_checkpoint": (
                    file_contract(baseline_detector_checkpoint)
                    if baseline_detector_checkpoint is not None
                    else None
                ),
            },
        },
    )
    image_cache = RGBImageCache(image_cache_mb)
    preprocessing_is_fixed = not any(
        (
            scale_x_min is not None,
            y_pad_min is not None,
            x_pad_min is not None,
            tune_baseline_crop,
            baseline_can_be_active and baseline_line_pad_min is not None,
            baseline_can_be_active and baseline_line_pad_px_min is not None,
            baseline_can_be_active and baseline_max_angle_min is not None,
            baseline_can_be_active and tune_baseline_deskew,
            baseline_can_be_active and baseline_detector_threshold_min is not None,
        )
    )
    inference_cache = None
    inference_errors = None
    if cache_neural_outputs and preprocessing_is_fixed:
        configure_vertical_segmentation(
            vertical_segmentation,
            cut_threshold=cut_threshold if cut_threshold is not None else cut_threshold_min,
            cut_min_width=cut_min_width if cut_min_width is not None else cut_min_width_min,
            cut_max_width=cut_max_width if cut_max_width is not None else cut_max_width_min,
            cut_smooth_radius=(
                cut_smooth_radius
                if cut_smooth_radius is not None
                else cut_smooth_radius_min
            ),
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
        print(f"Caching vertical segmentation logits for {len(jobs)} images...")
        inference_cache, inference_errors = infer_segment_images(
            vertical_segmentation,
            jobs,
            batch_size=batch_size,
            log_every=log_every,
            image_loader=image_cache.load,
        )
    print(
        "The vertical segmentation summary above shows neutral initialization; "
        "trial preprocessing is applied immediately before evaluation."
    )

    def objective(trial) -> float:
        trial_baseline_crop = (
            trial.suggest_categorical("baseline_crop", [False, True])
            if tune_baseline_crop
            else baseline_crop
        )
        tune_active_line_pad = bool(trial_baseline_crop) and baseline_line_pad_min is not None
        tune_active_line_pad_px = bool(trial_baseline_crop) and baseline_line_pad_px_min is not None
        tune_active_max_angle = bool(trial_baseline_crop) and baseline_max_angle_min is not None
        tune_active_detector = bool(trial_baseline_crop) and baseline_detector_checkpoint is not None

        trial_baseline_line_pad = baseline_line_pad
        trial_baseline_line_pad_px = baseline_line_pad_px
        trial_baseline_max_angle = baseline_max_angle
        trial_baseline_detector_threshold = baseline_detector_threshold
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
        trial_cut_threshold = require_float_parameter(
            trial,
            "cut_threshold",
            cut_threshold,
            cut_threshold_min,
            cut_threshold_max,
        )
        trial_cut_min_width = require_int_parameter(
            trial,
            "cut_min_width",
            cut_min_width,
            cut_min_width_min,
            cut_min_width_max,
        )
        trial_cut_max_width = require_int_parameter(
            trial,
            "cut_max_width",
            cut_max_width,
            cut_max_width_min,
            cut_max_width_max,
        )
        trial_cut_smooth_radius = require_int_parameter(
            trial,
            "cut_smooth_radius",
            cut_smooth_radius,
            cut_smooth_radius_min,
            cut_smooth_radius_max,
        )
        trial_scale_x = require_float_parameter(
            trial, "scale_x", scale_x, scale_x_min, scale_x_max
        )
        trial_y_pad = require_float_parameter(trial, "y_pad", y_pad, y_pad_min, y_pad_max)
        trial_x_pad = require_float_parameter(trial, "x_pad", x_pad, x_pad_min, x_pad_max)
        configure_vertical_segmentation(
            vertical_segmentation,
            cut_threshold=trial_cut_threshold,
            cut_min_width=trial_cut_min_width,
            cut_max_width=trial_cut_max_width,
            cut_smooth_radius=trial_cut_smooth_radius,
            scale_x=trial_scale_x,
            y_pad=trial_y_pad,
            x_pad=trial_x_pad,
            baseline_crop=bool(trial_baseline_crop),
            baseline_line_pad=trial_baseline_line_pad,
            baseline_line_pad_px=trial_baseline_line_pad_px,
            baseline_deskew=bool(trial_baseline_deskew),
            baseline_max_angle=trial_baseline_max_angle,
            baseline_detector_threshold=trial_baseline_detector_threshold,
        )
        metrics = evaluate_with_vertical_segmentation(
            base_rows=base_rows,
            jobs=jobs,
            vertical_segmentation=vertical_segmentation,
            output_csv=None,
            batch_size=batch_size,
            log_every=0,
            verbose=False,
            cut_tolerance_px=cut_tolerance_px,
            image_loader=image_cache.load,
            inference_cache=inference_cache,
            inference_errors=inference_errors,
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                trial.set_user_attr(key, value)
        if trials_output is not None:
            append_trial_log(trials_output, trial.number, metrics, metric_name)
        return float(metrics[metric_name])

    print(
        "Optuna vertical segmentation search: "
        f"trials={trials}, metric={metric_name}, "
        f"cut_threshold={cut_threshold if cut_threshold_min is None else f'[{cut_threshold_min}, {cut_threshold_max}]'}, "
        f"cut_min_width={cut_min_width if cut_min_width_min is None else f'[{cut_min_width_min}, {cut_min_width_max}]'}, "
        f"cut_max_width={cut_max_width if cut_max_width_min is None else f'[{cut_max_width_min}, {cut_max_width_max}]'}, "
        f"cut_smooth_radius={cut_smooth_radius if cut_smooth_radius_min is None else f'[{cut_smooth_radius_min}, {cut_smooth_radius_max}]'}, "
        f"scale_x={scale_x if scale_x_min is None else f'[{scale_x_min}, {scale_x_max}]'}, "
        f"y_pad={y_pad if y_pad_min is None else f'[{y_pad_min}, {y_pad_max}]'}, "
        f"x_pad={x_pad if x_pad_min is None else f'[{x_pad_min}, {x_pad_max}]'}, "
        f"baseline_detector={baseline_detector_checkpoint}, "
        f"tune_baseline_crop={tune_baseline_crop}, "
        f"tune_baseline_deskew={tune_baseline_deskew}, "
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
    for name, fixed in fixed_params.items():
        if fixed is not None:
            best_params[name] = fixed
    defaults = {
        "cut_threshold": cut_threshold,
        "cut_min_width": cut_min_width,
        "cut_max_width": cut_max_width,
        "cut_smooth_radius": cut_smooth_radius,
        "scale_x": scale_x,
        "y_pad": y_pad,
        "x_pad": x_pad,
        "baseline_crop": baseline_crop,
        "baseline_line_pad": baseline_line_pad,
        "baseline_line_pad_px": baseline_line_pad_px,
        "baseline_deskew": baseline_deskew,
        "baseline_max_angle": baseline_max_angle,
        "baseline_detector_threshold": baseline_detector_threshold,
    }
    for name, default in defaults.items():
        if default is not None:
            best_params.setdefault(name, default)
    best_value = float(best_trial.value)
    print(f"Best params: {best_params}, {metric_name}={best_value:.8f}")

    configure_vertical_segmentation(
        vertical_segmentation,
        cut_threshold=float(best_params["cut_threshold"]),
        cut_min_width=int(best_params["cut_min_width"]),
        cut_max_width=int(best_params["cut_max_width"]),
        cut_smooth_radius=int(best_params["cut_smooth_radius"]),
        scale_x=float(best_params["scale_x"]),
        y_pad=float(best_params["y_pad"]),
        x_pad=float(best_params["x_pad"]),
        baseline_crop=bool(best_params["baseline_crop"]),
        baseline_line_pad=float(best_params["baseline_line_pad"]),
        baseline_line_pad_px=float(best_params["baseline_line_pad_px"]),
        baseline_deskew=bool(best_params["baseline_deskew"]),
        baseline_max_angle=float(best_params["baseline_max_angle"]),
        baseline_detector_threshold=float(best_params["baseline_detector_threshold"]),
    )
    final_metrics = evaluate_with_vertical_segmentation(
        base_rows=base_rows,
        jobs=jobs,
        vertical_segmentation=vertical_segmentation,
        output_csv=output_csv,
        batch_size=batch_size,
        log_every=log_every,
        verbose=True,
        cut_tolerance_px=cut_tolerance_px,
        image_loader=image_cache.load,
        inference_cache=inference_cache,
        inference_errors=inference_errors,
    )
    final_metrics["optuna_trials"] = trials
    final_metrics["optuna_metric"] = metric_name
    final_metrics["optuna_best_value"] = best_value
    final_metrics["optuna_image_cache"] = image_cache.stats()
    final_metrics["optuna_neural_output_cache"] = inference_cache is not None
    cache_stats = image_cache.stats()
    print(
        "Optuna runtime reuse: "
        "segmenter_loads=1, "
        f"neural_forward_passes={len(jobs) if inference_cache is not None else 'per trial'}, "
        f"image_cache_hits={cache_stats['hits']}, "
        f"misses={cache_stats['misses']}, "
        f"memory={cache_stats['megabytes']:.1f} MB"
    )
    return final_metrics
