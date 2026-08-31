from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence


from fcn_ocr.evaluation.config import (
    evaluation_parameter_range,
)
from fcn_ocr.evaluation.reporting import (
    save_and_print_inference_command,
)

from .fcn_ocr_optimization import optimize_preprocess
from .fcn_ocr_runner import build_rows_and_jobs, evaluate

from .fcn_ocr_arguments import parse_args, resolve_inference_args

def _common_eval_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "baseline_crop": args.baseline_crop,
        "baseline_deskew": args.baseline_deskew,
        "baseline_max_angle": args.baseline_max_angle,
        "baseline_line_pad": args.baseline_line_pad,
        "baseline_line_pad_px": args.baseline_line_pad_px,
        "baseline_detector_checkpoint": Path(args.baseline_detector_checkpoint) if args.baseline_detector_checkpoint else None,
        "baseline_detector_threshold": args.baseline_detector_threshold,
        "vertical_segmentation_checkpoint": Path(args.vertical_segmentation_checkpoint) if args.vertical_segmentation_checkpoint else None,
        "decode_with_vertical_segmentation": args.decode_with_vertical_segmentation,
        "vertical_segmentation_scale_x": args.vertical_segmentation_scale_x,
        "vertical_segmentation_y_pad": args.vertical_segmentation_y_pad,
        "vertical_segmentation_x_pad": args.vertical_segmentation_x_pad,
        "vertical_segmentation_cut_threshold": args.vertical_segmentation_cut_threshold,
        "vertical_segmentation_cut_min_width": args.vertical_segmentation_cut_min_width,
        "vertical_segmentation_cut_max_width": args.vertical_segmentation_cut_max_width,
        "vertical_segmentation_cut_smooth_radius": args.vertical_segmentation_cut_smooth_radius,
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
    has_vertical_segmentation = bool(metrics.get("vertical_segmentation_checkpoint"))
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
                "enabled": bool(metrics["decode_with_vertical_segmentation"] and has_vertical_segmentation),
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
    if has_vertical_segmentation:
        config_data["vertical_segmentation"] = {
            "checkpoint": str(
                Path(metrics["vertical_segmentation_checkpoint"]).expanduser().resolve()
            ),
            "preprocessing": {
                "scale_x": metrics["vertical_segmentation_scale_x"],
                "y_pad": metrics["vertical_segmentation_y_pad"],
                "x_pad": metrics["vertical_segmentation_x_pad"],
            },
            "cut_threshold": metrics["vertical_segmentation_cut_threshold"],
            "cut_min_width": metrics["vertical_segmentation_cut_min_width"],
            "cut_max_width": metrics["vertical_segmentation_cut_max_width"],
            "cut_smooth_radius": metrics["vertical_segmentation_cut_smooth_radius"],
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
        range_names = (
            "scale_x",
            "y_pad",
            "x_pad",
            "vertical_segmentation_scale_x",
            "vertical_segmentation_y_pad",
            "vertical_segmentation_x_pad",
            "baseline_detector_threshold",
            "baseline_line_pad",
            "baseline_line_pad_px",
            "vertical_segmentation_cut_threshold",
            "vertical_segmentation_cut_min_width",
            "vertical_segmentation_cut_max_width",
            "vertical_segmentation_cut_smooth_radius",
            "center_fraction",
            "min_score_width",
        )
        ranges = {
            name: evaluation_parameter_range(args, name)
            for name in range_names
        }
        metrics = optimize_preprocess(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            trials=args.optuna_trials,
            scale_x=args.scale_x,
            y_pad=args.y_pad,
            scale_x_min=ranges["scale_x"][0],
            scale_x_max=ranges["scale_x"][1],
            y_pad_min=ranges["y_pad"][0],
            y_pad_max=ranges["y_pad"][1],
            x_pad=args.x_pad,
            metric_name=args.optuna_metric,
            log_every=args.log_every,
            trials_output=trials_output,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            progress=args.optuna_progress,
            x_pad_min=ranges["x_pad"][0],
            x_pad_max=ranges["x_pad"][1],
            vertical_segmentation_scale_x_min=ranges["vertical_segmentation_scale_x"][0],
            vertical_segmentation_scale_x_max=ranges["vertical_segmentation_scale_x"][1],
            vertical_segmentation_y_pad_min=ranges["vertical_segmentation_y_pad"][0],
            vertical_segmentation_y_pad_max=ranges["vertical_segmentation_y_pad"][1],
            vertical_segmentation_x_pad_min=ranges["vertical_segmentation_x_pad"][0],
            vertical_segmentation_x_pad_max=ranges["vertical_segmentation_x_pad"][1],
            baseline_detector_threshold_min=ranges["baseline_detector_threshold"][0],
            baseline_detector_threshold_max=ranges["baseline_detector_threshold"][1],
            baseline_line_pad_min=ranges["baseline_line_pad"][0],
            baseline_line_pad_max=ranges["baseline_line_pad"][1],
            baseline_line_pad_px_min=ranges["baseline_line_pad_px"][0],
            baseline_line_pad_px_max=ranges["baseline_line_pad_px"][1],
            vertical_segmentation_cut_threshold_min=ranges["vertical_segmentation_cut_threshold"][0],
            vertical_segmentation_cut_threshold_max=ranges["vertical_segmentation_cut_threshold"][1],
            vertical_segmentation_cut_min_width_min=ranges["vertical_segmentation_cut_min_width"][0],
            vertical_segmentation_cut_min_width_max=ranges["vertical_segmentation_cut_min_width"][1],
            vertical_segmentation_cut_max_width_min=ranges["vertical_segmentation_cut_max_width"][0],
            vertical_segmentation_cut_max_width_max=ranges["vertical_segmentation_cut_max_width"][1],
            vertical_segmentation_cut_smooth_radius_min=ranges["vertical_segmentation_cut_smooth_radius"][0],
            vertical_segmentation_cut_smooth_radius_max=ranges["vertical_segmentation_cut_smooth_radius"][1],
            center_fraction_min=ranges["center_fraction"][0],
            center_fraction_max=ranges["center_fraction"][1],
            min_score_width_min=ranges["min_score_width"][0],
            min_score_width_max=ranges["min_score_width"][1],
            optuna_seed=args.optuna_seed,
            image_cache_mb=args.optuna_image_cache_mb,
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


def main(argv: Sequence[str] | None = None) -> None:
    args = resolve_inference_args(parse_args(argv))
    run_evaluation(args)
