from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence


from fcn_ocr.evaluation.config import (
    evaluation_parameter_range,
    parse_args_with_evaluation_config,
)
from fcn_ocr.evaluation.reporting import (
    save_and_print_inference_command,
)

from .vertical_segmentation_optimization import optimize
from .vertical_segmentation_runner import build_rows_and_jobs, evaluate

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune or evaluate vertical segmentation.")
    parser.add_argument("--config", default=None, help="Evaluation YAML config.")
    parser.add_argument(
        "--json",
        default=None,
        help="Label Studio export JSON or manual markup JSON created by tools.annotation.server.",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Override images directory stored in manual markup JSON; required for Label Studio JSON.",
    )
    parser.add_argument("--checkpoint", default=None, help="Vertical segmentation checkpoint.")
    parser.add_argument("--out", default="vertical_segmentation_length_metrics.csv", help="Output CSV path.")
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
    parser.add_argument("--cut-min-width", type=int, default=None)
    parser.add_argument("--cut-max-width", type=int, default=None)
    parser.add_argument("--cut-smooth-radius", type=int, default=None)
    parser.add_argument("--scale-x", type=float, default=0.0)
    parser.add_argument("--y-pad", type=float, default=0.0)
    parser.add_argument("--x-pad", type=float, default=0.0)
    parser.add_argument("--baseline-crop", action="store_true")
    parser.add_argument("--baseline-line-pad", type=float, default=0.08)
    parser.add_argument("--baseline-line-pad-px", type=float, default=0.0)
    parser.add_argument(
        "--baseline-deskew",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--optuna-trials-out", default=None)
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument("--optuna-storage", default=None)
    parser.add_argument("--optuna-seed", type=int, default=0)
    parser.add_argument(
        "--optuna-image-cache-mb",
        type=float,
        default=512.0,
        help="RAM limit for decoded source images reused between trials; 0 disables caching.",
    )
    parser.add_argument(
        "--optuna-cache-neural-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache FCN logits when preprocessing is fixed, so cut postprocessing "
            "trials do not rerun the network."
        ),
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
            "baseline_detector_checkpoint",
            "optuna_trials_out",
        ),
        required_fields=("json", "checkpoint"),
        parameter_ranges={
            "cut_threshold": (None, None),
            "cut_min_width": (None, None),
            "cut_max_width": (None, None),
            "cut_smooth_radius": (None, None),
            "scale_x": (None, None),
            "y_pad": (None, None),
            "x_pad": (None, None),
            "baseline_line_pad": (None, None),
            "baseline_line_pad_px": (None, None),
            "baseline_max_angle": (None, None),
            "baseline_detector_threshold": (None, None),
        },
        tunable_booleans=(
            "baseline_crop",
            "baseline_deskew",
        ),
        argv=argv,
    )


def _print_inference_command(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    images_dir = Path(args.images) if args.images else None
    _, jobs = build_rows_and_jobs(Path(args.json), images_dir, args.limit)
    image_path = str(jobs[0][1]) if jobs else "<IMAGE_PATH>"
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
            "deskew": metrics["baseline_deskew"],
            "max_angle": metrics["baseline_max_angle"],
            "line_pad": metrics["baseline_line_pad"],
            "line_pad_px": metrics["baseline_line_pad_px"],
        },
        "vertical_segmentation": {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "preprocessing": {
                "scale_x": metrics["scale_x"],
                "y_pad": metrics["y_pad"],
                "x_pad": metrics["x_pad"],
            },
            "cut_threshold": metrics["cut_threshold"],
            "cut_min_width": metrics["cut_min_width"],
            "cut_max_width": metrics["cut_max_width"],
            "cut_smooth_radius": metrics["cut_smooth_radius"],
        },
    }
    print(
        "Generated config contains baseline_detection and "
        "vertical_segmentation sections."
    )
    save_and_print_inference_command(config_data, args.out, image_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    images_dir = Path(args.images) if args.images else None

    if args.optuna_trials > 0:
        range_names = (
            "cut_threshold",
            "cut_min_width",
            "cut_max_width",
            "cut_smooth_radius",
            "scale_x",
            "y_pad",
            "x_pad",
            "baseline_line_pad",
            "baseline_line_pad_px",
            "baseline_max_angle",
            "baseline_detector_threshold",
        )
        ranges = {
            name: evaluation_parameter_range(args, name)
            for name in range_names
        }
        metrics = optimize(
            json_path=Path(args.json),
            images_dir=images_dir,
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            trials=args.optuna_trials,
            metric_name=args.optuna_metric,
            log_every=args.log_every,
            trials_output=Path(args.optuna_trials_out) if args.optuna_trials_out else None,
            cut_threshold=args.cut_threshold,
            cut_threshold_min=ranges["cut_threshold"][0],
            cut_threshold_max=ranges["cut_threshold"][1],
            cut_min_width=args.cut_min_width,
            cut_min_width_min=ranges["cut_min_width"][0],
            cut_min_width_max=ranges["cut_min_width"][1],
            cut_max_width=args.cut_max_width,
            cut_max_width_min=ranges["cut_max_width"][0],
            cut_max_width_max=ranges["cut_max_width"][1],
            cut_smooth_radius=args.cut_smooth_radius,
            cut_smooth_radius_min=ranges["cut_smooth_radius"][0],
            cut_smooth_radius_max=ranges["cut_smooth_radius"][1],
            scale_x=args.scale_x,
            scale_x_min=ranges["scale_x"][0],
            scale_x_max=ranges["scale_x"][1],
            y_pad=args.y_pad,
            y_pad_min=ranges["y_pad"][0],
            y_pad_max=ranges["y_pad"][1],
            x_pad=args.x_pad,
            x_pad_min=ranges["x_pad"][0],
            x_pad_max=ranges["x_pad"][1],
            tune_baseline_crop=args.optuna_tune_baseline_crop,
            tune_baseline_deskew=args.optuna_tune_baseline_deskew,
            baseline_crop=args.baseline_crop,
            baseline_line_pad=args.baseline_line_pad,
            baseline_line_pad_px=args.baseline_line_pad_px,
            baseline_deskew=args.baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
            baseline_detector_checkpoint=Path(args.baseline_detector_checkpoint) if args.baseline_detector_checkpoint else None,
            baseline_detector_threshold=args.baseline_detector_threshold,
            baseline_line_pad_min=ranges["baseline_line_pad"][0],
            baseline_line_pad_max=ranges["baseline_line_pad"][1],
            baseline_line_pad_px_min=ranges["baseline_line_pad_px"][0],
            baseline_line_pad_px_max=ranges["baseline_line_pad_px"][1],
            baseline_max_angle_min=ranges["baseline_max_angle"][0],
            baseline_max_angle_max=ranges["baseline_max_angle"][1],
            baseline_detector_threshold_min=ranges["baseline_detector_threshold"][0],
            baseline_detector_threshold_max=ranges["baseline_detector_threshold"][1],
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            cut_tolerance_px=args.cut_tolerance_px,
            progress=args.optuna_progress,
            optuna_seed=args.optuna_seed,
            image_cache_mb=args.optuna_image_cache_mb,
            cache_neural_outputs=args.optuna_cache_neural_outputs,
        )
    else:
        metrics = evaluate(
            json_path=Path(args.json),
            images_dir=images_dir,
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            log_every=args.log_every,
            cut_threshold=args.cut_threshold,
            cut_min_width=args.cut_min_width,
            cut_max_width=args.cut_max_width,
            cut_smooth_radius=args.cut_smooth_radius,
            scale_x=args.scale_x,
            y_pad=args.y_pad,
            x_pad=args.x_pad,
            baseline_crop=args.baseline_crop,
            baseline_line_pad=args.baseline_line_pad,
            baseline_line_pad_px=args.baseline_line_pad_px,
            baseline_deskew=args.baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
            baseline_detector_checkpoint=Path(args.baseline_detector_checkpoint) if args.baseline_detector_checkpoint else None,
            baseline_detector_threshold=args.baseline_detector_threshold,
            cut_tolerance_px=args.cut_tolerance_px,
        )
    _print_inference_command(args, metrics)
