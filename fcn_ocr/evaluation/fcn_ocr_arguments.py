from __future__ import annotations

import argparse
from typing import Any, Sequence


from fcn_ocr import InferenceConfig
from fcn_ocr.evaluation.config import (
    parse_args_with_evaluation_config,
)



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FCN OCR on a Label Studio JSON export.")
    parser.add_argument("--config", default=None, help="Evaluation YAML config.")
    parser.add_argument("--json", default=None, help="Path to Label Studio export JSON.")
    parser.add_argument("--images", default=None, help="Folder with images.")
    parser.add_argument(
        "--inference-config",
        default=None,
        help="Optional inference YAML used as the fixed baseline, vertical segmentation, and OCR configuration.",
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

    parser.add_argument("--vertical-segmentation-checkpoint", default=None)
    parser.add_argument(
        "--decode-with-vertical-segmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--vertical-segmentation-scale-x",
        type=float,
        default=None,
        help="Normalized horizontal scale used only by vertical segmentation.",
    )
    parser.add_argument(
        "--vertical-segmentation-y-pad",
        type=float,
        default=None,
        help="Normalized vertical padding/crop used only by vertical segmentation.",
    )
    parser.add_argument(
        "--vertical-segmentation-x-pad",
        type=float,
        default=None,
        help="Normalized horizontal padding used only by vertical segmentation.",
    )
    parser.add_argument("--vertical-segmentation-cut-threshold", type=float, default=None)
    parser.add_argument("--vertical-segmentation-cut-min-width", type=int, default=None)
    parser.add_argument("--vertical-segmentation-cut-max-width", type=int, default=None)
    parser.add_argument("--vertical-segmentation-cut-smooth-radius", type=int, default=None)
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
        help="Inference batch size for OCR with or without vertical segmentation.",
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
    parser.add_argument("--optuna-seed", type=int, default=0)
    parser.add_argument(
        "--optuna-image-cache-mb",
        type=float,
        default=512.0,
        help="RAM limit for decoded source images reused between trials; 0 disables caching.",
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
            "inference_config",
            "checkpoint",
            "out",
            "baseline_detector_checkpoint",
            "vertical_segmentation_checkpoint",
            "optuna_trials_out",
        ),
        required_fields=("json", "images"),
        parameter_ranges={
            "scale_x": (None, None),
            "y_pad": (None, None),
            "x_pad": (None, None),
            "vertical_segmentation_scale_x": (None, None),
            "vertical_segmentation_y_pad": (None, None),
            "vertical_segmentation_x_pad": (None, None),
            "baseline_detector_threshold": (None, None),
            "baseline_line_pad": (None, None),
            "baseline_line_pad_px": (None, None),
            "vertical_segmentation_cut_threshold": (None, None),
            "vertical_segmentation_cut_min_width": (None, None),
            "vertical_segmentation_cut_max_width": (None, None),
            "vertical_segmentation_cut_smooth_radius": (None, None),
            "center_fraction": (None, None),
            "min_score_width": (None, None),
        },
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
    vertical_segmentation = config.vertical_segmentation if config is not None else None
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

    args.vertical_segmentation_checkpoint = _first_defined(
        args.vertical_segmentation_checkpoint,
        vertical_segmentation.checkpoint if vertical_segmentation is not None else None,
    )
    vertical_segmentation_preprocess = (
        vertical_segmentation.preprocessing
        if vertical_segmentation is not None
        else None
    )
    args.vertical_segmentation_scale_x = float(
        _first_defined(
            args.vertical_segmentation_scale_x,
            vertical_segmentation_preprocess.scale_x if vertical_segmentation_preprocess is not None else None,
            0.0,
        )
    )
    args.vertical_segmentation_y_pad = float(
        _first_defined(
            args.vertical_segmentation_y_pad,
            vertical_segmentation_preprocess.y_pad if vertical_segmentation_preprocess is not None else None,
            0.0,
        )
    )
    args.vertical_segmentation_x_pad = float(
        _first_defined(
            args.vertical_segmentation_x_pad,
            vertical_segmentation_preprocess.x_pad if vertical_segmentation_preprocess is not None else None,
            0.0,
        )
    )
    for name in (
        "cut_threshold",
        "cut_min_width",
        "cut_max_width",
        "cut_smooth_radius",
    ):
        argument_name = f"vertical_segmentation_{name}"
        config_value = getattr(vertical_segmentation, name) if vertical_segmentation is not None else None
        setattr(
            args,
            argument_name,
            _first_defined(getattr(args, argument_name), config_value),
        )

    args.decode_with_vertical_segmentation = bool(
        _first_defined(
            args.decode_with_vertical_segmentation,
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
    if args.decode_with_vertical_segmentation and args.vertical_segmentation_checkpoint is None:
        raise ValueError(
            "Vertical segmentation decoding requires vertical_segmentation.checkpoint in --inference-config "
            "or --vertical-segmentation-checkpoint"
        )
    args.inference_debug_top_k = config.debug.top_k if config is not None else 8
    return args
