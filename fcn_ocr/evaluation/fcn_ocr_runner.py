from __future__ import annotations

from copy import deepcopy
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from fcn_ocr import InferenceConfig, FCNPipeline
from fcn_ocr.evaluation import OCR_RESULT_FIELDS, compute_ocr_metrics, load_label_studio_samples
from fcn_ocr.evaluation.reporting import (
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
    print(
        "Decode with vertical segmentation: "
        f"{metrics['decode_with_vertical_segmentation']}"
    )
    if metrics.get("vertical_segmentation_checkpoint"):
        print(f"Vertical segmentation checkpoint:     {metrics['vertical_segmentation_checkpoint']}")
        print(f"Vertical segmentation scale_x:        {metrics['vertical_segmentation_scale_x']:+.5f}")
        print(f"Vertical segmentation y_pad:          {metrics['vertical_segmentation_y_pad']:+.5f}")
        print(f"Vertical segmentation x_pad:          {metrics['vertical_segmentation_x_pad']:.5f}")
        print(f"Vertical segmentation cut threshold:  {metrics['vertical_segmentation_cut_threshold']:.5f}")
        print(f"Vertical segmentation cut min width:  {metrics['vertical_segmentation_cut_min_width']}")
        print(f"Vertical segmentation cut max width:  {metrics['vertical_segmentation_cut_max_width']}")
        print(f"Vertical segmentation smooth radius:  {metrics['vertical_segmentation_cut_smooth_radius']}")
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


class OCRPipelinePool:
    """Owns one model pipeline for all trials in a single evaluation run."""

    def __init__(self, verbose_on_create: bool = False) -> None:
        self.pipeline: FCNPipeline | None = None
        self.verbose_on_create = verbose_on_create
        self.loads = 0

    def acquire(
        self,
        config: InferenceConfig,
        *,
        verbose: bool,
    ) -> FCNPipeline:
        if self.pipeline is None:
            self.pipeline = FCNPipeline(
                config,
                verbose=verbose or self.verbose_on_create,
            )
            self.loads += 1
        else:
            configure_evaluation_pipeline(self.pipeline, config)
        return self.pipeline


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
    pipeline_pool: OCRPipelinePool | None = None,
    image_loader: Callable[[Path], Image.Image] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if decode_with_vertical_segmentation and vertical_segmentation_checkpoint is None:
        raise ValueError("decode_with_vertical_segmentation requires vertical_segmentation_checkpoint")

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
                "enabled": decode_with_vertical_segmentation,
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
    if decode_with_vertical_segmentation:
        config_data["vertical_segmentation"] = {
            "checkpoint": vertical_segmentation_checkpoint,
            "preprocessing": {
                "scale_x": vertical_segmentation_scale_x,
                "y_pad": vertical_segmentation_y_pad,
                "x_pad": vertical_segmentation_x_pad,
            },
            "cut_threshold": vertical_segmentation_cut_threshold,
            "cut_min_width": vertical_segmentation_cut_min_width,
            "cut_max_width": vertical_segmentation_cut_max_width,
            "cut_smooth_radius": vertical_segmentation_cut_smooth_radius,
        }

    inference_config = InferenceConfig.model_validate(config_data)
    if pipeline_pool is None:
        pipeline = FCNPipeline(inference_config, verbose=verbose)
    else:
        pipeline = pipeline_pool.acquire(inference_config, verbose=verbose)
    path_results, batch_metrics = pipeline.recognize_paths_text(
        [path for _, path in jobs],
        batch_size=batch_size,
        log_every=log_every,
        image_loader=image_loader,
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
    metrics["decode_with_vertical_segmentation"] = bool(decode_with_vertical_segmentation)
    metrics["vertical_segmentation_checkpoint"] = str(vertical_segmentation_checkpoint) if vertical_segmentation_checkpoint else ""
    metrics["vertical_segmentation_scale_x"] = float(vertical_segmentation_scale_x)
    metrics["vertical_segmentation_y_pad"] = float(vertical_segmentation_y_pad)
    metrics["vertical_segmentation_x_pad"] = float(vertical_segmentation_x_pad)
    metrics["decode_method"] = str(decode_method)
    metrics["decode_top_k"] = int(decode_top_k)
    metrics["center_fraction"] = float(center_fraction)
    metrics["min_score_width"] = int(min_score_width)
    metrics["cut_weight"] = float(cut_weight)
    metrics["ocr_weight"] = float(ocr_weight)
    metrics["width_weight"] = float(width_weight)
    metrics["skip_cut_penalty"] = float(skip_cut_penalty)
    metrics["glyph_width_prior"] = glyph_width_prior or {}
    if pipeline.vertical_segmenter is not None:
        metrics["vertical_segmentation_cut_threshold"] = float(
            pipeline.vertical_segmenter.cut_threshold
        )
        metrics["vertical_segmentation_cut_min_width"] = int(
            pipeline.vertical_segmenter.cut_min_width
        )
        metrics["vertical_segmentation_cut_max_width"] = int(
            pipeline.vertical_segmenter.cut_max_width
        )
        metrics["vertical_segmentation_cut_smooth_radius"] = int(
            pipeline.vertical_segmenter.cut_smooth_radius
        )
    else:
        metrics["vertical_segmentation_cut_threshold"] = float(vertical_segmentation_cut_threshold or 0.0)
        metrics["vertical_segmentation_cut_min_width"] = int(vertical_segmentation_cut_min_width or 0)
        metrics["vertical_segmentation_cut_max_width"] = int(vertical_segmentation_cut_max_width or 0)
        metrics["vertical_segmentation_cut_smooth_radius"] = int(vertical_segmentation_cut_smooth_radius or 0)

    if output_csv is not None:
        write_csv_rows(rows, output_csv, OCR_RESULT_FIELDS)

    if verbose:
        print_metrics(metrics, output_csv)

    return metrics


def configure_evaluation_pipeline(
    pipeline: FCNPipeline,
    config: InferenceConfig,
) -> None:
    """Apply trial parameters without reloading any FCN checkpoint."""

    if pipeline.recognizer is None or config.fcn_ocr is None:
        raise ValueError("OCR evaluation pipeline requires an fcn_ocr stage")
    if pipeline.recognizer.checkpoint_path.resolve() != config.fcn_ocr.checkpoint.resolve():
        raise ValueError("Cannot replace the OCR checkpoint inside an Optuna study")

    ocr_preprocess = config.fcn_ocr.preprocessing
    pipeline.recognizer.scale_x = float(ocr_preprocess.scale_x)
    pipeline.recognizer.y_pad = float(ocr_preprocess.y_pad)
    pipeline.recognizer.x_pad = float(ocr_preprocess.x_pad)

    baseline_config = config.baseline_detection
    baseline_enabled = baseline_config is not None and baseline_config.enabled
    if baseline_enabled != (pipeline.baseline_processor is not None):
        raise ValueError("Cannot enable or disable baseline detection inside an Optuna study")
    if baseline_enabled:
        assert baseline_config is not None
        assert pipeline.baseline_processor is not None
        detector = pipeline.baseline_processor
        if detector.checkpoint_path.resolve() != baseline_config.detector_checkpoint.resolve():
            raise ValueError("Cannot replace the baseline checkpoint inside an Optuna study")
        detector.baseline_detector_threshold = float(baseline_config.detector_threshold)
        detector.baseline_deskew = bool(baseline_config.deskew)
        detector.baseline_max_angle = float(baseline_config.max_angle)
        detector.baseline_line_pad = float(baseline_config.line_pad)
        detector.baseline_line_pad_px = float(baseline_config.line_pad_px)

    vertical_config = config.vertical_segmentation
    if (vertical_config is not None) != (pipeline.vertical_segmenter is not None):
        raise ValueError(
            "Cannot enable or disable vertical segmentation inside an Optuna study"
        )
    if vertical_config is not None:
        assert pipeline.vertical_segmenter is not None
        segmenter = pipeline.vertical_segmenter
        if segmenter.checkpoint_path.resolve() != vertical_config.checkpoint.resolve():
            raise ValueError(
                "Cannot replace the vertical segmentation checkpoint inside an Optuna study"
            )
        preprocess = vertical_config.preprocessing
        segmenter.scale_x = float(preprocess.scale_x)
        segmenter.y_pad = float(preprocess.y_pad)
        segmenter.x_pad = float(preprocess.x_pad)
        segmenter.cut_threshold = segmenter._resolve_cut_threshold(
            vertical_config.cut_threshold
        )
        segmenter.cut_min_width = segmenter._resolve_non_negative_int(
            vertical_config.cut_min_width,
            "cut_min_width",
            default=segmenter.cut_min_width,
            min_value=1,
        )
        segmenter.cut_max_width = segmenter._resolve_non_negative_int(
            vertical_config.cut_max_width,
            "cut_max_width",
            default=segmenter.cut_max_width,
            min_value=0,
        )
        segmenter.cut_smooth_radius = segmenter._resolve_non_negative_int(
            vertical_config.cut_smooth_radius,
            "cut_smooth_radius",
            default=segmenter.cut_smooth_radius,
            min_value=0,
        )

    pipeline.config = config
    pipeline.decode = config.fcn_ocr.decode


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
        vertical_segmentation_checkpoint=vertical_segmentation_checkpoint,
        decode_with_vertical_segmentation=decode_with_vertical_segmentation,
        vertical_segmentation_scale_x=vertical_segmentation_scale_x,
        vertical_segmentation_y_pad=vertical_segmentation_y_pad,
        vertical_segmentation_x_pad=vertical_segmentation_x_pad,
        vertical_segmentation_cut_threshold=vertical_segmentation_cut_threshold,
        vertical_segmentation_cut_min_width=vertical_segmentation_cut_min_width,
        vertical_segmentation_cut_max_width=vertical_segmentation_cut_max_width,
        vertical_segmentation_cut_smooth_radius=vertical_segmentation_cut_smooth_radius,
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

