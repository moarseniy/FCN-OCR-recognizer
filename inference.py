from __future__ import annotations

import argparse
from pathlib import Path
import random

from PIL import Image
import yaml

from fcn_ocr import (
    ClassConfidence,
    CutDecodingResult,
    DecodedSymbol,
    InferenceConfig,
    OCRPipeline,
    OCRPipelineResult,
    RecognitionResult,
    TextRecognizer,
    VerticalSegmentationResult,
    VerticalSegmentator,
    save_debug_image,
    tensor_to_pil,
)
from synth_generators.line_generator.dataset import SingleLineDataset, SingleLineDatasetConfig


__all__ = [
    "ClassConfidence",
    "CutDecodingResult",
    "DecodedSymbol",
    "InferenceConfig",
    "OCRPipeline",
    "OCRPipelineResult",
    "RecognitionResult",
    "TextRecognizer",
    "VerticalSegmentationResult",
    "VerticalSegmentator",
    "load_dataset_config",
    "main",
    "save_debug_image",
    "tensor_to_pil",
]


def load_dataset_config(config_path: str | Path) -> SingleLineDatasetConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configured FCN inference pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Inference YAML; omitted baseline, segmentator, or OCR sections are skipped.",
    )
    parser.add_argument("--image", help="Path to an image file for recognition.")
    parser.add_argument(
        "--generation-config",
        default=None,
        help="Generation config used only with --sample-index.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Recognize a generated synthetic sample instead of --image.",
    )
    parser.add_argument(
        "--save-sample",
        default="temp.png",
        help="Where to save the generated sample image in --sample-index mode.",
    )
    parser.add_argument("--show-raw", action="store_true", help="Print raw OCR timestep predictions.")
    parser.add_argument(
        "--debug-image",
        default=None,
        help="Optional path to save an annotated inference debug image.",
    )
    return parser.parse_args()


def _load_source_image(
    args: argparse.Namespace,
    pipeline: OCRPipeline,
) -> tuple[Image.Image, str, str | None]:
    if args.image:
        with Image.open(args.image) as image_file:
            return image_file.convert("RGB"), str(args.image), None

    if args.generation_config is None:
        raise ValueError("--image or --generation-config is required")
    sample_index = args.sample_index if args.sample_index is not None else 0
    dataset_config = load_dataset_config(args.generation_config)
    recognizer = pipeline.recognizer or pipeline.segmentator
    if recognizer is None:
        raise ValueError(
            "Synthetic --sample-index inference requires an ocr or segmentator section"
        )
    dataset_config = dataset_config.model_copy(
        update={
            "alphabet": recognizer.alphabet,
            "sample_alphabet": recognizer.alphabet,
            "channels": recognizer.in_channels,
            "image_height": recognizer.image_height,
        }
    )
    dataset = SingleLineDataset(dataset_config)
    rng = random.Random((dataset_config.seed or 0) + sample_index)
    sample = dataset.generate_sample(rng)
    source_image = tensor_to_pil(sample.image).convert("RGB")
    source_image.save(args.save_sample)
    print(f"Synthetic sample index: {sample_index}")
    print(f"Saved sample image: {args.save_sample}")
    print(f"Expected text: '{sample.text}'")
    return source_image, f"synthetic sample index {sample_index}", sample.text


def _debug_metadata(
    config_path: Path,
    pipeline: OCRPipeline,
    result: OCRPipelineResult,
    source: str,
    expected_text: str | None,
) -> dict:
    config = pipeline.config
    metadata = {
        "source": source,
        "inference_config": str(config_path),
        "debug_top_k": config.debug.top_k,
        "expected_text": expected_text,
        "baseline_shared": True,
        "baseline_enabled": bool(config.baseline is not None and config.baseline.enabled),
        "segmentator_enabled": config.segmentator is not None,
        "ocr_enabled": config.ocr is not None,
    }
    active_model = pipeline.recognizer or pipeline.segmentator or pipeline.baseline_processor
    if active_model is not None:
        metadata["device"] = str(active_model.device)
    metadata.update(result.baseline_preprocess_debug.metadata)
    if config.ocr is not None:
        metadata["checkpoint"] = str(config.ocr.checkpoint)
        metadata["ocr_preprocessing"] = config.ocr.preprocessing.model_dump()
    if result.ocr_preprocess_debug is not None:
        metadata["ocr_preprocessing_runtime"] = result.ocr_preprocess_debug.metadata
    if config.segmentator is not None:
        metadata["segmentator_checkpoint"] = str(config.segmentator.checkpoint)
        metadata["segmentator_preprocessing"] = config.segmentator.preprocessing.model_dump()
    if result.segmentator_preprocess_debug is not None:
        metadata["segmentator_preprocessing_runtime"] = (
            result.segmentator_preprocess_debug.metadata
        )
    if result.cut_decoding is not None:
        metadata.update(
            {
                "legacy_cuts_text": result.cut_decoding.text,
                "legacy_cuts_symbols": len(result.cut_decoding.symbols),
                "legacy_cuts_raw_cuts": len(result.cut_decoding.cuts),
                "legacy_cuts_decode_center_fraction": config.decode.center_fraction,
                "legacy_cuts_decode_min_score_width": config.decode.min_score_width,
            }
        )
    return metadata


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    pipeline = OCRPipeline(config_path, verbose=True)
    source_image, source_label, expected_text = _load_source_image(args, pipeline)
    result = pipeline.recognize_pil(source_image, collect_debug=bool(args.debug_image))

    if args.image:
        print(f"Image: {args.image}")
    if result.segmentation is not None:
        print(
            "Segmentator: "
            f"{len(result.segmentation.cut_positions or [])} cuts, "
            f"{len(result.segmentation.raw_indices)} timesteps"
        )
    if result.cut_decoding is not None:
        print(f"Recognized text (legacy+cuts): '{result.cut_decoding.text}'")
    if result.recognition is not None:
        print(f"Recognized text (raw OCR): '{result.recognition.text}'")
    else:
        print("OCR stage: skipped")
    if result.segmentation is None:
        print("Vertical segmentator stage: skipped")
    baseline_status = result.baseline_preprocess_debug.metadata.get(
        "baseline_status",
        "unknown",
    )
    print(f"Baseline stage: {baseline_status}")

    if args.debug_image:
        metadata = _debug_metadata(
            config_path,
            pipeline,
            result,
            source_label,
            expected_text,
        )
        save_debug_image(
            source_image,
            result.recognition,
            args.debug_image,
            metadata,
            baseline_output_image=result.baseline_image,
            baseline_preprocess_images=result.baseline_preprocess_debug.images,
            baseline_enabled=bool(
                pipeline.config.baseline is not None
                and pipeline.config.baseline.enabled
            ),
            network_input_image=(
                tensor_to_pil(result.ocr_input)
                if result.ocr_input is not None
                else None
            ),
            preprocess_images=(
                result.ocr_preprocess_debug.images
                if result.ocr_preprocess_debug is not None
                else None
            ),
            ocr_enabled=pipeline.config.ocr is not None,
            segmentation_result=result.segmentation,
            segmentator_input_image=(
                tensor_to_pil(result.segmentator_input)
                if result.segmentator_input is not None
                else None
            ),
            segmentator_preprocess_images=(
                result.segmentator_preprocess_debug.images
                if result.segmentator_preprocess_debug is not None
                else None
            ),
            segmentator_enabled=pipeline.config.segmentator is not None,
            cut_decoding_result=result.cut_decoding,
        )
        print(f"Saved debug image: {args.debug_image}")

    if args.show_raw:
        if result.recognition is None:
            raise ValueError("--show-raw requires an ocr section in the inference config")
        print(f"Raw indices: {result.recognition.raw_indices}")
        print(f"Raw chars: {result.recognition.raw_chars}")
        print(
            "Raw confidences: "
            f"{[round(confidence, 6) for confidence in result.recognition.raw_confidences]}"
        )


if __name__ == "__main__":
    main()
