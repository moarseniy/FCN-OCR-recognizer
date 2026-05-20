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
    "RecognitionResult",
    "TextRecognizer",
    "VerticalSegmentationResult",
    "VerticalSegmentator",
    "load_dataset_config",
    "main",
    "save_debug_image",
    "tensor_to_pil",
]


def load_dataset_config(
    config_path: str | Path,
) -> SingleLineDatasetConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r") as file:
        return SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCN OCR inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--segmentator-checkpoint",
        default=None,
        help="Optional vertical segmentator checkpoint. If --debug-image is set, its gap/cut map is rendered too.",
    )
    parser.add_argument(
        "--segmentator-gap-threshold",
        type=float,
        default=None,
        help="Override segmentator gap/cut probability threshold from checkpoint config.",
    )
    parser.add_argument(
        "--segmentator-min-gap-width",
        type=int,
        default=None,
        help="Override minimum gap run width, or peak min distance for cut-projection segmentators.",
    )
    parser.add_argument(
        "--segmentator-merge-gap-width",
        type=int,
        default=None,
        help="Override maximum non-gap distance for merging nearby gap runs in output timesteps.",
    )
    parser.add_argument(
        "--segmentator-cut-postprocess",
        choices=("peaks", "widths"),
        default=None,
        help="Cut-projection postprocessing: raw local peaks or width-constrained cuts.",
    )
    parser.add_argument(
        "--segmentator-cut-min-width",
        type=int,
        default=None,
        help="Minimum distance between cut-projection cuts in output timesteps.",
    )
    parser.add_argument(
        "--segmentator-cut-max-width",
        type=int,
        default=None,
        help="Maximum allowed distance between neighboring cuts; 0 disables cut insertion.",
    )
    parser.add_argument(
        "--segmentator-cut-candidate-threshold",
        type=float,
        default=None,
        help="Lower threshold for candidate cut peaks used by width-constrained postprocessing.",
    )
    parser.add_argument(
        "--segmentator-cut-smooth-radius",
        type=int,
        default=None,
        help="Triangular smoothing radius for cut projection scores before peak selection.",
    )
    parser.add_argument(
        "--decode-with-segmentator",
        action="store_true",
        help="Also decode a legacy OCR checkpoint by averaging OCR probabilities inside segmentator cut intervals.",
    )
    parser.add_argument(
        "--segmentator-decode-top-k",
        type=int,
        default=8,
        help="Number of OCR class candidates to keep per legacy+cuts interval in --debug-image.",
    )
    parser.add_argument("--image", help="Path to an image file for recognition.")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config for --sample-index mode.",
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
    parser.add_argument("--device", default=None, help="Device to use: cuda or cpu.")
    parser.add_argument(
        "--scale-x",
        type=float,
        default=0.0,
        help="Normalized horizontal inference scale. 0.2 stretches width by 20%%, -0.2 squeezes by 20%%.",
    )
    parser.add_argument(
        "--y-pad",
        type=float,
        default=0.0,
        help="Normalized vertical inference padding/crop before resize. 0.2 pads, -0.2 crops.",
    )
    parser.add_argument(
        "--x-pad",
        type=float,
        default=0.0,
        help="Normalized symmetric horizontal inference padding after resize/scale. 0.05 adds 5%% width on each side.",
    )
    parser.add_argument(
        "--baseline-crop",
        action="store_true",
        help="Detect a text baseline, deskew/crop vertically by it, then apply y-pad and resize.",
    )
    parser.add_argument(
        "--baseline-top-pad",
        type=float,
        default=0.12,
        help="Extra top margin for --baseline-crop as a fraction of text height above the baseline.",
    )
    parser.add_argument(
        "--baseline-bottom-pad",
        type=float,
        default=0.18,
        help="Extra bottom margin for --baseline-crop as a fraction of text height above the baseline.",
    )
    parser.add_argument(
        "--no-baseline-deskew",
        action="store_true",
        help="Disable baseline-based deskew while keeping baseline crop enabled.",
    )
    parser.add_argument(
        "--baseline-max-angle",
        type=float,
        default=12.0,
        help="Reject baseline crop if the detected baseline angle is larger than this many degrees.",
    )
    parser.add_argument("--show-raw", action="store_true", help="Print raw timestep predictions.")
    parser.add_argument(
        "--debug-image",
        default=None,
        help="Optional path to save an annotated inference debug image.",
    )
    parser.add_argument(
        "--debug-top-k",
        type=int,
        default=8,
        help="Number of class-confidence candidates to show per decoded symbol in --debug-image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    recognizer = TextRecognizer(
        checkpoint_path,
        args.device,
        verbose=True,
        scale_x=args.scale_x,
        y_pad=args.y_pad,
        x_pad=args.x_pad,
        baseline_crop=args.baseline_crop,
        baseline_top_pad=args.baseline_top_pad,
        baseline_bottom_pad=args.baseline_bottom_pad,
        baseline_deskew=not args.no_baseline_deskew,
        baseline_max_angle=args.baseline_max_angle,
    )
    segmentator = None
    segmentator_checkpoint_path = None
    if args.segmentator_checkpoint:
        segmentator_checkpoint_path = Path(args.segmentator_checkpoint)
        if not segmentator_checkpoint_path.exists():
            raise FileNotFoundError(f"Segmentator checkpoint not found: {segmentator_checkpoint_path}")
        segmentator = VerticalSegmentator(
            segmentator_checkpoint_path,
            args.device,
            verbose=True,
            scale_x=args.scale_x,
            y_pad=args.y_pad,
            x_pad=args.x_pad,
            baseline_crop=args.baseline_crop,
            baseline_top_pad=args.baseline_top_pad,
            baseline_bottom_pad=args.baseline_bottom_pad,
            baseline_deskew=not args.no_baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
            gap_threshold=args.segmentator_gap_threshold,
            min_gap_width=args.segmentator_min_gap_width,
            merge_gap_width=args.segmentator_merge_gap_width,
            cut_postprocess=args.segmentator_cut_postprocess,
            cut_min_width=args.segmentator_cut_min_width,
            cut_max_width=args.segmentator_cut_max_width,
            cut_candidate_threshold=args.segmentator_cut_candidate_threshold,
            cut_smooth_radius=args.segmentator_cut_smooth_radius,
        )
    if args.decode_with_segmentator and segmentator is None:
        raise ValueError("--decode-with-segmentator requires --segmentator-checkpoint")

    segmentation_result = None
    segmentator_input_image = None
    cut_decoding_result = None

    if args.image:
        with Image.open(args.image) as image_file:
            source_image = image_file.convert("RGB")
        if args.debug_image:
            input_tensor, preprocess_debug = recognizer.preprocess_image_debug(args.image)
        else:
            input_tensor = recognizer.preprocess_image(args.image)
            preprocess_debug = None
        network_input_image = tensor_to_pil(input_tensor)
        result, ocr_logits = recognizer.recognize_tensor_debug_with_logits(input_tensor, top_k=args.debug_top_k)
        print(f"Image: {args.image}")
        debug_metadata = {
            "source": str(args.image),
            "checkpoint": str(checkpoint_path),
            "device": str(recognizer.device),
            "scale_x": args.scale_x,
            "y_pad": args.y_pad,
            "x_pad": args.x_pad,
            "debug_top_k": args.debug_top_k,
        }
        if preprocess_debug is not None:
            debug_metadata.update(preprocess_debug.metadata)
    else:
        if args.config is None:
            raise ValueError("--config is required when using --sample-index mode")
        sample_index = args.sample_index if args.sample_index is not None else 0
        dataset_config = load_dataset_config(args.config)
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
        source_image = tensor_to_pil(sample.image)
        source_image.save(args.save_sample)

        if args.debug_image:
            input_tensor, preprocess_debug = recognizer.preprocess_pil_debug(source_image)
        else:
            input_tensor = recognizer.preprocess_pil(source_image)
            preprocess_debug = None
        network_input_image = tensor_to_pil(input_tensor)
        result, ocr_logits = recognizer.recognize_tensor_debug_with_logits(input_tensor, top_k=args.debug_top_k)
        print(f"Synthetic sample index: {sample_index}")
        print(f"Saved sample image: {args.save_sample}")
        print(f"Expected text: '{sample.text}'")
        debug_metadata = {
            "source": f"synthetic sample index {sample_index}",
            "checkpoint": str(checkpoint_path),
            "device": str(recognizer.device),
            "scale_x": args.scale_x,
            "y_pad": args.y_pad,
            "x_pad": args.x_pad,
            "expected_text": sample.text,
            "debug_top_k": args.debug_top_k,
        }
        if preprocess_debug is not None:
            debug_metadata.update(preprocess_debug.metadata)

    if segmentator is not None:
        segmentator_input_tensor = segmentator.preprocess_pil(source_image)
        segmentator_input_image = tensor_to_pil(segmentator_input_tensor)
        segmentation_result = segmentator.segment_tensor_debug(segmentator_input_tensor)
        debug_metadata["segmentator_checkpoint"] = str(segmentator_checkpoint_path)
        if segmentation_result.mode == "cut_projection":
            print(
                "Segmentator: "
                f"{len(segmentation_result.cut_positions or [])} cuts, "
                f"{len(segmentation_result.raw_indices)} timesteps"
            )
        else:
            print(
                "Segmentator: "
                f"{sum(1 for run in segmentation_result.runs if run.label == 1)} gap runs, "
                f"{len(segmentation_result.raw_indices)} timesteps"
            )
        if args.decode_with_segmentator:
            text_bounds = recognizer.text_x_bounds_from_tensor(input_tensor)
            if text_bounds["ok"]:
                debug_metadata["text_x_bounds"] = (int(text_bounds["left"]), int(text_bounds["right"]))
                debug_metadata["text_x_bounds_confidence"] = float(text_bounds["confidence"])
                text_x_bounds = (int(text_bounds["left"]), int(text_bounds["right"]))
            else:
                debug_metadata["text_x_bounds_status"] = text_bounds["status"]
                text_x_bounds = None
            cut_decoding_result = recognizer.decode_legacy_with_cuts(
                ocr_logits,
                segmentation_result,
                input_width=int(input_tensor.shape[-1]),
                top_k=args.segmentator_decode_top_k,
                text_x_bounds=text_x_bounds,
            )
            debug_metadata["legacy_cuts_text"] = cut_decoding_result.text
            debug_metadata["legacy_cuts_symbols"] = len(cut_decoding_result.symbols)
            debug_metadata["legacy_cuts_raw_cuts"] = len(cut_decoding_result.cuts)
            print(f"Recognized text (legacy+cuts): '{cut_decoding_result.text}'")

    print(f"Recognized text: '{result.text}'")

    if args.debug_image:
        save_debug_image(
            source_image,
            result,
            args.debug_image,
            debug_metadata,
            network_input_image=network_input_image,
            preprocess_images=preprocess_debug.images if preprocess_debug is not None else None,
            segmentation_result=segmentation_result,
            segmentator_input_image=segmentator_input_image,
            cut_decoding_result=cut_decoding_result,
        )
        print(f"Saved debug image: {args.debug_image}")

    if args.show_raw:
        print(f"Raw indices: {result.raw_indices}")
        print(f"Raw chars: {result.raw_chars}")
        print(f"Raw confidences: {[round(confidence, 6) for confidence in result.raw_confidences]}")


if __name__ == "__main__":
    main()
