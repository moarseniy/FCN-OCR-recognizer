from __future__ import annotations

import argparse
from pathlib import Path
import random

from PIL import Image
import yaml

from fcn_ocr import (
    ClassConfidence,
    DecodedSymbol,
    RecognitionResult,
    TextRecognizer,
    save_debug_image,
    tensor_to_pil,
)
from synth_generators.line_generator.dataset import SingleLineDataset, SingleLineDatasetConfig


DEFAULT_CONFIG = "synth_generators/line_generator/configs/example_001.yaml"
__all__ = [
    "ClassConfidence",
    "DecodedSymbol",
    "RecognitionResult",
    "TextRecognizer",
    "load_dataset_config",
    "main",
    "save_debug_image",
    "tensor_to_pil",
]


def load_dataset_config(
    config_path: str | Path | None,
    checkpoint_config: dict | None = None,
) -> SingleLineDatasetConfig:
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found: {path}")
        with path.open("r") as file:
            return SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), path)

    if checkpoint_config:
        return SingleLineDatasetConfig.model_validate(checkpoint_config)

    default_path = Path(DEFAULT_CONFIG)
    with default_path.open("r") as file:
        return SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), default_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCN OCR inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--image", help="Path to an image file for recognition.")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config for --sample-index mode. Defaults to checkpoint config, then example config.",
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
    )

    if args.image:
        with Image.open(args.image) as image_file:
            source_image = image_file.convert("RGB")
        input_tensor = recognizer.preprocess_image(args.image)
        network_input_image = tensor_to_pil(input_tensor)
        result = recognizer.recognize_tensor_debug(input_tensor, top_k=args.debug_top_k)
        print(f"Image: {args.image}")
        debug_metadata = {
            "source": str(args.image),
            "checkpoint": str(checkpoint_path),
            "device": str(recognizer.device),
            "scale_x": args.scale_x,
            "y_pad": args.y_pad,
            "debug_top_k": args.debug_top_k,
        }
    else:
        sample_index = args.sample_index if args.sample_index is not None else 0
        dataset_config = load_dataset_config(args.config, recognizer.checkpoint.get("config"))
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

        input_tensor = recognizer.preprocess_pil(source_image)
        network_input_image = tensor_to_pil(input_tensor)
        result = recognizer.recognize_tensor_debug(input_tensor, top_k=args.debug_top_k)
        print(f"Synthetic sample index: {sample_index}")
        print(f"Saved sample image: {args.save_sample}")
        print(f"Expected text: '{sample.text}'")
        debug_metadata = {
            "source": f"synthetic sample index {sample_index}",
            "checkpoint": str(checkpoint_path),
            "device": str(recognizer.device),
            "scale_x": args.scale_x,
            "y_pad": args.y_pad,
            "expected_text": sample.text,
            "debug_top_k": args.debug_top_k,
        }

    print(f"Recognized text: '{result.text}'")

    if args.debug_image:
        save_debug_image(source_image, result, args.debug_image, debug_metadata, network_input_image=network_input_image)
        print(f"Saved debug image: {args.debug_image}")

    if args.show_raw:
        print(f"Raw indices: {result.raw_indices}")
        print(f"Raw chars: {result.raw_chars}")
        print(f"Raw confidences: {[round(confidence, 6) for confidence in result.raw_confidences]}")


if __name__ == "__main__":
    main()
