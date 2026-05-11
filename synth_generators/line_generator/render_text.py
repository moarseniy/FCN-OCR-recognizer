from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image
import yaml

from .dataset import SingleLineDataset, SingleLineDatasetConfig


def tensor_to_image(sample_tensor) -> Image.Image:
    array = (sample_tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a synthetic OCR line with an explicit text.")
    parser.add_argument("--text", required=True, help="Text to render.")
    parser.add_argument(
        "--config",
        default="synth_generators/line_generator/configs/example.yaml",
        help="Path to line generator YAML config.",
    )
    parser.add_argument("--output", default="rendered_text.png", help="Output image path.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for font/background/jitter.")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as file:
        config = SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), config_path)

    dataset = SingleLineDataset(config)
    rng = random.Random(args.seed)
    sample = dataset.generate_text_sample(args.text, rng)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(sample.image).save(output_path)

    print(f"Saved {output_path}")
    print(f"Text: {sample.text!r}")
    print(f"Length: {sample.length}")


if __name__ == "__main__":
    main()
