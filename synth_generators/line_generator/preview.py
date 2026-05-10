from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image

from .dataset import SingleLineDataset, SingleLineDatasetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one synthetic OCR line preview.")
    parser.add_argument(
        "--config",
        default="synth_generators/line_generator/example_config.yaml",
        help="Path to a SingleLineDataset YAML config.",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index to render.")
    parser.add_argument("--output", default="synthetic_line_preview.png", help="Output PNG path.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = SingleLineDatasetConfig.model_validate(yaml.safe_load(file))

    dataset = SingleLineDataset(config)
    image, target, length = dataset[args.index]

    array = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
    if array.shape[2] == 1:
        array = array[:, :, 0]

    output_path = Path(args.output)
    Image.fromarray(array).save(output_path)
    print(f"Saved {output_path}")
    print(f"target shape: {tuple(target.shape)}, length: {int(length)}")


if __name__ == "__main__":
    main()
