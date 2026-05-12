from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
import yaml

from .dataset import SingleLineDataset, SingleLineDatasetConfig


def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    return (image.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def save_chunk(dataset: SingleLineDataset, start: int, end: int, output_dir: Path, chunk_idx: int) -> dict:
    images = []
    texts = []

    for index in range(start, end):
        sample = dataset.generate_sample_from_index(index)
        images.append(image_to_uint8(sample.image))
        texts.append(sample.text)

    filename = f"chunk_{chunk_idx:06d}.pt"
    torch.save(
        {
            "images": torch.stack(images, dim=0).contiguous(),
            "texts": texts,
        },
        output_dir / filename,
    )
    return {"file": filename, "samples": end - start}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic OCR line dataset into uint8 torch chunks.")
    parser.add_argument("--config", required=True, help="Path to generation YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    with config_path.open("r") as file:
        config_data = yaml.safe_load(file)
    generation_config = SingleLineDatasetConfig.model_validate_with_paths(config_data, config_path)
    if generation_config.output_dir is None:
        raise ValueError("Generation config must contain output_dir")

    output_dir = Path(generation_config.output_dir)
    if output_dir.exists():
        if not generation_config.overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}. Set overwrite: true to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SingleLineDataset(generation_config)

    total = len(dataset)
    for chunk_idx, start in enumerate(range(0, total, generation_config.chunk_size)):
        end = min(start + generation_config.chunk_size, total)
        chunk = save_chunk(dataset, start, end, output_dir, chunk_idx)
        print(f"saved {chunk['file']} [{start}:{end}]")
    print(f"Saved {total} samples to {output_dir}")


if __name__ == "__main__":
    main()
