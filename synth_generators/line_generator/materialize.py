from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch
import yaml

from .dataset import SUPPORTED_AUGMENTATIONS, SingleLineDataset, SingleLineDatasetConfig


def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    return (image.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def save_chunk(dataset: SingleLineDataset, start: int, end: int, output_dir: Path, chunk_idx: int) -> dict:
    images = []
    targets = []
    lengths = []
    texts = []

    for index in range(start, end):
        sample = dataset.generate_sample_from_index(index)
        images.append(image_to_uint8(sample.image))
        targets.append(sample.target.cpu())
        lengths.append(torch.tensor(sample.length, dtype=torch.long))
        texts.append(sample.text)

    filename = f"chunk_{chunk_idx:06d}.pt"
    torch.save(
        {
            "images": torch.stack(images, dim=0).contiguous(),
            "targets": torch.stack(targets, dim=0).contiguous(),
            "lengths": torch.stack(lengths, dim=0).contiguous(),
            "texts": texts,
        },
        output_dir / filename,
    )
    return {"file": filename, "samples": end - start}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize synthetic OCR lines into uint8 torch chunks.")
    parser.add_argument("--config", required=True, help="Path to SingleLineDataset YAML config.")
    parser.add_argument("--output-dir", required=True, help="Directory for saved .pt chunks.")
    parser.add_argument("--samples", type=int, default=None, help="Override config sample count.")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Samples per chunk file.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir before writing.")
    parser.add_argument(
        "--with-augmentations",
        dest="with_augmentations",
        action="store_true",
        help="Apply CPU augmentations while materializing. By default chunks are rendered clean.",
    )
    parser.add_argument(
        "--no-augmentations",
        dest="with_augmentations",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(with_augmentations=False)
    return parser.parse_args()


def without_augmentations(config_data: dict) -> dict:
    render_config = dict(config_data)
    render_config["noise_std"] = 0.0
    render_config["blur_radius"] = 0.0
    render_config["max_rotation_degrees"] = 0.0
    render_config["augmentation_probabilities"] = {name: 0.0 for name in SUPPORTED_AUGMENTATIONS}
    return render_config


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r") as file:
        config_data = yaml.safe_load(file)
    if args.samples is not None:
        config_data["samples"] = args.samples
    render_config_data = config_data if args.with_augmentations else without_augmentations(config_data)
    render_config = SingleLineDatasetConfig.model_validate_with_paths(render_config_data, config_path)
    dataset = SingleLineDataset(render_config)

    total = len(dataset)
    for chunk_idx, start in enumerate(range(0, total, args.chunk_size)):
        end = min(start + args.chunk_size, total)
        chunk = save_chunk(dataset, start, end, output_dir, chunk_idx)
        print(f"saved {chunk['file']} [{start}:{end}]")
    print(f"Saved {total} samples to {output_dir}")


if __name__ == "__main__":
    main()
