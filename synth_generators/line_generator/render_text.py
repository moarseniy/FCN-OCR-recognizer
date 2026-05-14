from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import textwrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

try:
    from .chunk_dataset import load_chunk_metadata
    from .dataset import SingleLineDataset, SingleLineDatasetConfig
    from .gpu_augmentations import GpuTextAugmenter
except ImportError:
    from chunk_dataset import load_chunk_metadata
    from dataset import SingleLineDataset, SingleLineDatasetConfig
    from gpu_augmentations import GpuTextAugmenter


def tensor_to_image(sample_tensor: torch.Tensor) -> Image.Image:
    tensor = sample_tensor.detach().cpu()
    if tensor.dtype == torch.uint8:
        if tensor.ndim == 2:
            array = tensor.numpy()
        elif tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            array = tensor.permute(1, 2, 0).numpy()
            if array.shape[2] == 1:
                array = array[:, :, 0]
        else:
            raise ValueError(f"Unsupported uint8 image tensor shape: {tuple(tensor.shape)}")
        return Image.fromarray(array)

    if tensor.ndim != 3:
        raise ValueError(f"Unsupported float image tensor shape: {tuple(tensor.shape)}")
    array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def scale_preview_image(
    image: Image.Image,
    min_width: int,
    max_width: int,
) -> tuple[Image.Image, float]:
    if image.width <= 0:
        return image, 1.0

    scale = 1.0
    if min_width > 0 and image.width < min_width:
        scale = min_width / image.width
    if max_width > 0 and image.width * scale > max_width:
        scale = max_width / image.width

    if scale == 1.0:
        return image, scale

    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resampling = Image.Resampling.NEAREST if scale > 1.0 else Image.Resampling.BICUBIC
    return image.resize(new_size, resampling), scale


def tensor_to_float_image(sample_tensor: torch.Tensor) -> torch.Tensor:
    tensor = sample_tensor.detach().cpu()
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float().clamp(0.0, 1.0)

    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        return tensor
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        return tensor.permute(2, 0, 1)
    raise ValueError(f"Unsupported image tensor shape: {tuple(tensor.shape)}")


def load_config(config_path: Path, chunks_dir: Path | None = None) -> SingleLineDatasetConfig:
    with config_path.open("r") as file:
        raw_config = yaml.safe_load(file) or {}

    config_data = {}
    if chunks_dir is not None:
        config_data.update(load_chunk_metadata(chunks_dir))
    config_data.update(raw_config)
    config = SingleLineDatasetConfig.model_validate_with_paths(config_data, config_path)
    if config.alphabet is None:
        config = config.model_copy(update={"alphabet": config.sample_alphabet})
    return config


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def apply_augmentations(
    image: torch.Tensor,
    config: SingleLineDatasetConfig,
    device: torch.device,
    enabled: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    image = tensor_to_float_image(image)
    if not enabled:
        return image, []

    augmenter = GpuTextAugmenter(config)
    batch = image.unsqueeze(0).to(device)
    augmented, metadata = augmenter.augment_with_metadata(batch)
    return augmented[0].detach().cpu(), metadata[0]


def load_chunk_sample(chunks_dir: Path, index: int) -> tuple[torch.Tensor, str, dict[str, Any]]:
    chunk_paths = sorted(chunks_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.pt files found in {chunks_dir}")

    if index < 0:
        total = sum(read_chunk_size(path) for path in chunk_paths)
        index += total

    offset = 0
    for path in chunk_paths:
        chunk = load_torch_chunk(path)
        images = chunk["images"]
        texts = chunk["texts"]
        sample_count = int(images.shape[0])
        if offset <= index < offset + sample_count:
            local_index = index - offset
            return images[local_index], str(texts[local_index]), {
                "chunk_file": str(path),
                "chunk_local_index": local_index,
                "global_index": index,
            }
        offset += sample_count

    raise IndexError(f"Chunk sample index out of range: {index}")


def read_chunk_size(path: Path) -> int:
    return int(load_torch_chunk(path)["images"].shape[0])


def load_torch_chunk(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def normalize_text(text: str, config: SingleLineDatasetConfig) -> str:
    return config.space_char.join(part for part in text.split(config.space_char) if part)


def annotation_lines(metadata: dict[str, Any]) -> list[str]:
    lines = [
        f"source: {metadata['source']}",
        f"text: {metadata['text']!r}",
        f"image: {metadata['image_size'][0]}x{metadata['image_size'][1]}",
        f"preview: {metadata['preview_size'][0]}x{metadata['preview_size'][1]} scale={metadata['preview_scale']:.3f}",
        f"seed: {metadata['seed']}",
        f"device: {metadata['device']}",
    ]
    if metadata["source"] == "chunk":
        lines.append(f"chunk: {metadata['chunk_file']}[{metadata['chunk_local_index']}]")

    augmentations = metadata["augmentations"]
    if not augmentations:
        lines.append("augmentations: none")
        return lines

    lines.append("augmentations:")
    for augmentation in augmentations:
        params = json.dumps(augmentation["params"], ensure_ascii=False, sort_keys=True)
        lines.append(f"  {augmentation['name']}: {params}")
    return lines


def annotate_image(image: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    char_width = max(1, draw_probe.textbbox((0, 0), "M", font=font)[2])
    max_chars = max(16, (image.width - 16) // char_width)
    wrapped_lines: list[str] = []
    for line in annotation_lines(metadata):
        wrapped = textwrap.wrap(line, width=max_chars, subsequent_indent="    ") or [line]
        wrapped_lines.extend(wrapped)

    text_bbox = draw_probe.textbbox((0, 0), "Ag", font=font)
    line_height = int(text_bbox[3] - text_bbox[1]) + 5
    padding = 8
    panel_height = padding * 2 + line_height * len(wrapped_lines)

    canvas = Image.new("RGB", (image.width, image.height + panel_height), color=(245, 245, 245))
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, image.height, image.width, image.height + panel_height), fill=(245, 245, 245))

    y = image.height + padding
    for line in wrapped_lines:
        draw.text((padding, y), line, fill=(20, 20, 20), font=font)
        y += line_height
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an OCR line from text or from an offline chunk sample.")
    parser.add_argument("--text", help="Text to render. Mutually exclusive with --chunks-dir.")
    parser.add_argument("--chunks-dir", help="Directory with offline chunk_*.pt files.")
    parser.add_argument("--index", type=int, default=0, help="Sample index for --chunks-dir.")
    parser.add_argument(
        "--config",
        default="synth_generators/line_generator/configs/example_001.yaml",
        help="Path to line generator YAML config.",
    )
    parser.add_argument("--output", default="rendered_text.png", help="Output image path.")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Output JSON path. Defaults to the output image path with .json suffix.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for render/augmentation sampling.")
    parser.add_argument("--device", default="auto", help="Augmentation device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--preview-min-width", type=int, default=512, help="Scale rendered preview up to this width; 0 disables.")
    parser.add_argument("--preview-max-width", type=int, default=1280, help="Scale rendered preview down to this width; 0 disables.")
    parser.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-augmentations", action="store_true", help="Disable render-time augmentations.")
    args = parser.parse_args()

    if bool(args.text) == bool(args.chunks_dir):
        parser.error("Pass exactly one of --text or --chunks-dir.")

    config_path = Path(args.config)
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else None
    config = load_config(config_path, chunks_dir)
    dataset = None if chunks_dir else SingleLineDataset(config)
    rng = random.Random(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    source_metadata: dict[str, Any] = {}
    if chunks_dir:
        image_tensor, text, source_metadata = load_chunk_sample(chunks_dir, args.index)
        text = normalize_text(text, config)
        image_tensor, augmentations = apply_augmentations(
            image_tensor,
            config,
            device,
            enabled=not args.no_augmentations,
        )
        source = "chunk"
    else:
        if dataset is None:
            raise RuntimeError("dataset must be initialized for text rendering")
        sample = dataset.generate_text_sample(args.text, rng)
        text = sample.text
        image_tensor, augmentations = apply_augmentations(
            sample.image,
            config,
            device,
            enabled=not args.no_augmentations,
        )
        source = "text"
    image = tensor_to_image(image_tensor)
    preview_image, preview_scale = scale_preview_image(
        image,
        min_width=args.preview_min_width,
        max_width=args.preview_max_width,
    )

    metadata = {
        "source": source,
        "text": text,
        "image_size": [image.width, image.height],
        "preview_size": [preview_image.width, preview_image.height],
        "preview_scale": preview_scale,
        "seed": args.seed,
        "config": str(config_path),
        "device": str(device),
        "augmentations": augmentations,
        **source_metadata,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = annotate_image(preview_image, metadata) if args.annotate else preview_image
    output_image.save(output_path)

    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved image: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Text: {text!r}")
    print(f"Image size: {image.width}x{image.height}")
    print(f"Preview size: {preview_image.width}x{preview_image.height} (scale {preview_scale:.3f})")
    print(f"Augmentations: {len(augmentations)}")


if __name__ == "__main__":
    main()
