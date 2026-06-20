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
    from .run_directories import is_timestamped_directory, latest_timestamped_directory
except ImportError:
    from chunk_dataset import load_chunk_metadata
    from dataset import SingleLineDataset, SingleLineDatasetConfig
    from gpu_augmentations import GpuTextAugmenter
    from run_directories import is_timestamped_directory, latest_timestamped_directory


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


def target_to_float(target: torch.Tensor | None) -> torch.Tensor | None:
    if target is None:
        return None
    output = target.detach().cpu().float()
    if target.dtype == torch.uint8:
        output = output / 255.0
    return output.clamp(0.0, 1.0).contiguous()


def dense_target_to_long(target: torch.Tensor | None) -> torch.Tensor | None:
    if target is None:
        return None
    output = target.detach().cpu().long()
    if output.ndim != 1:
        raise ValueError(f"dense target must have shape (W,), got {tuple(output.shape)}")
    return output.contiguous()


def load_config(config_path: Path, chunks_dir: Path | None = None) -> SingleLineDatasetConfig:
    with config_path.open("r") as file:
        raw_config = yaml.safe_load(file) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    is_training_config = "chunks_dir" in raw_config or "loss_mode" in raw_config
    if is_training_config:
        if chunks_dir is None:
            raise ValueError(
                "A training config can be used only with --chunks-dir; "
                "chunk metadata provides its alphabet and image dimensions"
            )
        from train import dataset_config_from_training_config, load_training_config

        training_config, _ = load_training_config(config_path)
        return dataset_config_from_training_config(
            training_config,
            load_chunk_metadata(chunks_dir),
        )

    config_data = {}
    if chunks_dir is not None:
        config_data.update(
            {
                key: value
                for key, value in load_chunk_metadata(chunks_dir).items()
                if key in SingleLineDatasetConfig.model_fields
            }
        )
    config_data.update(raw_config)
    config = SingleLineDatasetConfig.model_validate_with_paths(config_data, config_path)
    if config.alphabet is None:
        config = config.model_copy(update={"alphabet": config.sample_alphabet})
    return config


def resolve_chunks_dir(chunks_dir: Path) -> Path:
    if is_timestamped_directory(chunks_dir):
        return chunks_dir
    latest_dir = latest_timestamped_directory(chunks_dir, required_file="metadata.yaml")
    return latest_dir or chunks_dir


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def apply_augmentations(
    image: torch.Tensor,
    config: SingleLineDatasetConfig,
    device: torch.device,
    enabled: bool,
    dense_target: torch.Tensor | None = None,
    cut_projection_target: torch.Tensor | None = None,
    baseline_target: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    list[dict[str, Any]],
]:
    image = tensor_to_float_image(image)
    dense_target = dense_target_to_long(dense_target)
    cut_projection_target = target_to_float(cut_projection_target)
    baseline_target = target_to_float(baseline_target)
    if not enabled:
        return image, dense_target, cut_projection_target, baseline_target, []

    augmenter = GpuTextAugmenter(config)
    batch = image.unsqueeze(0).to(device)
    augmented, metadata = augmenter.augment_with_metadata(batch)
    if dense_target is not None:
        dense_target = augmenter.apply_metadata_to_targets(
            dense_target.unsqueeze(0).to(device),
            "dense_symbols",
            metadata,
        )[0].detach().cpu().long()
    if cut_projection_target is not None:
        cut_projection_target = augmenter.apply_metadata_to_targets(
            cut_projection_target.unsqueeze(0).to(device),
            "cut_projection",
            metadata,
        )[0].detach().cpu()
    if baseline_target is not None:
        baseline_target = augmenter.apply_metadata_to_targets(
            baseline_target.unsqueeze(0).to(device),
            "baseline_heatmap",
            metadata,
        )[0].detach().cpu()
    return (
        augmented[0].detach().cpu(),
        dense_target,
        cut_projection_target,
        baseline_target,
        metadata[0],
    )


def load_chunk_sample(
    chunks_dir: Path,
    index: int,
) -> tuple[
    torch.Tensor,
    str,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    dict[str, Any],
]:
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
            dense_targets = chunk.get("dense_targets")
            cut_targets = chunk.get("cut_projection_targets")
            baseline_targets = chunk.get("baseline_targets")
            return (
                images[local_index],
                str(texts[local_index]),
                dense_targets[local_index] if dense_targets is not None else None,
                cut_targets[local_index] if cut_targets is not None else None,
                baseline_targets[local_index] if baseline_targets is not None else None,
                {
                    "chunk_file": str(path),
                    "chunk_local_index": local_index,
                    "global_index": index,
                },
            )
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


def _blend_line_mask(
    image_array: np.ndarray,
    line_mask: np.ndarray,
    color: tuple[int, int, int],
    opacity: float,
) -> None:
    alpha = line_mask.astype(np.float32)[..., None] * float(opacity)
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    image_array[:] = image_array * (1.0 - alpha) + color_array * alpha


def _baseline_centerline(heatmap: np.ndarray) -> np.ndarray:
    line_mask = np.zeros_like(heatmap, dtype=bool)
    if heatmap.size == 0:
        return line_mask

    column_scores = heatmap.max(axis=0)
    active_columns = column_scores > 0.0
    active_x = np.flatnonzero(active_columns)
    if active_x.size:
        center_y = heatmap[:, active_x].argmax(axis=0)
        line_mask[center_y, active_x] = True
    return line_mask


def _projection_centerlines(projection: np.ndarray, height: int) -> np.ndarray:
    line_mask = np.zeros((height, projection.shape[0]), dtype=bool)
    if projection.size == 0:
        return line_mask

    left = np.concatenate(([float("-inf")], projection[:-1]))
    right = np.concatenate((projection[1:], [float("-inf")]))
    maxima = (projection > 0.0) & (projection >= left) & (projection >= right)
    indices = np.flatnonzero(maxima)
    if indices.size == 0:
        return line_mask

    run_start = 0
    while run_start < indices.size:
        run_end = run_start
        while run_end + 1 < indices.size and indices[run_end + 1] == indices[run_end] + 1:
            run_end += 1
        run = indices[run_start : run_end + 1]
        values = projection[run]
        best = run[values == values.max()]
        center = int(best[len(best) // 2])
        line_mask[:, center] = True
        run_start = run_end + 1
    return line_mask


def describe_dense_target(
    dense_target: torch.Tensor | None,
    alphabet: str,
    space_char: str,
) -> tuple[list[dict[str, Any]], bool, bool]:
    dense = dense_target_to_long(dense_target)
    if dense is None:
        return [], False, False
    dense_array = dense.numpy()
    if dense_array.size and (dense_array.min() < 0 or dense_array.max() >= len(alphabet)):
        raise ValueError("dense target contains a class index outside the configured alphabet")

    runs: list[dict[str, Any]] = []
    run_start = 0
    for x in range(1, dense_array.size + 1):
        if x < dense_array.size and dense_array[x] == dense_array[run_start]:
            continue
        class_index = int(dense_array[run_start])
        char = alphabet[class_index]
        runs.append(
            {
                "char": "␠" if char == space_char else char,
                "class_index": class_index,
                "start": run_start,
                "end": x,
            }
        )
        run_start = x

    space_index = alphabet.index(space_char)
    has_left_space = bool(dense_array.size and dense_array[0] == space_index)
    has_right_space = bool(dense_array.size and dense_array[-1] == space_index)
    return runs, has_left_space, has_right_space


def overlay_full_markup(
    image: Image.Image,
    cut_projection_target: torch.Tensor | None,
    baseline_target: torch.Tensor | None,
) -> tuple[Image.Image, dict[str, Any]]:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    height, width = image_array.shape[:2]
    markup_metadata: dict[str, Any] = {
        "cut_projection": cut_projection_target is not None,
        "baseline": baseline_target is not None,
        "legend": {
            "cut_projection": "green",
            "baseline_top": "red",
            "baseline_bottom": "blue",
        },
    }

    if baseline_target is not None:
        baseline = target_to_float(baseline_target)
        if baseline is None or baseline.ndim != 3 or baseline.shape[0] != 2:
            raise ValueError(
                "baseline markup target must have shape (2, H, W), "
                f"got {None if baseline is None else tuple(baseline.shape)}"
            )
        if tuple(baseline.shape[-2:]) != (height, width):
            raise ValueError(
                f"baseline markup shape {tuple(baseline.shape[-2:])} "
                f"does not match image shape {(height, width)}"
            )
        baseline_array = baseline.numpy()
        _blend_line_mask(
            image_array,
            _baseline_centerline(baseline_array[0]),
            (255, 45, 45),
            opacity=0.92,
        )
        _blend_line_mask(
            image_array,
            _baseline_centerline(baseline_array[1]),
            (45, 105, 255),
            opacity=0.92,
        )

    if cut_projection_target is not None:
        cut_projection = target_to_float(cut_projection_target)
        if cut_projection is None or cut_projection.ndim != 1:
            raise ValueError(
                "cut projection markup target must have shape (W,), "
                f"got {None if cut_projection is None else tuple(cut_projection.shape)}"
            )
        if cut_projection.shape[0] != width:
            raise ValueError(
                f"cut projection markup width {cut_projection.shape[0]} "
                f"does not match image width {width}"
            )
        cut_lines = _projection_centerlines(cut_projection.numpy(), height)
        _blend_line_mask(image_array, cut_lines, (30, 230, 80), opacity=0.92)

    return Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8)), markup_metadata


def annotation_lines(metadata: dict[str, Any]) -> list[str]:
    lines = [
        f"source: {metadata['source']}",
        f"text: {metadata['text']!r}",
    ]
    dense_runs = metadata.get("dense_runs")
    if dense_runs:
        run_text = " ".join(
            f"{run['char']}[{run['start']}:{run['end']}]" for run in dense_runs
        )
        lines.append(f"dense: {run_text}")
    lines.extend([
        f"image: {metadata['image_size'][0]}x{metadata['image_size'][1]}",
        f"seed: {metadata['seed']}",
        f"device: {metadata['device']}",
    ])
    if metadata["source"] == "chunk":
        lines.append(f"chunk: {metadata['chunk_file']}[{metadata['chunk_local_index']}]")

    markup = metadata.get("full_markup")
    if markup is not None:
        lines.append(
            "markup: "
            f"cuts={'green' if markup['cut_projection'] else 'absent'}, "
            f"top baseline={'red' if markup['baseline'] else 'absent'}, "
            f"bottom baseline={'blue' if markup['baseline'] else 'absent'}"
        )

    augmentations = metadata["augmentations"]
    if not augmentations:
        lines.append("augmentations: none")
        return lines

    lines.append("augmentations:")
    for augmentation in augmentations:
        params = json.dumps(augmentation["params"], ensure_ascii=False, sort_keys=True)
        lines.append(f"  {augmentation['name']}: {params}")
    return lines


def annotate_image(image: Image.Image, metadata: dict[str, Any], canvas_width: int) -> Image.Image:
    font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    padding = 12
    image_width = image.width + padding * 2
    canvas_width = max(canvas_width, image_width)
    char_width = max(1, draw_probe.textbbox((0, 0), "M", font=font)[2])
    max_chars = max(24, (canvas_width - padding * 2) // char_width)
    wrapped_lines: list[str] = []
    for line in annotation_lines(metadata):
        wrapped = textwrap.wrap(line, width=max_chars, subsequent_indent="    ") or [line]
        wrapped_lines.extend(wrapped)

    text_bbox = draw_probe.textbbox((0, 0), "Ag", font=font)
    line_height = int(text_bbox[3] - text_bbox[1]) + 5
    image_band_height = image.height + padding * 2
    panel_height = padding * 2 + line_height * len(wrapped_lines)

    canvas = Image.new("RGB", (canvas_width, image_band_height + panel_height), color=(245, 245, 245))
    canvas.paste(image.convert("RGB"), (padding, padding))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (padding - 1, padding - 1, padding + image.width, padding + image.height),
        outline=(180, 180, 180),
    )
    draw.rectangle((0, image_band_height, canvas_width, image_band_height + panel_height), fill=(245, 245, 245))

    y = image_band_height + padding
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
        required=True,
        help="Generation config, or training config when rendering from --chunks-dir.",
    )
    parser.add_argument("--output", default="rendered_text.png", help="Output image path.")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Output JSON path. Defaults to the output image path with .json suffix.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for render/augmentation sampling.")
    parser.add_argument("--device", default="auto", help="Augmentation device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--canvas-width", type=int, default=900, help="Annotated output width without scaling the source crop.")
    parser.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-augmentations", action="store_true", help="Disable render-time augmentations.")
    parser.add_argument(
        "--show-full-markup",
        "--show-full--markup",
        action="store_true",
        help="Overlay available cut-projection and top/bottom baseline targets on the rendered image.",
    )
    args = parser.parse_args()

    if bool(args.text) == bool(args.chunks_dir):
        parser.error("Pass exactly one of --text or --chunks-dir.")

    config_path = Path(args.config)
    chunks_dir = resolve_chunks_dir(Path(args.chunks_dir)) if args.chunks_dir else None
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
        (
            image_tensor,
            text,
            dense_target,
            cut_projection_target,
            baseline_target,
            source_metadata,
        ) = load_chunk_sample(chunks_dir, args.index)
        text = normalize_text(text, config)
        (
            image_tensor,
            dense_target,
            cut_projection_target,
            baseline_target,
            augmentations,
        ) = apply_augmentations(
            image_tensor,
            config,
            device,
            enabled=not args.no_augmentations,
            dense_target=dense_target,
            cut_projection_target=cut_projection_target,
            baseline_target=baseline_target,
        )
        source = "chunk"
    else:
        if dataset is None:
            raise RuntimeError("dataset must be initialized for text rendering")
        sample = dataset.generate_text_sample(args.text, rng)
        text = sample.text
        dense_target = sample.dense_target
        (
            image_tensor,
            dense_target,
            cut_projection_target,
            baseline_target,
            augmentations,
        ) = apply_augmentations(
            sample.image,
            config,
            device,
            enabled=not args.no_augmentations,
            dense_target=dense_target,
            cut_projection_target=sample.cut_projection_target,
            baseline_target=sample.baseline_target,
        )
        source = "text"
    alphabet = config.alphabet or config.sample_alphabet
    dense_runs, has_left_space, has_right_space = describe_dense_target(
        dense_target,
        alphabet,
        config.space_char,
    )
    display_text = text
    if has_left_space and not display_text.startswith(config.space_char):
        display_text = config.space_char + display_text
    if has_right_space and not display_text.endswith(config.space_char):
        display_text += config.space_char

    image = tensor_to_image(image_tensor)
    full_markup = None
    if args.show_full_markup:
        image, full_markup = overlay_full_markup(
            image,
            cut_projection_target=cut_projection_target,
            baseline_target=baseline_target,
        )

    metadata = {
        "source": source,
        "text": display_text,
        "dense_runs": dense_runs,
        "image_size": [image.width, image.height],
        "seed": args.seed,
        "config": str(config_path),
        "device": str(device),
        "augmentations": augmentations,
        **source_metadata,
    }
    if full_markup is not None:
        metadata["full_markup"] = full_markup

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = annotate_image(image, metadata, args.canvas_width) if args.annotate else image
    metadata["output_size"] = [output_image.width, output_image.height]
    output_image.save(output_path)

    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved image: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Text: {display_text!r}")
    if dense_runs:
        print(
            "Dense: "
            + " ".join(
                f"{run['char']}[{run['start']}:{run['end']}]"
                for run in dense_runs
            )
        )
    print(f"Image size: {image.width}x{image.height}")
    print(f"Output size: {output_image.width}x{output_image.height}")
    print(f"Augmentations: {len(augmentations)}")
    if full_markup is not None:
        print(
            "Markup: "
            f"cuts={'yes' if full_markup['cut_projection'] else 'no'}, "
            f"baseline={'yes' if full_markup['baseline'] else 'no'}"
        )


if __name__ == "__main__":
    main()
