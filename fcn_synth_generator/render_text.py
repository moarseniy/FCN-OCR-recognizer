from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import textwrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

from fcn_augmentations import AugmentationConfig, GpuTextAugmenter
from fcn_tasks import (
    BASELINE_DETECTION_TASK,
    FCN_OCR_TASK,
    VERTICAL_SEGMENTATION_TASK,
)

from .chunk_dataset import load_torch_chunk, validate_chunk_payload
from .chunk_metadata import load_chunk_metadata
from .dataset import SingleLineDataset, SingleLineDatasetConfig
from .run_directories import is_timestamped_directory, latest_timestamped_directory


@dataclass(frozen=True)
class RenderConfig:
    dataset: SingleLineDatasetConfig
    augmentations: AugmentationConfig


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
            raise ValueError(
                f"Unsupported uint8 image tensor shape: {tuple(tensor.shape)}"
            )
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


def fcn_ocr_target_to_long(target: torch.Tensor | None) -> torch.Tensor | None:
    if target is None:
        return None
    output = target.detach().cpu().long()
    if output.ndim != 1:
        raise ValueError(
            f"OCR target must have shape (W,), got {tuple(output.shape)}"
        )
    return output.contiguous()


def load_config(
    config_path: Path, chunks_dir: Path | None = None
) -> RenderConfig:
    with config_path.open("r") as file:
        raw_config = yaml.safe_load(file) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    is_training_config = "chunks_dir" in raw_config
    if is_training_config:
        if chunks_dir is None:
            raise ValueError(
                "A training config can be used only with --chunks-dir; "
                "chunk metadata provides its alphabet and image dimensions"
            )
        from fcn_training import (
            dataset_config_from_chunk_metadata,
            load_training_config,
        )

        training_config, _ = load_training_config(config_path)
        dataset_config = dataset_config_from_chunk_metadata(
            training_config,
            load_chunk_metadata(chunks_dir),
        )
        augmentation_config = AugmentationConfig.from_alphabet(
            alphabet=dataset_config.alphabet,
            space_char=dataset_config.space_char,
            background=dataset_config.background,
            probabilities=training_config.augmentation_probabilities,
            parameters=training_config.augmentations,
        )
        return RenderConfig(dataset_config, augmentation_config)

    supplied_config = SingleLineDatasetConfig.model_validate_with_paths(
        raw_config, config_path
    )
    if chunks_dir is None:
        return RenderConfig(
            supplied_config,
            AugmentationConfig.from_alphabet(
                alphabet=supplied_config.alphabet,
                space_char=supplied_config.space_char,
                background=supplied_config.background,
            ),
        )

    metadata = load_chunk_metadata(chunks_dir)
    immutable_fields = (
        "alphabet",
        "space_char",
        "image_height",
        "image_width",
        "channels",
        "background",
    )
    mismatches = []
    for field_name in immutable_fields:
        supplied_value = getattr(supplied_config, field_name)
        metadata_value = getattr(metadata, field_name)
        if supplied_value != metadata_value:
            mismatches.append(
                f"{field_name}: config={supplied_value!r}, metadata={metadata_value!r}"
            )
    if mismatches:
        raise ValueError(
            "Generation config does not match the chunk contract: "
            + "; ".join(mismatches)
        )

    config_data = metadata.dataset_config_data()
    config_data.update({"seed": supplied_config.seed})
    dataset_config = SingleLineDatasetConfig.model_validate(config_data)
    return RenderConfig(
        dataset_config,
        AugmentationConfig.from_alphabet(
            alphabet=dataset_config.alphabet,
            space_char=dataset_config.space_char,
            background=dataset_config.background,
        ),
    )


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
    target: torch.Tensor,
    task: str,
    config: AugmentationConfig,
    device: torch.device,
    enabled: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[dict[str, Any]],
]:
    image = tensor_to_float_image(image)
    if task == FCN_OCR_TASK:
        prepared_target = fcn_ocr_target_to_long(target)
    else:
        prepared_target = target_to_float(target)
    if prepared_target is None:
        raise ValueError("target is required")
    if not enabled:
        return image, prepared_target, []

    augmenter = GpuTextAugmenter(config)
    batch = image.unsqueeze(0).to(device)
    augmented, metadata = augmenter.augment_with_metadata(batch)
    prepared_target = augmenter.apply_metadata_to_targets(
        prepared_target.unsqueeze(0).to(device),
        task,
        metadata,
    )[0].detach().cpu()
    if task == FCN_OCR_TASK:
        prepared_target = prepared_target.long()
    return (
        augmented[0].detach().cpu(),
        prepared_target,
        metadata[0],
    )


def load_chunk_sample(
    chunks_dir: Path,
    index: int,
) -> tuple[
    torch.Tensor,
    str,
    torch.Tensor,
    str,
    dict[str, Any],
]:
    metadata = load_chunk_metadata(chunks_dir)

    if index < 0:
        index += metadata.samples
    if index < 0 or index >= metadata.samples:
        raise IndexError(f"Chunk sample index out of range: {index}")

    offset = 0
    for entry in metadata.chunks:
        path = chunks_dir / entry.file
        if not offset <= index < offset + entry.samples:
            offset += entry.samples
            continue
        chunk = load_torch_chunk(path)
        validate_chunk_payload(chunk, path, metadata, entry)
        images = chunk["images"]
        texts = chunk["texts"]
        local_index = index - offset
        return (
            images[local_index],
            str(texts[local_index]),
            chunk["targets"][local_index],
            metadata.task,
            {
                "chunk_file": str(path),
                "chunk_local_index": local_index,
                "global_index": index,
            },
        )

    raise IndexError(f"Chunk sample index out of range: {index}")


def normalize_text(text: str, config: SingleLineDatasetConfig) -> str:
    return config.space_char.join(
        part for part in text.split(config.space_char) if part
    )


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
        while (
            run_end + 1 < indices.size and indices[run_end + 1] == indices[run_end] + 1
        ):
            run_end += 1
        run = indices[run_start : run_end + 1]
        values = projection[run]
        best = run[values == values.max()]
        center = int(best[len(best) // 2])
        line_mask[:, center] = True
        run_start = run_end + 1
    return line_mask


def describe_fcn_ocr_target(
    fcn_ocr_target: torch.Tensor | None,
    alphabet: str,
    space_char: str,
) -> tuple[list[dict[str, Any]], bool, bool]:
    ocr_labels = fcn_ocr_target_to_long(fcn_ocr_target)
    if ocr_labels is None:
        return [], False, False
    target_array = ocr_labels.numpy()
    if target_array.size and (
        target_array.min() < 0 or target_array.max() >= len(alphabet)
    ):
        raise ValueError(
            "OCR target contains a class index outside the configured alphabet"
        )

    runs: list[dict[str, Any]] = []
    run_start = 0
    for x in range(1, target_array.size + 1):
        if x < target_array.size and target_array[x] == target_array[run_start]:
            continue
        class_index = int(target_array[run_start])
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
    has_left_space = bool(target_array.size and target_array[0] == space_index)
    has_right_space = bool(target_array.size and target_array[-1] == space_index)
    return runs, has_left_space, has_right_space


def overlay_full_markup(
    image: Image.Image,
    target: torch.Tensor,
    task: str,
) -> tuple[Image.Image, dict[str, Any]]:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    height, width = image_array.shape[:2]
    markup_metadata: dict[str, Any] = {
        "vertical_segmentation": task == VERTICAL_SEGMENTATION_TASK,
        "baseline_detection": task == BASELINE_DETECTION_TASK,
        "legend": {
            "vertical_segmentation": "green",
            "baseline_top": "red",
            "baseline_bottom": "blue",
        },
    }

    if task == BASELINE_DETECTION_TASK:
        baseline = target_to_float(target)
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

    if task == VERTICAL_SEGMENTATION_TASK:
        vertical_scores = target_to_float(target)
        if vertical_scores is None or vertical_scores.ndim != 1:
            raise ValueError(
                "vertical segmentation markup target must have shape (W,), "
                f"got {None if vertical_scores is None else tuple(vertical_scores.shape)}"
            )
        if vertical_scores.shape[0] != width:
            raise ValueError(
                f"vertical segmentation markup width {vertical_scores.shape[0]} "
                f"does not match image width {width}"
            )
        cut_lines = _projection_centerlines(vertical_scores.numpy(), height)
        _blend_line_mask(image_array, cut_lines, (30, 230, 80), opacity=0.92)

    return Image.fromarray(
        np.clip(image_array, 0, 255).astype(np.uint8)
    ), markup_metadata


def annotation_lines(metadata: dict[str, Any]) -> list[str]:
    lines = [
        f"source: {metadata['source']}",
        f"text: {metadata['text']!r}",
    ]
    ocr_runs = metadata.get("ocr_runs")
    if ocr_runs:
        run_text = " ".join(
            f"{run['char']}[{run['start']}:{run['end']}]" for run in ocr_runs
        )
        lines.append(f"fcn_ocr: {run_text}")
    lines.extend(
        [
            f"image: {metadata['image_size'][0]}x{metadata['image_size'][1]}",
            f"seed: {metadata['seed']}",
            f"device: {metadata['device']}",
        ]
    )
    if metadata["source"] == "chunk":
        lines.append(
            f"chunk: {metadata['chunk_file']}[{metadata['chunk_local_index']}]"
        )

    markup = metadata.get("full_markup")
    if markup is not None:
        lines.append(
            "markup: "
            "vertical segmentation="
            f"{'green' if markup['vertical_segmentation'] else 'absent'}, "
            "top baseline="
            f"{'red' if markup['baseline_detection'] else 'absent'}, "
            "bottom baseline="
            f"{'blue' if markup['baseline_detection'] else 'absent'}"
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


def annotate_image(
    image: Image.Image, metadata: dict[str, Any], canvas_width: int
) -> Image.Image:
    font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    padding = 12
    image_width = image.width + padding * 2
    canvas_width = max(canvas_width, image_width)
    char_width = max(1, draw_probe.textbbox((0, 0), "M", font=font)[2])
    max_chars = max(24, (canvas_width - padding * 2) // char_width)
    wrapped_lines: list[str] = []
    for line in annotation_lines(metadata):
        wrapped = textwrap.wrap(line, width=max_chars, subsequent_indent="    ") or [
            line
        ]
        wrapped_lines.extend(wrapped)

    text_bbox = draw_probe.textbbox((0, 0), "Ag", font=font)
    line_height = int(text_bbox[3] - text_bbox[1]) + 5
    image_band_height = image.height + padding * 2
    panel_height = padding * 2 + line_height * len(wrapped_lines)

    canvas = Image.new(
        "RGB", (canvas_width, image_band_height + panel_height), color=(245, 245, 245)
    )
    canvas.paste(image.convert("RGB"), (padding, padding))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (padding - 1, padding - 1, padding + image.width, padding + image.height),
        outline=(180, 180, 180),
    )
    draw.rectangle(
        (0, image_band_height, canvas_width, image_band_height + panel_height),
        fill=(245, 245, 245),
    )

    y = image_band_height + padding
    for line in wrapped_lines:
        draw.text((padding, y), line, fill=(20, 20, 20), font=font)
        y += line_height
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an OCR line from text or from an offline chunk sample."
    )
    parser.add_argument(
        "--text", help="Text to render. Mutually exclusive with --chunks-dir."
    )
    parser.add_argument("--chunks-dir", help="Directory with offline chunk_*.pt files.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Chunk sample index, or generated crop index for --text with line_crops.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Generation config, or training config when rendering from --chunks-dir.",
    )
    parser.add_argument(
        "--output", default="rendered_text.png", help="Output image path."
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Output JSON path. Defaults to the output image path with .json suffix.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for render/augmentation sampling.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Augmentation device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=900,
        help="Annotated output width without scaling the source crop.",
    )
    parser.add_argument(
        "--annotate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--no-augmentations",
        action="store_true",
        help="Disable render-time augmentations.",
    )
    parser.add_argument(
        "--show-full-markup",
        action="store_true",
        help="Overlay available cut-projection and top/bottom baseline targets on the rendered image.",
    )
    args = parser.parse_args()

    if bool(args.text) == bool(args.chunks_dir):
        parser.error("Pass exactly one of --text or --chunks-dir.")

    config_path = Path(args.config)
    chunks_dir = resolve_chunks_dir(Path(args.chunks_dir)) if args.chunks_dir else None
    render_config = load_config(config_path, chunks_dir)
    config = render_config.dataset
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
            target,
            task,
            source_metadata,
        ) = load_chunk_sample(chunks_dir, args.index)
        text = normalize_text(text, config)
        (
            image_tensor,
            target,
            augmentations,
        ) = apply_augmentations(
            image_tensor,
            target,
            task,
            render_config.augmentations,
            device,
            enabled=not args.no_augmentations,
        )
        source = "chunk"
    else:
        if dataset is None:
            raise RuntimeError("dataset must be initialized for text rendering")
        if config.line_crops:
            samples = dataset.generate_text_crops(args.text, rng)
            crop_index = args.index if args.index >= 0 else len(samples) + args.index
            if crop_index < 0 or crop_index >= len(samples):
                raise IndexError(
                    f"crop index {args.index} is out of range; "
                    f"explicit text produced {len(samples)} crops"
                )
            sample = samples[crop_index]
            source_metadata = {
                "source_text": normalize_text(args.text, config),
                "crop_index": crop_index,
                "crop_count": len(samples),
            }
        else:
            sample = dataset.generate_text_sample(args.text, rng)
        text = sample.text
        task = config.task
        (
            image_tensor,
            target,
            augmentations,
        ) = apply_augmentations(
            sample.image,
            sample.target,
            task,
            render_config.augmentations,
            device,
            enabled=not args.no_augmentations,
        )
        source = "text"
    alphabet = config.alphabet
    ocr_runs, has_left_space, has_right_space = describe_fcn_ocr_target(
        target if task == FCN_OCR_TASK else None,
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
            target=target,
            task=task,
        )

    metadata = {
        "source": source,
        "task": task,
        "text": display_text,
        "ocr_runs": ocr_runs,
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
    output_image = (
        annotate_image(image, metadata, args.canvas_width) if args.annotate else image
    )
    metadata["output_size"] = [output_image.width, output_image.height]
    output_image.save(output_path)

    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output_path.with_suffix(".json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved image: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Text: {display_text!r}")
    if ocr_runs:
        print(
            "FCN OCR: "
            + " ".join(
                f"{run['char']}[{run['start']}:{run['end']}]" for run in ocr_runs
            )
        )
    print(f"Image size: {image.width}x{image.height}")
    print(f"Output size: {output_image.width}x{output_image.height}")
    print(f"Augmentations: {len(augmentations)}")
    if full_markup is not None:
        print(
            "Markup: "
            "vertical_segmentation="
            f"{'yes' if full_markup['vertical_segmentation'] else 'no'}, "
            "baseline_detection="
            f"{'yes' if full_markup['baseline_detection'] else 'no'}"
        )


if __name__ == "__main__":
    main()
