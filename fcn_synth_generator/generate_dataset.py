from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
import shutil
from typing import Any, Iterable

import torch
import yaml

from fcn_tasks import FCN_OCR_TASK

from .dataset import GeneratedLineSample, SingleLineDataset, SingleLineDatasetConfig
from .chunk_metadata import (
    CHUNK_FORMAT,
    CHUNK_METADATA_VERSION,
    GENERATION_CONFIG_FILENAME,
    ChunkMetadata,
    save_chunk_metadata,
)
from .run_directories import timestamped_directory


def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    return (image.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def save_chunk(
    samples: Iterable[GeneratedLineSample],
    output_dir: Path,
    chunk_idx: int,
    task: str,
    alphabet: str,
) -> dict[str, Any]:
    images = []
    texts = []
    targets = []

    for sample in samples:
        images.append(image_to_uint8(sample.image))
        texts.append(sample.text)
        target = sample.target.detach().cpu()
        if task == FCN_OCR_TASK:
            targets.append(target.to(torch.int16))
        else:
            targets.append(
                (target.clamp(0.0, 1.0) * 255.0)
                .round()
                .to(torch.uint8)
            )

    if not images:
        raise ValueError("cannot save an empty chunk")

    filename = f"chunk_{chunk_idx:06d}.pt"
    chunk = {
        "images": torch.stack(images, dim=0).contiguous(),
        "texts": texts,
        "targets": torch.stack(targets, dim=0).contiguous(),
    }
    target_class_counts = None
    if task == FCN_OCR_TASK:
        stacked_targets = chunk["targets"]
        if int(stacked_targets.min()) < 0 or int(
            stacked_targets.max()
        ) >= len(alphabet):
            raise ValueError(
                "OCR target contains an index outside the generation alphabet"
            )
        target_class_counts = torch.bincount(
            stacked_targets.long().flatten(),
            minlength=len(alphabet),
        ).tolist()

    torch.save(chunk, output_dir / filename)
    return {
        "file": filename,
        "samples": len(images),
        "text_char_counts": dict(Counter("".join(texts))),
        "target_class_counts": target_class_counts,
        "max_observed_text_length": max(len(text) for text in texts),
    }


def chunk_seed(base_seed: int | None, start: int) -> int | None:
    if base_seed is None:
        return None
    return base_seed + start


def iter_chunk_specs(total: int, chunk_size: int) -> Iterable[tuple[int, int, int]]:
    for chunk_idx, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size, total)
        yield chunk_idx, start, end


def worker_config_data(
    config: SingleLineDatasetConfig,
    font_paths: list[str],
    background_paths: list[str],
    sample_count: int,
    seed: int | None,
) -> dict:
    data = config.model_dump(exclude_none=True)
    data.update(
        {
            "samples": sample_count,
            "seed": seed,
            "font_paths": font_paths,
            "font_dir": None,
            "font_check": False,
            "background_paths": background_paths,
            "background_dir": None,
        }
    )
    return data


def generate_chunk_worker(task: dict) -> dict:
    torch.set_num_threads(1)
    config = SingleLineDatasetConfig.model_validate(task["config"])
    dataset = SingleLineDataset(config)
    samples = list(islice(dataset.iter_generated_samples(), task["sample_count"]))
    if len(samples) != task["sample_count"]:
        raise RuntimeError(
            f"Generator stopped after {len(samples)} samples, expected {task['sample_count']}"
        )
    chunk = save_chunk(
        samples,
        Path(task["output_dir"]),
        task["chunk_idx"],
        task=config.task,
        alphabet=config.alphabet,
    )
    chunk["crop_statistics"] = dict(dataset.crop_statistics)
    return chunk


def build_metadata(
    config: SingleLineDatasetConfig, chunks: list[dict[str, Any]]
) -> ChunkMetadata:
    alphabet = config.alphabet
    text_char_counts: Counter[str] = Counter()
    target_class_counts = [0] * len(alphabet) if config.task == FCN_OCR_TASK else None
    max_observed_text_length = 0
    manifest = []
    for chunk in chunks:
        manifest.append({"file": chunk["file"], "samples": chunk["samples"]})
        text_char_counts.update(chunk["text_char_counts"])
        max_observed_text_length = max(
            max_observed_text_length,
            int(chunk["max_observed_text_length"]),
        )
        if target_class_counts is not None:
            chunk_counts = chunk["target_class_counts"]
            if chunk_counts is None or len(chunk_counts) != len(target_class_counts):
                raise ValueError("chunk OCR class statistics do not match alphabet")
            target_class_counts = [
                total + int(count)
                for total, count in zip(target_class_counts, chunk_counts)
            ]

    return ChunkMetadata.model_validate(
        {
            "format": CHUNK_FORMAT,
            "version": CHUNK_METADATA_VERSION,
            "task": config.task,
            "alphabet": alphabet,
            "space_char": config.space_char,
            "samples": config.samples,
            "image_height": config.image_height,
            "image_width": config.image_width,
            "channels": config.channels,
            "background": config.background,
            "word_count_min": config.word_count_min,
            "word_count_max": config.word_count_max,
            "word_length_min": config.word_length_min,
            "word_length_max": config.word_length_max,
            "crop_stride": config.crop_stride,
            "min_crop_text_length": config.min_crop_text_length,
            "max_crop_text_length": config.max_crop_text_length,
            "edge_char_min_visible_ratio": config.edge_char_min_visible_ratio,
            "edge_fragment_max_visible_ratio": config.edge_fragment_max_visible_ratio,
            "neighbor_lines_probability": config.neighbor_lines_probability,
            "neighbor_line_min_crop_ratio": config.neighbor_line_min_crop_ratio,
            "neighbor_line_visible_ratio_min": config.neighbor_line_visible_ratio_min,
            "neighbor_line_gap_min": config.neighbor_line_gap_min,
            "neighbor_line_gap_max": config.neighbor_line_gap_max,
            "ink_spacing_enabled": config.ink_spacing_enabled,
            "ink_spacing_min_char_gap_px": config.ink_spacing_min_char_gap_px,
            "ink_spacing_touch_gap_px": config.ink_spacing_touch_gap_px,
            "ink_spacing_touch_probability": config.ink_spacing_touch_probability,
            "image_dtype": "uint8",
            "target_dtype": "int16" if config.task == FCN_OCR_TASK else "uint8",
            "fcn_ocr_target_edge_bounds": (
                "ink" if config.task == FCN_OCR_TASK else None
            ),
            "vertical_segmentation_target_radius": config.vertical_segmentation_target_radius,
            "vertical_segmentation_include_margins": config.vertical_segmentation_include_margins,
            "baseline_detection_target_radius": config.baseline_detection_target_radius,
            "chunk_size": config.chunk_size,
            "chunk_count": len(chunks),
            "chunks": manifest,
            "text_char_counts": {
                char: int(text_char_counts.get(char, 0)) for char in alphabet
            },
            "target_class_counts": target_class_counts,
            "max_observed_text_length": max_observed_text_length,
        }
    )


def save_metadata(
    config: SingleLineDatasetConfig, chunks: list[dict], output_dir: Path
) -> None:
    metadata_path = save_chunk_metadata(build_metadata(config, chunks), output_dir)
    print(f"saved {metadata_path.name}")


def generate_chunks_sequential(
    dataset: SingleLineDataset,
    output_dir: Path,
    total: int,
    chunk_size: int,
) -> list[dict]:
    chunks = []
    sample_iter = dataset.iter_generated_samples()
    saved = 0
    for chunk_idx, start, end in iter_chunk_specs(total, chunk_size):
        previous_statistics = dataset.crop_statistics.copy()
        chunk_samples = list(islice(sample_iter, end - start))
        if len(chunk_samples) != end - start:
            raise RuntimeError(
                f"Generator stopped after {saved} samples, expected {total}"
            )

        chunk = save_chunk(
            chunk_samples,
            output_dir,
            chunk_idx,
            task=dataset.config.task,
            alphabet=dataset.alphabet,
        )
        chunk["crop_statistics"] = dict(dataset.crop_statistics - previous_statistics)
        chunks.append(chunk)
        saved += chunk["samples"]
        print(f"saved {chunk['file']} [{start}:{start + chunk['samples']}]")
    return chunks


def generate_chunks_parallel(
    config: SingleLineDatasetConfig,
    font_paths: list[str],
    background_paths: list[str],
    output_dir: Path,
    total: int,
    chunk_size: int,
    num_workers: int,
) -> list[dict]:
    specs = list(iter_chunk_specs(total, chunk_size))
    chunks_by_index: dict[int, dict] = {}

    print(f"Parallel generation: {num_workers} workers, {len(specs)} chunks")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for chunk_idx, start, end in specs:
            task = {
                "chunk_idx": chunk_idx,
                "start": start,
                "end": end,
                "sample_count": end - start,
                "output_dir": str(output_dir),
                "config": worker_config_data(
                    config,
                    font_paths,
                    background_paths,
                    sample_count=end - start,
                    seed=chunk_seed(config.seed, start),
                ),
            }
            future = executor.submit(generate_chunk_worker, task)
            futures[future] = (chunk_idx, start, end)

        for future in as_completed(futures):
            chunk_idx, start, end = futures[future]
            chunk = future.result()
            chunks_by_index[chunk_idx] = chunk
            print(f"saved {chunk['file']} [{start}:{end}]")

    return [chunks_by_index[chunk_idx] for chunk_idx, _, _ in specs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic OCR line dataset into uint8 torch chunks."
    )
    parser.add_argument(
        "--config", required=True, help="Path to generation YAML config."
    )
    return parser.parse_args()


def resolve_output_dir(config_path: Path, configured_output_dir: str) -> Path:
    output_root = Path(configured_output_dir)
    dataset_name = config_path.stem
    if output_root.name == dataset_name:
        return output_root
    if output_root.name == "line_chunks":
        return output_root.parent / dataset_name
    return output_root / dataset_name


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    with config_path.open("r") as file:
        config_data = yaml.safe_load(file)
    generation_config = SingleLineDatasetConfig.model_validate_with_paths(
        config_data, config_path
    )
    if generation_config.output_dir is None:
        raise ValueError("Generation config must contain output_dir")

    base_output_dir = resolve_output_dir(config_path, generation_config.output_dir)
    output_dir = timestamped_directory(base_output_dir)
    if output_dir.exists():
        if not generation_config.overwrite:
            raise FileExistsError(
                f"Output dir already exists: {output_dir}. Set overwrite: true to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset output: {output_dir}")
    generation_config_snapshot = output_dir / GENERATION_CONFIG_FILENAME
    shutil.copy2(config_path, generation_config_snapshot)
    print(f"Generation config saved to {generation_config_snapshot}")

    dataset = SingleLineDataset(generation_config)

    total = len(dataset)
    if generation_config.num_workers > 0:
        max_workers = min(
            generation_config.num_workers,
            len(list(iter_chunk_specs(total, generation_config.chunk_size))),
        )
        chunks = generate_chunks_parallel(
            generation_config,
            dataset.font_paths,
            dataset.background_paths,
            output_dir,
            total,
            generation_config.chunk_size,
            max_workers,
        )
    else:
        chunks = generate_chunks_sequential(
            dataset,
            output_dir,
            total,
            generation_config.chunk_size,
        )
    save_metadata(generation_config, chunks, output_dir)
    crop_statistics: Counter[str] = Counter({key: 0 for key in dataset.crop_statistics})
    for chunk in chunks:
        crop_statistics.update(chunk["crop_statistics"])
    print(f"Crop planning: {dict(crop_statistics)}")
    print(f"Saved {total} samples to {output_dir}")


if __name__ == "__main__":
    main()
