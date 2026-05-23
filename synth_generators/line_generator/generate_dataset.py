from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
import shutil
from typing import Iterable

import torch
import yaml

from .dataset import GeneratedLineSample, SingleLineDataset, SingleLineDatasetConfig
from .chunk_dataset import CHUNK_METADATA_FILENAME


def image_to_uint8(image: torch.Tensor) -> torch.Tensor:
    return (image.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def save_chunk(
    samples: Iterable[GeneratedLineSample],
    output_dir: Path,
    chunk_idx: int,
    save_dense_targets: bool = False,
) -> dict:
    images = []
    texts = []
    dense_targets = []
    cut_projection_targets = []

    for sample in samples:
        images.append(image_to_uint8(sample.image))
        texts.append(sample.text)
        if save_dense_targets:
            if sample.dense_target is None:
                raise RuntimeError("sample does not contain dense_target")
            dense_targets.append(sample.dense_target.detach().cpu().to(torch.int16))
        if sample.cut_projection_target is not None:
            cut_projection_targets.append(
                (sample.cut_projection_target.detach().cpu().clamp(0.0, 1.0) * 255.0)
                .round()
                .to(torch.uint8)
            )

    if not images:
        raise ValueError("cannot save an empty chunk")

    filename = f"chunk_{chunk_idx:06d}.pt"
    chunk = {
        "images": torch.stack(images, dim=0).contiguous(),
        "texts": texts,
    }
    if save_dense_targets:
        chunk["dense_targets"] = torch.stack(dense_targets, dim=0).contiguous()
    if cut_projection_targets:
        if len(cut_projection_targets) != len(images):
            raise RuntimeError("only some samples contain cut_projection_target")
        chunk["cut_projection_targets"] = torch.stack(cut_projection_targets, dim=0).contiguous()

    torch.save(chunk, output_dir / filename)
    return {"file": filename, "samples": len(images)}


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
    data = config.model_dump()
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
    return save_chunk(
        samples,
        Path(task["output_dir"]),
        task["chunk_idx"],
        save_dense_targets=bool(task["save_dense_targets"]),
    )


def build_metadata(config: SingleLineDatasetConfig, chunks: list[dict]) -> dict:
    return {
        "format": "fcn_ocr_line_chunks",
        "version": 1,
        "alphabet": config.alphabet or config.sample_alphabet,
        "sample_alphabet": config.sample_alphabet,
        "space_char": config.space_char,
        "samples": config.samples,
        "image_height": config.image_height,
        "image_width": config.image_width,
        "channels": config.channels,
        "background": config.background,
        "min_text_length": config.min_text_length,
        "max_text_length": config.max_text_length,
        "line_crops": config.line_crops,
        "word_count_min": config.word_count_min,
        "word_count_max": config.word_count_max,
        "word_length_min": config.word_length_min,
        "word_length_max": config.word_length_max,
        "crop_stride": config.crop_stride,
        "min_crop_text_length": config.min_crop_text_length,
        "edge_char_min_visible_ratio": config.edge_char_min_visible_ratio,
        "edge_fragment_max_visible_ratio": config.edge_fragment_max_visible_ratio,
        "dense_targets": config.save_dense_targets,
        "cut_projection_targets": config.save_cut_projection_targets,
        "cut_projection_peak_radius": config.cut_projection_peak_radius,
        "cut_projection_include_margins": config.cut_projection_include_margins,
        "dtype": "uint8",
        "chunk_size": config.chunk_size,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def save_metadata(config: SingleLineDatasetConfig, chunks: list[dict], output_dir: Path) -> None:
    metadata_path = output_dir / CHUNK_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(build_metadata(config, chunks), file, allow_unicode=True, sort_keys=False)
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
        chunk_samples = list(islice(sample_iter, end - start))
        if len(chunk_samples) != end - start:
            raise RuntimeError(f"Generator stopped after {saved} samples, expected {total}")

        chunk = save_chunk(
            chunk_samples,
            output_dir,
            chunk_idx,
            save_dense_targets=dataset.config.save_dense_targets,
        )
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
                "save_dense_targets": config.save_dense_targets,
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
    parser = argparse.ArgumentParser(description="Generate synthetic OCR line dataset into uint8 torch chunks.")
    parser.add_argument("--config", required=True, help="Path to generation YAML config.")
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
    config_path = Path(args.config)

    with config_path.open("r") as file:
        config_data = yaml.safe_load(file)
    generation_config = SingleLineDatasetConfig.model_validate_with_paths(config_data, config_path)
    if generation_config.output_dir is None:
        raise ValueError("Generation config must contain output_dir")

    output_dir = resolve_output_dir(config_path, generation_config.output_dir)
    if output_dir.exists():
        if not generation_config.overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}. Set overwrite: true to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SingleLineDataset(generation_config)

    total = len(dataset)
    if generation_config.num_workers > 0:
        max_workers = min(generation_config.num_workers, len(list(iter_chunk_specs(total, generation_config.chunk_size))))
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
    print(f"Saved {total} samples to {output_dir}")


if __name__ == "__main__":
    main()
