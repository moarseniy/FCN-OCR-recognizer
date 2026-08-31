from __future__ import annotations

from pathlib import Path

import torch

from fcn_synth_generator.chunk_dataset import ChunkedLineDataset, load_chunk_metadata
from fcn_synth_generator.chunk_metadata import CHUNK_METADATA_FILENAME, ChunkMetadata
from fcn_synth_generator.dataset import SingleLineDatasetConfig
from fcn_synth_generator.run_directories import (
    is_timestamped_directory,
    latest_timestamped_directory,
)

from .config import TrainingConfig
from .tasks import all_training_task_config_fields, get_training_task


def dataset_config_from_chunk_metadata(
    config: TrainingConfig,
    metadata: ChunkMetadata,
) -> SingleLineDatasetConfig:
    data = metadata.dataset_config_data()
    data.update(
        {
            "seed": config.seed,
        }
    )
    return SingleLineDatasetConfig.model_validate(data)


def effective_training_config_data(
    config: TrainingConfig, dataset_config: SingleLineDatasetConfig
) -> dict:
    task = get_training_task(config.task)
    foreign_task_fields = all_training_task_config_fields() - task.config_fields
    data = config.model_dump(exclude=foreign_task_fields)
    data.update(
        {
            "alphabet": dataset_config.alphabet,
            "space_char": dataset_config.space_char,
            "max_text_length": dataset_config.max_text_length,
            "channels": dataset_config.channels,
            "image_height": dataset_config.image_height,
            "image_width": dataset_config.image_width,
            "background": dataset_config.background,
        }
    )
    return data


def resolve_chunks_dir(configured_dir: str | Path) -> Path:
    base_dir = Path(configured_dir)
    if is_timestamped_directory(base_dir):
        return base_dir
    latest_dir = latest_timestamped_directory(
        base_dir,
        required_file=CHUNK_METADATA_FILENAME,
    )
    return latest_dir or base_dir


def load_dataset_from_config(
    config: TrainingConfig,
) -> tuple[torch.utils.data.Dataset, SingleLineDatasetConfig]:
    task = get_training_task(config.task)
    chunks_dir = resolve_chunks_dir(config.chunks_dir)
    config.chunks_dir = str(chunks_dir)
    metadata = load_chunk_metadata(chunks_dir)
    metadata.require_task(task.name)
    dataset_config = dataset_config_from_chunk_metadata(config, metadata)
    dataset = ChunkedLineDataset(
        chunks_dir,
        cache_size=config.chunk_cache_size,
        config=dataset_config,
        task=task.name,
    )
    print(f"Dataset source: chunks ({chunks_dir})")
    print(
        f"Dataset metadata: {chunks_dir / CHUNK_METADATA_FILENAME} "
        f"(format={metadata.format}, version={metadata.version}, task={metadata.task})"
    )
    return dataset, dataset_config


def printable_char(char: str) -> str:
    if char == " ":
        return "<space>"
    if char == "\t":
        return "<tab>"
    if char == "\n":
        return "<newline>"
    return char


def validate_and_log_alphabet(
    dataset: ChunkedLineDataset,
    alphabet: str,
    max_text_length: int,
    checkpoint_dir: str | Path,
) -> None:
    metadata = dataset.metadata
    if alphabet != metadata.alphabet:
        raise ValueError(
            "Training alphabet order differs from chunk metadata; class indices would be corrupted"
        )
    text_counts = metadata.text_char_counts
    target_counts = metadata.target_class_counts
    unused_chars = [char for char in alphabet if text_counts.get(char, 0) == 0]

    stats_path = Path(checkpoint_dir) / "alphabet_stats.tsv"
    with stats_path.open("w") as file:
        file.write("class_index\tchar\ttext_count\ttarget_count\n")
        for class_index, char in enumerate(alphabet):
            target_count = (
                "" if target_counts is None else str(target_counts[class_index])
            )
            file.write(
                f"{class_index}\t{printable_char(char)}\t"
                f"{text_counts.get(char, 0)}\t{target_count}\n"
            )

    print("\nAlphabet/data check:")
    print(f"  Metadata samples:       {metadata.samples}")
    print(f"  Dataset task:           {metadata.task}")
    print("  Alphabet order:         exact metadata order")
    print(
        f"  Unique chars in text:   {sum(count > 0 for count in text_counts.values())}"
    )
    print(f"  Max text length:        {metadata.max_observed_text_length}")
    print(f"  Stats file:             {stats_path}")
    print("  Per-char counts:")
    for class_index, char in enumerate(alphabet):
        target_suffix = (
            ""
            if target_counts is None
            else f", fcn_ocr_target={target_counts[class_index]}"
        )
        print(
            f"    [{class_index:>3}] {printable_char(char):>9}: "
            f"text={text_counts.get(char, 0)}{target_suffix}"
        )

    if unused_chars:
        printable = ", ".join(printable_char(char) for char in unused_chars)
        print(f"  Alphabet chars absent in data: {printable}")
    if metadata.max_observed_text_length > max_text_length:
        raise ValueError(
            f"Data contains text length {metadata.max_observed_text_length}, "
            f"but training max_text_length is {max_text_length}"
        )


__all__ = [
    "dataset_config_from_chunk_metadata",
    "effective_training_config_data",
    "load_dataset_from_config",
    "printable_char",
    "resolve_chunks_dir",
    "validate_and_log_alphabet",
]
