from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable

from fcn_synth_generator.chunk_metadata import GENERATION_CONFIG_FILENAME
from fcn_synth_generator.run_directories import (
    is_timestamped_directory,
    latest_timestamped_directory,
    timestamped_directory,
)


TRAINING_CONFIG_FILENAME = "training_config.yaml"
EpochCallback = Callable[[dict[str, Any]], None]


def resolve_checkpoint_dir(
    configured_dir: str | Path,
    resume: bool = False,
) -> Path:
    base_dir = Path(configured_dir)
    if not resume:
        return timestamped_directory(base_dir)
    if is_timestamped_directory(base_dir):
        return base_dir

    latest_dir = latest_timestamped_directory(
        base_dir,
        required_file="latest_checkpoint.pth",
    )
    if latest_dir is not None:
        return latest_dir
    if (base_dir / "latest_checkpoint.pth").is_file():
        return base_dir
    return timestamped_directory(base_dir)


def save_experiment_config_snapshots(
    training_config_path: str | Path,
    chunks_dir: str | Path,
    checkpoint_dir: str | Path,
) -> tuple[Path, Path]:
    checkpoint_dir = Path(checkpoint_dir)
    training_config_path = Path(training_config_path).expanduser().resolve()

    training_snapshot = checkpoint_dir / TRAINING_CONFIG_FILENAME
    if training_config_path != training_snapshot.resolve():
        shutil.copy2(training_config_path, training_snapshot)
    print(f"Training config saved to {training_snapshot}")

    chunks_dir = Path(chunks_dir)
    generation_source = chunks_dir / GENERATION_CONFIG_FILENAME
    if not generation_source.is_file():
        raise FileNotFoundError(
            "Dataset directory must contain its generation config: "
            f"{generation_source}. Regenerate the dataset with the current generator."
        )
    generation_snapshot = checkpoint_dir / GENERATION_CONFIG_FILENAME
    if generation_source.resolve() != generation_snapshot.resolve():
        shutil.copy2(generation_source, generation_snapshot)
    print(f"Generation config saved to {generation_snapshot}")
    return training_snapshot, generation_snapshot


__all__ = [
    "EpochCallback",
    "TRAINING_CONFIG_FILENAME",
    "resolve_checkpoint_dir",
    "save_experiment_config_snapshots",
]
