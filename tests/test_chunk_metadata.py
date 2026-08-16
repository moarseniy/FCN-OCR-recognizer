from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest
import torch
import yaml

from fcn_synth_generator.chunk_dataset import ChunkedLineDataset
from fcn_synth_generator.chunk_metadata import (
    CHUNK_FORMAT,
    CHUNK_METADATA_VERSION,
    ChunkMetadata,
    load_chunk_metadata,
    save_chunk_metadata,
)
from fcn_synth_generator.dataset import SingleLineDatasetConfig
from fcn_synth_generator.generate_dataset import build_metadata
from train import (
    TrainingConfig,
    dataset_config_from_chunk_metadata,
    load_dataset_from_config,
)


ALPHABET = " AB"


def _chunk_payload(*, image_dtype: torch.dtype = torch.uint8) -> dict:
    ocr_targets = torch.zeros((2, 32), dtype=torch.int16)
    ocr_targets[0, 8:24] = 1
    ocr_targets[1, 8:24] = 2
    return {
        "images": torch.zeros((2, 1, 16, 32), dtype=image_dtype),
        "texts": ["A", "B"],
        "ocr_targets": ocr_targets,
    }


def _metadata_data(*, version: int = CHUNK_METADATA_VERSION) -> dict:
    ocr_counts = [32, 16, 16]
    data = {
        "format": CHUNK_FORMAT,
        "version": version,
        "alphabet": ALPHABET,
        "space_char": " ",
        "samples": 2,
        "image_height": 16,
        "image_width": 32,
        "channels": 1,
        "background": 255,
        "min_text_length": 1,
        "max_text_length": 2,
        "line_crops": True,
        "word_count_min": 1,
        "word_count_max": 1,
        "word_length_min": 1,
        "word_length_max": 2,
        "crop_stride": 32,
        "min_crop_text_length": 1,
        "edge_char_min_visible_ratio": 0.75,
        "edge_fragment_max_visible_ratio": 0.25,
        "neighbor_lines_probability": 0.0,
        "neighbor_line_min_crop_ratio": 0.65,
        "neighbor_line_visible_ratio_min": 0.06,
        "neighbor_line_gap_min": 0,
        "neighbor_line_gap_max": 5,
        "ocr_targets": True,
        "cut_projection_targets": False,
        "cut_projection_peak_radius": 1,
        "cut_projection_include_margins": True,
        "baseline_targets": False,
        "baseline_target_radius": 1,
        "dtype": "uint8",
        "chunk_size": 2,
        "chunk_count": 1,
        "chunks": [{"file": "chunk_000000.pt", "samples": 2}],
    }
    if version == CHUNK_METADATA_VERSION:
        data.update(
            {
                "ocr_target_dtype": "int16",
                "cut_projection_target_dtype": None,
                "baseline_target_dtype": None,
                "text_char_counts": {" ": 0, "A": 1, "B": 1},
                "ocr_class_counts": ocr_counts,
                "max_observed_text_length": 1,
            }
        )
    return data


def _write_dataset(root: Path) -> ChunkMetadata:
    root.mkdir()
    torch.save(_chunk_payload(), root / "chunk_000000.pt")
    metadata = ChunkMetadata.model_validate(_metadata_data())
    save_chunk_metadata(metadata, root)
    return metadata


def _dataset_config(metadata: ChunkMetadata) -> SingleLineDatasetConfig:
    return SingleLineDatasetConfig.model_validate(metadata.dataset_config_data())


def test_dataset_initialization_uses_manifest_counts_without_loading_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "chunks"
    metadata = _write_dataset(root)

    def fail_if_loaded(path: Path) -> dict:
        raise AssertionError(f"chunk was loaded during initialization: {path}")

    monkeypatch.setattr(
        "fcn_synth_generator.chunk_dataset.load_torch_chunk",
        fail_if_loaded,
    )
    dataset = ChunkedLineDataset(root, config=_dataset_config(metadata))

    assert len(dataset) == 2
    assert dataset.chunks == [{"file": "chunk_000000.pt", "samples": 2}]


def test_dataset_rejects_chunk_files_absent_from_manifest(tmp_path: Path) -> None:
    root = tmp_path / "chunks"
    metadata = _write_dataset(root)
    torch.save(_chunk_payload(), root / "chunk_000001.pt")

    with pytest.raises(ValueError, match="unexpected=.*chunk_000001.pt"):
        ChunkedLineDataset(root, config=_dataset_config(metadata))


def test_dataset_validates_tensor_dtype_when_chunk_is_first_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chunks"
    metadata = _write_dataset(root)
    torch.save(_chunk_payload(image_dtype=torch.float32), root / "chunk_000000.pt")
    dataset = ChunkedLineDataset(root, config=_dataset_config(metadata))

    with pytest.raises(ValueError, match="images dtype must be uint8"):
        _ = dataset[0]


def test_loader_rejects_old_metadata_and_requires_regeneration(tmp_path: Path) -> None:
    root = tmp_path / "chunks"
    root.mkdir()
    with (root / "metadata.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_metadata_data(version=1), file)

    with pytest.raises(ValueError, match="Regenerate this dataset"):
        load_chunk_metadata(root)


def test_training_config_cannot_override_dataset_contract() -> None:
    metadata = ChunkMetadata.model_validate(_metadata_data())

    with pytest.raises(
        ValidationError, match="Extra inputs are not permitted"
    ) as error:
        TrainingConfig.model_validate(
            {
                "architecture": "fcn_ocr",
                "chunks_dir": "/unused",
                "alphabet": " BA",
            }
        )
    assert error.value.errors()[0]["loc"] == ("alphabet",)

    training = TrainingConfig.model_validate(
        {
            "architecture": "fcn_ocr",
            "chunks_dir": "/unused",
            "seed": 123,
            "augmentation_probabilities": {"x_pad": 0.5},
            "augmentations": {"x_pad": {"pad": 0.1}},
        }
    )
    dataset_config = dataset_config_from_chunk_metadata(training, metadata)

    assert dataset_config.alphabet == ALPHABET
    assert dataset_config.image_height == 16
    assert dataset_config.image_width == 32
    assert dataset_config.channels == 1
    assert dataset_config.seed == 123
    assert dataset_config.augmentation_probabilities == {"x_pad": 0.5}


def test_removed_config_vocabulary_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SingleLineDatasetConfig.model_validate(
            {
                "alphabet": ALPHABET,
                "sample_alphabet": ALPHABET,
            }
        )

    with pytest.raises(ValidationError, match="loss_mode must be one of"):
        TrainingConfig.model_validate(
            {
                "architecture": "fcn_ocr",
                "chunks_dir": "/unused",
                "loss_mode": "removed_task",
            }
        )


def test_generation_writer_builds_a_complete_current_contract() -> None:
    config = SingleLineDatasetConfig(
        alphabet=ALPHABET,
        samples=2,
        image_height=16,
        image_width=32,
        min_text_length=1,
        max_text_length=2,
        chunk_size=2,
        save_ocr_targets=True,
    )

    metadata = build_metadata(
        config,
        [
            {
                "file": "chunk_000000.pt",
                "samples": 2,
                "text_char_counts": {"A": 1, "B": 1},
                "ocr_class_counts": [32, 16, 16],
                "max_observed_text_length": 1,
            }
        ],
    )

    assert metadata.version == CHUNK_METADATA_VERSION
    assert metadata.samples == 2
    assert metadata.ocr_target_dtype == "int16"
    assert metadata.ocr_class_counts == [32, 16, 16]
    assert metadata.text_char_counts == {" ": 0, "A": 1, "B": 1}


def test_training_loader_builds_model_input_config_only_from_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chunks"
    _write_dataset(root)
    training = TrainingConfig.model_validate(
        {
            "architecture": "fcn_ocr",
            "chunks_dir": str(root),
            "loss_mode": "fcn_ocr",
            "augmentation_probabilities": {"x_pad": 0.25},
        }
    )

    dataset, dataset_config = load_dataset_from_config(training)

    assert len(dataset) == 2
    assert dataset_config.alphabet == ALPHABET
    assert dataset_config.image_height == 16
    assert dataset_config.image_width == 32
    assert dataset_config.augmentation_probabilities == {"x_pad": 0.25}
