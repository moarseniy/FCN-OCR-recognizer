from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from fcn_tasks import (
    BASELINE_DETECTION_TASK,
    FCN_OCR_TASK,
    VERTICAL_SEGMENTATION_TASK,
    normalize_task_name,
)


CHUNK_FORMAT = "fcn_synth_dataset"
CHUNK_METADATA_VERSION = 3
CHUNK_METADATA_FILENAME = "metadata.yaml"
GENERATION_CONFIG_FILENAME = "generation_config.yaml"
CHUNK_FILENAME_PATTERN = re.compile(r"chunk_\d{6}\.pt")


class ChunkManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    samples: int = Field(ge=1)

    @field_validator("file")
    @classmethod
    def file_must_be_a_chunk_basename(cls, value: str) -> str:
        if Path(value).name != value or CHUNK_FILENAME_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "chunk file must match chunk_XXXXXX.pt and contain no directory"
            )
        return value


class ChunkMetadata(BaseModel):
    """Versioned contract between dataset generation, training, and rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[CHUNK_FORMAT]
    version: Literal[CHUNK_METADATA_VERSION]
    task: str
    alphabet: str
    space_char: str
    samples: int = Field(ge=1)
    image_height: int = Field(ge=1)
    image_width: int = Field(ge=1)
    channels: int = Field(ge=1, le=3)
    background: int = Field(ge=0, le=255)
    word_count_min: int = Field(ge=1)
    word_count_max: int = Field(ge=1)
    word_length_min: int = Field(ge=1)
    word_length_max: int = Field(ge=1)
    crop_stride: int | None = Field(default=None, ge=1)
    min_crop_text_length: int = Field(ge=1)
    max_crop_text_length: int = Field(ge=1)
    edge_char_min_visible_ratio: float = Field(ge=0.0, le=1.0)
    edge_fragment_max_visible_ratio: float = Field(ge=0.0, le=1.0)
    neighbor_lines_probability: float = Field(ge=0.0, le=1.0)
    neighbor_line_min_crop_ratio: float = Field(ge=0.0, le=1.0)
    neighbor_line_visible_ratio_min: float = Field(ge=0.0, le=1.0)
    neighbor_line_gap_min: int = Field(ge=0)
    neighbor_line_gap_max: int = Field(ge=0)
    ink_spacing_enabled: bool = False
    ink_spacing_min_char_gap_px: float = 0.0
    ink_spacing_touch_gap_px: float = Field(default=0.5, ge=0.0)
    ink_spacing_touch_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    image_dtype: Literal["uint8"]
    target_dtype: Literal["int16", "uint8"]
    fcn_ocr_target_edge_bounds: Literal["ink"] | None = None
    vertical_segmentation_target_radius: int | None = Field(default=None, ge=0)
    vertical_segmentation_include_margins: bool | None = None
    baseline_detection_target_radius: int | None = Field(default=None, ge=0)
    chunk_size: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    chunks: list[ChunkManifestEntry]
    text_char_counts: dict[str, int]
    target_class_counts: list[int] | None
    max_observed_text_length: int = Field(ge=0)

    @field_validator("alphabet")
    @classmethod
    def alphabets_must_be_nonempty_and_unique(cls, value: str) -> str:
        if not value:
            raise ValueError("alphabet must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("alphabet characters must be unique and ordered")
        return value

    @field_validator("task")
    @classmethod
    def task_must_be_current(cls, value: str) -> str:
        return normalize_task_name(value)

    @field_validator("space_char")
    @classmethod
    def space_char_must_be_one_character(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("space_char must contain exactly one character")
        return value

    @field_validator("text_char_counts")
    @classmethod
    def text_counts_must_be_nonnegative(
        cls, value: dict[str, int]
    ) -> dict[str, int]:
        for char, count in value.items():
            if len(char) != 1:
                raise ValueError("text_char_counts keys must contain one character")
            if count < 0:
                raise ValueError("text_char_counts values must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "ChunkMetadata":
        if self.space_char not in self.alphabet:
            raise ValueError("space_char must be present in alphabet")
        if self.max_crop_text_length < self.min_crop_text_length:
            raise ValueError(
                "max_crop_text_length must be >= min_crop_text_length"
            )
        if self.word_count_max < self.word_count_min:
            raise ValueError("word_count_max must be >= word_count_min")
        if self.word_length_max < self.word_length_min:
            raise ValueError("word_length_max must be >= word_length_min")
        if self.edge_fragment_max_visible_ratio > self.edge_char_min_visible_ratio:
            raise ValueError(
                "edge fragment visibility cannot exceed accepted edge character visibility"
            )
        if self.neighbor_line_gap_max < self.neighbor_line_gap_min:
            raise ValueError("neighbor_line_gap_max must be >= neighbor_line_gap_min")
        if self.max_observed_text_length > self.max_crop_text_length:
            raise ValueError(
                "max_observed_text_length exceeds max_crop_text_length"
            )

        files = [entry.file for entry in self.chunks]
        if len(files) != len(set(files)):
            raise ValueError("chunk manifest contains duplicate files")
        if self.chunk_count != len(self.chunks):
            raise ValueError("chunk_count does not match the manifest length")
        if self.samples != sum(entry.samples for entry in self.chunks):
            raise ValueError("samples does not match the sum of manifest sample counts")

        if set(self.text_char_counts) != set(self.alphabet):
            raise ValueError(
                "text_char_counts keys must exactly match the dataset alphabet"
            )
        if self.target_class_counts is not None:
            if len(self.target_class_counts) != len(self.alphabet):
                raise ValueError("target_class_counts length must match alphabet length")
            if any(count < 0 for count in self.target_class_counts):
                raise ValueError("target_class_counts values must be non-negative")

        if self.task == FCN_OCR_TASK:
            if (
                self.target_dtype != "int16"
                or self.target_class_counts is None
                or self.fcn_ocr_target_edge_bounds != "ink"
            ):
                raise ValueError(
                    "fcn_ocr requires int16 targets, ink edge bounds, and target_class_counts"
                )
            expected_ocr_values = self.samples * self.image_width
            if sum(self.target_class_counts) != expected_ocr_values:
                raise ValueError(
                    "target_class_counts sum does not match samples * image_width"
                )
        elif self.fcn_ocr_target_edge_bounds is not None or self.target_class_counts is not None:
            raise ValueError(
                "fcn_ocr target metadata is present for a different task"
            )

        if self.task == VERTICAL_SEGMENTATION_TASK:
            if (
                self.target_dtype != "uint8"
                or self.vertical_segmentation_target_radius is None
                or self.vertical_segmentation_include_margins is None
            ):
                raise ValueError(
                    "vertical_segmentation requires uint8 targets and its target parameters"
                )
        elif (
            self.vertical_segmentation_target_radius is not None
            or self.vertical_segmentation_include_margins is not None
        ):
            raise ValueError(
                "vertical segmentation target metadata is present for a different task"
            )

        if self.task == BASELINE_DETECTION_TASK:
            if self.target_dtype != "uint8" or self.baseline_detection_target_radius is None:
                raise ValueError(
                    "baseline_detection requires uint8 targets and target radius"
                )
        elif self.baseline_detection_target_radius is not None:
            raise ValueError(
                "baseline detection target metadata is present for a different task"
            )
        return self

    def require_task(self, task: str) -> None:
        task = normalize_task_name(task)
        if self.task != task:
            raise ValueError(
                f"Dataset task is {self.task!r}, requested training task is {task!r}"
            )

    def dataset_config_data(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "alphabet": self.alphabet,
            "space_char": self.space_char,
            "samples": self.samples,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "channels": self.channels,
            "background": self.background,
            "word_count_min": self.word_count_min,
            "word_count_max": self.word_count_max,
            "word_length_min": self.word_length_min,
            "word_length_max": self.word_length_max,
            "crop_stride": self.crop_stride,
            "min_crop_text_length": self.min_crop_text_length,
            "max_crop_text_length": self.max_crop_text_length,
            "edge_char_min_visible_ratio": self.edge_char_min_visible_ratio,
            "edge_fragment_max_visible_ratio": self.edge_fragment_max_visible_ratio,
            "neighbor_lines_probability": self.neighbor_lines_probability,
            "neighbor_line_min_crop_ratio": self.neighbor_line_min_crop_ratio,
            "neighbor_line_visible_ratio_min": self.neighbor_line_visible_ratio_min,
            "neighbor_line_gap_min": self.neighbor_line_gap_min,
            "neighbor_line_gap_max": self.neighbor_line_gap_max,
            "ink_spacing_enabled": self.ink_spacing_enabled,
            "ink_spacing_min_char_gap_px": self.ink_spacing_min_char_gap_px,
            "ink_spacing_touch_gap_px": self.ink_spacing_touch_gap_px,
            "ink_spacing_touch_probability": self.ink_spacing_touch_probability,
            "chunk_size": self.chunk_size,
        } | self._task_config_data()

    def _task_config_data(self) -> dict[str, Any]:
        if self.task == VERTICAL_SEGMENTATION_TASK:
            return {
                "vertical_segmentation_target_radius": self.vertical_segmentation_target_radius,
                "vertical_segmentation_include_margins": self.vertical_segmentation_include_margins,
            }
        if self.task == BASELINE_DETECTION_TASK:
            return {
                "baseline_detection_target_radius": self.baseline_detection_target_radius,
            }
        return {}


def load_chunk_metadata(
    root_dir: str | Path,
) -> ChunkMetadata:
    root = Path(root_dir).expanduser().resolve()
    metadata_path = root / CHUNK_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Required chunk metadata not found: {metadata_path}. "
            "Offline datasets without metadata are not supported."
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Chunk metadata must be a YAML mapping: {metadata_path}")

    dataset_format = raw.get("format")
    if dataset_format != CHUNK_FORMAT:
        raise ValueError(
            f"Unsupported dataset format {dataset_format!r} in {metadata_path}. "
            "Regenerate this dataset with the current generate_dataset command."
        )
    version = raw.get("version")
    if version != CHUNK_METADATA_VERSION:
        raise ValueError(
            f"Unsupported dataset metadata version {version!r} in {metadata_path}. "
            "Regenerate this dataset with the current generate_dataset command."
        )

    return ChunkMetadata.model_validate(raw)


def save_chunk_metadata(metadata: ChunkMetadata, root_dir: str | Path) -> Path:
    metadata_path = Path(root_dir) / CHUNK_METADATA_FILENAME
    temporary_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            metadata.model_dump(mode="json"),
            file,
            allow_unicode=True,
            sort_keys=False,
        )
    temporary_path.replace(metadata_path)
    return metadata_path
