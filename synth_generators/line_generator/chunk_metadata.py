from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


CHUNK_FORMAT = "fcn_ocr_line_chunks"
CHUNK_METADATA_VERSION = 2
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
    alphabet: str
    space_char: str
    samples: int = Field(ge=1)
    image_height: int = Field(ge=1)
    image_width: int = Field(ge=1)
    channels: int = Field(ge=1, le=3)
    background: int = Field(ge=0, le=255)
    min_text_length: int = Field(ge=1)
    max_text_length: int = Field(ge=1)
    line_crops: bool
    word_count_min: int = Field(ge=1)
    word_count_max: int = Field(ge=1)
    word_length_min: int = Field(ge=1)
    word_length_max: int = Field(ge=1)
    crop_stride: int | None = Field(default=None, ge=1)
    min_crop_text_length: int = Field(ge=1)
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
    ocr_targets: bool
    ocr_target_edge_bounds: Literal["ink"] = "ink"
    cut_projection_targets: bool
    cut_projection_peak_radius: int = Field(ge=0)
    cut_projection_include_margins: bool
    baseline_targets: bool
    baseline_target_radius: int = Field(ge=0)
    dtype: Literal["uint8"]
    ocr_target_dtype: Literal["int16"] | None = None
    cut_projection_target_dtype: Literal["uint8"] | None = None
    baseline_target_dtype: Literal["uint8"] | None = None
    chunk_size: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    chunks: list[ChunkManifestEntry]
    text_char_counts: dict[str, int]
    ocr_class_counts: list[int] | None
    max_observed_text_length: int = Field(ge=0)

    @field_validator("alphabet")
    @classmethod
    def alphabets_must_be_nonempty_and_unique(cls, value: str) -> str:
        if not value:
            raise ValueError("alphabet must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("alphabet characters must be unique and ordered")
        return value

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
        if self.max_text_length < self.min_text_length:
            raise ValueError("max_text_length must be >= min_text_length")
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
        if self.ocr_class_counts is not None:
            if len(self.ocr_class_counts) != len(self.alphabet):
                raise ValueError("ocr_class_counts length must match alphabet length")
            if any(count < 0 for count in self.ocr_class_counts):
                raise ValueError("ocr_class_counts values must be non-negative")

        if self.ocr_targets:
            if self.ocr_target_dtype != "int16" or self.ocr_class_counts is None:
                raise ValueError(
                    "OCR targets require int16 dtype and ocr_class_counts"
                )
            expected_ocr_values = self.samples * self.image_width
            if sum(self.ocr_class_counts) != expected_ocr_values:
                raise ValueError(
                    "ocr_class_counts sum does not match samples * image_width"
                )
        elif self.ocr_target_dtype is not None or self.ocr_class_counts is not None:
            raise ValueError(
                "OCR target metadata is present while ocr_targets is false"
            )

        expected_cut_dtype = "uint8" if self.cut_projection_targets else None
        if self.cut_projection_target_dtype != expected_cut_dtype:
            raise ValueError(
                "cut projection target dtype does not match target presence"
            )
        expected_baseline_dtype = "uint8" if self.baseline_targets else None
        if self.baseline_target_dtype != expected_baseline_dtype:
            raise ValueError("baseline target dtype does not match target presence")
        return self

    def require_target(self, target_format: str) -> None:
        available = {
            "fcn_ocr": self.ocr_targets,
            "cut_projection": self.cut_projection_targets,
            "baseline_heatmap": self.baseline_targets,
        }
        if target_format not in available:
            raise ValueError(f"Unknown target format: {target_format}")
        if not available[target_format]:
            raise ValueError(
                f"Dataset metadata does not provide required target format {target_format!r}"
            )

    def dataset_config_data(self) -> dict[str, Any]:
        return {
            "alphabet": self.alphabet,
            "space_char": self.space_char,
            "samples": self.samples,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "channels": self.channels,
            "background": self.background,
            "min_text_length": self.min_text_length,
            "max_text_length": self.max_text_length,
            "line_crops": self.line_crops,
            "word_count_min": self.word_count_min,
            "word_count_max": self.word_count_max,
            "word_length_min": self.word_length_min,
            "word_length_max": self.word_length_max,
            "crop_stride": self.crop_stride,
            "min_crop_text_length": self.min_crop_text_length,
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
            "save_ocr_targets": self.ocr_targets,
            "save_cut_projection_targets": self.cut_projection_targets,
            "cut_projection_peak_radius": self.cut_projection_peak_radius,
            "cut_projection_include_margins": self.cut_projection_include_margins,
            "save_baseline_targets": self.baseline_targets,
            "baseline_target_radius": self.baseline_target_radius,
            "chunk_size": self.chunk_size,
        }


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
