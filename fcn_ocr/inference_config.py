from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


class InferencePreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_x: float = Field(default=0.0, gt=-0.95)
    y_pad: float = Field(default=0.0, gt=-0.95)
    x_pad: float = Field(default=0.0, ge=0.0)


class BaselineDetectionInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    detector_checkpoint: Path | None = None
    detector_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    deskew: bool = True
    max_angle: float = Field(default=12.0, gt=0.0)
    line_pad: float = Field(default=0.08, ge=0.0)
    line_pad_px: float = Field(default=0.0, ge=0.0)


class GlyphWidthPriorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    weight: float = Field(default=0.0, ge=0.0)
    normalize_by: Literal["input_height", "median_cell_width"] = "input_height"
    ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @field_validator("ranges")
    @classmethod
    def ranges_must_be_valid(cls, value: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        for group, bounds in value.items():
            if not isinstance(group, str) or group == "":
                raise ValueError("glyph_width_prior range keys must be non-empty strings")
            low, high = bounds
            if low < 0.0 or high < 0.0:
                raise ValueError("glyph_width_prior ranges must be non-negative")
            if high < low:
                raise ValueError("glyph_width_prior range upper bound must be >= lower bound")
        return value


class OCRDecodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["cells", "dp"] = "cells"
    top_k: int = Field(default=8, ge=1)
    center_fraction: float = Field(default=0.6, gt=0.0, le=1.0)
    min_score_width: int = Field(default=1, ge=1)
    cut_weight: float = Field(default=1.0, ge=0.0)
    ocr_weight: float = Field(default=1.0, ge=0.0)
    width_weight: float = Field(default=0.05, ge=0.0)
    skip_cut_penalty: float = Field(default=0.35, ge=0.0)
    glyph_width_prior: GlyphWidthPriorConfig = Field(default_factory=GlyphWidthPriorConfig)


class FCNOCRInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Path
    preprocessing: InferencePreprocessingConfig = Field(default_factory=InferencePreprocessingConfig)
    decode: OCRDecodeConfig = Field(default_factory=OCRDecodeConfig)


class VerticalSegmentationInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Path
    preprocessing: InferencePreprocessingConfig = Field(default_factory=InferencePreprocessingConfig)
    cut_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    cut_min_width: int | None = Field(default=None, ge=1)
    cut_max_width: int | None = Field(default=None, ge=0)
    cut_smooth_radius: int | None = Field(default=None, ge=0)


class DebugInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=8, ge=1)


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str | None = None
    baseline_detection: BaselineDetectionInferenceConfig | None = None
    fcn_ocr: FCNOCRInferenceConfig | None = None
    vertical_segmentation: VerticalSegmentationInferenceConfig | None = None
    debug: DebugInferenceConfig = Field(default_factory=DebugInferenceConfig)

    @model_validator(mode="after")
    def validate_pipeline(self) -> "InferenceConfig":
        baseline = self.baseline_detection
        has_baseline_detection = baseline is not None and baseline.enabled
        if (
            not has_baseline_detection
            and self.fcn_ocr is None
            and self.vertical_segmentation is None
        ):
            raise ValueError(
                "Inference config must enable at least one task: "
                "baseline_detection, vertical_segmentation, or fcn_ocr"
            )
        if (
            self.fcn_ocr is not None
            and self.fcn_ocr.decode.enabled
            and self.vertical_segmentation is None
        ):
            raise ValueError(
                "fcn_ocr.decode.enabled requires a vertical_segmentation section"
            )
        if has_baseline_detection and baseline.detector_checkpoint is None:
            raise ValueError(
                "An enabled baseline_detection section requires detector_checkpoint"
            )
        return self

    @classmethod
    def load(cls, config_path: str | Path) -> "InferenceConfig":
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Inference config not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        config = cls.model_validate(data)
        return config._resolve_paths(path.parent)

    def _resolve_paths(self, config_dir: Path) -> "InferenceConfig":
        def resolve(value: Path | None) -> Path | None:
            if value is None:
                return None
            path = value.expanduser()
            return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()

        updates: dict[str, Any] = {}
        if self.baseline_detection is not None:
            updates["baseline_detection"] = self.baseline_detection.model_copy(
                update={
                    "detector_checkpoint": resolve(
                        self.baseline_detection.detector_checkpoint
                    )
                }
            )
        if self.fcn_ocr is not None:
            updates["fcn_ocr"] = self.fcn_ocr.model_copy(
                update={"checkpoint": resolve(self.fcn_ocr.checkpoint)}
            )
        if self.vertical_segmentation is not None:
            updates["vertical_segmentation"] = self.vertical_segmentation.model_copy(
                update={"checkpoint": resolve(self.vertical_segmentation.checkpoint)}
            )
        return self.model_copy(update=updates)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def save(self, config_path: str | Path) -> Path:
        path = Path(config_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                self.to_dict(),
                file,
                allow_unicode=True,
                sort_keys=False,
            )
        return path
