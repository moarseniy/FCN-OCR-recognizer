from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


class InferencePreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_x: float = Field(default=0.0, gt=-0.95)
    y_pad: float = Field(default=0.0, gt=-0.95)
    x_pad: float = Field(default=0.0, ge=0.0)


class BaselineInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    detector_checkpoint: Path | None = None
    detector_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    deskew: bool = True
    max_angle: float = Field(default=12.0, gt=0.0)
    strict_lines: bool = True
    line_pad: float = Field(default=0.08, ge=0.0)
    line_pad_px: float = Field(default=0.0, ge=0.0)


class OCRDecodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    top_k: int = Field(default=8, ge=1)
    center_fraction: float = Field(default=0.6, gt=0.0, le=1.0)
    min_score_width: int = Field(default=1, ge=1)


class OCRInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Path
    preprocessing: InferencePreprocessingConfig = Field(default_factory=InferencePreprocessingConfig)
    decode: OCRDecodeConfig = Field(default_factory=OCRDecodeConfig)


class SegmentatorInferenceConfig(BaseModel):
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
    baseline: BaselineInferenceConfig | None = None
    ocr: OCRInferenceConfig | None = None
    segmentator: SegmentatorInferenceConfig | None = None
    debug: DebugInferenceConfig = Field(default_factory=DebugInferenceConfig)

    @model_validator(mode="after")
    def validate_pipeline(self) -> "InferenceConfig":
        has_baseline = self.baseline is not None and self.baseline.enabled
        if not has_baseline and self.ocr is None and self.segmentator is None:
            raise ValueError("Inference config must enable at least one stage: baseline, segmentator, or ocr")
        if self.ocr is not None and self.ocr.decode.enabled and self.segmentator is None:
            raise ValueError("ocr.decode.enabled requires a segmentator section")
        if has_baseline and self.baseline.detector_checkpoint is None and self.ocr is None and self.segmentator is None:
            raise ValueError(
                "A standalone baseline stage requires baseline.detector_checkpoint"
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
        if self.baseline is not None:
            updates["baseline"] = self.baseline.model_copy(
                update={"detector_checkpoint": resolve(self.baseline.detector_checkpoint)}
            )
        if self.ocr is not None:
            updates["ocr"] = self.ocr.model_copy(
                update={"checkpoint": resolve(self.ocr.checkpoint)}
            )
        if self.segmentator is not None:
            updates["segmentator"] = self.segmentator.model_copy(
                update={"checkpoint": resolve(self.segmentator.checkpoint)}
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
