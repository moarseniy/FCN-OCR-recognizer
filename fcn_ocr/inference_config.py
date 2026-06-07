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


class OCRInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Path
    preprocessing: InferencePreprocessingConfig = Field(default_factory=InferencePreprocessingConfig)


class SegmentatorInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Path | None = None
    preprocessing: InferencePreprocessingConfig = Field(default_factory=InferencePreprocessingConfig)
    cut_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    peak_min_distance: int | None = Field(default=None, ge=1)
    cut_postprocess: str | None = None
    cut_min_width: int | None = Field(default=None, ge=1)
    cut_max_width: int | None = Field(default=None, ge=0)
    cut_candidate_threshold: float | None = Field(default=None, ge=0.0, lt=1.0)
    cut_smooth_radius: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_postprocess(self) -> "SegmentatorInferenceConfig":
        if self.cut_postprocess not in {None, "peaks", "widths"}:
            raise ValueError("cut_postprocess must be 'peaks' or 'widths'")
        return self


class SegmentatorDecodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    top_k: int = Field(default=8, ge=1)
    center_fraction: float = Field(default=0.6, gt=0.0, le=1.0)
    min_score_width: int = Field(default=1, ge=1)


class DebugInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=8, ge=1)


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str | None = None
    baseline: BaselineInferenceConfig = Field(default_factory=BaselineInferenceConfig)
    ocr: OCRInferenceConfig
    segmentator: SegmentatorInferenceConfig = Field(default_factory=SegmentatorInferenceConfig)
    decode: SegmentatorDecodeConfig = Field(default_factory=SegmentatorDecodeConfig)
    debug: DebugInferenceConfig = Field(default_factory=DebugInferenceConfig)

    @model_validator(mode="after")
    def validate_pipeline(self) -> "InferenceConfig":
        if self.decode.enabled and self.segmentator.checkpoint is None:
            raise ValueError("decode.enabled requires segmentator.checkpoint")
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

        return self.model_copy(
            update={
                "baseline": self.baseline.model_copy(
                    update={"detector_checkpoint": resolve(self.baseline.detector_checkpoint)}
                ),
                "ocr": self.ocr.model_copy(
                    update={"checkpoint": resolve(self.ocr.checkpoint)}
                ),
                "segmentator": self.segmentator.model_copy(
                    update={"checkpoint": resolve(self.segmentator.checkpoint)}
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

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
