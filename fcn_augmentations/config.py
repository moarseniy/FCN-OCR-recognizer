from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


SUPPORTED_AUGMENTATIONS = (
    "cycle_shift",
    "preprocess_geometry",
    "strong_blur",
    "motion_blur",
    "scale",
    "darkening",
    "vertical_fade",
    "noise",
    "projective",
    "rotate",
    "x_pad",
    "crop_x",
    "crop_y",
    "rescale_quality",
    "random_line",
    "baseline_line",
    "morphology",
    "unsharp_mask",
    "brightness",
    "contrast",
    "invert",
)


def validate_augmentation_probabilities(
    value: dict[str, float],
) -> dict[str, float]:
    unknown = sorted(set(value) - set(SUPPORTED_AUGMENTATIONS))
    if unknown:
        raise ValueError(f"unknown augmentations: {unknown}")
    for name, probability in value.items():
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability for {name} must be between 0 and 1")
    return value


def validate_augmentation_parameters(
    value: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    unknown = sorted(set(value) - set(SUPPORTED_AUGMENTATIONS))
    if unknown:
        raise ValueError(f"unknown augmentation configs: {unknown}")
    return value


class AugmentationConfig(BaseModel):
    """Runtime contract for the torch augmentation pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    space_index: int = Field(ge=0)
    background: int = Field(default=255, ge=0, le=255)
    probabilities: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("probabilities")
    @classmethod
    def probabilities_must_be_valid(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        return validate_augmentation_probabilities(value)

    @field_validator("parameters")
    @classmethod
    def parameters_must_be_valid(
        cls, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        return validate_augmentation_parameters(value)

    @classmethod
    def from_alphabet(
        cls,
        *,
        alphabet: str,
        space_char: str,
        background: int = 255,
        probabilities: dict[str, float] | None = None,
        parameters: dict[str, dict[str, Any]] | None = None,
    ) -> "AugmentationConfig":
        if space_char not in alphabet:
            raise ValueError("space_char must be present in alphabet")
        return cls(
            space_index=alphabet.index(space_char),
            background=background,
            probabilities=probabilities or {},
            parameters=parameters or {},
        )


__all__ = [
    "AugmentationConfig",
    "SUPPORTED_AUGMENTATIONS",
    "validate_augmentation_parameters",
    "validate_augmentation_probabilities",
]
