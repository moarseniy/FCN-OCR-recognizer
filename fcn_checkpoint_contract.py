from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fcn_tasks import normalize_task_name, task_output_channels


CHECKPOINT_FORMAT = "fcn_model_checkpoint"
CHECKPOINT_VERSION = 1


class FCNModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: str
    architecture_params: dict[str, Any]
    in_channels: int = Field(ge=1)
    num_classes: int = Field(ge=1)
    task: str

    @field_validator("task")
    @classmethod
    def task_must_be_supported(cls, value: str) -> str:
        return normalize_task_name(value)


def validate_checkpoint_contract(payload: Any) -> FCNModelConfig:
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint must contain a mapping, got {type(payload).__name__}"
        )
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint format must be {CHECKPOINT_FORMAT!r}; regenerate the checkpoint"
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version must be {CHECKPOINT_VERSION}; regenerate the checkpoint"
        )

    alphabet = payload.get("alphabet")
    training_config = payload.get("config")
    if not isinstance(alphabet, str) or not alphabet:
        raise ValueError("Checkpoint alphabet must be a non-empty string")
    if not isinstance(training_config, dict):
        raise ValueError("Checkpoint config must be a mapping")

    model_config = FCNModelConfig.model_validate(payload.get("model_config"))
    expected_classes = task_output_channels(model_config.task, alphabet)
    if model_config.num_classes != expected_classes:
        raise ValueError(
            f"Checkpoint task {model_config.task!r} expects {expected_classes} "
            f"output classes, got {model_config.num_classes}"
        )
    return model_config


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "FCNModelConfig",
    "validate_checkpoint_contract",
]
