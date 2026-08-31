from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from fcn_augmentations import (
    validate_augmentation_parameters,
    validate_augmentation_probabilities,
)
from fcn_architectures import available_architectures, normalize_architecture_name

from .tasks import (
    all_training_task_config_fields,
    available_training_tasks,
    get_training_task,
)


SUPPORTED_SCHEDULERS = ("none", "reduce_on_plateau", "cosine", "step")
SUPPORTED_OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str
    architecture_params: dict[str, Any] = Field(default_factory=dict)
    chunks_dir: str = Field(min_length=1)

    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=128, ge=1)
    batch_count: int | None = Field(default=None, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    optimizer: str = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    adam_eps: float = Field(default=1e-8, gt=0.0)
    sgd_momentum: float = Field(default=0.9, ge=0.0)
    sgd_nesterov: bool = False
    rmsprop_alpha: float = Field(default=0.99, gt=0.0, lt=1.0)
    rmsprop_momentum: float = Field(default=0.0, ge=0.0)
    rmsprop_eps: float = Field(default=1e-8, gt=0.0)

    task: str
    fcn_ocr_crop_left: int = Field(default=6, ge=0)
    fcn_ocr_crop_right: int = Field(default=5, ge=0)
    fcn_ocr_strict_width: bool = False
    fcn_ocr_target_min_majority: float = Field(default=0.6, ge=0.0, le=1.0)
    fcn_ocr_space_weight: float = Field(default=1.0, gt=0.0)
    vertical_segmentation_crop_left: int = Field(default=0, ge=0)
    vertical_segmentation_crop_right: int = Field(default=0, ge=0)
    vertical_segmentation_strict_width: bool = True
    vertical_segmentation_loss: str = "mse"
    vertical_segmentation_positive_weight: float = Field(default=1.0, ge=1.0)
    baseline_detection_strict_size: bool = True
    baseline_detection_loss: str = "bce"
    baseline_detection_positive_weight: float = Field(default=4.0, ge=1.0)

    scheduler: str = "reduce_on_plateau"
    scheduler_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    scheduler_patience: int = Field(default=3, ge=0)
    scheduler_min_lr: float = Field(default=1e-6, ge=0.0)
    scheduler_threshold: float = Field(default=1e-4, ge=0.0)
    scheduler_cooldown: int = Field(default=0, ge=0)
    scheduler_t_max: int | None = Field(default=None, ge=1)
    scheduler_eta_min: float = Field(default=1e-6, ge=0.0)
    scheduler_step_size: int = Field(default=10, ge=1)
    scheduler_gamma: float = Field(default=0.5, gt=0.0)

    checkpoint_dir: str = "checkpoints"
    max_train_batches: int | None = None
    max_val_batches: int | None = 50
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = 0
    resume: bool = False

    num_workers: int = Field(default=0, ge=0)
    drop_last: bool = False
    prefetch_factor: int = Field(default=2, ge=1)
    persistent_workers: bool = True
    chunk_cache_size: int = Field(default=2, ge=1)
    chunk_aware_batches: bool = True

    log_every: int = Field(default=1, ge=0)
    preview_samples: int = Field(default=0, ge=0)
    preview_dir: str = "input_previews"

    gpu_augmentations: bool = True
    gpu_augment_val: bool = False
    augmentation_probabilities: dict[str, float] = Field(default_factory=dict)
    augmentations: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def model_validate_with_paths(
        cls, data: Any, config_path: str | Path
    ) -> "TrainingConfig":
        data = dict(data)
        config_dir = Path(config_path).resolve().parent
        for key in ("chunks_dir", "checkpoint_dir", "preview_dir"):
            value = data.get(key)
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = config_dir / path
                data[key] = str(path.resolve())
        return cls.model_validate(data)

    @field_validator("scheduler")
    @classmethod
    def scheduler_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        if value not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SUPPORTED_SCHEDULERS}")
        return value

    @field_validator("optimizer")
    @classmethod
    def optimizer_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        if value not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {SUPPORTED_OPTIMIZERS}")
        return value

    @field_validator("architecture")
    @classmethod
    def architecture_must_be_supported(cls, value: str) -> str:
        value = normalize_architecture_name(value)
        if value not in available_architectures():
            raise ValueError(f"architecture must be one of {available_architectures()}")
        return value

    @field_validator("task")
    @classmethod
    def task_must_be_supported(cls, value: str) -> str:
        value = value.lower()
        supported = available_training_tasks()
        if value not in supported:
            raise ValueError(f"task must be one of {supported}")
        return value

    @field_validator("vertical_segmentation_loss", "baseline_detection_loss")
    @classmethod
    def task_loss_name_must_be_normalized(cls, value: str) -> str:
        return value.lower()

    @model_validator(mode="after")
    def task_fields_must_match_selected_task(self) -> "TrainingConfig":
        task = get_training_task(self.task)
        explicit_task_fields = self.model_fields_set & all_training_task_config_fields()
        unexpected = sorted(explicit_task_fields - task.config_fields)
        if unexpected:
            raise ValueError(
                f"training task {task.name!r} does not accept config fields: {unexpected}"
            )
        task.validate_config(self)
        return self

    @field_validator("augmentation_probabilities")
    @classmethod
    def augmentation_probabilities_must_be_valid(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        return validate_augmentation_probabilities(value)

    @field_validator("augmentations")
    @classmethod
    def augmentations_must_be_known(
        cls, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        return validate_augmentation_parameters(value)


def load_training_config(config_path: str | Path) -> tuple[TrainingConfig, dict]:
    with Path(config_path).open("r") as file:
        config_data = yaml.safe_load(file)
    return TrainingConfig.model_validate_with_paths(
        config_data, config_path
    ), config_data


__all__ = [
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_SCHEDULERS",
    "TrainingConfig",
    "load_training_config",
]
