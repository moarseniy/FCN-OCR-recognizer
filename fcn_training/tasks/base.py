from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from fcn_synth_generator.dataset import SingleLineDatasetConfig


class TrainingTask(ABC):
    """Complete training-time contract for one target representation."""

    name: str
    config_fields: frozenset[str]

    def validate_config(self, config: Any) -> None:
        """Validate values whose meaning belongs exclusively to this task."""

    @abstractmethod
    def num_outputs(self, alphabet: str) -> int:
        """Return the number of output channels/classes required by the model."""

    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> torch.Tensor:
        """Compute the task loss for one batch."""

    @abstractmethod
    def validate_model(
        self,
        model: torch.nn.Module,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> None:
        """Validate that model output geometry satisfies the task contract."""

    @abstractmethod
    def summary_lines(
        self,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> list[str]:
        """Return task-specific lines for the startup log."""


def output_width_for_model(model: torch.nn.Module, width: int) -> int:
    if hasattr(model, "output_width_for_input_width"):
        return int(model.output_width_for_input_width(width))

    output_width = int(width)
    for module in model.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        kernel = module.kernel_size[1]
        stride = module.stride[1]
        padding = module.padding[1]
        dilation = module.dilation[1]
        output_width = (
            output_width + 2 * padding - dilation * (kernel - 1) - 1
        ) // stride + 1
    return output_width


def print_model_width(model: torch.nn.Module, dataset_config: SingleLineDatasetConfig) -> int:
    output_width = output_width_for_model(model, dataset_config.image_width)
    print(
        f"Model output width: {output_width} for input width {dataset_config.image_width}"
    )
    return output_width
