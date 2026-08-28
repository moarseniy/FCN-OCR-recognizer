from __future__ import annotations

from typing import Any

import torch

from fcn_synth_generator.dataset import SingleLineDatasetConfig
from fcn_tasks import VERTICAL_SEGMENTATION_TASK
from loss import vertical_segmentation_loss

from .base import TrainingTask, print_model_width


class VerticalSegmentationTrainingTask(TrainingTask):
    name = VERTICAL_SEGMENTATION_TASK
    config_fields = frozenset(
        {
            "vertical_segmentation_crop_left",
            "vertical_segmentation_crop_right",
            "vertical_segmentation_strict_width",
            "vertical_segmentation_loss",
            "vertical_segmentation_positive_weight",
        }
    )

    def validate_config(self, config: Any) -> None:
        supported = ("mse", "smooth_l1", "bce")
        if config.vertical_segmentation_loss not in supported:
            raise ValueError(f"vertical_segmentation_loss must be one of {supported}")

    def num_outputs(self, alphabet: str) -> int:
        return 1

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> torch.Tensor:
        return vertical_segmentation_loss(
            logits,
            targets,
            crop_left=config.vertical_segmentation_crop_left,
            crop_right=config.vertical_segmentation_crop_right,
            strict_width=config.vertical_segmentation_strict_width,
            loss=config.vertical_segmentation_loss,
            positive_weight=config.vertical_segmentation_positive_weight,
        )

    def validate_model(
        self,
        model: torch.nn.Module,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> None:
        output_width = print_model_width(model, dataset_config)
        target_width = (
            dataset_config.image_width
            - config.vertical_segmentation_crop_left
            - config.vertical_segmentation_crop_right
        )
        if target_width <= 0:
            raise ValueError(
                "Vertical segmentation target crop is empty: "
                f"image_width={dataset_config.image_width}, "
                "vertical_segmentation_crop_left="
                f"{config.vertical_segmentation_crop_left}, "
                "vertical_segmentation_crop_right="
                f"{config.vertical_segmentation_crop_right}"
            )
        print(f"Vertical segmentation target width: {target_width}")
        if config.vertical_segmentation_strict_width and output_width != target_width:
            raise ValueError(
                "vertical_segmentation_strict_width requires model output width to match "
                f"target width, but architecture={config.architecture!r} gives "
                f"T={output_width} while targets have width {target_width}. Use a "
                "width-preserving architecture such as vertical_segmentation_fcn, or set "
                "vertical_segmentation_strict_width: false."
            )

    def summary_lines(
        self,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> list[str]:
        return [
            "Batch targets: vertical character-boundary scores from generator/chunks",
            "Vertical segmentation target crop: "
            f"[{config.vertical_segmentation_crop_left}, "
            f"-{config.vertical_segmentation_crop_right}]",
            f"Vertical segmentation loss: {config.vertical_segmentation_loss} "
            f"positive_weight={config.vertical_segmentation_positive_weight:g}",
        ]
