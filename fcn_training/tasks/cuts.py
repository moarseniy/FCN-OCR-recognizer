from __future__ import annotations

from typing import Any

import torch

from fcn_synth_generator.dataset import SingleLineDatasetConfig
from loss import cut_projection_loss

from .base import TrainingTask, print_model_width


class CutProjectionTrainingTask(TrainingTask):
    name = "cut_projection"
    target_format = "cut_projection"
    config_fields = frozenset(
        {
            "cut_projection_crop_left",
            "cut_projection_crop_right",
            "cut_projection_strict_width",
            "cut_projection_loss",
            "cut_projection_positive_weight",
        }
    )

    def validate_config(self, config: Any) -> None:
        supported = ("mse", "smooth_l1", "bce")
        if config.cut_projection_loss not in supported:
            raise ValueError(f"cut_projection_loss must be one of {supported}")

    def num_outputs(self, alphabet: str) -> int:
        return 1

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> torch.Tensor:
        return cut_projection_loss(
            logits,
            targets,
            crop_left=config.cut_projection_crop_left,
            crop_right=config.cut_projection_crop_right,
            strict_width=config.cut_projection_strict_width,
            loss=config.cut_projection_loss,
            positive_weight=config.cut_projection_positive_weight,
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
            - config.cut_projection_crop_left
            - config.cut_projection_crop_right
        )
        if target_width <= 0:
            raise ValueError(
                "Cut projection target crop is empty: "
                f"image_width={dataset_config.image_width}, "
                f"cut_projection_crop_left={config.cut_projection_crop_left}, "
                f"cut_projection_crop_right={config.cut_projection_crop_right}"
            )
        print(f"Cut projection target width: {target_width}")
        if config.cut_projection_strict_width and output_width != target_width:
            raise ValueError(
                "cut_projection_strict_width requires model output width to match target width, "
                f"but architecture={config.architecture!r} gives T={output_width} while "
                f"targets have width {target_width}. Use a width-preserving architecture such as "
                "vertical_segmentator_fcn, or set cut_projection_strict_width: false."
            )

    def summary_lines(
        self,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> list[str]:
        return [
            "Batch targets: cut projection heatmaps from generator/chunks",
            "Cut projection crop: "
            f"[{config.cut_projection_crop_left}, -{config.cut_projection_crop_right}]",
            f"Cut projection loss: {config.cut_projection_loss} "
            f"positive_weight={config.cut_projection_positive_weight:g}",
        ]
