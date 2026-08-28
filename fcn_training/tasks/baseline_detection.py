from __future__ import annotations

from typing import Any

import torch

from fcn_synth_generator.dataset import SingleLineDatasetConfig
from fcn_tasks import BASELINE_DETECTION_TASK
from loss import baseline_detection_loss

from .base import TrainingTask, print_model_width


class BaselineDetectionTrainingTask(TrainingTask):
    name = BASELINE_DETECTION_TASK
    config_fields = frozenset(
        {
            "baseline_detection_strict_size",
            "baseline_detection_loss",
            "baseline_detection_positive_weight",
        }
    )

    def validate_config(self, config: Any) -> None:
        supported = ("bce", "mse", "smooth_l1")
        if config.baseline_detection_loss not in supported:
            raise ValueError(f"baseline_detection_loss must be one of {supported}")

    def num_outputs(self, alphabet: str) -> int:
        return 2

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> torch.Tensor:
        return baseline_detection_loss(
            logits,
            targets,
            strict_size=config.baseline_detection_strict_size,
            loss=config.baseline_detection_loss,
            positive_weight=config.baseline_detection_positive_weight,
        )

    def validate_model(
        self,
        model: torch.nn.Module,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> None:
        output_width = print_model_width(model, dataset_config)
        print(
            "Baseline detection target shape: "
            f"2x{dataset_config.image_height}x{dataset_config.image_width}"
        )
        if (
            config.baseline_detection_strict_size
            and output_width != dataset_config.image_width
        ):
            raise ValueError(
                "baseline_detection_strict_size requires model output width to match image "
                f"width, but architecture={config.architecture!r} gives T={output_width} "
                f"while targets have width {dataset_config.image_width}. Use a "
                "width-preserving 2D architecture such as baseline_detection_fcn, or set "
                "baseline_detection_strict_size: false."
            )

        was_training = model.training
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        try:
            model.eval()
            with torch.no_grad():
                output = model(
                    torch.zeros(
                        1,
                        dataset_config.channels,
                        dataset_config.image_height,
                        dataset_config.image_width,
                        device=device,
                    )
                )
        finally:
            if was_training:
                model.train()

        print(f"Model output shape: {tuple(output.shape)}")
        if output.dim() != 4 or output.size(1) != 2:
            raise ValueError(
                "baseline_detection requires model output shaped (B, 2, H, W), "
                f"got {tuple(output.shape)} from architecture={config.architecture!r}."
            )
        if config.baseline_detection_strict_size and output.shape[-2:] != (
            dataset_config.image_height,
            dataset_config.image_width,
        ):
            raise ValueError(
                "baseline_detection_strict_size requires model output HxW to match target "
                f"HxW, got output={tuple(output.shape[-2:])}, "
                f"target={(dataset_config.image_height, dataset_config.image_width)}."
            )

    def summary_lines(
        self,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> list[str]:
        return [
            "Batch targets: two-channel top/bottom baseline maps from chunks",
            f"Baseline detection loss: {config.baseline_detection_loss} "
            f"positive_weight={config.baseline_detection_positive_weight:g} "
            f"strict_size={config.baseline_detection_strict_size}",
        ]
