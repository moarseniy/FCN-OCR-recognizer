from __future__ import annotations

from typing import Any

import torch

from fcn_synth_generator.dataset import SingleLineDatasetConfig
from loss import baseline_heatmap_loss

from .base import TrainingTask, print_model_width


class BaselineHeatmapTrainingTask(TrainingTask):
    name = "baseline_heatmap"
    target_format = "baseline_heatmap"
    config_fields = frozenset(
        {
            "baseline_heatmap_strict_size",
            "baseline_heatmap_loss",
            "baseline_heatmap_positive_weight",
        }
    )

    def validate_config(self, config: Any) -> None:
        supported = ("bce", "mse", "smooth_l1")
        if config.baseline_heatmap_loss not in supported:
            raise ValueError(f"baseline_heatmap_loss must be one of {supported}")

    def num_outputs(self, alphabet: str) -> int:
        return 2

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> torch.Tensor:
        return baseline_heatmap_loss(
            logits,
            targets,
            strict_size=config.baseline_heatmap_strict_size,
            loss=config.baseline_heatmap_loss,
            positive_weight=config.baseline_heatmap_positive_weight,
        )

    def validate_model(
        self,
        model: torch.nn.Module,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> None:
        output_width = print_model_width(model, dataset_config)
        print(
            "Baseline heatmap target shape: "
            f"2x{dataset_config.image_height}x{dataset_config.image_width}"
        )
        if (
            config.baseline_heatmap_strict_size
            and output_width != dataset_config.image_width
        ):
            raise ValueError(
                "baseline_heatmap_strict_size requires model output width to match image width, "
                f"but architecture={config.architecture!r} gives T={output_width} while "
                f"targets have width {dataset_config.image_width}. Use a width-preserving "
                "2D architecture such as baseline_detector_fcn, or set "
                "baseline_heatmap_strict_size: false."
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
                "baseline_heatmap requires a model output shaped (B, 2, H, W), "
                f"got {tuple(output.shape)} from architecture={config.architecture!r}."
            )
        if config.baseline_heatmap_strict_size and output.shape[-2:] != (
            dataset_config.image_height,
            dataset_config.image_width,
        ):
            raise ValueError(
                "baseline_heatmap_strict_size requires model output HxW to match target HxW, "
                f"got output={tuple(output.shape[-2:])}, "
                f"target={(dataset_config.image_height, dataset_config.image_width)}."
            )

    def summary_lines(
        self,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> list[str]:
        return [
            "Batch targets: two-channel top/bottom baseline heatmaps from chunks",
            f"Baseline heatmap loss: {config.baseline_heatmap_loss} "
            f"positive_weight={config.baseline_heatmap_positive_weight:g} "
            f"strict_size={config.baseline_heatmap_strict_size}",
        ]
