from __future__ import annotations

from typing import Any

import torch

from fcn_synth_generator.dataset import SingleLineDatasetConfig
from loss import fcn_ocr_loss

from .base import TrainingTask, print_model_width


class OcrTrainingTask(TrainingTask):
    name = "fcn_ocr"
    target_format = "fcn_ocr"
    config_fields = frozenset(
        {
            "ocr_crop_left",
            "ocr_crop_right",
            "ocr_strict_width",
            "ocr_target_min_majority",
            "ocr_space_weight",
        }
    )

    def num_outputs(self, alphabet: str) -> int:
        return len(alphabet)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> torch.Tensor:
        space_index = (
            dataset_config.alphabet.index(dataset_config.space_char)
            if dataset_config.space_char in dataset_config.alphabet
            else None
        )
        return fcn_ocr_loss(
            logits,
            targets,
            crop_left=config.ocr_crop_left,
            crop_right=config.ocr_crop_right,
            strict_width=config.ocr_strict_width,
            label_min_majority=config.ocr_target_min_majority,
            space_index=space_index,
            space_weight=config.ocr_space_weight,
        )

    def validate_model(
        self,
        model: torch.nn.Module,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> None:
        output_width = print_model_width(model, dataset_config)
        target_width = (
            dataset_config.image_width - config.ocr_crop_left - config.ocr_crop_right
        )
        if target_width <= 0:
            raise ValueError(
                "FCN OCR target crop is empty: "
                f"image_width={dataset_config.image_width}, "
                f"ocr_crop_left={config.ocr_crop_left}, "
                f"ocr_crop_right={config.ocr_crop_right}"
            )
        print(f"FCN OCR target width: {target_width}")
        if config.ocr_strict_width and output_width != target_width:
            raise ValueError(
                "ocr_strict_width requires model output width to match target width, "
                f"but architecture={config.architecture!r} gives T={output_width} while "
                f"targets have width {target_width}. Set ocr_strict_width: false to use "
                "majority-bin target alignment."
            )

    def summary_lines(
        self,
        config: Any,
        dataset_config: SingleLineDatasetConfig,
    ) -> list[str]:
        space_index = (
            dataset_config.alphabet.index(dataset_config.space_char)
            if dataset_config.space_char in dataset_config.alphabet
            else None
        )
        return [
            f"FCN OCR target crop: [{config.ocr_crop_left}, -{config.ocr_crop_right}]",
            "FCN OCR target alignment: majority_bins "
            f"min_majority={config.ocr_target_min_majority:g}",
            f"FCN OCR space weight: {config.ocr_space_weight:g} "
            f"(space index: {space_index})",
            "Batch targets: one OCR class per input X-position",
        ]
