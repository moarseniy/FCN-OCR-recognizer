from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fcn_tasks import BASELINE_DETECTION_TASK

from .baseline_processing import NeuralBaselineMixin
from .model_runner import FCNModelRunner
from .results import PreprocessDebug


class BaselineDetector(FCNModelRunner, NeuralBaselineMixin):
    """Standalone neural top/bottom baseline detector."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        threshold: float = 0.35,
        deskew: bool = True,
        max_angle: float = 12.0,
        line_pad: float = 0.08,
        line_pad_px: float = 0.0,
        background: int = 255,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if max_angle <= 0.0:
            raise ValueError("max_angle must be > 0")
        if line_pad < 0.0 or line_pad_px < 0.0:
            raise ValueError("line padding must be >= 0")

        super().__init__(
            checkpoint_path,
            expected_task=BASELINE_DETECTION_TASK,
            device=device,
        )
        self.baseline_detector_checkpoint = self.checkpoint_path
        self.baseline_detector_threshold = float(threshold)
        self.baseline_crop = True
        self.baseline_deskew = bool(deskew)
        self.baseline_line_pad = float(line_pad)
        self.baseline_line_pad_px = float(line_pad_px)
        self.baseline_max_angle = float(max_angle)
        self.preprocess_fill = int(background)
        self.baseline_detector_model = self.model
        self.baseline_detector_in_channels = self.in_channels
        self.baseline_detector_image_height = int(self.training_config["image_height"])
        self.baseline_detector_architecture = self.architecture

    def print_summary(self) -> None:
        print(f"Baseline detector checkpoint: {self.baseline_detector_checkpoint}")
        print(f"Baseline detector device:     {self.device}")
        print(f"Baseline detector threshold:  {self.baseline_detector_threshold:.4f}")

    def detect(self, image: Image.Image) -> dict[str, Any]:
        mode = "RGB" if self.baseline_detector_in_channels == 3 else "L"
        source = image if image.mode == mode else image.convert(mode)
        return self._detect_baseline_neural(source)

    def heatmaps(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, float, float]:
        mode = "RGB" if self.baseline_detector_in_channels == 3 else "L"
        source = image if image.mode == mode else image.convert(mode)
        heatmaps, _, _, scale_x, scale_y = self._baseline_detector_heatmaps(source)
        return heatmaps, scale_x, scale_y

    def detect_from_heatmaps(
        self,
        image: Image.Image,
        heatmap_data: tuple[np.ndarray, float, float],
    ) -> dict[str, Any]:
        mode = "RGB" if self.baseline_detector_in_channels == 3 else "L"
        source = image if image.mode == mode else image.convert(mode)
        heatmaps, scale_x, scale_y = heatmap_data
        cleaned_mask = self._baseline_score_mask(heatmaps, source.size)
        foreground_pixels = int(np.count_nonzero(cleaned_mask))
        return self._detect_baseline_from_heatmaps(
            source,
            heatmaps,
            cleaned_mask,
            foreground_pixels,
            scale_x,
            scale_y,
        )

    def prepare_baseline_image(
        self,
        image: Image.Image,
        collect_debug: bool = False,
    ) -> tuple[Image.Image, PreprocessDebug]:
        source = image.convert("RGB")
        prepared, debug = self._apply_baseline_crop(
            source,
            collect_debug=collect_debug,
        )
        return prepared, PreprocessDebug(
            metadata={"baseline_crop": True, **debug.metadata},
            images=debug.images,
        )
