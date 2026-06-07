from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch

from .recognizer import TextRecognizer


class BaselineDetector(TextRecognizer):
    """Standalone neural top/bottom baseline detector."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        threshold: float = 0.35,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.baseline_detector_checkpoint = Path(checkpoint_path)
        self.baseline_detector_threshold = float(threshold)
        self.baseline_line_pad = 0.0
        self.baseline_line_pad_px = 0.0
        self.baseline_strict_lines = True
        self.baseline_max_angle = 12.0
        self.baseline_detector_model = None
        self.baseline_detector_in_channels = 1
        self.baseline_detector_image_height = 0
        self.baseline_detector_architecture = ""
        self._load_baseline_detector()

    def print_summary(self) -> None:
        print(f"Baseline detector checkpoint: {self.baseline_detector_checkpoint}")
        print(f"Baseline detector device:     {self.device}")
        print(f"Baseline detector threshold:  {self.baseline_detector_threshold:.4f}")

    def detect(self, image: Image.Image) -> dict[str, Any]:
        source = image.convert("RGB" if self.baseline_detector_in_channels == 3 else "L")
        return self._detect_baseline_neural(source)
