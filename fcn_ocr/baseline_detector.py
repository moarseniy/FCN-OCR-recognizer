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
        rectify: str = "lines",
        curve_smooth_radius: int = 8,
        curve_min_coverage: float = 0.25,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if rectify not in {"lines", "curved"}:
            raise ValueError("rectify must be 'lines' or 'curved'")
        if curve_smooth_radius < 0:
            raise ValueError("curve_smooth_radius must be >= 0")
        if not 0.0 <= curve_min_coverage <= 1.0:
            raise ValueError("curve_min_coverage must be between 0 and 1")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.baseline_detector_checkpoint = Path(checkpoint_path)
        self.baseline_detector_threshold = float(threshold)
        self.baseline_rectify = rectify
        self.baseline_curve_smooth_radius = int(curve_smooth_radius)
        self.baseline_curve_min_coverage = float(curve_min_coverage)
        self.baseline_line_pad = 0.0
        self.baseline_line_pad_px = 0.0
        self.baseline_strict_lines = True
        self.baseline_top_pad = 0.0
        self.baseline_bottom_pad = 0.0
        self.baseline_detector_model = None
        self.baseline_detector_in_channels = 1
        self.baseline_detector_image_height = 0
        self.baseline_detector_architecture = ""
        self._load_baseline_detector()

    def print_summary(self) -> None:
        print(f"Baseline detector checkpoint: {self.baseline_detector_checkpoint}")
        print(f"Baseline detector device:     {self.device}")
        print(f"Baseline detector threshold:  {self.baseline_detector_threshold:.4f}")
        print(f"Baseline detector rectify:    {self.baseline_rectify}")
        print(f"Curve smooth radius:          {self.baseline_curve_smooth_radius}")
        print(f"Curve minimum coverage:       {self.baseline_curve_min_coverage:.4f}")

    def detect(self, image: Image.Image) -> dict[str, Any]:
        source = image.convert("RGB" if self.baseline_detector_in_channels == 3 else "L")
        if self.baseline_rectify == "curved":
            return self._detect_baseline_curves(source)
        return self._detect_baseline_neural(source)

