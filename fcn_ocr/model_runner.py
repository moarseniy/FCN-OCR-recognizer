from __future__ import annotations

from pathlib import Path

import torch

from .baseline_processing import NeuralBaselineMixin
from .checkpoint import load_fcn_checkpoint
from .preprocessing import ImagePreprocessingMixin


class FCNModelRunner:
    """Loads one FCN checkpoint and executes its tensor forward pass."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_task: str,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        loaded = load_fcn_checkpoint(self.checkpoint_path, self.device)
        if loaded.task != expected_task:
            raise ValueError(
                f"Expected a {expected_task!r} checkpoint, got task={loaded.task!r}"
            )

        self.checkpoint = loaded.payload
        self.training_config = loaded.training_config
        self.task = loaded.task
        self.alphabet = loaded.alphabet
        self.architecture = loaded.architecture
        self.architecture_params = loaded.architecture_params
        self.in_channels = loaded.in_channels
        self.num_classes = loaded.num_classes
        self.model = loaded.model

    @torch.no_grad()
    def logits_from_tensor(
        self,
        image_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return logits, tuple(image_tensor.shape)

    def output_width_for_input_width(self, width: int) -> int:
        if hasattr(self.model, "output_width_for_input_width"):
            return int(self.model.output_width_for_input_width(width))

        output_width = int(width)
        for module in self.model.modules():
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


class PreprocessedFCNRunner(
    FCNModelRunner,
    ImagePreprocessingMixin,
    NeuralBaselineMixin,
):
    """Adds line-image preprocessing to an FCN model runner."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_task: str,
        device: str | None = None,
        scale_x: float = 0.0,
        y_pad: float = 0.0,
        x_pad: float = 0.0,
        baseline_crop: bool = False,
        baseline_deskew: bool = True,
        baseline_max_angle: float = 12.0,
        baseline_line_pad: float = 0.08,
        baseline_line_pad_px: float = 0.0,
        baseline_detector_checkpoint: str | Path | None = None,
        baseline_detector_threshold: float = 0.35,
    ) -> None:
        if scale_x <= -0.95:
            raise ValueError("scale_x must be > -0.95")
        if y_pad <= -0.95:
            raise ValueError("y_pad must be > -0.95")
        if x_pad < 0.0:
            raise ValueError("x_pad must be >= 0")
        if baseline_line_pad < 0.0:
            raise ValueError("baseline_line_pad must be >= 0")
        if baseline_line_pad_px < 0.0:
            raise ValueError("baseline_line_pad_px must be >= 0")
        if baseline_max_angle <= 0.0:
            raise ValueError("baseline_max_angle must be > 0")
        if not 0.0 < baseline_detector_threshold < 1.0:
            raise ValueError("baseline_detector_threshold must be between 0 and 1")

        super().__init__(
            checkpoint_path,
            expected_task=expected_task,
            device=device,
        )
        self.scale_x = float(scale_x)
        self.y_pad = float(y_pad)
        self.x_pad = float(x_pad)
        self.baseline_crop = bool(baseline_crop)
        self.baseline_deskew = bool(baseline_deskew)
        self.baseline_max_angle = float(baseline_max_angle)
        self.baseline_line_pad = float(baseline_line_pad)
        self.baseline_line_pad_px = float(baseline_line_pad_px)
        self.baseline_detector_checkpoint = (
            Path(baseline_detector_checkpoint) if baseline_detector_checkpoint else None
        )
        self.baseline_detector_threshold = float(baseline_detector_threshold)
        self.baseline_detector_model: torch.nn.Module | None = None
        self.baseline_detector_in_channels = 1
        self.baseline_detector_image_height = 0
        self.baseline_detector_architecture = ""

        if self.baseline_crop and self.baseline_detector_checkpoint is None:
            raise ValueError("baseline_crop requires baseline_detector_checkpoint")

        self.image_height = int(self.training_config["image_height"])
        self.preprocess_fill = int(self.training_config["background"])
        if self.baseline_detector_checkpoint is not None:
            self._load_baseline_detector()
