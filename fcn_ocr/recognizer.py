from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
import torch

from .baseline_processing import NeuralBaselineMixin
from .checkpoint import load_fcn_checkpoint
from .cut_processing import CutProcessingMixin
from .decoding import FCNOCRDecodingMixin
from .preprocessing import ImagePreprocessingMixin
from .results import RecognitionResult


class TextRecognizer(
    ImagePreprocessingMixin,
    NeuralBaselineMixin,
    CutProcessingMixin,
    FCNOCRDecodingMixin,
):
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        verbose: bool = False,
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
    ):
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

        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        loaded = load_fcn_checkpoint(self.checkpoint_path, self.device)
        self.checkpoint = loaded.payload
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

        self.alphabet = loaded.alphabet
        self.idx_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

        checkpoint_config = loaded.training_config
        self.architecture = loaded.architecture
        self.architecture_params = loaded.architecture_params
        self.in_channels = loaded.in_channels
        self.num_classes = loaded.num_classes
        self.loss_mode = loaded.loss_mode
        self.target_format = loaded.target_format
        self.space_char = str(checkpoint_config["space_char"])
        self.space_idx = (
            self.alphabet.index(self.space_char)
            if self.space_char in self.alphabet
            else None
        )
        self.image_height = int(checkpoint_config["image_height"])
        self.preprocess_fill = int(checkpoint_config["background"])
        if self.loss_mode == "fcn_ocr":
            self.ocr_crop_left = int(checkpoint_config["ocr_crop_left"])
            self.ocr_crop_right = int(checkpoint_config["ocr_crop_right"])
        else:
            self.ocr_crop_left = 0
            self.ocr_crop_right = 0

        self.model = loaded.model

        if self.baseline_detector_checkpoint is not None:
            self._load_baseline_detector()

        if verbose:
            self.print_summary()

    def print_summary(self) -> None:
        print(f"Using device: {self.device}")
        print(
            f"Model loaded from epoch {self.checkpoint['epoch']}, "
            f"loss: {float(self.checkpoint['loss']):.8f}"
        )
        print(f"Architecture: {self.architecture}")
        if self.architecture_params:
            print(f"Architecture params: {self.architecture_params}")
        print(f"Alphabet size: {len(self.alphabet)}")
        print(f"Loss mode: {self.loss_mode}")
        if self.loss_mode == "fcn_ocr":
            print(f"FCN OCR crop: [{self.ocr_crop_left}, -{self.ocr_crop_right}]")
        print(f"Preprocess scale_x: {self.scale_x:+.4f}")
        print(f"Preprocess y_pad:   {self.y_pad:+.4f}")
        print(f"Preprocess x_pad:   {self.x_pad:.4f}")
        print(f"Model-internal baseline crop: {self.baseline_crop}")
        if self.baseline_crop:
            print(
                f"  deskew={self.baseline_deskew}, max_angle={self.baseline_max_angle:.2f}, "
                f"line_pad={self.baseline_line_pad:.3f}, "
                f"line_pad_px={self.baseline_line_pad_px:.1f}"
            )
            if self.baseline_detector_model is not None:
                print(
                    "  neural_detector="
                    f"{self.baseline_detector_checkpoint} "
                    f"threshold={self.baseline_detector_threshold:.3f} "
                    f"architecture={self.baseline_detector_architecture}"
                )

    @torch.no_grad()
    def logits_from_tensor(
        self, image_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return logits, tuple(image_tensor.shape)

    @torch.no_grad()
    def recognize_tensor_debug_with_logits(
        self,
        image_tensor: torch.Tensor,
        top_k: int = 8,
    ) -> tuple[RecognitionResult, torch.Tensor]:
        logits, input_shape = self.logits_from_tensor(image_tensor)
        return self.analyze_logits(logits, input_shape=input_shape, top_k=top_k), logits

    @torch.no_grad()
    def recognize_tensor_debug(
        self, image_tensor: torch.Tensor, top_k: int = 8
    ) -> RecognitionResult:
        logits, input_shape = self.logits_from_tensor(image_tensor)
        return self.analyze_logits(logits, input_shape=input_shape, top_k=top_k)

    @torch.no_grad()
    def recognize_tensor(self, image_tensor: torch.Tensor) -> tuple[str, list[int]]:
        logits, _ = self.logits_from_tensor(image_tensor)
        return self.decode_predictions(logits)

    def recognize(self, image_path: str | Path) -> tuple[str, list[int]]:
        return self.recognize_tensor(self.preprocess_image(image_path))

    def recognize_image_debug(
        self, image_path: str | Path, top_k: int = 8
    ) -> RecognitionResult:
        return self.recognize_tensor_debug(
            self.preprocess_image(image_path), top_k=top_k
        )

    def recognize_paths(
        self, image_paths: Iterable[str | Path], top_k: int = 8
    ) -> list[tuple[Path, RecognitionResult]]:
        results: list[tuple[Path, RecognitionResult]] = []
        for image_path in image_paths:
            path = Path(image_path)
            results.append((path, self.recognize_image_debug(path, top_k=top_k)))
        return results

    @torch.no_grad()
    def recognize_paths_text(
        self,
        image_paths: Iterable[str | Path],
        batch_size: int = 32,
    ) -> list[tuple[Path, str]]:
        paths = [Path(image_path) for image_path in image_paths]
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        results: list[tuple[Path, str]] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors: list[torch.Tensor] = []
            output_lengths: list[int] = []
            max_width = 0

            for path in batch_paths:
                with Image.open(path) as image:
                    tensor = self._preprocess_pil_3d(image)
                tensors.append(tensor)
                max_width = max(max_width, tensor.size(2))
                output_lengths.append(self.output_width_for_input_width(tensor.size(2)))

            if not tensors:
                continue

            batch = torch.ones(
                (len(tensors), self.in_channels, self.image_height, max_width),
                dtype=tensors[0].dtype,
                device=self.device,
            )
            for batch_index, tensor in enumerate(tensors):
                batch[batch_index, :, :, : tensor.size(2)] = tensor

            logits = self.model(batch)
            pred_ids = logits.argmax(dim=1)
            decoded = self.decode_pred_ids_batch(pred_ids, output_lengths)
            results.extend(
                (path, text) for path, (text, _) in zip(batch_paths, decoded)
            )

        return results
