from __future__ import annotations

from pathlib import Path

import torch

from .recognizer import TextRecognizer
from .results import SegmentationRun, VerticalSegmentationResult


class VerticalSegmentator(TextRecognizer):
    """Binary FCN segmentator for vertical gaps between characters."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        verbose: bool = False,
        scale_x: float = 0.0,
        y_pad: float = 0.0,
        baseline_crop: bool = False,
        baseline_top_pad: float = 0.12,
        baseline_bottom_pad: float = 0.18,
        baseline_deskew: bool = True,
        baseline_max_angle: float = 12.0,
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            verbose=False,
            scale_x=scale_x,
            y_pad=y_pad,
            baseline_crop=baseline_crop,
            baseline_top_pad=baseline_top_pad,
            baseline_bottom_pad=baseline_bottom_pad,
            baseline_deskew=baseline_deskew,
            baseline_max_angle=baseline_max_angle,
        )
        self.legacy_target_mode = str(
            self.checkpoint.get("model_config", {}).get(
                "legacy_target_mode",
                self.checkpoint.get("config", {}).get("legacy_target_mode", ""),
            )
        ).lower()
        if self.num_classes != 2:
            raise ValueError(
                f"VerticalSegmentator expects a binary checkpoint with num_classes=2, got {self.num_classes}"
            )
        if self.legacy_target_mode and self.legacy_target_mode != "binary_gaps":
            raise ValueError(
                "VerticalSegmentator expects legacy_target_mode=binary_gaps, "
                f"got {self.legacy_target_mode!r}"
            )

        if verbose:
            self.print_summary()

    def print_summary(self) -> None:
        epoch = self.checkpoint.get("epoch", "?")
        loss = self.checkpoint.get("loss")
        loss_text = f", loss: {loss:.8f}" if isinstance(loss, float) else ""
        print(f"Segmentator loaded from epoch {epoch}{loss_text}")
        print(f"Segmentator checkpoint: {self.checkpoint_path}")
        print(f"Segmentator device: {self.device}")
        print(f"Segmentator input height: {self.image_height}")
        print(f"Segmentator classes: non-gap=0, gap=1")

    def analyze_segmentation_logits(
        self,
        logits: torch.Tensor,
        input_shape: tuple[int, ...],
    ) -> VerticalSegmentationResult:
        if logits.size(1) != 2:
            raise ValueError(f"Expected binary segmentation logits with 2 classes, got {tuple(logits.shape)}")

        probs = torch.softmax(logits, dim=1)
        confidences, pred_ids = probs.max(dim=1)
        gap_probs = probs[:, 1, :]

        raw_indices = [int(value) for value in pred_ids[0].detach().cpu().tolist()]
        raw_confidences = [float(value) for value in confidences[0].detach().cpu().tolist()]
        gap_probabilities = [float(value) for value in gap_probs[0].detach().cpu().tolist()]
        runs = self._make_runs(raw_indices, raw_confidences, gap_probabilities)

        return VerticalSegmentationResult(
            raw_indices=raw_indices,
            raw_confidences=raw_confidences,
            gap_probabilities=gap_probabilities,
            runs=runs,
            input_shape=input_shape,
            logits_shape=tuple(logits.shape),
        )

    @staticmethod
    def _make_runs(
        raw_indices: list[int],
        raw_confidences: list[float],
        gap_probabilities: list[float],
    ) -> list[SegmentationRun]:
        if not raw_indices:
            return []

        runs: list[SegmentationRun] = []
        start = 0
        label = raw_indices[0]
        for timestep, value in enumerate(raw_indices[1:], start=1):
            if value == label:
                continue
            runs.append(VerticalSegmentator._run_from_slice(label, start, timestep - 1, raw_confidences, gap_probabilities))
            start = timestep
            label = value
        runs.append(VerticalSegmentator._run_from_slice(label, start, len(raw_indices) - 1, raw_confidences, gap_probabilities))
        return runs

    @staticmethod
    def _run_from_slice(
        label: int,
        start: int,
        end: int,
        raw_confidences: list[float],
        gap_probabilities: list[float],
    ) -> SegmentationRun:
        count = end - start + 1
        avg_confidence = sum(raw_confidences[start : end + 1]) / count
        avg_gap_probability = sum(gap_probabilities[start : end + 1]) / count
        return SegmentationRun(
            label=int(label),
            kind="gap" if label == 1 else "non-gap",
            start=int(start),
            end=int(end),
            confidence=float(avg_confidence),
            gap_probability=float(avg_gap_probability),
        )

    @torch.no_grad()
    def segment_tensor_debug(self, image_tensor: torch.Tensor) -> VerticalSegmentationResult:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return self.analyze_segmentation_logits(logits, input_shape=tuple(image_tensor.shape))

