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
        gap_threshold: float | None = None,
        min_gap_width: int | None = None,
        merge_gap_width: int | None = None,
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

        checkpoint_config = self.checkpoint.get("config", {})
        self.gap_threshold = self._resolve_gap_threshold(gap_threshold, checkpoint_config)
        self.min_gap_width = self._resolve_non_negative_int(
            min_gap_width,
            checkpoint_config,
            "segmentator_min_gap_width",
            default=1,
            min_value=1,
        )
        self.merge_gap_width = self._resolve_non_negative_int(
            merge_gap_width,
            checkpoint_config,
            "segmentator_merge_gap_width",
            default=0,
            min_value=0,
        )

        if verbose:
            self.print_summary()

    @staticmethod
    def _resolve_gap_threshold(value: float | None, config: dict) -> float:
        resolved = float(config.get("segmentator_gap_threshold", 0.5) if value is None else value)
        if not 0.0 < resolved < 1.0:
            raise ValueError("segmentator gap_threshold must be between 0 and 1")
        return resolved

    @staticmethod
    def _resolve_non_negative_int(
        value: int | None,
        config: dict,
        key: str,
        default: int,
        min_value: int,
    ) -> int:
        resolved = int(config.get(key, default) if value is None else value)
        if resolved < min_value:
            raise ValueError(f"{key} must be >= {min_value}")
        return resolved

    def print_summary(self) -> None:
        epoch = self.checkpoint.get("epoch", "?")
        loss = self.checkpoint.get("loss")
        loss_text = f", loss: {loss:.8f}" if isinstance(loss, float) else ""
        print(f"Segmentator loaded from epoch {epoch}{loss_text}")
        print(f"Segmentator checkpoint: {self.checkpoint_path}")
        print(f"Segmentator device: {self.device}")
        print(f"Segmentator input height: {self.image_height}")
        print(f"Segmentator classes: non-gap=0, gap=1")
        print(
            "Segmentator params: "
            f"gap_threshold={self.gap_threshold:.3f}, "
            f"min_gap_width={self.min_gap_width}, "
            f"merge_gap_width={self.merge_gap_width}"
        )

    def analyze_segmentation_logits(
        self,
        logits: torch.Tensor,
        input_shape: tuple[int, ...],
    ) -> VerticalSegmentationResult:
        if logits.size(1) != 2:
            raise ValueError(f"Expected binary segmentation logits with 2 classes, got {tuple(logits.shape)}")

        probs = torch.softmax(logits, dim=1)
        gap_probs = probs[:, 1, :]

        gap_probabilities = [float(value) for value in gap_probs[0].detach().cpu().tolist()]
        raw_indices = [1 if value >= self.gap_threshold else 0 for value in gap_probabilities]
        raw_indices = self._postprocess_labels(raw_indices)
        raw_confidences = [
            float(gap_probability if label == 1 else 1.0 - gap_probability)
            for label, gap_probability in zip(raw_indices, gap_probabilities)
        ]
        runs = self._make_runs(raw_indices, raw_confidences, gap_probabilities)

        return VerticalSegmentationResult(
            raw_indices=raw_indices,
            raw_confidences=raw_confidences,
            gap_probabilities=gap_probabilities,
            runs=runs,
            gap_threshold=self.gap_threshold,
            min_gap_width=self.min_gap_width,
            merge_gap_width=self.merge_gap_width,
            input_shape=input_shape,
            logits_shape=tuple(logits.shape),
        )

    def _postprocess_labels(self, labels: list[int]) -> list[int]:
        if not labels:
            return labels

        output = list(labels)
        if self.merge_gap_width > 0:
            output = self._merge_close_gap_runs(output, self.merge_gap_width)
        if self.min_gap_width > 1:
            output = self._drop_short_gap_runs(output, self.min_gap_width)
        return output

    @classmethod
    def _merge_close_gap_runs(cls, labels: list[int], max_distance: int) -> list[int]:
        output = list(labels)
        runs = cls._label_runs(output)
        for prev_run, current_run, next_run in zip(runs, runs[1:], runs[2:]):
            if prev_run[0] == 1 and current_run[0] == 0 and next_run[0] == 1:
                distance = current_run[2] - current_run[1] + 1
                if distance <= max_distance:
                    for index in range(current_run[1], current_run[2] + 1):
                        output[index] = 1
        return output

    @classmethod
    def _drop_short_gap_runs(cls, labels: list[int], min_width: int) -> list[int]:
        output = list(labels)
        for label, start, end in cls._label_runs(output):
            if label != 1:
                continue
            width = end - start + 1
            if width < min_width:
                for index in range(start, end + 1):
                    output[index] = 0
        return output

    @staticmethod
    def _label_runs(labels: list[int]) -> list[tuple[int, int, int]]:
        if not labels:
            return []

        runs: list[tuple[int, int, int]] = []
        start = 0
        label = labels[0]
        for index, value in enumerate(labels[1:], start=1):
            if value == label:
                continue
            runs.append((label, start, index - 1))
            start = index
            label = value
        runs.append((label, start, len(labels) - 1))
        return runs

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
