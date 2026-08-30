from __future__ import annotations

from pathlib import Path

import torch

from fcn_tasks import VERTICAL_SEGMENTATION_TASK

from .model_runner import PreprocessedFCNRunner
from .results import SegmentationRun, VerticalSegmentationResult


class VerticalSegmenter(PreprocessedFCNRunner):
    """FCN model that predicts vertical character-boundary scores."""

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
        cut_threshold: float | None = None,
        cut_min_width: int | None = None,
        cut_max_width: int | None = None,
        cut_smooth_radius: int | None = None,
    ):
        super().__init__(
            checkpoint_path,
            expected_task=VERTICAL_SEGMENTATION_TASK,
            device=device,
            scale_x=scale_x,
            y_pad=y_pad,
            x_pad=x_pad,
            baseline_crop=baseline_crop,
            baseline_deskew=baseline_deskew,
            baseline_max_angle=baseline_max_angle,
            baseline_line_pad=baseline_line_pad,
            baseline_line_pad_px=baseline_line_pad_px,
            baseline_detector_checkpoint=baseline_detector_checkpoint,
            baseline_detector_threshold=baseline_detector_threshold,
        )

        self.cut_threshold = self._resolve_cut_threshold(cut_threshold)
        self.cut_min_width = self._resolve_non_negative_int(
            cut_min_width,
            "cut_min_width",
            default=1,
            min_value=1,
        )
        self.cut_max_width = self._resolve_non_negative_int(
            cut_max_width,
            "cut_max_width",
            default=0,
            min_value=0,
        )
        self.cut_smooth_radius = self._resolve_non_negative_int(
            cut_smooth_radius,
            "cut_smooth_radius",
            default=0,
            min_value=0,
        )

        if verbose:
            self.print_summary()

    @staticmethod
    def _resolve_cut_threshold(value: float | None) -> float:
        resolved = 0.5 if value is None else float(value)
        if not 0.0 < resolved < 1.0:
            raise ValueError("vertical_segmentation cut threshold must be between 0 and 1")
        return resolved

    @staticmethod
    def _resolve_non_negative_int(
        value: int | None,
        key: str,
        default: int,
        min_value: int,
    ) -> int:
        resolved = default if value is None else int(value)
        if resolved < min_value:
            raise ValueError(f"{key} must be >= {min_value}")
        return resolved

    def print_summary(self) -> None:
        print(
            f"Vertical segmentation loaded from epoch {self.checkpoint['epoch']}, "
            f"loss: {float(self.checkpoint['loss']):.8f}"
        )
        print(f"Vertical segmentation checkpoint: {self.checkpoint_path}")
        print(f"Vertical segmentation device: {self.device}")
        print(f"Vertical segmentation input height: {self.image_height}")
        print(f"Vertical segmentation preprocess: scale_x={self.scale_x:+.4f}, y_pad={self.y_pad:+.4f}, x_pad={self.x_pad:.4f}")
        print(f"Vertical segmentation task: {self.task}")
        print("Vertical segmentation output: one character-boundary score per column")
        print(
            "Vertical segmentation params: "
            f"cut_threshold={self.cut_threshold:.3f}, "
            f"cut_min_width={self.cut_min_width}, "
            f"cut_max_width={self.cut_max_width}, "
            f"smooth_radius={self.cut_smooth_radius}"
        )

    def analyze_segmentation_logits(
        self,
        logits: torch.Tensor,
        input_shape: tuple[int, ...],
    ) -> VerticalSegmentationResult:
        if logits.size(1) != 1:
            raise ValueError(f"Cut vertical_segmentation expects logits with one channel, got {tuple(logits.shape)}")
        return self._analyze_vertical_segmentation_logits(logits, input_shape)

    def _analyze_vertical_segmentation_logits(
        self,
        logits: torch.Tensor,
        input_shape: tuple[int, ...],
    ) -> VerticalSegmentationResult:
        cut_scores_tensor = torch.sigmoid(logits[:, 0, :])
        cut_scores = [float(value) for value in cut_scores_tensor[0].detach().cpu().tolist()]
        postprocess_scores = self._smooth_scores(cut_scores, self.cut_smooth_radius)
        cut_positions = self._select_cut_peaks(
            postprocess_scores,
            threshold=self.cut_threshold,
            min_distance=self.cut_min_width,
        )
        cut_positions = self._apply_cut_width_constraints(
            cut_positions,
            postprocess_scores,
            min_width=self.cut_min_width,
            max_width=self.cut_max_width,
        )
        cut_set = set(cut_positions)
        raw_indices = [1 if index in cut_set else 0 for index in range(len(cut_scores))]
        raw_confidences = [
            float(score if label == 1 else 1.0 - score)
            for label, score in zip(raw_indices, cut_scores)
        ]
        runs = self._make_runs(raw_indices, raw_confidences, cut_scores)

        return VerticalSegmentationResult(
            raw_indices=raw_indices,
            raw_confidences=raw_confidences,
            cut_scores=cut_scores,
            runs=runs,
            cut_threshold=self.cut_threshold,
            input_shape=input_shape,
            logits_shape=tuple(logits.shape),
            cut_positions=cut_positions,
            cut_min_width=self.cut_min_width,
            cut_max_width=self.cut_max_width,
            cut_smooth_radius=self.cut_smooth_radius,
        )

    @staticmethod
    def _smooth_scores(scores: list[float], radius: int) -> list[float]:
        if radius <= 0 or len(scores) <= 2:
            return list(scores)

        smoothed: list[float] = []
        for index in range(len(scores)):
            total = 0.0
            weight_total = 0.0
            for offset in range(-radius, radius + 1):
                position = index + offset
                if position < 0 or position >= len(scores):
                    continue
                weight = float(radius + 1 - abs(offset))
                total += scores[position] * weight
                weight_total += weight
            smoothed.append(total / max(1.0, weight_total))
        return smoothed

    @classmethod
    def _apply_cut_width_constraints(
        cls,
        cuts: list[int],
        scores: list[float],
        min_width: int,
        max_width: int,
    ) -> list[int]:
        if not scores:
            return []

        output = cls._enforce_min_cut_width(sorted(set(cuts)), scores, min_width)
        if max_width > 0:
            output = cls._insert_missing_cuts_by_width(
                output,
                scores,
                min_width,
                max_width,
            )
            output = cls._enforce_min_cut_width(output, scores, min_width)
        return output

    @staticmethod
    def _enforce_min_cut_width(cuts: list[int], scores: list[float], min_width: int) -> list[int]:
        output = sorted(set(cuts))
        if min_width <= 1:
            return output

        changed = True
        while changed and len(output) > 1:
            changed = False
            for index in range(1, len(output)):
                if output[index] - output[index - 1] >= min_width:
                    continue
                left = output[index - 1]
                right = output[index]
                remove_index = index - 1 if scores[left] <= scores[right] else index
                output.pop(remove_index)
                changed = True
                break
        return output

    @classmethod
    def _insert_missing_cuts_by_width(
        cls,
        cuts: list[int],
        scores: list[float],
        min_width: int,
        max_width: int,
    ) -> list[int]:
        output = sorted(set(cuts))
        if len(output) < 2:
            return output

        while True:
            widest_interval: tuple[int, int] | None = None
            widest_distance = 0
            for left, right in zip(output, output[1:]):
                distance = right - left
                if distance > max_width and distance > widest_distance:
                    widest_interval = (left, right)
                    widest_distance = distance

            if widest_interval is None:
                return output

            left, right = widest_interval
            lower = left + min_width
            upper = right - min_width
            if lower > upper:
                return output

            interval_positions = [
                position for position in range(lower, upper + 1)
                if position not in output
            ]
            if not interval_positions:
                return output

            center = (left + right) * 0.5
            chosen = max(
                interval_positions,
                key=lambda position: (scores[position], -abs(position - center)),
            )
            output.append(int(chosen))
            output = cls._enforce_min_cut_width(sorted(set(output)), scores, min_width)

    @staticmethod
    def _select_cut_peaks(
        scores: list[float],
        threshold: float,
        min_distance: int,
    ) -> list[int]:
        if not scores:
            return []

        candidates: list[tuple[float, int]] = []
        last_index = len(scores) - 1
        for index, score in enumerate(scores):
            if score < threshold:
                continue
            left = scores[index - 1] if index > 0 else float("-inf")
            right = scores[index + 1] if index < last_index else float("-inf")
            if score >= left and score >= right:
                candidates.append((float(score), index))

        selected: list[int] = []
        for _, index in sorted(candidates, reverse=True):
            if all(abs(index - previous) >= min_distance for previous in selected):
                selected.append(index)
        return sorted(selected)

    @staticmethod
    def _make_runs(
        raw_indices: list[int],
        raw_confidences: list[float],
        cut_scores: list[float],
    ) -> list[SegmentationRun]:
        if not raw_indices:
            return []

        runs: list[SegmentationRun] = []
        start = 0
        label = raw_indices[0]
        for timestep, value in enumerate(raw_indices[1:], start=1):
            if value == label:
                continue
            runs.append(VerticalSegmenter._run_from_slice(label, start, timestep - 1, raw_confidences, cut_scores))
            start = timestep
            label = value
        runs.append(VerticalSegmenter._run_from_slice(label, start, len(raw_indices) - 1, raw_confidences, cut_scores))
        return runs

    @staticmethod
    def _run_from_slice(
        label: int,
        start: int,
        end: int,
        raw_confidences: list[float],
        cut_scores: list[float],
    ) -> SegmentationRun:
        count = end - start + 1
        avg_confidence = sum(raw_confidences[start : end + 1]) / count
        avg_score = sum(cut_scores[start : end + 1]) / count
        return SegmentationRun(
            label=int(label),
            kind="cut" if label == 1 else "background",
            start=int(start),
            end=int(end),
            confidence=float(avg_confidence),
            score=float(avg_score),
        )

    @torch.no_grad()
    def segment_tensor_debug(self, image_tensor: torch.Tensor) -> VerticalSegmentationResult:
        logits, input_shape = self.logits_from_tensor(image_tensor)
        return self.analyze_segmentation_logits(logits, input_shape=input_shape)
