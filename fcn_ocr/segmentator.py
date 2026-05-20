from __future__ import annotations

from pathlib import Path

import torch

from .recognizer import TextRecognizer
from .results import SegmentationRun, VerticalSegmentationResult


class VerticalSegmentator(TextRecognizer):
    """FCN segmentator for vertical gaps or cut-point projections."""

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
        cut_postprocess: str | None = None,
        cut_min_width: int | None = None,
        cut_max_width: int | None = None,
        cut_candidate_threshold: float | None = None,
        cut_smooth_radius: int | None = None,
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
        checkpoint_config = self.checkpoint.get("config", {})
        model_config = self.checkpoint.get("model_config", {})
        self.legacy_target_mode = str(
            self.checkpoint.get("model_config", {}).get(
                "legacy_target_mode",
                checkpoint_config.get("legacy_target_mode", ""),
            )
        ).lower()
        self.target_format = str(
            model_config.get("target_format", checkpoint_config.get("target_format", self.legacy_target_mode))
        ).lower()
        if self.loss_mode == "cut_projection" or self.num_classes == 1:
            self.target_format = "cut_projection"
        elif self.target_format in {"", "none"}:
            self.target_format = "binary_gaps"

        if self.target_format == "cut_projection":
            if self.num_classes != 1:
                raise ValueError(
                    f"Cut projection segmentator expects num_classes=1, got {self.num_classes}"
                )
        elif self.num_classes != 2:
            raise ValueError(
                f"VerticalSegmentator expects a binary checkpoint with num_classes=2, got {self.num_classes}"
            )
        if self.target_format != "cut_projection" and self.legacy_target_mode and self.legacy_target_mode != "binary_gaps":
            raise ValueError(
                "VerticalSegmentator expects legacy_target_mode=binary_gaps, "
                f"got {self.legacy_target_mode!r}"
            )

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
        self.peak_min_distance = self._resolve_non_negative_int(
            min_gap_width,
            checkpoint_config,
            "segmentator_peak_min_distance",
            default=self.min_gap_width,
            min_value=1,
        )
        self.cut_postprocess = self._resolve_cut_postprocess(cut_postprocess, checkpoint_config)
        effective_cut_min_width = cut_min_width if cut_min_width is not None else min_gap_width
        self.cut_min_width = self._resolve_non_negative_int(
            effective_cut_min_width,
            checkpoint_config,
            "segmentator_cut_min_width",
            default=self.peak_min_distance,
            min_value=1,
        )
        self.cut_max_width = self._resolve_non_negative_int(
            cut_max_width,
            checkpoint_config,
            "segmentator_cut_max_width",
            default=0,
            min_value=0,
        )
        self.cut_candidate_threshold = self._resolve_cut_candidate_threshold(
            cut_candidate_threshold,
            checkpoint_config,
        )
        self.cut_smooth_radius = self._resolve_non_negative_int(
            cut_smooth_radius,
            checkpoint_config,
            "segmentator_cut_smooth_radius",
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
    def _resolve_cut_candidate_threshold(value: float | None, config: dict) -> float:
        resolved = float(config.get("segmentator_cut_candidate_threshold", 0.1) if value is None else value)
        if not 0.0 <= resolved < 1.0:
            raise ValueError("segmentator cut candidate threshold must be in [0, 1)")
        return resolved

    @staticmethod
    def _resolve_cut_postprocess(value: str | None, config: dict) -> str:
        resolved = str(config.get("segmentator_cut_postprocess", "widths") if value is None else value).lower()
        if resolved not in {"peaks", "widths"}:
            raise ValueError("segmentator cut postprocess must be 'peaks' or 'widths'")
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
        print(f"Segmentator mode: {self.target_format}")
        if self.target_format == "cut_projection":
            print("Segmentator output: cut projection, one sigmoid score per column")
            print(
                "Segmentator params: "
                f"cut_threshold={self.gap_threshold:.3f}, "
                f"peak_min_distance={self.peak_min_distance}, "
                f"postprocess={self.cut_postprocess}, "
                f"cut_min_width={self.cut_min_width}, "
                f"cut_max_width={self.cut_max_width}, "
                f"candidate_threshold={self.cut_candidate_threshold:.3f}, "
                f"smooth_radius={self.cut_smooth_radius}"
            )
        else:
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
        if logits.size(1) == 1:
            return self._analyze_cut_projection_logits(logits, input_shape)
        if logits.size(1) != 2:
            raise ValueError(f"Expected segmentation logits with 1 or 2 classes, got {tuple(logits.shape)}")

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
            mode="binary_gaps",
            cut_positions=None,
            peak_min_distance=None,
            candidate_cut_positions=None,
            cut_postprocess=None,
            cut_candidate_threshold=None,
            cut_min_width=None,
            cut_max_width=None,
            cut_smooth_radius=None,
        )

    def _analyze_cut_projection_logits(
        self,
        logits: torch.Tensor,
        input_shape: tuple[int, ...],
    ) -> VerticalSegmentationResult:
        cut_scores_tensor = torch.sigmoid(logits[:, 0, :])
        cut_scores = [float(value) for value in cut_scores_tensor[0].detach().cpu().tolist()]
        postprocess_scores = self._smooth_scores(cut_scores, self.cut_smooth_radius)
        candidate_positions = self._select_cut_peaks(
            postprocess_scores,
            threshold=self.cut_candidate_threshold,
            min_distance=self.peak_min_distance,
        )
        cut_positions = self._select_cut_peaks(
            postprocess_scores,
            threshold=self.gap_threshold,
            min_distance=self.peak_min_distance,
        )
        if self.cut_postprocess == "widths":
            cut_positions = self._postprocess_cut_widths(
                cut_positions,
                candidate_positions,
                postprocess_scores,
                min_width=self.cut_min_width,
                max_width=self.cut_max_width,
                candidate_threshold=self.cut_candidate_threshold,
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
            gap_probabilities=cut_scores,
            runs=runs,
            gap_threshold=self.gap_threshold,
            min_gap_width=self.min_gap_width,
            merge_gap_width=self.merge_gap_width,
            input_shape=input_shape,
            logits_shape=tuple(logits.shape),
            mode="cut_projection",
            cut_positions=cut_positions,
            peak_min_distance=self.peak_min_distance,
            candidate_cut_positions=candidate_positions,
            cut_postprocess=self.cut_postprocess,
            cut_candidate_threshold=self.cut_candidate_threshold,
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
    def _postprocess_cut_widths(
        cls,
        cuts: list[int],
        candidates: list[int],
        scores: list[float],
        min_width: int,
        max_width: int,
        candidate_threshold: float,
    ) -> list[int]:
        if not scores:
            return []

        output = cls._enforce_min_cut_width(sorted(set(cuts)), scores, min_width)
        if max_width > 0:
            output = cls._insert_missing_cuts_by_width(
                output,
                candidates,
                scores,
                min_width,
                max_width,
                candidate_threshold,
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
        candidates: list[int],
        scores: list[float],
        min_width: int,
        max_width: int,
        candidate_threshold: float,
    ) -> list[int]:
        output = sorted(set(cuts))
        candidate_set = set(candidates)
        width = len(scores)

        while True:
            boundaries = [0, *output, width - 1]
            widest_interval: tuple[int, int] | None = None
            widest_distance = 0
            for left, right in zip(boundaries, boundaries[1:]):
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

            interval_candidates = [
                candidate for candidate in candidate_set
                if lower <= candidate <= upper and candidate not in output
            ]
            if not interval_candidates:
                interval_candidates = [
                    position for position in range(lower, upper + 1)
                    if position not in output and scores[position] >= candidate_threshold
                ]
            if not interval_candidates:
                return output

            center = (left + right) * 0.5
            chosen = max(
                interval_candidates,
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
