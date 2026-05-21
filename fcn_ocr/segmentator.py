from __future__ import annotations

from pathlib import Path

import torch

from .recognizer import TextRecognizer
from .results import SegmentationRun, VerticalSegmentationResult


class VerticalSegmentator(TextRecognizer):
    """FCN segmentator for vertical cut-point projections."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        verbose: bool = False,
        scale_x: float = 0.0,
        y_pad: float = 0.0,
        x_pad: float = 0.0,
        baseline_crop: bool = False,
        baseline_top_pad: float = 0.12,
        baseline_bottom_pad: float = 0.18,
        baseline_deskew: bool = True,
        baseline_max_angle: float = 12.0,
        cut_threshold: float | None = None,
        peak_min_distance: int | None = None,
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
            x_pad=x_pad,
            baseline_crop=baseline_crop,
            baseline_top_pad=baseline_top_pad,
            baseline_bottom_pad=baseline_bottom_pad,
            baseline_deskew=baseline_deskew,
            baseline_max_angle=baseline_max_angle,
        )
        checkpoint_config = self.checkpoint.get("config", {})
        model_config = self.checkpoint.get("model_config", {})
        self.target_format = str(
            model_config.get("target_format", checkpoint_config.get("target_format", ""))
        ).lower()
        if self.loss_mode == "cut_projection" or self.num_classes == 1:
            self.target_format = "cut_projection"

        if self.target_format != "cut_projection" or self.num_classes != 1:
            raise ValueError(
                "VerticalSegmentator expects a cut_projection checkpoint with one output channel; "
                f"got target_format={self.target_format!r}, num_classes={self.num_classes}"
            )

        self.cut_threshold = self._resolve_cut_threshold(cut_threshold, checkpoint_config)
        self.peak_min_distance = self._resolve_non_negative_int(
            peak_min_distance,
            checkpoint_config,
            "segmentator_peak_min_distance",
            default=1,
            min_value=1,
        )
        self.cut_postprocess = self._resolve_cut_postprocess(cut_postprocess, checkpoint_config)
        self.cut_min_width = self._resolve_non_negative_int(
            cut_min_width,
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
    def _resolve_cut_threshold(value: float | None, config: dict) -> float:
        resolved = float(config.get("segmentator_cut_threshold", 0.5) if value is None else value)
        if not 0.0 < resolved < 1.0:
            raise ValueError("segmentator cut threshold must be between 0 and 1")
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
        print(f"Segmentator preprocess: scale_x={self.scale_x:+.4f}, y_pad={self.y_pad:+.4f}, x_pad={self.x_pad:.4f}")
        print(f"Segmentator mode: {self.target_format}")
        print("Segmentator output: cut projection, one sigmoid score per column")
        print(
            "Segmentator params: "
            f"cut_threshold={self.cut_threshold:.3f}, "
            f"peak_min_distance={self.peak_min_distance}, "
            f"postprocess={self.cut_postprocess}, "
            f"cut_min_width={self.cut_min_width}, "
            f"cut_max_width={self.cut_max_width}, "
            f"candidate_threshold={self.cut_candidate_threshold:.3f}, "
            f"smooth_radius={self.cut_smooth_radius}"
        )

    def analyze_segmentation_logits(
        self,
        logits: torch.Tensor,
        input_shape: tuple[int, ...],
    ) -> VerticalSegmentationResult:
        if logits.size(1) != 1:
            raise ValueError(f"Cut segmentator expects logits with one channel, got {tuple(logits.shape)}")
        return self._analyze_cut_projection_logits(logits, input_shape)

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
            threshold=self.cut_threshold,
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
            cut_scores=cut_scores,
            runs=runs,
            cut_threshold=self.cut_threshold,
            peak_min_distance=self.peak_min_distance,
            input_shape=input_shape,
            logits_shape=tuple(logits.shape),
            mode="cut_projection",
            cut_positions=cut_positions,
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
            runs.append(VerticalSegmentator._run_from_slice(label, start, timestep - 1, raw_confidences, cut_scores))
            start = timestep
            label = value
        runs.append(VerticalSegmentator._run_from_slice(label, start, len(raw_indices) - 1, raw_confidences, cut_scores))
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
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return self.analyze_segmentation_logits(logits, input_shape=tuple(image_tensor.shape))
