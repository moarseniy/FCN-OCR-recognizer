from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .results import ClassConfidence, VerticalSegmentationResult


class CutProcessingMixin:
    @staticmethod
    def _segmentation_cut_positions(
        segmentation_result: VerticalSegmentationResult,
    ) -> list[int]:
        return [int(position) for position in segmentation_result.cut_positions or []]

    def _map_input_boundary_to_ocr(
        self, boundary: float, input_width: int, ocr_width: int
    ) -> int:
        if input_width <= 0 or ocr_width <= 0:
            return 0
        left = min(max(0, self.ocr_crop_left), max(0, input_width - 1))
        right = max(left + 1, input_width - max(0, self.ocr_crop_right))
        mapped = int(
            round(
                (float(boundary) - float(left)) * float(ocr_width) / float(right - left)
            )
        )
        return max(0, min(ocr_width, mapped))

    @staticmethod
    def _source_x_profile(source_x: np.ndarray) -> np.ndarray:
        if source_x.ndim != 2 or source_x.shape[1] == 0:
            raise ValueError("source_x map must have shape (H, W) with W > 0")
        profile = np.full(source_x.shape[1], np.nan, dtype=np.float64)
        for column in range(source_x.shape[1]):
            values = source_x[:, column]
            valid = values[values >= 0.0]
            if valid.size:
                profile[column] = float(np.median(valid))
        return profile

    @classmethod
    def _source_x_for_timestep(
        cls,
        position: int,
        output_width: int,
        source_x: np.ndarray,
    ) -> float | None:
        if output_width <= 0:
            return None
        profile = cls._source_x_profile(source_x)
        input_position = (float(position) + 0.5) * float(profile.size) / float(
            output_width
        ) - 0.5
        column = max(0, min(profile.size - 1, int(round(input_position))))
        value = profile[column]
        return float(value) if np.isfinite(value) else None

    @classmethod
    def _input_x_for_source_x(
        cls,
        source_position: float,
        source_x: np.ndarray,
        edge: str | None = None,
    ) -> float | None:
        profile = cls._source_x_profile(source_x)
        valid_indices = np.flatnonzero(np.isfinite(profile))
        if valid_indices.size == 0:
            return None
        distances = np.abs(profile[valid_indices] - float(source_position))
        best_distance = float(distances.min())
        best = valid_indices[np.isclose(distances, best_distance, rtol=0.0, atol=1e-6)]
        if edge == "left":
            column = int(best.min())
        elif edge == "right":
            column = int(best.max())
        else:
            column = int(round(float(np.mean(best))))
        return float(column) + 0.5

    @staticmethod
    def _central_decode_span(
        start: int,
        end: int,
        center_fraction: float,
        min_width: int,
    ) -> tuple[int, int]:
        if end <= start:
            return start, end
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError("center_fraction must be in (0, 1]")
        if min_width < 1:
            raise ValueError("min_width must be >= 1")

        width = end - start
        if center_fraction >= 1.0 or width <= 1:
            return start, end

        score_width = int(round(float(width) * center_fraction))
        score_width = max(1, min(width, max(int(min_width), score_width)))
        center = (float(start) + float(end)) * 0.5
        score_start = int(round(center - float(score_width) * 0.5))
        score_start = max(start, min(end - score_width, score_start))
        score_end = score_start + score_width
        return int(score_start), int(score_end)

    def _map_segmentator_cuts_to_ocr_boundaries(
        self,
        raw_cuts: list[int],
        *,
        segmentator_width: int,
        input_width: int,
        ocr_width: int,
        ocr_source_x: np.ndarray | None,
        segmentator_source_x: np.ndarray | None,
    ) -> list[int]:
        boundaries = []
        use_coordinate_maps = (
            ocr_source_x is not None and segmentator_source_x is not None
        )
        for cut_index, position in enumerate(raw_cuts):
            input_position: float
            if use_coordinate_maps:
                source_position = self._source_x_for_timestep(
                    position,
                    segmentator_width,
                    segmentator_source_x,
                )
                edge = (
                    "left"
                    if cut_index == 0
                    else "right"
                    if cut_index == len(raw_cuts) - 1
                    else None
                )
                mapped_input = (
                    self._input_x_for_source_x(source_position, ocr_source_x, edge=edge)
                    if source_position is not None
                    else None
                )
                if mapped_input is not None:
                    input_position = mapped_input
                else:
                    input_position = (
                        (float(position) + 0.5)
                        * float(input_width)
                        / float(segmentator_width)
                        if segmentator_width > 0
                        else 0.0
                    )
            elif segmentator_width > 0:
                input_position = int(
                    round(
                        (float(position) + 0.5)
                        * float(input_width)
                        / float(segmentator_width)
                    )
                )
            else:
                input_position = 0
            boundaries.append(
                self._map_input_boundary_to_ocr(input_position, input_width, ocr_width)
            )
        return boundaries

    @staticmethod
    def _candidate_cut_positions_from_scores(
        segmentation_result: VerticalSegmentationResult,
    ) -> list[int]:
        scores = segmentation_result.cut_scores
        if not scores:
            return []

        threshold = float(segmentation_result.cut_threshold)
        candidates = {
            int(position)
            for position in (segmentation_result.cut_positions or [])
            if 0 <= int(position) < len(scores)
        }
        last_index = len(scores) - 1
        for index, score in enumerate(scores):
            if float(score) < threshold:
                continue
            left = scores[index - 1] if index > 0 else float("-inf")
            right = scores[index + 1] if index < last_index else float("-inf")
            if float(score) >= float(left) and float(score) >= float(right):
                candidates.add(index)
        return sorted(candidates)

    @staticmethod
    def _width_prior_score(width: int, min_width: int, max_width: int) -> float:
        if max_width <= min_width:
            return 0.0
        midpoint = 0.5 * float(min_width + max_width)
        half_range = max(1.0, 0.5 * float(max_width - min_width))
        return max(0.0, 1.0 - abs(float(width) - midpoint) / half_range)

    @staticmethod
    def _safe_cut_score(scores: list[float], position: int) -> float:
        if not scores:
            return 0.0
        position = max(0, min(len(scores) - 1, int(position)))
        return float(scores[position])

    @staticmethod
    def _glyph_prior_field(
        config: dict[str, Any] | Any | None, name: str, default: Any
    ) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    def _glyph_width_prior_is_active(self, config: dict[str, Any] | Any | None) -> bool:
        if config is None:
            return False
        ranges = self._glyph_prior_field(config, "ranges", {})
        weight = float(self._glyph_prior_field(config, "weight", 0.0) or 0.0)
        return (
            bool(self._glyph_prior_field(config, "enabled", False))
            and weight > 0.0
            and bool(ranges)
        )

    def _glyph_width_bounds_for_char(
        self,
        char: str,
        config: dict[str, Any] | Any | None,
    ) -> tuple[float, float] | None:
        ranges = self._glyph_prior_field(config, "ranges", {}) or {}
        default_bounds = None
        for group, bounds in ranges.items():
            if group == "~":
                default_bounds = bounds
                continue
            if char in str(group):
                low, high = bounds
                return float(low), float(high)
        if default_bounds is None:
            return None
        low, high = default_bounds
        return float(low), float(high)

    def _glyph_width_prior_adjustment(
        self,
        class_index: int,
        cell_width: float,
        denominator: float,
        config: dict[str, Any] | Any | None,
    ) -> tuple[float, float | None]:
        if not self._glyph_width_prior_is_active(config):
            return 0.0, None
        char = self.idx_to_char.get(int(class_index))
        if char is None:
            return 0.0, None
        bounds = self._glyph_width_bounds_for_char(char, config)
        if bounds is None:
            return 0.0, None

        low, high = bounds
        denominator = max(1e-6, float(denominator))
        ratio = max(0.0, float(cell_width)) / denominator
        if low <= ratio <= high:
            return 0.0, ratio

        span = max(1e-6, high - low)
        distance = (low - ratio) / span if ratio < low else (ratio - high) / span
        penalty = min(4.0, distance * distance)
        weight = float(self._glyph_prior_field(config, "weight", 0.0) or 0.0)
        return -weight * penalty, ratio

    @staticmethod
    def _cell_width_in_input_pixels(
        start: int,
        end: int,
        *,
        input_width: int,
        ocr_width: int,
        source_start: int,
        source_end: int,
        segmentator_width: int,
    ) -> float:
        if end > start and ocr_width > 0:
            return max(0.0, float(end - start) * float(input_width) / float(ocr_width))
        if segmentator_width > 0:
            return max(
                0.0,
                float(source_end - source_start)
                * float(input_width)
                / float(segmentator_width),
            )
        return 0.0

    def _median_cell_width_in_input_pixels(
        self,
        cuts: list[int],
        boundaries: list[int],
        *,
        input_width: int,
        ocr_width: int,
        segmentator_width: int,
    ) -> float | None:
        widths = [
            self._cell_width_in_input_pixels(
                start,
                end,
                input_width=input_width,
                ocr_width=ocr_width,
                source_start=source_start,
                source_end=source_end,
                segmentator_width=segmentator_width,
            )
            for (start, end), (source_start, source_end) in zip(
                zip(boundaries, boundaries[1:]),
                zip(cuts, cuts[1:]),
            )
        ]
        positive_widths = [width for width in widths if width > 0.0]
        if not positive_widths:
            return None
        return float(np.median(np.asarray(positive_widths, dtype=np.float64)))

    def _glyph_width_denominator(
        self,
        config: dict[str, Any] | Any | None,
        *,
        input_height: int,
        median_cell_width: float | None,
    ) -> float:
        normalize_by = str(
            self._glyph_prior_field(config, "normalize_by", "input_height")
        ).lower()
        if normalize_by == "input_height":
            return max(1.0, float(input_height))
        if normalize_by == "median_cell_width":
            return max(
                1e-6,
                float(
                    median_cell_width if median_cell_width is not None else input_height
                ),
            )
        raise ValueError(
            "glyph_width_prior.normalize_by must be 'input_height' or 'median_cell_width'"
        )

    def _rank_class_scores(
        self,
        class_scores: torch.Tensor,
        *,
        cell_width: float,
        input_height: int,
        median_cell_width: float | None,
        glyph_width_prior: dict[str, Any] | Any | None,
        top_k: int,
        ocr_weight: float = 1.0,
    ) -> tuple[int, float, float, float | None, float, list[ClassConfidence]]:
        raw_scores = class_scores.detach().cpu().float().tolist()
        prior_active = self._glyph_width_prior_is_active(glyph_width_prior)
        denominator = (
            self._glyph_width_denominator(
                glyph_width_prior,
                input_height=input_height,
                median_cell_width=median_cell_width,
            )
            if prior_active
            else max(1.0, float(input_height))
        )
        ranked: list[tuple[float, float, int, float, float | None]] = []
        for class_index, raw_score in enumerate(raw_scores):
            if class_index not in self.idx_to_char:
                continue
            prior_score, ratio = self._glyph_width_prior_adjustment(
                class_index,
                cell_width,
                denominator,
                glyph_width_prior,
            )
            total_score = float(ocr_weight) * float(raw_score) + prior_score
            ranked.append(
                (total_score, float(raw_score), class_index, prior_score, ratio)
            )

        if not ranked:
            raise ValueError("No OCR classes are available for decoding")
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_k = max(1, min(int(top_k), len(ranked)))
        candidates = [
            ClassConfidence(
                label=self.class_label(class_index),
                confidence=raw_score,
                class_index=class_index,
                score=total_score if prior_active else None,
            )
            for total_score, raw_score, class_index, _, _ in ranked[:top_k]
        ]
        total_score, raw_score, class_index, prior_score, ratio = ranked[0]
        return class_index, raw_score, prior_score, ratio, total_score, candidates
