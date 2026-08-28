from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .results import (
    ClassConfidence,
    CutDecodedSymbol,
    CutDecodingResult,
    DecodedSymbol,
    RecognitionResult,
    VerticalSegmentationResult,
    display_char,
)


class FCNOCRDecodingMixin:
    def class_label(self, index: int) -> str:
        return display_char(self.idx_to_char.get(index, f"<{index}>"))

    def decode_predictions(self, logits: torch.Tensor) -> tuple[str, list[int]]:
        pred_ids = logits.argmax(dim=1)
        return self.decode_pred_ids_batch(pred_ids)[0]

    def decode_pred_ids_batch(
        self,
        pred_ids: torch.Tensor,
        input_lengths: list[int] | torch.Tensor | None = None,
    ) -> list[tuple[str, list[int]]]:
        decoded: list[tuple[str, list[int]]] = []
        if input_lengths is None:
            lengths = [pred_ids.size(1)] * pred_ids.size(0)
        elif isinstance(input_lengths, torch.Tensor):
            lengths = [int(length) for length in input_lengths.detach().cpu().tolist()]
        else:
            lengths = [int(length) for length in input_lengths]

        for row, length in zip(pred_ids, lengths):
            raw_ids = row[: max(0, length)].detach().cpu().tolist()
            collapsed_ids: list[int] = []
            previous_id: int | None = None
            for class_index in raw_ids:
                if class_index != previous_id:
                    collapsed_ids.append(class_index)
                previous_id = class_index

            text = "".join(
                self.idx_to_char[class_index]
                for class_index in collapsed_ids
                if class_index in self.idx_to_char
            )
            decoded.append((text, raw_ids))
        return decoded

    def analyze_logits(
        self, logits: torch.Tensor, input_shape: tuple[int, ...], top_k: int = 8
    ) -> RecognitionResult:
        probs = torch.softmax(logits, dim=1)
        confidences, pred_ids = probs.max(dim=1)
        top_k = max(1, min(int(top_k), probs.size(1)))
        top_confidences, top_indices = probs.topk(top_k, dim=1)

        raw_indices = pred_ids[0].detach().cpu().tolist()
        raw_confidences = confidences[0].detach().cpu().tolist()
        raw_chars = [self.class_label(idx) for idx in raw_indices]
        top_candidates_by_timestep: list[list[ClassConfidence]] = []
        for timestep in range(pred_ids.size(1)):
            timestep_candidates: list[ClassConfidence] = []
            for rank in range(top_k):
                class_index = int(top_indices[0, rank, timestep].detach().cpu().item())
                confidence = float(
                    top_confidences[0, rank, timestep].detach().cpu().item()
                )
                timestep_candidates.append(
                    ClassConfidence(
                        label=self.class_label(class_index),
                        confidence=confidence,
                        class_index=class_index,
                    )
                )
            top_candidates_by_timestep.append(timestep_candidates)

        decoded_symbols: list[DecodedSymbol] = []
        keep = torch.ones_like(pred_ids[0], dtype=torch.bool)
        if keep.numel() > 1:
            keep[1:] = pred_ids[0, 1:] != pred_ids[0, :-1]

        for timestep in keep.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
            class_index = raw_indices[timestep]
            char = self.idx_to_char.get(class_index)
            if char is None:
                continue
            decoded_symbols.append(
                DecodedSymbol(
                    char=char,
                    confidence=float(raw_confidences[timestep]),
                    timestep=int(timestep),
                    class_index=int(class_index),
                    candidates=top_candidates_by_timestep[timestep],
                )
            )

        text = "".join(symbol.char for symbol in decoded_symbols)
        return RecognitionResult(
            text=text,
            raw_indices=raw_indices,
            raw_confidences=[float(confidence) for confidence in raw_confidences],
            raw_chars=raw_chars,
            decoded_symbols=decoded_symbols,
            top_candidates_by_timestep=top_candidates_by_timestep,
            input_shape=input_shape,
            logits_shape=tuple(logits.shape),
        )

    def decode_fcn_ocr_with_cuts(
        self,
        logits: torch.Tensor,
        segmentation_result: VerticalSegmentationResult,
        input_width: int | None = None,
        input_height: int | None = None,
        top_k: int = 8,
        center_fraction: float = 0.6,
        min_score_width: int = 1,
        ocr_source_x: np.ndarray | None = None,
        segmentator_source_x: np.ndarray | None = None,
        glyph_width_prior: dict[str, Any] | Any | None = None,
    ) -> CutDecodingResult:
        if logits.dim() != 3 or logits.size(0) != 1:
            raise ValueError(
                f"FCN OCR+cuts decoding expects logits shape (1, C, T), got {tuple(logits.shape)}"
            )
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError("center_fraction must be in (0, 1]")
        if min_score_width < 1:
            raise ValueError("min_score_width must be >= 1")

        probs = torch.softmax(logits, dim=1)[0]
        ocr_width = int(probs.size(1))
        segmentator_width = len(segmentation_result.raw_indices)
        input_width = int(
            input_width
            if input_width is not None
            else segmentation_result.input_shape[-1]
        )
        input_height = int(
            input_height
            if input_height is not None
            else segmentation_result.input_shape[-2]
        )
        if ocr_width <= 0:
            return CutDecodingResult(
                text="",
                symbols=[],
                cuts=[],
                boundaries=[],
                input_width=input_width,
                ocr_width=ocr_width,
                segmentator_width=segmentator_width,
            )

        raw_cuts = sorted(
            {
                position
                for position in self._segmentation_cut_positions(segmentation_result)
                if 0 <= position < max(0, segmentator_width)
            }
        )
        # Cut lines are explicit cell boundaries: only consecutive pairs are decoded.
        boundaries = self._map_segmentator_cuts_to_ocr_boundaries(
            raw_cuts,
            segmentator_width=segmentator_width,
            input_width=input_width,
            ocr_width=ocr_width,
            ocr_source_x=ocr_source_x,
            segmentator_source_x=segmentator_source_x,
        )
        use_coordinate_maps = (
            ocr_source_x is not None and segmentator_source_x is not None
        )
        intervals = list(zip(boundaries, boundaries[1:]))
        source_intervals = list(zip(raw_cuts, raw_cuts[1:]))
        median_cell_width = self._median_cell_width_in_input_pixels(
            raw_cuts,
            boundaries,
            input_width=input_width,
            ocr_width=ocr_width,
            segmentator_width=segmentator_width,
        )
        top_k = max(1, min(int(top_k), probs.size(0)))

        symbols: list[CutDecodedSymbol] = []
        for (start, end), (source_start, source_end) in zip(
            intervals, source_intervals
        ):
            if end > start:
                score_start, score_end = self._central_decode_span(
                    start,
                    end,
                    center_fraction=center_fraction,
                    min_width=min_score_width,
                )
            else:
                input_center: float
                if use_coordinate_maps:
                    left_source = self._source_x_for_timestep(
                        source_start,
                        segmentator_width,
                        segmentator_source_x,
                    )
                    right_source = self._source_x_for_timestep(
                        source_end,
                        segmentator_width,
                        segmentator_source_x,
                    )
                    mapped_center = None
                    if left_source is not None and right_source is not None:
                        mapped_center = self._input_x_for_source_x(
                            (left_source + right_source) * 0.5,
                            ocr_source_x,
                        )
                    if mapped_center is not None:
                        input_center = mapped_center
                    else:
                        source_center = (
                            float(source_start) + float(source_end) + 1.0
                        ) * 0.5
                        input_center = (
                            source_center
                            * float(input_width)
                            / max(1.0, float(segmentator_width))
                        )
                else:
                    source_center = (
                        float(source_start) + float(source_end) + 1.0
                    ) * 0.5
                    input_center = (
                        source_center
                        * float(input_width)
                        / max(1.0, float(segmentator_width))
                    )
                score_start = self._map_input_boundary_to_ocr(
                    input_center, input_width, ocr_width
                )
                score_start = max(0, min(ocr_width - 1, score_start))
                score_end = score_start + 1
            if score_end <= score_start:
                continue
            scores = probs[:, score_start:score_end].mean(dim=1)
            cell_width = self._cell_width_in_input_pixels(
                start,
                end,
                input_width=input_width,
                ocr_width=ocr_width,
                source_start=source_start,
                source_end=source_end,
                segmentator_width=segmentator_width,
            )
            (
                class_index,
                raw_score,
                glyph_width_score,
                glyph_width_ratio,
                adjusted_score,
                candidates,
            ) = self._rank_class_scores(
                scores,
                cell_width=cell_width,
                input_height=input_height,
                median_cell_width=median_cell_width,
                glyph_width_prior=glyph_width_prior,
                top_k=top_k,
            )
            char = self.idx_to_char.get(class_index)
            if char is None:
                raise ValueError(
                    f"OCR class index {class_index} is not present in the checkpoint alphabet"
                )

            symbols.append(
                CutDecodedSymbol(
                    char=char,
                    confidence=raw_score,
                    class_index=class_index,
                    start=int(start),
                    end=int(end),
                    source_start=int(source_start),
                    source_end=int(source_end),
                    candidates=candidates,
                    score_start=int(score_start),
                    score_end=int(score_end),
                    glyph_width_ratio=glyph_width_ratio,
                    glyph_width_score=glyph_width_score,
                    adjusted_score=adjusted_score,
                )
            )

        return CutDecodingResult(
            text="".join(symbol.char for symbol in symbols),
            symbols=symbols,
            cuts=raw_cuts,
            boundaries=boundaries,
            input_width=input_width,
            ocr_width=ocr_width,
            segmentator_width=segmentator_width,
            decode_method="cells",
        )

    def decode_fcn_ocr_with_cuts_dp(
        self,
        logits: torch.Tensor,
        segmentation_result: VerticalSegmentationResult,
        input_width: int | None = None,
        input_height: int | None = None,
        top_k: int = 8,
        center_fraction: float = 0.6,
        min_score_width: int = 1,
        ocr_source_x: np.ndarray | None = None,
        segmentator_source_x: np.ndarray | None = None,
        cut_weight: float = 1.0,
        ocr_weight: float = 1.0,
        width_weight: float = 0.05,
        skip_cut_penalty: float = 0.35,
        glyph_width_prior: dict[str, Any] | Any | None = None,
    ) -> CutDecodingResult:
        if logits.dim() != 3 or logits.size(0) != 1:
            raise ValueError(
                f"FCN OCR+cuts DP decoding expects logits shape (1, C, T), got {tuple(logits.shape)}"
            )
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError("center_fraction must be in (0, 1]")
        if min_score_width < 1:
            raise ValueError("min_score_width must be >= 1")
        if (
            cut_weight < 0.0
            or ocr_weight < 0.0
            or width_weight < 0.0
            or skip_cut_penalty < 0.0
        ):
            raise ValueError("DP decode weights must be non-negative")

        probs = torch.softmax(logits, dim=1)[0]
        ocr_width = int(probs.size(1))
        segmentator_width = len(segmentation_result.raw_indices)
        input_width = int(
            input_width
            if input_width is not None
            else segmentation_result.input_shape[-1]
        )
        input_height = int(
            input_height
            if input_height is not None
            else segmentation_result.input_shape[-2]
        )
        if ocr_width <= 0 or segmentator_width <= 0:
            return CutDecodingResult(
                text="",
                symbols=[],
                cuts=[],
                boundaries=[],
                input_width=input_width,
                ocr_width=ocr_width,
                segmentator_width=segmentator_width,
                decode_method="dp",
                path_score=None,
            )

        candidate_cuts = self._candidate_cut_positions_from_scores(segmentation_result)
        if len(candidate_cuts) < 2:
            raise ValueError(
                "FCN OCR DP decoding requires at least two candidate cuts, "
                f"got {len(candidate_cuts)}"
            )

        boundaries = self._map_segmentator_cuts_to_ocr_boundaries(
            candidate_cuts,
            segmentator_width=segmentator_width,
            input_width=input_width,
            ocr_width=ocr_width,
            ocr_source_x=ocr_source_x,
            segmentator_source_x=segmentator_source_x,
        )
        boundary_by_index = {
            index: boundary for index, boundary in enumerate(boundaries)
        }
        top_k = max(1, min(int(top_k), probs.size(0)))
        min_width = max(1, int(segmentation_result.cut_min_width or 1))
        max_width = max(0, int(segmentation_result.cut_max_width or 0))
        median_cuts = sorted(
            {
                position
                for position in self._segmentation_cut_positions(segmentation_result)
                if 0 <= position < max(0, segmentator_width)
            }
        )
        if len(median_cuts) < 2:
            median_cuts = candidate_cuts
            median_boundaries = boundaries
        else:
            median_boundaries = self._map_segmentator_cuts_to_ocr_boundaries(
                median_cuts,
                segmentator_width=segmentator_width,
                input_width=input_width,
                ocr_width=ocr_width,
                ocr_source_x=ocr_source_x,
                segmentator_source_x=segmentator_source_x,
            )
        median_cell_width = self._median_cell_width_in_input_pixels(
            median_cuts,
            median_boundaries,
            input_width=input_width,
            ocr_width=ocr_width,
            segmentator_width=segmentator_width,
        )

        if max_width > 0:
            start_window = max_width
            end_window = max_width
            start_indices = [
                index for index, cut in enumerate(candidate_cuts) if cut <= start_window
            ]
            end_indices = [
                index
                for index, cut in enumerate(candidate_cuts)
                if cut >= segmentator_width - 1 - end_window
            ]
        else:
            start_indices = [0]
            end_indices = [len(candidate_cuts) - 1]
        if not start_indices:
            start_indices = [0]
        if not end_indices:
            end_indices = [len(candidate_cuts) - 1]
        end_index_set = set(end_indices)

        edge_cache: dict[tuple[int, int], tuple[float, CutDecodedSymbol]] = {}

        def score_edge(
            left_index: int, right_index: int
        ) -> tuple[float, CutDecodedSymbol] | None:
            key = (left_index, right_index)
            if key in edge_cache:
                return edge_cache[key]
            source_start = int(candidate_cuts[left_index])
            source_end = int(candidate_cuts[right_index])
            width = source_end - source_start
            if width < min_width:
                return None
            if max_width > 0 and width > max_width:
                return None

            start = int(boundary_by_index[left_index])
            end = int(boundary_by_index[right_index])
            if end > start:
                score_start, score_end = self._central_decode_span(
                    start,
                    end,
                    center_fraction=center_fraction,
                    min_width=min_score_width,
                )
            else:
                center = max(0, min(ocr_width - 1, int(round((start + end) * 0.5))))
                score_start, score_end = center, center + 1
            if score_end <= score_start:
                return None

            class_scores = probs[:, score_start:score_end].mean(dim=1)
            cell_width = self._cell_width_in_input_pixels(
                start,
                end,
                input_width=input_width,
                ocr_width=ocr_width,
                source_start=source_start,
                source_end=source_end,
                segmentator_width=segmentator_width,
            )
            (
                class_index,
                ocr_score,
                glyph_width_score,
                glyph_width_ratio,
                adjusted_score,
                candidates,
            ) = self._rank_class_scores(
                class_scores,
                cell_width=cell_width,
                input_height=input_height,
                median_cell_width=median_cell_width,
                glyph_width_prior=glyph_width_prior,
                top_k=top_k,
                ocr_weight=ocr_weight,
            )
            char = self.idx_to_char.get(class_index)
            if char is None:
                raise ValueError(
                    f"OCR class index {class_index} is not present in the checkpoint alphabet"
                )

            left_cut_score = self._safe_cut_score(
                segmentation_result.cut_scores, source_start
            )
            right_cut_score = self._safe_cut_score(
                segmentation_result.cut_scores, source_end
            )
            cut_score = 0.5 * (left_cut_score + right_cut_score)
            width_score = self._width_prior_score(width, min_width, max_width)
            skipped_cut_count = max(0, right_index - left_index - 1)
            edge_score = (
                float(cut_weight) * cut_score
                + float(ocr_weight) * ocr_score
                + float(width_weight) * width_score
                + float(glyph_width_score)
                - float(skip_cut_penalty) * float(skipped_cut_count)
            )
            symbol = CutDecodedSymbol(
                char=char,
                confidence=ocr_score,
                class_index=class_index,
                start=start,
                end=end,
                source_start=source_start,
                source_end=source_end,
                candidates=candidates,
                score_start=int(score_start),
                score_end=int(score_end),
                glyph_width_ratio=glyph_width_ratio,
                glyph_width_score=glyph_width_score,
                adjusted_score=adjusted_score,
            )
            edge_cache[key] = (edge_score, symbol)
            return edge_cache[key]

        states: dict[
            tuple[int, int],
            tuple[float, tuple[int, int] | None, CutDecodedSymbol | None],
        ] = {(index, 0): (0.0, None, None) for index in start_indices}
        candidate_count = len(candidate_cuts)
        for right_index in range(candidate_count):
            for left_index in range(right_index):
                scored = score_edge(left_index, right_index)
                if scored is None:
                    continue
                edge_score, symbol = scored
                previous_states = [
                    (count, value)
                    for (state_index, count), value in states.items()
                    if state_index == left_index
                ]
                for count, (previous_score, _, _) in previous_states:
                    next_key = (right_index, count + 1)
                    next_score = previous_score + edge_score
                    if next_key not in states or next_score > states[next_key][0]:
                        states[next_key] = (
                            next_score,
                            (left_index, count),
                            symbol,
                        )

        final_items = [
            (key, value)
            for key, value in states.items()
            if key[0] in end_index_set and key[1] > 0
        ]
        if not final_items:
            raise ValueError(
                "FCN OCR DP decoding could not build a valid path through the cut candidates"
            )

        best_key, (best_score, _, _) = max(
            final_items,
            key=lambda item: (item[1][0], item[0][1]),
        )
        path_score = best_score / float(best_key[1])
        symbols_reversed: list[CutDecodedSymbol] = []
        cut_indices_reversed: list[int] = [best_key[0]]
        current_key: tuple[int, int] | None = best_key
        while current_key is not None:
            _, previous_key, symbol = states[current_key]
            if symbol is not None:
                symbols_reversed.append(symbol)
            if previous_key is not None:
                cut_indices_reversed.append(previous_key[0])
            current_key = previous_key
        symbols = list(reversed(symbols_reversed))
        path_cut_indices = list(reversed(cut_indices_reversed))
        path_cuts = [candidate_cuts[index] for index in path_cut_indices]
        path_boundaries = [boundaries[index] for index in path_cut_indices]

        return CutDecodingResult(
            text="".join(symbol.char for symbol in symbols),
            symbols=symbols,
            cuts=path_cuts,
            boundaries=path_boundaries,
            input_width=input_width,
            ocr_width=ocr_width,
            segmentator_width=segmentator_width,
            decode_method="dp",
            path_score=float(path_score),
        )
