from __future__ import annotations

from fcn_ocr.segmentator import VerticalSegmentator

from tests.helpers import (
    make_lightweight_recognizer,
    make_segmentation,
    two_cell_logits,
)


def test_cut_min_width_keeps_the_stronger_of_close_candidates() -> None:
    scores = [0.0, 0.0, 0.6, 0.0, 0.9, 0.0]

    cuts = VerticalSegmentator._apply_cut_width_constraints(
        [2, 4],
        scores,
        min_width=3,
        max_width=0,
    )

    assert cuts == [4]


def test_cut_max_width_inserts_the_strongest_candidate_inside_large_cell() -> None:
    scores = [0.0] * 11
    scores[5] = 0.8

    cuts = VerticalSegmentator._apply_cut_width_constraints(
        [0, 10],
        scores,
        min_width=2,
        max_width=6,
    )

    assert cuts == [0, 5, 10]


def test_cells_decoder_reads_exactly_one_symbol_between_each_cut_pair() -> None:
    recognizer = make_lightweight_recognizer()
    segmentation = make_segmentation([0, 4, 8], width=9)

    result = recognizer.decode_legacy_with_cuts(
        two_cell_logits(),
        segmentation,
        input_width=9,
        input_height=8,
        center_fraction=1.0,
    )

    assert result.text == "AB"
    assert result.cuts == [0, 4, 8]
    # Cut positions are segmentator columns, not direct OCR boundaries. The
    # current center-based mapping sends the last column 8 to OCR boundary 7.
    assert [(symbol.start, symbol.end) for symbol in result.symbols] == [(0, 4), (4, 7)]
    assert result.decode_method == "cells"


def test_dp_decoder_uses_the_same_two_cells_when_widths_force_adjacent_cuts() -> None:
    recognizer = make_lightweight_recognizer()
    segmentation = make_segmentation(
        [0, 4, 8],
        width=9,
        cut_min_width=3,
        cut_max_width=4,
    )

    result = recognizer.decode_legacy_with_cuts_dp(
        two_cell_logits(),
        segmentation,
        input_width=9,
        input_height=8,
        center_fraction=1.0,
    )

    assert result.text == "AB"
    assert result.cuts == [0, 4, 8]
    assert result.decode_method == "dp"
    assert result.path_score is not None


def test_dp_decoder_currently_falls_back_to_cells_when_it_has_one_cut() -> None:
    recognizer = make_lightweight_recognizer()
    segmentation = make_segmentation([4], width=9)

    result = recognizer.decode_legacy_with_cuts_dp(
        two_cell_logits(),
        segmentation,
        input_width=9,
        input_height=8,
    )

    assert result.text == ""
    assert result.decode_method == "dp_fallback_cells"
