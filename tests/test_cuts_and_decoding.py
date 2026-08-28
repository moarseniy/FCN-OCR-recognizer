from __future__ import annotations

import pytest

from evaluate_vertical_segmentation import configure_segmentator
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


def test_evaluation_configures_current_segmentator_parameters() -> None:
    segmentator = VerticalSegmentator.__new__(VerticalSegmentator)
    segmentator.cut_threshold = 0.5
    segmentator.cut_min_width = 1
    segmentator.cut_max_width = 0
    segmentator.cut_smooth_radius = 0

    configure_segmentator(
        segmentator,
        cut_threshold=0.7,
        cut_min_width=4,
        cut_max_width=24,
        cut_smooth_radius=2,
        scale_x=-0.2,
        y_pad=0.1,
        x_pad=0.03,
        baseline_crop=True,
        baseline_line_pad=0.08,
        baseline_line_pad_px=1.0,
        baseline_deskew=True,
        baseline_max_angle=12.0,
        baseline_detector_threshold=0.35,
    )

    assert segmentator.cut_threshold == 0.7
    assert segmentator.cut_min_width == 4
    assert segmentator.cut_max_width == 24
    assert segmentator.cut_smooth_radius == 2
    assert segmentator.scale_x == -0.2
    assert segmentator.x_pad == 0.03


def test_cells_decoder_reads_exactly_one_symbol_between_each_cut_pair() -> None:
    recognizer = make_lightweight_recognizer()
    segmentation = make_segmentation([0, 4, 8], width=9)

    result = recognizer.decode_fcn_ocr_with_cuts(
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

    result = recognizer.decode_fcn_ocr_with_cuts_dp(
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


def test_dp_decoder_rejects_input_with_fewer_than_two_candidate_cuts() -> None:
    recognizer = make_lightweight_recognizer()
    segmentation = make_segmentation([4], width=9)

    with pytest.raises(ValueError, match="at least two candidate cuts"):
        recognizer.decode_fcn_ocr_with_cuts_dp(
            two_cell_logits(),
            segmentation,
            input_width=9,
            input_height=8,
        )
