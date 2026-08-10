from __future__ import annotations

import numpy as np

from fcn_ocr.recognizer import TextRecognizer


def _baseline_box_recognizer(*, strict: bool = True) -> TextRecognizer:
    recognizer = TextRecognizer.__new__(TextRecognizer)
    recognizer.baseline_strict_lines = strict
    recognizer.baseline_line_pad = 0.1
    recognizer.baseline_line_pad_px = 2.0
    return recognizer


def test_strict_paired_baseline_crop_uses_both_lines_and_combined_padding() -> None:
    recognizer = _baseline_box_recognizer()
    xs = np.arange(0, 20, dtype=np.float64)
    ys = np.linspace(12.0, 18.0, num=20, dtype=np.float64)

    result = recognizer._paired_baseline_crop_box(
        top_slope=0.0,
        top_intercept=10.0,
        bottom_slope=0.0,
        bottom_intercept=20.0,
        xs=xs,
        ys=ys,
        image_width=20,
    )

    assert result == ((0, 7, 20, 24), 17)


def test_strict_paired_baseline_crop_rejects_crossed_lines() -> None:
    recognizer = _baseline_box_recognizer(strict=True)
    xs = np.arange(0, 10, dtype=np.float64)
    ys = np.linspace(10.0, 12.0, num=10, dtype=np.float64)

    result = recognizer._paired_baseline_crop_box(
        top_slope=0.0,
        top_intercept=14.0,
        bottom_slope=0.0,
        bottom_intercept=12.0,
        xs=xs,
        ys=ys,
        image_width=10,
    )

    assert result is None
