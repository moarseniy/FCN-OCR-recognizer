from __future__ import annotations

import torch
import pytest

from loss import _align_logits_and_labels, fcn_ocr_targets_to_labels
from fcn_synth_generator.dataset import (
    SingleLineDataset,
    SingleLineDatasetConfig,
    TextRenderStyle,
)


def _dataset_without_rendering() -> SingleLineDataset:
    config = SingleLineDatasetConfig(
        task="fcn_ocr",
        alphabet=" AB",
    )
    dataset = SingleLineDataset.__new__(SingleLineDataset)
    dataset.config = config
    dataset.alphabet = config.alphabet
    dataset.char_to_index = {char: index for index, char in enumerate(config.alphabet)}
    return dataset


def test_ocr_target_marks_pixels_outside_visible_ink_as_space() -> None:
    dataset = _dataset_without_rendering()

    labels = dataset._encode_fcn_ocr_targets(
        spans=[("A", 2.0, 5.0), ("B", 5.0, 8.0)],
        ink_spans=[("A", 2.0, 5.0), ("B", 5.0, 8.0)],
        width=10,
    )

    assert labels.tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 0, 0]


def test_ocr_target_uses_nearest_character_between_logical_spans() -> None:
    dataset = _dataset_without_rendering()

    labels = dataset._encode_fcn_ocr_targets(
        spans=[("A", 1.0, 3.0), ("B", 5.0, 7.0)],
        ink_spans=[("A", 1.0, 3.0), ("B", 5.0, 7.0)],
        width=8,
    )

    assert labels.tolist() == [0, 1, 1, 1, 2, 2, 2, 0]


def test_explicit_text_resamples_font_and_style_after_ink_spacing_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_without_rendering()
    style = TextRenderStyle(char_spacing=0.0, word_spacing_multiplier=1.0)
    rendered = (torch.zeros(1, 48, 64), [], [], 10.0, 30.0)
    expected = object()
    accepted = iter((False, True))
    render_calls = 0

    monkeypatch.setattr(dataset, "_sample_text_style", lambda rng: style)
    monkeypatch.setattr(
        dataset,
        "_load_font_that_fits",
        lambda text, rng, sampled_style: object(),
    )

    def render_text(text, font, rng, sampled_style):
        nonlocal render_calls
        render_calls += 1
        return rendered

    monkeypatch.setattr(dataset, "_render_text", render_text)
    monkeypatch.setattr(
        dataset,
        "_accept_ink_spacing",
        lambda cut_spans, rng: next(accepted),
    )
    monkeypatch.setattr(dataset, "_make_sample", lambda *args: expected)

    sample = dataset.generate_text_sample("AB")

    assert sample is expected
    assert render_calls == 2


def test_majority_alignment_keeps_clear_bins_and_ignores_ambiguous_bins() -> None:
    logits = torch.zeros((1, 3, 2))
    labels = torch.tensor([[1, 2, 2, 2]])

    _, aligned = _align_logits_and_labels(
        logits,
        labels,
        strict_width=False,
        label_min_majority=0.6,
        ignore_index=-100,
    )

    assert aligned.tolist() == [[-100, 2]]


def test_fcn_ocr_crop_only_removes_configured_columns() -> None:
    labels = torch.tensor([[0, 1, 1, 2, 2, 0]])

    cropped = fcn_ocr_targets_to_labels(labels, crop_left=1, crop_right=1)

    assert cropped.tolist() == [[1, 1, 2, 2]]


def test_fcn_ocr_targets_reject_removed_image_map_format() -> None:
    targets = torch.zeros((1, 1, 4, 8), dtype=torch.long)

    with pytest.raises(ValueError, match=r"shape \(B, W\)"):
        fcn_ocr_targets_to_labels(targets)
