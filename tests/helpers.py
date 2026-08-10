from __future__ import annotations

import torch

from fcn_ocr.recognizer import TextRecognizer
from fcn_ocr.results import VerticalSegmentationResult


def make_lightweight_recognizer(alphabet: str = " AB") -> TextRecognizer:
    """Build a decoder-only recognizer without loading a checkpoint."""
    recognizer = TextRecognizer.__new__(TextRecognizer)
    recognizer.loss_mode = "legacy_logreg"
    recognizer.alphabet = alphabet
    recognizer.idx_to_char = dict(enumerate(alphabet))
    recognizer.legacy_crop_left = 0
    recognizer.legacy_crop_right = 0
    return recognizer


def make_segmentation(
    cut_positions: list[int],
    *,
    width: int,
    threshold: float = 0.5,
    cut_min_width: int = 1,
    cut_max_width: int = 0,
) -> VerticalSegmentationResult:
    cut_scores = [0.05] * width
    for position in cut_positions:
        cut_scores[position] = 0.95
    cut_set = set(cut_positions)
    raw_indices = [int(index in cut_set) for index in range(width)]
    raw_confidences = [
        cut_scores[index] if label else 1.0 - cut_scores[index]
        for index, label in enumerate(raw_indices)
    ]
    return VerticalSegmentationResult(
        raw_indices=raw_indices,
        raw_confidences=raw_confidences,
        cut_scores=cut_scores,
        runs=[],
        cut_threshold=threshold,
        input_shape=(1, 3, 8, width),
        logits_shape=(1, 1, width),
        cut_positions=cut_positions,
        cut_min_width=cut_min_width,
        cut_max_width=cut_max_width,
        cut_smooth_radius=0,
    )


def two_cell_logits() -> torch.Tensor:
    logits = torch.full((1, 3, 8), -4.0)
    logits[:, 1, :4] = 4.0
    logits[:, 2, 4:] = 4.0
    return logits
