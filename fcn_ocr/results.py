from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


BLANK_SYMBOL = "∅"
SPACE_SYMBOL = "␠"


@dataclass(frozen=True)
class ClassConfidence:
    label: str
    confidence: float
    class_index: int


@dataclass(frozen=True)
class DecodedSymbol:
    char: str
    confidence: float
    timestep: int
    class_index: int
    candidates: list[ClassConfidence]


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    raw_indices: list[int]
    raw_confidences: list[float]
    raw_chars: list[str]
    decoded_symbols: list[DecodedSymbol]
    top_candidates_by_timestep: list[list[ClassConfidence]]
    input_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]


@dataclass(frozen=True)
class SegmentationRun:
    label: int
    kind: str
    start: int
    end: int
    confidence: float
    gap_probability: float


@dataclass(frozen=True)
class VerticalSegmentationResult:
    raw_indices: list[int]
    raw_confidences: list[float]
    gap_probabilities: list[float]
    runs: list[SegmentationRun]
    gap_threshold: float
    min_gap_width: int
    merge_gap_width: int
    input_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    mode: str = "binary_gaps"
    cut_positions: list[int] | None = None
    peak_min_distance: int | None = None


@dataclass(frozen=True)
class PreprocessDebug:
    metadata: dict[str, Any]
    images: list[tuple[str, Image.Image]]


def display_char(char: str) -> str:
    if char == " ":
        return SPACE_SYMBOL
    if char == "\t":
        return "<tab>"
    if char == "\n":
        return "<newline>"
    return char
