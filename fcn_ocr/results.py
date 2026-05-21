from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


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
class CutDecodedSymbol:
    char: str
    confidence: float
    class_index: int
    start: int
    end: int
    source_start: int
    source_end: int
    candidates: list[ClassConfidence]


@dataclass(frozen=True)
class CutDecodingResult:
    text: str
    symbols: list[CutDecodedSymbol]
    cuts: list[int]
    boundaries: list[int]
    input_width: int
    ocr_width: int
    segmentator_width: int


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
    score: float


@dataclass(frozen=True)
class VerticalSegmentationResult:
    raw_indices: list[int]
    raw_confidences: list[float]
    cut_scores: list[float]
    runs: list[SegmentationRun]
    cut_threshold: float
    peak_min_distance: int
    input_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    mode: str = "cut_projection"
    cut_positions: list[int] | None = None
    candidate_cut_positions: list[int] | None = None
    cut_postprocess: str | None = None
    cut_candidate_threshold: float | None = None
    cut_min_width: int | None = None
    cut_max_width: int | None = None
    cut_smooth_radius: int | None = None


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
