from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import torch
from torch.utils.data import Dataset


DEFAULT_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/comfortaa/Comfortaa-Regular.ttf",
)

DEFAULT_BACKGROUND_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc")
FONT_REPORT_LIMIT = 12

SUPPORTED_AUGMENTATIONS = (
    "cycle_shift",
    "preprocess_geometry",
    "strong_blur",
    "motion_blur",
    "scale",
    "darkening",
    "noise",
    "projective",
    "rotate",
    "crop_x",
    "crop_y",
    "random_line",
    "morphology",
    "unsharp_mask",
    "gaussian_blur",
    "gaussian_noise",
    "brightness",
    "contrast",
    "invert",
)


class SingleLineDatasetConfig(BaseModel):
    """Config for a simple fully-convolutional single-line OCR dataset."""

    model_config = ConfigDict(extra="ignore")

    sample_alphabet: str = " 0123456789abcdefghijklmnopqrstuvwxyz"
    alphabet: str | None = None
    space_char: str = " "
    samples: int = Field(default=10_000, ge=1)
    image_height: int = Field(default=48, ge=16)
    image_width: int = Field(default=256, ge=32)
    min_text_length: int = Field(default=4, ge=1)
    max_text_length: int = Field(default=16, ge=1)
    line_crops: bool = False
    word_count_min: int = Field(default=2, ge=1)
    word_count_max: int = Field(default=8, ge=1)
    word_length_min: int = Field(default=2, ge=1)
    word_length_max: int = Field(default=8, ge=1)
    crop_stride: int | None = Field(default=None, ge=1)
    min_crop_text_length: int = Field(default=1, ge=1)
    font_paths: list[str] | None = None
    font_dir: str | None = None
    font_check: bool = True
    font_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_FONT_EXTENSIONS))
    font_size_min: int = Field(default=24, ge=6)
    font_size_max: int = Field(default=34, ge=6)
    char_spacing_min: float = 0.0
    char_spacing_max: float = 0.0
    word_spacing_multiplier_min: float = Field(default=1.0, gt=0.0)
    word_spacing_multiplier_max: float = Field(default=1.0, gt=0.0)
    channels: int = Field(default=3, ge=1, le=3)
    seed: int | None = None
    background: int = Field(default=255, ge=0, le=255)
    background_paths: list[str] | None = None
    background_dir: str | None = None
    background_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_BACKGROUND_EXTENSIONS))
    foreground_min: int = Field(default=0, ge=0, le=255)
    foreground_max: int = Field(default=60, ge=0, le=255)
    noise_std: float = Field(default=0.0, ge=0.0)
    blur_radius: float = Field(default=0.0, ge=0.0)
    max_rotation_degrees: float = Field(default=0.0, ge=0.0)
    augmentation_probabilities: dict[str, float] = Field(default_factory=dict)
    augmentations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    horizontal_padding: int = Field(default=8, ge=0)
    output_dir: str | None = None
    chunk_size: int = Field(default=1024, ge=1)
    num_workers: int = Field(default=0, ge=0)
    overwrite: bool = False
    save_dense_targets: bool = False
    save_cut_projection_targets: bool = False
    cut_projection_peak_radius: int = Field(default=1, ge=0)
    cut_projection_include_margins: bool = False

    @model_validator(mode="before")
    @classmethod
    def fill_alphabet_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)
        if data.get("sample_alphabet") is None and data.get("alphabet") is not None:
            data["sample_alphabet"] = data["alphabet"]
        if data.get("alphabet") is None and data.get("sample_alphabet") is not None:
            data["alphabet"] = data["sample_alphabet"]
        return data

    @classmethod
    def model_validate_with_paths(cls, data: Any, config_path: str | Path | None = None) -> "SingleLineDatasetConfig":
        data = dict(data)
        if config_path is None:
            return cls.model_validate(data)

        config_dir = Path(config_path).resolve().parent
        data["font_paths"] = cls._resolve_relative_paths(data.get("font_paths"), config_dir)
        data["background_paths"] = cls._resolve_relative_paths(data.get("background_paths"), config_dir)

        font_dir = data.get("font_dir")
        if font_dir:
            font_path = Path(font_dir)
            if not font_path.is_absolute():
                data["font_dir"] = str(config_dir / font_path)

        background_dir = data.get("background_dir")
        if background_dir:
            background_path = Path(background_dir)
            if not background_path.is_absolute():
                data["background_dir"] = str(config_dir / background_path)

        output_dir = data.get("output_dir")
        if output_dir:
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                data["output_dir"] = str(config_dir / output_path)

        return cls.model_validate(data)

    @staticmethod
    def _resolve_relative_paths(paths: list[str] | None, base_dir: Path) -> list[str] | None:
        if paths is None:
            return None
        resolved_paths = []
        for path in paths:
            path_obj = Path(path)
            resolved_paths.append(str(path_obj if path_obj.is_absolute() else base_dir / path_obj))
        return resolved_paths

    @field_validator("sample_alphabet")
    @classmethod
    def sample_alphabet_must_be_unique(cls, value: str) -> str:
        if not value:
            raise ValueError("sample_alphabet must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("sample_alphabet must contain unique characters")
        return value

    @field_validator("alphabet")
    @classmethod
    def alphabet_must_be_unique_and_cover_samples(cls, value: str | None, info) -> str | None:
        if value is None:
            return value
        if not value:
            raise ValueError("alphabet must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("alphabet must contain unique characters")
        sample_alphabet = info.data.get("sample_alphabet", "")
        missing = sorted(set(sample_alphabet) - set(value))
        if missing:
            raise ValueError(f"alphabet does not cover sample_alphabet chars: {missing}")
        return value

    @field_validator("space_char")
    @classmethod
    def space_char_must_be_one_char(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("space_char must contain exactly one character")
        return value

    @field_validator("max_text_length")
    @classmethod
    def max_length_must_be_valid(cls, value: int, info) -> int:
        min_length = info.data.get("min_text_length")
        if min_length is not None and value < min_length:
            raise ValueError("max_text_length must be >= min_text_length")
        return value

    @field_validator("word_count_max")
    @classmethod
    def max_word_count_must_be_valid(cls, value: int, info) -> int:
        min_count = info.data.get("word_count_min")
        if min_count is not None and value < min_count:
            raise ValueError("word_count_max must be >= word_count_min")
        return value

    @field_validator("word_length_max")
    @classmethod
    def max_word_length_must_be_valid(cls, value: int, info) -> int:
        min_length = info.data.get("word_length_min")
        if min_length is not None and value < min_length:
            raise ValueError("word_length_max must be >= word_length_min")
        return value

    @field_validator("font_size_max")
    @classmethod
    def max_font_size_must_be_valid(cls, value: int, info) -> int:
        min_size = info.data.get("font_size_min")
        if min_size is not None and value < min_size:
            raise ValueError("font_size_max must be >= font_size_min")
        return value

    @field_validator("char_spacing_max")
    @classmethod
    def max_char_spacing_must_be_valid(cls, value: float, info) -> float:
        min_spacing = info.data.get("char_spacing_min")
        if min_spacing is not None and value < min_spacing:
            raise ValueError("char_spacing_max must be >= char_spacing_min")
        return value

    @field_validator("word_spacing_multiplier_max")
    @classmethod
    def max_word_spacing_multiplier_must_be_valid(cls, value: float, info) -> float:
        min_multiplier = info.data.get("word_spacing_multiplier_min")
        if min_multiplier is not None and value < min_multiplier:
            raise ValueError("word_spacing_multiplier_max must be >= word_spacing_multiplier_min")
        return value

    @field_validator("foreground_max")
    @classmethod
    def foreground_range_must_be_valid(cls, value: int, info) -> int:
        min_value = info.data.get("foreground_min")
        if min_value is not None and value < min_value:
            raise ValueError("foreground_max must be >= foreground_min")
        return value

    @field_validator("background_extensions")
    @classmethod
    def background_extensions_must_be_valid(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("background_extensions must not be empty")
        return [extension if extension.startswith(".") else f".{extension}" for extension in value]

    @field_validator("font_extensions")
    @classmethod
    def font_extensions_must_be_valid(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("font_extensions must not be empty")
        return [extension if extension.startswith(".") else f".{extension}" for extension in value]

    @field_validator("augmentation_probabilities")
    @classmethod
    def augmentation_probabilities_must_be_valid(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - set(SUPPORTED_AUGMENTATIONS))
        if unknown:
            raise ValueError(f"unknown augmentations: {unknown}")
        for name, probability in value.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"probability for {name} must be between 0 and 1")
        return value

    @field_validator("augmentations")
    @classmethod
    def augmentations_must_be_known(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        unknown = sorted(set(value) - set(SUPPORTED_AUGMENTATIONS))
        if unknown:
            raise ValueError(f"unknown augmentation configs: {unknown}")
        return value


@dataclass(frozen=True)
class TextRenderStyle:
    char_spacing: float
    word_spacing_multiplier: float


@dataclass(frozen=True)
class GeneratedLineSample:
    text: str
    image: torch.Tensor
    target: torch.Tensor
    length: int
    dense_target: torch.Tensor | None
    cut_projection_target: torch.Tensor | None


class SingleLineDataset(Dataset):
    """Renders synthetic text lines with OCR sequence labels."""

    def __init__(self, config: SingleLineDatasetConfig, target_format: str = "text"):
        self.config = config
        self.target_format = target_format
        if self.target_format not in {"text", "dense_symbols", "cut_projection"}:
            raise ValueError("target_format must be 'text', 'dense_symbols', or 'cut_projection'")
        self.alphabet = config.alphabet or config.sample_alphabet
        self.char_to_index = {char: idx for idx, char in enumerate(self.alphabet)}
        if config.space_char not in self.char_to_index:
            raise ValueError("space_char must be present in sample_alphabet/alphabet")
        self.sample_alphabet = config.sample_alphabet
        if not self.sample_alphabet:
            raise ValueError("sample_alphabet must not be empty")
        if config.font_check:
            self.font_paths = self._resolve_font_paths(
                config.font_paths,
                config.font_dir,
                config.font_extensions,
                self.sample_alphabet,
            )
        else:
            self.font_paths = self._collect_unchecked_font_paths(
                config.font_paths,
                config.font_dir,
                config.font_extensions,
            )
        self.background_paths = self._resolve_background_paths(
            config.background_paths,
            config.background_dir,
            config.background_extensions,
        )

    def __len__(self) -> int:
        return self.config.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.generate_sample_from_index(index)
        if self.target_format == "dense_symbols":
            if sample.dense_target is None:
                raise RuntimeError("dense target was not generated for this sample")
            return sample.image, sample.dense_target, torch.tensor(-1, dtype=torch.long)
        if self.target_format == "cut_projection":
            if sample.cut_projection_target is None:
                raise RuntimeError("cut projection target was not generated for this sample")
            return sample.image, sample.cut_projection_target, torch.tensor(-1, dtype=torch.long)
        return sample.image, sample.target, torch.tensor(sample.length, dtype=torch.long)

    def generate_sample_from_index(self, index: int) -> GeneratedLineSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if self.config.line_crops:
            for sample_index, sample in enumerate(self.iter_generated_samples()):
                if sample_index == index:
                    return sample
            raise IndexError(index)

        rng = random.Random(self._sample_seed(index))
        return self.generate_sample(rng)

    def generate_sample(self, rng: random.Random | None = None) -> GeneratedLineSample:
        rng = rng or random.Random()
        if self.config.line_crops:
            return next(self._iter_line_crop_samples(rng))

        text, font, style = self._make_text_that_fits(rng)
        return self.generate_text_sample(text, rng, font, style)

    def iter_generated_samples(self) -> Iterable[GeneratedLineSample]:
        if not self.config.line_crops:
            for index in range(len(self)):
                yield self.generate_sample_from_index(index)
            return

        rng = random.Random(self.config.seed)
        produced = 0
        for sample in self._iter_line_crop_samples(rng):
            yield sample
            produced += 1
            if produced >= self.config.samples:
                return

    def generate_text_sample(
        self,
        text: str,
        rng: random.Random | None = None,
        font: ImageFont.FreeTypeFont | None = None,
        style: TextRenderStyle | None = None,
    ) -> GeneratedLineSample:
        rng = rng or random.Random()
        self._validate_text(text)
        text = self._normalize_spaces(text)
        if len(text) > self.config.max_text_length:
            raise ValueError(f"text length {len(text)} exceeds max_text_length={self.config.max_text_length}")
        style = style or self._sample_text_style(rng)
        font = font or self._load_font_that_fits(text, rng, style)
        image, spans = self._render_text(text, font, rng, style)
        return self._make_sample(image, text, spans)

    def _validate_text(self, text: str) -> None:
        text = self._normalize_spaces(text)
        if not text:
            raise ValueError("text must not be empty")
        missing = sorted(set(text) - set(self.sample_alphabet))
        if missing:
            raise ValueError(f"text contains chars outside sample_alphabet: {missing}")

    def _normalize_spaces(self, text: str) -> str:
        return self.config.space_char.join(part for part in text.split(self.config.space_char) if part)

    def _load_font_that_fits(
        self,
        text: str,
        rng: random.Random,
        style: TextRenderStyle | None = None,
    ) -> ImageFont.FreeTypeFont:
        for _ in range(100):
            font = self._load_font(rng)
            if self._text_fits(text, font, style):
                return font

        font_paths = list(self.font_paths)
        rng.shuffle(font_paths)
        for path in font_paths:
            font = ImageFont.truetype(path, self.config.font_size_min)
            if self._text_fits(text, font, style):
                return font

        raise ValueError(
            f"text does not fit image_width={self.config.image_width} "
            f"with horizontal_padding={self.config.horizontal_padding} "
            f"and font_size_min={self.config.font_size_min}: {text!r}"
        )

    def _sample_seed(self, index: int) -> int | None:
        if self.config.seed is None:
            return None
        return self.config.seed + index

    def _iter_line_crop_samples(self, rng: random.Random) -> Iterable[GeneratedLineSample]:
        while True:
            crops = self._generate_line_crop_samples(rng)
            for sample in crops:
                yield sample

    def _generate_line_crop_samples(self, rng: random.Random) -> list[GeneratedLineSample]:
        for _ in range(100):
            text = self._make_line_text(rng)
            style = self._sample_text_style(rng)
            font = self._load_font_for_line(text, rng)
            image, spans = self._render_long_text(text, font, rng, style)
            samples = self._slice_line_image(image, spans)
            if samples:
                return samples

        raise RuntimeError("failed to generate non-empty line crops")

    def _make_line_text(self, rng: random.Random) -> str:
        chars = [char for char in self.sample_alphabet if char != self.config.space_char]
        if not chars:
            raise ValueError("sample_alphabet must contain at least one non-space character for line_crops")

        word_count = rng.randint(self.config.word_count_min, self.config.word_count_max)
        words = []
        for _ in range(word_count):
            word_length = rng.randint(self.config.word_length_min, self.config.word_length_max)
            words.append("".join(rng.choice(chars) for _ in range(word_length)))
        return self.config.space_char.join(words)

    def _load_font_for_line(self, text: str, rng: random.Random) -> ImageFont.FreeTypeFont:
        for _ in range(100):
            font = self._load_font(rng)
            if self._text_height_fits(text, font):
                return font

        font_paths = list(self.font_paths)
        rng.shuffle(font_paths)
        for path in font_paths:
            font = ImageFont.truetype(path, self.config.font_size_min)
            if self._text_height_fits(text, font):
                return font

        raise ValueError(
            f"line text does not fit image_height={self.config.image_height} "
            f"with font_size_min={self.config.font_size_min}: {text!r}"
        )

    def _make_text_that_fits(self, rng: random.Random) -> tuple[str, ImageFont.FreeTypeFont, TextRenderStyle]:
        last_error: Exception | None = None

        for _ in range(1000):
            text_length = rng.randint(self.config.min_text_length, self.config.max_text_length)
            text = self._normalize_spaces("".join(rng.choice(self.sample_alphabet) for _ in range(text_length)))
            if not text:
                continue
            style = self._sample_text_style(rng)
            try:
                font = self._load_font_that_fits(text, rng, style)
                return text, font, style
            except ValueError as exc:
                last_error = exc

        details = f" Last fit error: {last_error}" if last_error is not None else ""
        raise RuntimeError(
            "failed to create a text sample that fits the configured image width. "
            "Decrease min_text_length/max_text_length/font_size_min, "
            "decrease horizontal_padding, or increase image_width."
            f"{details}"
        )

    def _render_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        rng: random.Random,
        style: TextRenderStyle,
    ) -> tuple[Image.Image, list[tuple[str, float, float]]]:
        cfg = self.config
        image = self._make_background(rng)
        draw = ImageDraw.Draw(image)

        bbox = self._text_bbox(text, font, style)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x_min = cfg.horizontal_padding - bbox[0]
        x_max = cfg.image_width - cfg.horizontal_padding - bbox[2]
        free_x = max(0, int(math.floor(x_max - x_min)))
        x = x_min + rng.randint(0, free_x)
        y_jitter = rng.randint(-2, 2)
        y_min = -bbox[1]
        y_max = cfg.image_height - bbox[3]
        y = (cfg.image_height - text_height) // 2 - bbox[1] + y_jitter
        y = min(max(y, y_min), y_max)
        fill = rng.randint(cfg.foreground_min, cfg.foreground_max)

        spans = self._draw_text(draw, float(x), float(y), text, font, fill, style)

        return image, spans

    def _render_long_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        rng: random.Random,
        style: TextRenderStyle,
    ) -> tuple[Image.Image, list[tuple[str, float, float]]]:
        cfg = self.config
        bbox = self._text_bbox(text, font, style)
        text_width = bbox[2] - bbox[0]
        text_advance = max(1.0, self._text_advance(text, font, style))
        x = cfg.horizontal_padding - bbox[0]
        crop_width = cfg.image_width
        stride = cfg.crop_stride or crop_width
        content_right = max(
            cfg.horizontal_padding + text_width,
            x + text_advance,
        )
        content_width = math.ceil(content_right + cfg.horizontal_padding)
        crop_count = max(1, math.ceil(max(0, content_width - crop_width) / stride) + 1)
        line_width = (crop_count - 1) * stride + crop_width

        image = self._make_background(rng, width=line_width, height=cfg.image_height)
        draw = ImageDraw.Draw(image)

        text_height = bbox[3] - bbox[1]
        y_jitter = rng.randint(-2, 2)
        y_min = -bbox[1]
        y_max = cfg.image_height - bbox[3]
        y = (cfg.image_height - text_height) // 2 - bbox[1] + y_jitter
        y = min(max(y, y_min), y_max)
        fill = rng.randint(cfg.foreground_min, cfg.foreground_max)

        spans = self._draw_text(draw, float(x), float(y), text, font, fill, style)
        return image, spans

    def _slice_line_image(
        self,
        image: Image.Image,
        spans: list[tuple[str, float, float]],
    ) -> list[GeneratedLineSample]:
        cfg = self.config
        stride = cfg.crop_stride or cfg.image_width
        samples: list[GeneratedLineSample] = []

        for left in range(0, image.width - cfg.image_width + 1, stride):
            right = left + cfg.image_width
            crop_spans = self._crop_spans(spans, left, right)
            text = "".join(char for char, _, _ in crop_spans)
            if len(text) < cfg.min_crop_text_length:
                continue
            if len(text) > cfg.max_text_length:
                continue
            crop = image.crop((left, 0, right, cfg.image_height))
            samples.append(self._make_sample(crop, text, crop_spans))

        return samples

    def _crop_spans(
        self,
        spans: list[tuple[str, float, float]],
        left: int,
        right: int,
    ) -> list[tuple[str, float, float]]:
        cropped_spans = []
        for char, start, end in spans:
            center = (start + end) * 0.5
            if left <= center < right:
                cropped_spans.append((char, start - left, end - left))
        return self._normalize_span_sequence(cropped_spans)

    def _normalize_span_sequence(
        self,
        spans: list[tuple[str, float, float]],
    ) -> list[tuple[str, float, float]]:
        space = self.config.space_char
        normalized = []
        previous_was_space = False
        for char, start, end in spans:
            is_space = char == space
            if is_space and (not normalized or previous_was_space):
                continue
            normalized.append((char, start, end))
            previous_was_space = is_space

        while normalized and normalized[-1][0] == space:
            normalized.pop()
        return normalized

    def _make_sample(
        self,
        image: Image.Image,
        text: str,
        spans: list[tuple[str, float, float]],
    ) -> GeneratedLineSample:
        target = self._encode_text(text)
        dense_target = None
        if self.target_format == "dense_symbols" or self.config.save_dense_targets:
            dense_target = self._encode_dense_symbols(spans, image.width)
        cut_projection_target = None
        if self.target_format == "cut_projection" or self.config.save_cut_projection_targets:
            cut_projection_target = self._encode_cut_projection(spans, image.width)
        length = len(text)

        if self.config.channels == 3:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        else:
            array = np.asarray(image.convert("L"), dtype=np.float32)
            tensor = torch.from_numpy(array).unsqueeze(0) / 255.0

        return GeneratedLineSample(
            text=text,
            image=tensor.contiguous(),
            target=target,
            length=length,
            dense_target=dense_target,
            cut_projection_target=cut_projection_target,
        )

    def _encode_text(self, text: str) -> torch.Tensor:
        target = torch.zeros(self.config.max_text_length, dtype=torch.long)
        encoded = torch.tensor([self.char_to_index[char] for char in text], dtype=torch.long)
        target[: len(encoded)] = encoded
        return target

    def _encode_dense_symbols(
        self,
        spans: list[tuple[str, float, float]],
        width: int,
    ) -> torch.Tensor:
        if not spans:
            raise ValueError("cannot encode dense symbols for an empty span list")

        labels = torch.empty(width, dtype=torch.long)
        centers = [0.5 * (start + end) for _, start, end in spans]
        last_span_index = len(spans) - 1
        space_index = self.char_to_index[self.config.space_char]
        left_text = spans[0][1]
        right_text = spans[-1][2]

        for x in range(width):
            position = x + 0.5
            if position < left_text or position >= right_text:
                labels[x] = space_index
                continue

            chosen_index = None
            for span_index, (_, start, end) in enumerate(spans):
                if start <= position < end:
                    chosen_index = span_index
                    break
            if chosen_index is None:
                chosen_index = min(
                    range(len(spans)),
                    key=lambda span_index: abs(centers[span_index] - position),
                )
            char = spans[min(chosen_index, last_span_index)][0]
            labels[x] = self.char_to_index[char]

        return labels

    def _encode_cut_projection(
        self,
        spans: list[tuple[str, float, float]],
        width: int,
    ) -> torch.Tensor:
        if not spans:
            raise ValueError("cannot encode cut projection for an empty span list")

        projection = torch.zeros(width, dtype=torch.float32)
        radius = self.config.cut_projection_peak_radius

        def mark_peak(center: float) -> None:
            center_index = int(round(center - 0.5))
            if radius == 0:
                if 0 <= center_index < width:
                    projection[center_index] = 1.0
                return

            for offset in range(-radius, radius + 1):
                x = center_index + offset
                if 0 <= x < width:
                    value = 1.0 - (abs(offset) / float(radius + 1))
                    projection[x] = max(float(projection[x]), value)

        if self.config.cut_projection_include_margins:
            mark_peak(spans[0][1])
            mark_peak(spans[-1][2])

        for (_, _, prev_end), (_, next_start, _) in zip(spans, spans[1:]):
            mark_peak((prev_end + next_start) * 0.5)

        return projection

    def _char_spans(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        x: float,
    ) -> list[tuple[str, float, float]]:
        spans: list[tuple[str, float, float]] = []
        for char_index, char in enumerate(text):
            start = x + float(font.getlength(text[:char_index]))
            end = x + float(font.getlength(text[: char_index + 1]))
            if end <= start:
                end = start + max(1.0, float(font.getlength(char)))
            spans.append((char, start, end))
        return spans

    def _sample_text_style(self, rng: random.Random) -> TextRenderStyle:
        cfg = self.config
        char_spacing = rng.uniform(cfg.char_spacing_min, cfg.char_spacing_max)
        word_spacing_multiplier = rng.uniform(
            cfg.word_spacing_multiplier_min,
            cfg.word_spacing_multiplier_max,
        )
        return TextRenderStyle(
            char_spacing=char_spacing,
            word_spacing_multiplier=word_spacing_multiplier,
        )

    def _has_custom_spacing(self, style: TextRenderStyle | None) -> bool:
        if style is None:
            return False
        return (
            abs(style.char_spacing) > 1e-6
            or abs(style.word_spacing_multiplier - 1.0) > 1e-6
        )

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: int,
        style: TextRenderStyle,
    ) -> list[tuple[str, float, float]]:
        if not self._has_custom_spacing(style):
            draw.text((x, y), text, font=font, fill=fill)
            return self._char_spans(text, font, x)

        spans, origins = self._styled_char_layout(text, font, x, style)
        for (char, _, _), origin in zip(spans, origins):
            if char != self.config.space_char:
                draw.text((origin, y), char, font=font, fill=fill)
        return spans

    def _styled_char_layout(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        x: float,
        style: TextRenderStyle,
    ) -> tuple[list[tuple[str, float, float]], list[float]]:
        spans: list[tuple[str, float, float]] = []
        origins: list[float] = []
        extra_before = 0.0

        for char_index, char in enumerate(text):
            prefix = text[:char_index]
            next_prefix = text[: char_index + 1]
            base_start = float(font.getlength(prefix))
            base_end = float(font.getlength(next_prefix))
            base_advance = max(1.0, base_end - base_start)
            origin = x + base_start + extra_before
            origins.append(origin)

            if char == self.config.space_char:
                span_width = max(1.0, base_advance * style.word_spacing_multiplier)
                extra_after = span_width - base_advance
            else:
                span_width = base_advance
                next_char = text[char_index + 1] if char_index + 1 < len(text) else None
                extra_after = (
                    style.char_spacing
                    if next_char is not None and next_char != self.config.space_char
                    else 0.0
                )

            start = origin
            end = max(start + 1.0, origin + span_width)
            spans.append((char, start, end))
            extra_before += extra_after

        return spans, origins

    def _styled_text_bbox(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        style: TextRenderStyle,
    ) -> tuple[float, float, float, float]:
        spans, origins = self._styled_char_layout(text, font, 0.0, style)
        left_values = [span[1] for span in spans]
        right_values = [span[2] for span in spans]
        top_values: list[float] = []
        bottom_values: list[float] = []

        for (char, _, _), origin in zip(spans, origins):
            if char == self.config.space_char:
                continue
            char_bbox = self._plain_text_bbox(char, font)
            left_values.append(origin + char_bbox[0])
            right_values.append(origin + char_bbox[2])
            top_values.append(float(char_bbox[1]))
            bottom_values.append(float(char_bbox[3]))

        if not top_values:
            plain_bbox = self._plain_text_bbox(text, font)
            top_values.append(float(plain_bbox[1]))
            bottom_values.append(float(plain_bbox[3]))

        return (
            min(left_values),
            min(top_values),
            max(right_values),
            max(bottom_values),
        )

    def _text_advance(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        style: TextRenderStyle | None = None,
    ) -> float:
        if not self._has_custom_spacing(style):
            return max(1.0, float(font.getlength(text)))
        spans, _ = self._styled_char_layout(text, font, 0.0, style)
        return max(1.0, spans[-1][2] if spans else 1.0)

    def _load_font(self, rng: random.Random, size: int | None = None) -> ImageFont.FreeTypeFont:
        font_size = size or rng.randint(self.config.font_size_min, self.config.font_size_max)
        path = rng.choice(self.font_paths)
        return ImageFont.truetype(path, font_size)

    def _make_background(self, rng: random.Random, width: int | None = None, height: int | None = None) -> Image.Image:
        cfg = self.config
        width = width or cfg.image_width
        height = height or cfg.image_height
        if not self.background_paths:
            return Image.new("L", (width, height), color=cfg.background)

        path = rng.choice(self.background_paths)
        with Image.open(path) as background_image:
            background_image = background_image.convert("L")
            return self._random_crop_or_resize_background(background_image, rng, width, height)

    def _random_crop_or_resize_background(
        self,
        image: Image.Image,
        rng: random.Random,
        target_width: int,
        target_height: int,
    ) -> Image.Image:
        scale = max(target_width / image.width, target_height / image.height)
        resized_width = max(target_width, int(round(image.width * scale)))
        resized_height = max(target_height, int(round(image.height * scale)))
        image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

        max_left = resized_width - target_width
        max_top = resized_height - target_height
        left = rng.randint(0, max_left) if max_left > 0 else 0
        top = rng.randint(0, max_top) if max_top > 0 else 0
        return image.crop((left, top, left + target_width, top + target_height))

    @staticmethod
    def _char_advances(text: str, font: ImageFont.FreeTypeFont) -> list[float]:
        return [max(1.0, float(font.getlength(char))) for char in text]

    def _text_fits(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        style: TextRenderStyle | None = None,
    ) -> bool:
        bbox = self._text_bbox(text, font, style)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        max_width = self.config.image_width - 2 * self.config.horizontal_padding
        return text_width <= max_width and text_height <= self.config.image_height

    def _text_height_fits(self, text: str, font: ImageFont.FreeTypeFont) -> bool:
        bbox = self._plain_text_bbox(text, font)
        return bbox[3] - bbox[1] <= self.config.image_height

    def _text_bbox(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        style: TextRenderStyle | None = None,
    ) -> tuple[float, float, float, float]:
        if self._has_custom_spacing(style):
            return self._styled_text_bbox(text, font, style)
        return tuple(float(value) for value in self._plain_text_bbox(text, font))

    @staticmethod
    def _plain_text_bbox(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
        draw = ImageDraw.Draw(Image.new("L", (1, 1)))
        return draw.textbbox((0, 0), text, font=font)

    @classmethod
    def _resolve_font_paths(
        cls,
        configured_paths: Iterable[str] | None,
        font_dir: str | None,
        extensions: Iterable[str],
        alphabet: str,
    ) -> list[str]:
        candidates, missing_paths = cls._collect_font_candidates(configured_paths, font_dir, extensions)
        accepted: list[str] = []
        rejected: list[dict[str, Any]] = []

        for path in candidates:
            missing_chars, error = cls._missing_font_chars(path, alphabet)
            if error is None and not missing_chars:
                accepted.append(str(path))
            else:
                rejected.append({"path": str(path), "missing": missing_chars, "error": error})

        cls._print_font_report(candidates, accepted, rejected, missing_paths, alphabet)
        if accepted:
            return accepted
        raise FileNotFoundError(
            "No usable font files cover the configured sample_alphabet. "
            "Pass font_dir/font_paths with fonts that contain every sample_alphabet character."
        )

    @classmethod
    def _collect_unchecked_font_paths(
        cls,
        configured_paths: Iterable[str] | None,
        font_dir: str | None,
        extensions: Iterable[str],
    ) -> list[str]:
        candidates, missing_paths = cls._collect_font_candidates(configured_paths, font_dir, extensions)
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths[:FONT_REPORT_LIMIT])
            if len(missing_paths) > FONT_REPORT_LIMIT:
                missing = f"{missing}, ... and {len(missing_paths) - FONT_REPORT_LIMIT} more"
            raise FileNotFoundError(f"Missing configured font paths: {missing}")
        return [str(path) for path in candidates]

    @classmethod
    def _collect_font_candidates(
        cls,
        configured_paths: Iterable[str] | None,
        font_dir: str | None,
        extensions: Iterable[str],
    ) -> tuple[list[Path], list[Path]]:
        paths: list[Path] = []
        missing_paths: list[Path] = []
        normalized_extensions = {extension.lower() for extension in extensions}

        if configured_paths is not None:
            for path in configured_paths:
                path_obj = Path(path)
                if path_obj.exists() and path_obj.is_file():
                    paths.append(path_obj)
                else:
                    missing_paths.append(path_obj)

        if font_dir is not None:
            root = Path(font_dir)
            if not root.exists():
                raise FileNotFoundError(f"font_dir does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"font_dir is not a directory: {root}")
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in normalized_extensions
            )

        if configured_paths is None and font_dir is None:
            paths.extend(Path(path) for path in DEFAULT_FONT_CANDIDATES)

        deduplicated: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                deduplicated.append(path)
                seen.add(resolved)

        if not deduplicated:
            raise FileNotFoundError(
                "No font files found. Pass font_dir or font_paths in the dataset config."
            )
        return deduplicated, missing_paths

    @classmethod
    def _missing_font_chars(cls, path: Path, alphabet: str) -> tuple[tuple[str, ...], str | None]:
        try:
            codepoints = cls._font_codepoints(path)
            if codepoints is None:
                return cls._missing_font_chars_with_pillow(path, alphabet), None
        except Exception as exc:
            return tuple(alphabet), f"{type(exc).__name__}: {exc}"
        missing_chars = tuple(char for char in alphabet if ord(char) not in codepoints)
        return missing_chars, None

    @staticmethod
    def _font_codepoints(path: Path) -> set[int] | None:
        try:
            from fontTools.ttLib import TTFont
        except ImportError:
            return None

        with TTFont(path, fontNumber=0, lazy=True) as font:
            if "cmap" not in font:
                return set()
            codepoints: set[int] = set()
            for table in font["cmap"].tables:
                codepoints.update(table.cmap.keys())
            return codepoints

    @classmethod
    def _missing_font_chars_with_pillow(cls, path: Path, alphabet: str) -> tuple[str, ...]:
        font = ImageFont.truetype(str(path), 32)
        missing_signatures = {
            cls._glyph_signature(font, char)
            for char in ("\ufffd", "\uffff")
        }
        missing_signatures.discard(None)

        missing_chars = []
        for char in alphabet:
            if char == " ":
                if font.getlength(char) <= 0:
                    missing_chars.append(char)
                continue

            signature = cls._glyph_signature(font, char)
            if signature is None or signature in missing_signatures:
                missing_chars.append(char)
        return tuple(missing_chars)

    @staticmethod
    def _glyph_signature(font: ImageFont.FreeTypeFont, char: str) -> tuple[tuple[int, int], tuple[int, int, int, int] | None, bytes] | None:
        try:
            mask = font.getmask(char)
        except Exception:
            return None
        return mask.size, mask.getbbox(), bytes(mask)

    @classmethod
    def _print_font_report(
        cls,
        candidates: list[Path],
        accepted: list[str],
        rejected: list[dict[str, Any]],
        missing_paths: list[Path],
        alphabet: str,
    ) -> None:
        print("\nFonts check")
        print(f"  sample_alphabet length: {len(alphabet)}")
        print(f"  candidates: {len(candidates)}")
        print(f"  accepted:   {len(accepted)}")
        print(f"  rejected:   {len(rejected)}")
        if missing_paths:
            print(f"  missing configured paths: {len(missing_paths)}")

        if accepted:
            print("  accepted sample:")
            for path in accepted[:FONT_REPORT_LIMIT]:
                print(f"    + {Path(path).name}")
            if len(accepted) > FONT_REPORT_LIMIT:
                print(f"    ... and {len(accepted) - FONT_REPORT_LIMIT} more")

        if rejected:
            print("  rejected sample:")
            for item in rejected[:FONT_REPORT_LIMIT]:
                path = Path(item["path"]).name
                if item["error"] is not None:
                    print(f"    - {path}: {item['error']}")
                    continue
                missing = cls._format_chars(item["missing"], limit=20)
                print(f"    - {path}: missing {missing}")
            if len(rejected) > FONT_REPORT_LIMIT:
                print(f"    ... and {len(rejected) - FONT_REPORT_LIMIT} more")

            missing_counter: dict[str, int] = {}
            for item in rejected:
                if item["error"] is not None:
                    continue
                for char in item["missing"]:
                    missing_counter[char] = missing_counter.get(char, 0) + 1
            if missing_counter:
                top_missing = sorted(missing_counter.items(), key=lambda item: (-item[1], item[0]))[:20]
                summary = ", ".join(f"{cls._printable_char(char)}:{count}" for char, count in top_missing)
                print(f"  most often missing chars: {summary}")

        if missing_paths:
            print("  missing configured paths sample:")
            for path in missing_paths[:FONT_REPORT_LIMIT]:
                print(f"    - {path}")
            if len(missing_paths) > FONT_REPORT_LIMIT:
                print(f"    ... and {len(missing_paths) - FONT_REPORT_LIMIT} more")

    @classmethod
    def _format_chars(cls, chars: Iterable[str], limit: int) -> str:
        chars = list(chars)
        rendered = ", ".join(cls._printable_char(char) for char in chars[:limit])
        if len(chars) > limit:
            rendered = f"{rendered}, ...(+{len(chars) - limit})"
        return rendered or "none"

    @staticmethod
    def _printable_char(char: str) -> str:
        if char == " ":
            return "<space>"
        if char == "\t":
            return "<tab>"
        if char == "\n":
            return "<newline>"
        return repr(char)

    @staticmethod
    def _resolve_background_paths(
        configured_paths: Iterable[str] | None,
        background_dir: str | None,
        extensions: Iterable[str],
    ) -> list[str]:
        paths: list[Path] = []
        missing_paths: list[Path] = []

        if configured_paths is not None:
            for path in configured_paths:
                path_obj = Path(path)
                if path_obj.exists() and path_obj.is_file():
                    paths.append(path_obj)
                else:
                    missing_paths.append(path_obj)

        if background_dir is not None:
            root = Path(background_dir)
            if not root.exists():
                raise FileNotFoundError(f"background_dir does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"background_dir is not a directory: {root}")

            normalized_extensions = {extension.lower() for extension in extensions}
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in normalized_extensions
            )

        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths[:FONT_REPORT_LIMIT])
            if len(missing_paths) > FONT_REPORT_LIMIT:
                missing = f"{missing}, ... and {len(missing_paths) - FONT_REPORT_LIMIT} more"
            raise FileNotFoundError(f"Missing configured background paths: {missing}")

        deduplicated: list[str] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                deduplicated.append(str(path))
                seen.add(resolved)

        if background_dir is not None and not deduplicated:
            raise FileNotFoundError(
                f"No background images found in {root}. "
                f"Supported extensions: {sorted(normalized_extensions)}"
            )
        return deduplicated
