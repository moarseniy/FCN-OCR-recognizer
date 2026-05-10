from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch
from torch.utils.data import Dataset


DEFAULT_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/comfortaa/Comfortaa-Regular.ttf",
)

SUPPORTED_AUGMENTATIONS = (
    "rotate",
    "gaussian_blur",
    "gaussian_noise",
    "brightness",
    "contrast",
    "invert",
)


class SingleLineDatasetConfig(BaseModel):
    """Config for a simple fully-convolutional single-line OCR dataset.

    In ``ctc`` mode the target is a padded text sequence. In ``column`` mode
    the target has one class id per image column; background columns are labeled
    with ``space_char``.
    """

    model_config = ConfigDict(extra="ignore")

    alphabet: str = " 0123456789abcdefghijklmnopqrstuvwxyz"
    sample_alphabet: str | None = None
    target_mode: Literal["ctc", "column"] = "ctc"
    space_char: str = " "
    samples: int = Field(default=10_000, ge=1)
    image_height: int = Field(default=48, ge=16)
    image_width: int = Field(default=256, ge=32)
    min_text_length: int = Field(default=4, ge=1)
    max_text_length: int = Field(default=16, ge=1)
    font_paths: list[str] | None = None
    font_size_min: int = Field(default=24, ge=6)
    font_size_max: int = Field(default=34, ge=6)
    channels: int = Field(default=3, ge=1, le=3)
    seed: int | None = None
    background: int = Field(default=255, ge=0, le=255)
    foreground_min: int = Field(default=0, ge=0, le=255)
    foreground_max: int = Field(default=60, ge=0, le=255)
    noise_std: float = Field(default=4.0, ge=0.0)
    blur_radius: float = Field(default=0.15, ge=0.0)
    max_rotation_degrees: float = Field(default=1.0, ge=0.0)
    augmentation_probabilities: dict[str, float] = Field(default_factory=dict)
    augmentations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    horizontal_padding: int = Field(default=8, ge=0)

    @field_validator("alphabet")
    @classmethod
    def alphabet_must_be_unique(cls, value: str) -> str:
        if not value:
            raise ValueError("alphabet must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("alphabet must contain unique characters")
        return value

    @field_validator("sample_alphabet")
    @classmethod
    def sample_alphabet_must_match_alphabet(cls, value: str | None, info) -> str | None:
        if value is None:
            return value
        alphabet = info.data.get("alphabet", "")
        missing = sorted(set(value) - set(alphabet))
        if missing:
            raise ValueError(f"sample_alphabet contains characters outside alphabet: {missing}")
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

    @field_validator("font_size_max")
    @classmethod
    def max_font_size_must_be_valid(cls, value: int, info) -> int:
        min_size = info.data.get("font_size_min")
        if min_size is not None and value < min_size:
            raise ValueError("font_size_max must be >= font_size_min")
        return value

    @field_validator("foreground_max")
    @classmethod
    def foreground_range_must_be_valid(cls, value: int, info) -> int:
        min_value = info.data.get("foreground_min")
        if min_value is not None and value < min_value:
            raise ValueError("foreground_max must be >= foreground_min")
        return value

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
class GeneratedLineSample:
    text: str
    image: torch.Tensor
    target: torch.Tensor
    length: int


class SingleLineDataset(Dataset):
    """Renders synthetic text lines with CTC-friendly sequence labels."""

    def __init__(self, config: SingleLineDatasetConfig):
        self.config = config
        self.char_to_index = {char: idx for idx, char in enumerate(config.alphabet)}
        if config.target_mode == "column" and config.space_char not in self.char_to_index:
            raise ValueError("column target_mode requires space_char to be present in alphabet")
        self.sample_alphabet = config.sample_alphabet or config.alphabet.replace(config.space_char, "")
        if not self.sample_alphabet:
            raise ValueError("sample_alphabet must not be empty")
        self.font_paths = self._resolve_font_paths(config.font_paths)

    def __len__(self) -> int:
        return self.config.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(self._sample_seed(index))
        sample = self.generate_sample(rng)
        return sample.image, sample.target, torch.tensor(sample.length, dtype=torch.long)

    def generate_sample(self, rng: random.Random | None = None) -> GeneratedLineSample:
        rng = rng or random.Random()
        text, font = self._make_text_that_fits(rng)
        image, x_offset, advances = self._render_text(text, font, rng)
        if self.config.target_mode == "ctc":
            target = self._encode_text(text)
            length = len(text)
        else:
            target = self._make_column_targets(text, advances, x_offset)
            length = self.config.image_width

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
        )

    def _sample_seed(self, index: int) -> int | None:
        if self.config.seed is None:
            return None
        return self.config.seed + index

    def _make_text_that_fits(self, rng: random.Random) -> tuple[str, ImageFont.FreeTypeFont]:
        max_width = self.config.image_width - 2 * self.config.horizontal_padding
        last_candidate: tuple[str, ImageFont.FreeTypeFont] | None = None

        for _ in range(100):
            text_length = rng.randint(self.config.min_text_length, self.config.max_text_length)
            text = "".join(rng.choice(self.sample_alphabet) for _ in range(text_length))
            font = self._load_font(rng)
            advances = self._char_advances(text, font)
            last_candidate = (text, font)
            if sum(advances) <= max_width:
                return text, font

        if last_candidate is None:
            raise RuntimeError("failed to create a text sample")

        text, _ = last_candidate
        smaller_font = self._load_font(rng, size=self.config.font_size_min)
        return text, smaller_font

    def _render_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        rng: random.Random,
    ) -> tuple[Image.Image, int, list[float]]:
        cfg = self.config
        image = Image.new("L", (cfg.image_width, cfg.image_height), color=cfg.background)
        draw = ImageDraw.Draw(image)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        free_x = max(0, cfg.image_width - text_width - 2 * cfg.horizontal_padding)
        x = cfg.horizontal_padding + rng.randint(0, free_x)
        y_jitter = rng.randint(-2, 2)
        y = max(0, (cfg.image_height - text_height) // 2 - bbox[1] + y_jitter)
        fill = rng.randint(cfg.foreground_min, cfg.foreground_max)

        draw.text((x, y), text, font=font, fill=fill)
        image = self._apply_augmentations(image, rng)

        return image, x, self._char_advances(text, font)

    def _encode_text(self, text: str) -> torch.Tensor:
        target = torch.zeros(self.config.max_text_length, dtype=torch.long)
        encoded = torch.tensor([self.char_to_index[char] for char in text], dtype=torch.long)
        target[: len(encoded)] = encoded
        return target

    def _make_column_targets(self, text: str, advances: list[float], x_offset: int) -> torch.Tensor:
        target = torch.full(
            (self.config.image_width,),
            fill_value=self.char_to_index[self.config.space_char],
            dtype=torch.long,
        )
        cursor = float(x_offset)
        for char, advance in zip(text, advances):
            start = max(0, int(round(cursor)))
            end = min(self.config.image_width, int(round(cursor + advance)))
            if end > start:
                target[start:end] = self.char_to_index[char]
            cursor += advance
        return target

    def _load_font(self, rng: random.Random, size: int | None = None) -> ImageFont.FreeTypeFont:
        font_size = size or rng.randint(self.config.font_size_min, self.config.font_size_max)
        path = rng.choice(self.font_paths)
        return ImageFont.truetype(path, font_size)

    def _apply_augmentations(self, image: Image.Image, rng: random.Random) -> Image.Image:
        probabilities = self._effective_augmentation_probabilities()
        for name in SUPPORTED_AUGMENTATIONS:
            probability = probabilities.get(name, 0.0)
            if probability <= 0.0 or rng.random() > probability:
                continue

            params = self.config.augmentations.get(name, {})
            if name == "rotate":
                image = self._augment_rotate(image, rng, params)
            elif name == "gaussian_blur":
                image = self._augment_gaussian_blur(image, rng, params)
            elif name == "gaussian_noise":
                image = self._augment_gaussian_noise(image, rng, params)
            elif name == "brightness":
                image = self._augment_brightness(image, rng, params)
            elif name == "contrast":
                image = self._augment_contrast(image, rng, params)
            elif name == "invert":
                image = ImageOps.invert(image)

        return image

    def _effective_augmentation_probabilities(self) -> dict[str, float]:
        if self.config.augmentation_probabilities:
            return self.config.augmentation_probabilities

        probabilities: dict[str, float] = {}
        if self.config.max_rotation_degrees:
            probabilities["rotate"] = 1.0
        if self.config.blur_radius:
            probabilities["gaussian_blur"] = 1.0
        if self.config.noise_std:
            probabilities["gaussian_noise"] = 1.0
        return probabilities

    def _augment_rotate(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        max_degrees = float(params.get("max_degrees", self.config.max_rotation_degrees))
        if max_degrees <= 0.0:
            return image
        angle = rng.uniform(-max_degrees, max_degrees)
        fillcolor = int(params.get("fillcolor", self.config.background))
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=fillcolor)

    def _augment_gaussian_blur(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        radius = self._sample_range(rng, params, "radius", self.config.blur_radius)
        if radius <= 0.0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _augment_gaussian_noise(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        std = self._sample_range(rng, params, "std", self.config.noise_std)
        if std <= 0.0:
            return image
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        array = np.asarray(image, dtype=np.float32)
        array += noise_rng.normal(0.0, std, size=array.shape)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")

    def _augment_brightness(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        factor = self._sample_range(rng, params, "factor", 1.0)
        return ImageEnhance.Brightness(image).enhance(factor)

    def _augment_contrast(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        factor = self._sample_range(rng, params, "factor", 1.0)
        return ImageEnhance.Contrast(image).enhance(factor)

    @staticmethod
    def _sample_range(rng: random.Random, params: dict[str, Any], name: str, default: float) -> float:
        if name in params:
            return float(params[name])

        min_name = f"{name}_min"
        max_name = f"{name}_max"
        if min_name in params or max_name in params:
            low = float(params.get(min_name, default))
            high = float(params.get(max_name, default))
            if high < low:
                low, high = high, low
            return rng.uniform(low, high)

        return float(default)

    @staticmethod
    def _char_advances(text: str, font: ImageFont.FreeTypeFont) -> list[float]:
        return [max(1.0, float(font.getlength(char))) for char in text]

    @staticmethod
    def _resolve_font_paths(configured_paths: Iterable[str] | None) -> list[str]:
        paths = list(configured_paths or DEFAULT_FONT_CANDIDATES)
        existing_paths = [str(Path(path)) for path in paths if Path(path).exists()]
        if existing_paths:
            return existing_paths
        raise FileNotFoundError(
            "No usable font files found. Pass font_paths in the dataset config."
        )
