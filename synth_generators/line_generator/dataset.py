from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch
from torch.utils.data import Dataset


DEFAULT_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/comfortaa/Comfortaa-Regular.ttf",
)


class SingleLineDatasetConfig(BaseModel):
    """Config for a simple fully-convolutional single-line OCR dataset.

    The dataset returns fixed-width images and padded text-sequence targets:
    ``image`` has shape ``(C, H, W)``, ``target`` has shape
    ``(max_text_length,)`` and ``length`` is the real text length.
    This is intended for CTC training on FCN outputs over image width.
    """

    model_config = ConfigDict(extra="ignore")

    alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyz"
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
    horizontal_padding: int = Field(default=8, ge=0)

    @field_validator("alphabet")
    @classmethod
    def alphabet_must_be_unique(cls, value: str) -> str:
        if not value:
            raise ValueError("alphabet must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("alphabet must contain unique characters")
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
        image = self._render_text(text, font, rng)
        target = self._encode_text(text)

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
            length=len(text),
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
            text = "".join(rng.choice(self.config.alphabet) for _ in range(text_length))
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
    ) -> Image.Image:
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

        if cfg.max_rotation_degrees:
            angle = rng.uniform(-cfg.max_rotation_degrees, cfg.max_rotation_degrees)
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=cfg.background)

        if cfg.blur_radius:
            image = image.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))

        if cfg.noise_std:
            noise_rng = np.random.default_rng(rng.randrange(2**32))
            array = np.asarray(image, dtype=np.float32)
            array += noise_rng.normal(0.0, cfg.noise_std, size=array.shape)
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")

        return image

    def _encode_text(self, text: str) -> torch.Tensor:
        target = torch.zeros(self.config.max_text_length, dtype=torch.long)
        encoded = torch.tensor([self.char_to_index[char] for char in text], dtype=torch.long)
        target[: len(encoded)] = encoded
        return target

    def _load_font(self, rng: random.Random, size: int | None = None) -> ImageFont.FreeTypeFont:
        font_size = size or rng.randint(self.config.font_size_min, self.config.font_size_max)
        path = rng.choice(self.font_paths)
        return ImageFont.truetype(path, font_size)

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
