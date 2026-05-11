from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable

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

DEFAULT_BACKGROUND_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

SUPPORTED_AUGMENTATIONS = (
    "cycle_shift",
    "strong_blur",
    "motion_blur",
    "scale",
    "darkening",
    "noise",
    "projective",
    "rotate",
    "crop_x",
    "crop_y",
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

    alphabet: str = " 0123456789abcdefghijklmnopqrstuvwxyz"
    sample_alphabet: str | None = None
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
    background_dir: str | None = None
    background_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_BACKGROUND_EXTENSIONS))
    foreground_min: int = Field(default=0, ge=0, le=255)
    foreground_max: int = Field(default=60, ge=0, le=255)
    noise_std: float = Field(default=4.0, ge=0.0)
    blur_radius: float = Field(default=0.15, ge=0.0)
    max_rotation_degrees: float = Field(default=1.0, ge=0.0)
    augmentation_probabilities: dict[str, float] = Field(default_factory=dict)
    augmentations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    horizontal_padding: int = Field(default=8, ge=0)
    output_dir: str | None = None
    chunk_size: int = Field(default=1024, ge=1)
    overwrite: bool = False
    apply_augmentations: bool = False

    @classmethod
    def model_validate_with_paths(cls, data: Any, config_path: str | Path | None = None) -> "SingleLineDatasetConfig":
        if config_path is None:
            return cls.model_validate(data)

        data = dict(data)
        config_dir = Path(config_path).resolve().parent
        data["font_paths"] = cls._resolve_relative_paths(data.get("font_paths"), config_dir)

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

    @field_validator("background_extensions")
    @classmethod
    def background_extensions_must_be_valid(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("background_extensions must not be empty")
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
        self.sample_alphabet = config.sample_alphabet or config.alphabet.replace(config.space_char, "")
        if not self.sample_alphabet:
            raise ValueError("sample_alphabet must not be empty")
        self.font_paths = self._resolve_font_paths(config.font_paths)
        self.background_paths = self._resolve_background_paths(
            config.background_dir,
            config.background_extensions,
        )

    def __len__(self) -> int:
        return self.config.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.generate_sample_from_index(index)
        return sample.image, sample.target, torch.tensor(sample.length, dtype=torch.long)

    def generate_sample_from_index(self, index: int) -> GeneratedLineSample:
        rng = random.Random(self._sample_seed(index))
        return self.generate_sample(rng)

    def generate_sample(self, rng: random.Random | None = None) -> GeneratedLineSample:
        rng = rng or random.Random()
        text, font = self._make_text_that_fits(rng)
        image = self._render_text(text, font, rng)
        target = self._encode_text(text)
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
    ) -> Image.Image:
        cfg = self.config
        image = self._make_background(rng)
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

    def _make_background(self, rng: random.Random) -> Image.Image:
        cfg = self.config
        if not self.background_paths:
            return Image.new("L", (cfg.image_width, cfg.image_height), color=cfg.background)

        path = rng.choice(self.background_paths)
        with Image.open(path) as background_image:
            background_image = background_image.convert("L")
            return self._random_crop_or_resize_background(background_image, rng)

    def _random_crop_or_resize_background(self, image: Image.Image, rng: random.Random) -> Image.Image:
        cfg = self.config
        target_width = cfg.image_width
        target_height = cfg.image_height

        scale = max(target_width / image.width, target_height / image.height)
        resized_width = max(target_width, int(round(image.width * scale)))
        resized_height = max(target_height, int(round(image.height * scale)))
        image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

        max_left = resized_width - target_width
        max_top = resized_height - target_height
        left = rng.randint(0, max_left) if max_left > 0 else 0
        top = rng.randint(0, max_top) if max_top > 0 else 0
        return image.crop((left, top, left + target_width, top + target_height))

    def _apply_augmentations(self, image: Image.Image, rng: random.Random) -> Image.Image:
        probabilities = self._effective_augmentation_probabilities()
        for name in SUPPORTED_AUGMENTATIONS:
            probability = probabilities.get(name, 0.0)
            if probability <= 0.0 or rng.random() > probability:
                continue

            params = self.config.augmentations.get(name, {})
            if name == "cycle_shift":
                image = self._augment_cycle_shift(image, rng, params)
            elif name == "strong_blur":
                image = self._augment_strong_blur(image, rng, params)
            elif name == "motion_blur":
                image = self._augment_motion_blur(image, rng, params)
            elif name == "scale":
                image = self._augment_scale(image, rng, params)
            elif name == "darkening":
                image = self._augment_darkening(image, rng, params)
            elif name == "noise":
                image = self._augment_noise(image, rng, params)
            elif name == "projective":
                image = self._augment_projective(image, rng, params)
            elif name == "rotate":
                image = self._augment_rotate(image, rng, params)
            elif name == "crop_x":
                image = self._augment_crop_x(image, rng, params)
            elif name == "crop_y":
                image = self._augment_crop_y(image, rng, params)
            elif name == "morphology":
                image = self._augment_morphology(image, rng, params)
            elif name == "unsharp_mask":
                image = self._augment_unsharp_mask(image, rng, params)
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
            probabilities["noise"] = 1.0
        return probabilities

    def _augment_cycle_shift(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        max_x = int(params.get("max_x", 0))
        max_y = int(params.get("max_y", 0))
        shift_x = rng.randint(-max_x, max_x) if max_x > 0 else 0
        shift_y = rng.randint(-max_y, max_y) if max_y > 0 else 0
        if shift_x == 0 and shift_y == 0:
            return image
        array = np.asarray(image)
        array = np.roll(array, shift=(shift_y, shift_x), axis=(0, 1))
        return Image.fromarray(array.astype(np.uint8), mode="L")

    def _augment_strong_blur(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        radius = self._sample_range(rng, params, "radius", 1.2)
        if radius <= 0.0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _augment_motion_blur(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        size = int(round(self._sample_range(rng, params, "size", 5)))
        if size <= 1:
            return image
        if size % 2 == 0:
            size += 1

        angle = self._sample_range(rng, params, "angle", 0.0)
        kernel = self._make_motion_kernel(size, angle)
        return self._filter_with_kernel(image, kernel)

    def _augment_scale(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        factor_x = self._sample_range(rng, params, "factor_x", self._sample_range(rng, params, "factor", 1.0))
        factor_y = self._sample_range(rng, params, "factor_y", self._sample_range(rng, params, "factor", 1.0))
        if factor_x <= 0.0 or factor_y <= 0.0:
            return image

        width, height = image.size
        scaled_width = max(1, int(round(width * factor_x)))
        scaled_height = max(1, int(round(height * factor_y)))
        scaled = image.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)
        return self._fit_to_canvas(scaled, width, height, int(params.get("fillcolor", self.config.background)))

    def _augment_darkening(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        factor = self._sample_range(rng, params, "factor", 0.75)
        return ImageEnhance.Brightness(image).enhance(factor)

    def _augment_noise(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        kind = params.get("kind", "gaussian")
        if kind == "salt_pepper":
            amount = self._sample_range(rng, params, "amount", 0.01)
            return self._augment_salt_pepper_noise(image, rng, amount)
        return self._augment_gaussian_noise(image, rng, params)

    def _augment_projective(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        max_dx = self._sample_range(rng, params, "max_dx", 4.0)
        max_dy = self._sample_range(rng, params, "max_dy", 2.0)
        width, height = image.size
        src = [(0, 0), (width, 0), (width, height), (0, height)]
        dst = [
            (rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)),
            (width + rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)),
            (width + rng.uniform(-max_dx, max_dx), height + rng.uniform(-max_dy, max_dy)),
            (rng.uniform(-max_dx, max_dx), height + rng.uniform(-max_dy, max_dy)),
        ]
        coefficients = self._find_perspective_coefficients(dst, src)
        fillcolor = int(params.get("fillcolor", self.config.background))
        return image.transform(
            image.size,
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=fillcolor,
        )

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

    def _augment_crop_x(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        max_left = int(round(self._sample_range(rng, params, "left", float(params.get("max_left", 0)))))
        max_right = int(round(self._sample_range(rng, params, "right", float(params.get("max_right", 0)))))
        crop_left = rng.randint(0, max(0, max_left))
        crop_right = rng.randint(0, max(0, max_right))
        if crop_left == 0 and crop_right == 0:
            return image
        width, height = image.size
        right = max(crop_left + 1, width - crop_right)
        cropped = image.crop((crop_left, 0, right, height))
        canvas = Image.new("L", (width, height), color=int(params.get("fillcolor", self.config.background)))
        canvas.paste(cropped, (0, 0))
        return canvas

    def _augment_crop_y(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        max_top = int(round(self._sample_range(rng, params, "top", float(params.get("max_top", 0)))))
        max_bottom = int(round(self._sample_range(rng, params, "bottom", float(params.get("max_bottom", 0)))))
        crop_top = rng.randint(0, max(0, max_top))
        crop_bottom = rng.randint(0, max(0, max_bottom))
        if crop_top == 0 and crop_bottom == 0:
            return image
        width, height = image.size
        bottom = max(crop_top + 1, height - crop_bottom)
        cropped = image.crop((0, crop_top, width, bottom))
        canvas = Image.new("L", (width, height), color=int(params.get("fillcolor", self.config.background)))
        canvas.paste(cropped, (0, 0))
        return canvas

    def _augment_morphology(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        size = int(round(self._sample_range(rng, params, "size", 3)))
        if size <= 1:
            return image
        if size % 2 == 0:
            size += 1

        operation = params.get("operation", "random")
        if operation == "random":
            operation = rng.choice(["dilate", "erode", "open", "close"])

        if operation == "dilate":
            return image.filter(ImageFilter.MinFilter(size=size))
        if operation == "erode":
            return image.filter(ImageFilter.MaxFilter(size=size))
        if operation == "open":
            return image.filter(ImageFilter.MaxFilter(size=size)).filter(ImageFilter.MinFilter(size=size))
        if operation == "close":
            return image.filter(ImageFilter.MinFilter(size=size)).filter(ImageFilter.MaxFilter(size=size))
        return image

    def _augment_unsharp_mask(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        radius = self._sample_range(rng, params, "radius", 1.0)
        percent = int(round(self._sample_range(rng, params, "percent", 120.0)))
        threshold = int(round(self._sample_range(rng, params, "threshold", 3.0)))
        return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    def _augment_brightness(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        factor = self._sample_range(rng, params, "factor", 1.0)
        return ImageEnhance.Brightness(image).enhance(factor)

    def _augment_contrast(self, image: Image.Image, rng: random.Random, params: dict[str, Any]) -> Image.Image:
        factor = self._sample_range(rng, params, "factor", 1.0)
        return ImageEnhance.Contrast(image).enhance(factor)

    def _augment_salt_pepper_noise(self, image: Image.Image, rng: random.Random, amount: float) -> Image.Image:
        if amount <= 0.0:
            return image
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        array = np.asarray(image, dtype=np.uint8).copy()
        mask = noise_rng.random(array.shape)
        amount = min(max(amount, 0.0), 1.0)
        array[mask < amount / 2.0] = 0
        array[(mask >= amount / 2.0) & (mask < amount)] = 255
        return Image.fromarray(array, mode="L")

    def _fit_to_canvas(self, image: Image.Image, width: int, height: int, fillcolor: int) -> Image.Image:
        canvas = Image.new("L", (width, height), color=fillcolor)

        crop_left = max(0, (image.width - width) // 2)
        crop_top = max(0, (image.height - height) // 2)
        image = image.crop((crop_left, crop_top, crop_left + min(width, image.width), crop_top + min(height, image.height)))

        paste_x = max(0, (width - image.width) // 2)
        paste_y = max(0, (height - image.height) // 2)
        canvas.paste(image, (paste_x, paste_y))
        return canvas

    @staticmethod
    def _make_motion_kernel(size: int, angle: float) -> np.ndarray:
        kernel = np.zeros((size, size), dtype=np.float32)
        center = (size - 1) / 2.0
        radians = np.deg2rad(angle)
        dx = np.cos(radians)
        dy = np.sin(radians)

        for step in np.linspace(-center, center, size):
            x = int(round(center + step * dx))
            y = int(round(center + step * dy))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0

        if kernel.sum() == 0:
            kernel[size // 2, :] = 1.0
        return kernel / kernel.sum()

    @staticmethod
    def _filter_with_kernel(image: Image.Image, kernel: np.ndarray) -> Image.Image:
        array = np.asarray(image, dtype=np.float32)
        kernel = np.asarray(kernel, dtype=np.float32)
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2
        padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        filtered = np.zeros_like(array, dtype=np.float32)

        for y in range(kernel.shape[0]):
            for x in range(kernel.shape[1]):
                weight = kernel[y, x]
                if weight == 0:
                    continue
                filtered += weight * padded[y : y + array.shape[0], x : x + array.shape[1]]

        return Image.fromarray(np.clip(filtered, 0, 255).astype(np.uint8), mode="L")

    @staticmethod
    def _find_perspective_coefficients(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list[float]:
        matrix = []
        vector = []
        for (src_x, src_y), (dst_x, dst_y) in zip(src, dst):
            matrix.append([src_x, src_y, 1, 0, 0, 0, -dst_x * src_x, -dst_x * src_y])
            matrix.append([0, 0, 0, src_x, src_y, 1, -dst_y * src_x, -dst_y * src_y])
            vector.extend([dst_x, dst_y])
        return np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64)).tolist()

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

    @staticmethod
    def _resolve_background_paths(background_dir: str | None, extensions: Iterable[str]) -> list[str]:
        if background_dir is None:
            return []

        root = Path(background_dir)
        if not root.exists():
            raise FileNotFoundError(f"background_dir does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"background_dir is not a directory: {root}")

        normalized_extensions = {extension.lower() for extension in extensions}
        paths = [
            str(path)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in normalized_extensions
        ]
        if not paths:
            raise FileNotFoundError(
                f"No background images found in {root}. "
                f"Supported extensions: {sorted(normalized_extensions)}"
            )
        return paths
