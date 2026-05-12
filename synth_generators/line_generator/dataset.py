from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    font_dir: str | None = None
    font_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_FONT_EXTENSIONS))
    font_size_min: int = Field(default=24, ge=6)
    font_size_max: int = Field(default=34, ge=6)
    channels: int = Field(default=3, ge=1, le=3)
    seed: int | None = None
    background: int = Field(default=255, ge=0, le=255)
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
    overwrite: bool = False

    @classmethod
    def model_validate_with_paths(cls, data: Any, config_path: str | Path | None = None) -> "SingleLineDatasetConfig":
        if config_path is None:
            return cls.model_validate(data)

        data = dict(data)
        config_dir = Path(config_path).resolve().parent
        data["font_paths"] = cls._resolve_relative_paths(data.get("font_paths"), config_dir)

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
        if config.space_char not in self.char_to_index:
            raise ValueError("space_char must be present in alphabet")
        self.sample_alphabet = config.sample_alphabet or config.alphabet
        if not self.sample_alphabet:
            raise ValueError("sample_alphabet must not be empty")
        self.font_paths = self._resolve_font_paths(
            config.font_paths,
            config.font_dir,
            config.font_extensions,
            config.alphabet,
        )
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
        return self.generate_text_sample(text, rng, font)

    def generate_text_sample(
        self,
        text: str,
        rng: random.Random | None = None,
        font: ImageFont.FreeTypeFont | None = None,
    ) -> GeneratedLineSample:
        rng = rng or random.Random()
        self._validate_text(text)
        text = self._normalize_spaces(text)
        if len(text) > self.config.max_text_length:
            raise ValueError(f"text length {len(text)} exceeds max_text_length={self.config.max_text_length}")
        font = font or self._load_font_that_fits(text, rng)
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

    def _validate_text(self, text: str) -> None:
        text = self._normalize_spaces(text)
        if not text:
            raise ValueError("text must not be empty")
        missing = sorted(set(text) - set(self.config.alphabet))
        if missing:
            raise ValueError(f"text contains chars outside alphabet: {missing}")

    def _normalize_spaces(self, text: str) -> str:
        return self.config.space_char.join(part for part in text.split(self.config.space_char) if part)

    def _load_font_that_fits(self, text: str, rng: random.Random) -> ImageFont.FreeTypeFont:
        max_width = self.config.image_width - 2 * self.config.horizontal_padding

        for _ in range(100):
            font = self._load_font(rng)
            if sum(self._char_advances(text, font)) <= max_width:
                return font

        font = self._load_font(rng, size=self.config.font_size_min)
        if sum(self._char_advances(text, font)) > max_width:
            raise ValueError(
                f"text does not fit image_width={self.config.image_width} "
                f"with horizontal_padding={self.config.horizontal_padding}: {text!r}"
            )
        return font

    def _sample_seed(self, index: int) -> int | None:
        if self.config.seed is None:
            return None
        return self.config.seed + index

    def _make_text_that_fits(self, rng: random.Random) -> tuple[str, ImageFont.FreeTypeFont]:
        max_width = self.config.image_width - 2 * self.config.horizontal_padding
        last_candidate: tuple[str, ImageFont.FreeTypeFont] | None = None

        for _ in range(100):
            text_length = rng.randint(self.config.min_text_length, self.config.max_text_length)
            text = self._normalize_spaces("".join(rng.choice(self.sample_alphabet) for _ in range(text_length)))
            if not text:
                continue
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

    @staticmethod
    def _char_advances(text: str, font: ImageFont.FreeTypeFont) -> list[float]:
        return [max(1.0, float(font.getlength(char))) for char in text]

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
            "No usable font files cover the configured alphabet. "
            "Pass font_dir/font_paths with fonts that contain every alphabet character."
        )

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
        except Exception as exc:
            return tuple(alphabet), f"{type(exc).__name__}: {exc}"
        missing_chars = tuple(char for char in alphabet if ord(char) not in codepoints)
        return missing_chars, None

    @staticmethod
    def _font_codepoints(path: Path) -> set[int]:
        try:
            from fontTools.ttLib import TTFont
        except ImportError as exc:
            raise RuntimeError("fontTools is required for reliable font alphabet checks") from exc

        with TTFont(path, fontNumber=0, lazy=True) as font:
            if "cmap" not in font:
                return set()
            codepoints: set[int] = set()
            for table in font["cmap"].tables:
                codepoints.update(table.cmap.keys())
            return codepoints

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
        print(f"  alphabet length: {len(alphabet)}")
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
