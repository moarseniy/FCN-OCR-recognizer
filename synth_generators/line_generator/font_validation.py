from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
import yaml

from .dataset import SingleLineDataset, SingleLineDatasetConfig


DEFAULT_OUTPUT_DIR = Path("output") / "font_validation"
DEFAULT_TITLE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render generation-config alphabet previews for every configured font."
    )
    parser.add_argument("--config", required=True, help="Path to generation YAML config.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to save font validation images.",
    )
    parser.add_argument("--font-size", type=int, default=44, help="Font size for alphabet glyphs.")
    parser.add_argument("--columns", type=int, default=12, help="Grid columns in each validation image.")
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Render only fonts that cover every character in the config alphabet.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke checks.")
    return parser.parse_args()


def load_config(config_path: str | Path) -> SingleLineDatasetConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return SingleLineDatasetConfig.model_validate_with_paths(data, path)


def load_label_font(size: int) -> ImageFont.ImageFont:
    path = Path(DEFAULT_TITLE_FONT)
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def sanitize_filename(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return stem or "font"


def display_char_label(char: str) -> str:
    if char == " ":
        return "<space>"
    if char == "\t":
        return "<tab>"
    if char == "\n":
        return "<newline>"
    return char


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + max(0, (right - left - width) // 2) - bbox[0]
    y = top + max(0, (bottom - top - height) // 2) - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def missing_chars_text(missing_chars: Iterable[str]) -> str:
    values = [display_char_label(char) for char in missing_chars]
    if not values:
        return "missing: none"
    preview = ", ".join(values[:24])
    if len(values) > 24:
        preview = f"{preview}, ... +{len(values) - 24}"
    return f"missing: {preview}"


def render_font_validation_image(
    font_path: Path,
    alphabet: str,
    missing_chars: tuple[str, ...],
    error: str | None,
    font_size: int,
    columns: int,
) -> Image.Image:
    columns = max(1, int(columns))
    cell_width = 116
    cell_height = 104
    margin = 24
    title_height = 112
    rows = max(1, (len(alphabet) + columns - 1) // columns)
    width = margin * 2 + columns * cell_width
    height = margin * 2 + title_height + rows * cell_height

    image = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(image)
    title_font = load_label_font(22)
    label_font = load_label_font(13)
    meta_font = load_label_font(14)

    title = font_path.name
    draw.text((margin, margin), title, font=title_font, fill=(25, 25, 25))
    draw.text((margin, margin + 34), str(font_path), font=meta_font, fill=(80, 80, 80))
    status = error or missing_chars_text(missing_chars)
    status_color = (150, 40, 40) if error or missing_chars else (35, 110, 55)
    draw.text((margin, margin + 58), status, font=meta_font, fill=status_color)
    draw.text(
        (margin, margin + 80),
        f"alphabet length: {len(alphabet)}",
        font=meta_font,
        fill=(80, 80, 80),
    )

    try:
        glyph_font = ImageFont.truetype(str(font_path), font_size)
    except Exception as exc:
        draw.text(
            (margin, margin + title_height),
            f"failed to load font with Pillow: {type(exc).__name__}: {exc}",
            font=meta_font,
            fill=(150, 40, 40),
        )
        return image

    grid_top = margin + title_height
    missing_set = set(missing_chars)
    for index, char in enumerate(alphabet):
        row = index // columns
        column = index % columns
        x0 = margin + column * cell_width
        y0 = grid_top + row * cell_height
        x1 = x0 + cell_width - 8
        y1 = y0 + cell_height - 8
        is_missing = char in missing_set
        fill = (255, 245, 245) if is_missing else (255, 255, 255)
        outline = (220, 120, 120) if is_missing else (205, 205, 205)
        draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline)

        glyph_box = (x0 + 8, y0 + 8, x1 - 8, y0 + 64)
        if char == " ":
            draw.line((glyph_box[0] + 16, y0 + 38, glyph_box[2] - 16, y0 + 38), fill=(150, 150, 150), width=2)
        else:
            draw_centered_text(draw, glyph_box, char, glyph_font, fill=(20, 20, 20))

        label = f"{index:02d} {display_char_label(char)}"
        draw_centered_text(draw, (x0 + 6, y0 + 70, x1 - 6, y1 - 8), label, label_font, fill=(70, 70, 70))

    return image


def unique_output_path(output_dir: Path, font_path: Path, index: int) -> Path:
    stem = sanitize_filename(font_path.stem)
    suffix = sanitize_filename(font_path.suffix.lstrip("."))
    filename = f"{index:04d}_{stem}"
    if suffix:
        filename = f"{filename}_{suffix}"
    return output_dir / f"{filename}.png"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    alphabet = config.sample_alphabet or config.alphabet
    if not alphabet:
        raise ValueError("generation config alphabet is empty")

    font_paths = SingleLineDataset._collect_unchecked_font_paths(
        config.font_paths,
        config.font_dir,
        config.font_extensions,
    )
    if args.limit is not None:
        font_paths = font_paths[: max(0, args.limit)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for index, font_path_str in enumerate(font_paths):
        font_path = Path(font_path_str)
        missing_chars, error = SingleLineDataset._missing_font_chars(font_path, alphabet)
        if args.accepted_only and (error is not None or missing_chars):
            skipped += 1
            continue
        image = render_font_validation_image(
            font_path=font_path,
            alphabet=alphabet,
            missing_chars=missing_chars,
            error=error,
            font_size=args.font_size,
            columns=args.columns,
        )
        image.save(unique_output_path(output_dir, font_path, index))
        saved += 1

    print(f"Font validation saved {saved} images to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} fonts because --accepted-only was set")


if __name__ == "__main__":
    main()
