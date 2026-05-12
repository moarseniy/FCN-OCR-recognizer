from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

from model import FullyConvTextRecognizer, decode_greedy_batch_tensor
from synth_generators.line_generator.dataset import SingleLineDataset, SingleLineDatasetConfig


DEFAULT_CONFIG = "synth_generators/line_generator/configs/example.yaml"
DEFAULT_DEBUG_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
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


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().float().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image[0]

    if image.shape[0] == 1:
        array = (image[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_debug_font(size: int = 14) -> ImageFont.ImageFont:
    font_path = Path(DEFAULT_DEBUG_FONT)
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def text_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str = "Ag") -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[3] - bbox[1]


def wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    char_width = max(1, draw.textbbox((0, 0), "M", font=font)[2])
    wrap_width = max(8, max_width // char_width)
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=wrap_width) or [""])
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 4,
) -> int:
    x, y = xy
    line_height = text_height(draw, font) + line_gap
    for line in wrapped_lines(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def display_char(char: str) -> str:
    if char == " ":
        return SPACE_SYMBOL
    if char == "\t":
        return "<tab>"
    if char == "\n":
        return "<newline>"
    return char


def format_confidence_pair(candidate: ClassConfidence) -> str:
    return f"{candidate.label} {candidate.confidence:.3f}"


def format_candidate_row(candidates: list[ClassConfidence]) -> str:
    return "    ".join(format_confidence_pair(candidate) for candidate in candidates)


def raw_timestep_summary(result: RecognitionResult) -> str:
    if not result.raw_indices:
        return "<empty>"

    runs: list[str] = []
    start = 0
    current_index = result.raw_indices[0]
    for timestep, class_index in enumerate(result.raw_indices[1:], start=1):
        if class_index == current_index:
            continue
        runs.append(format_raw_run(result, start, timestep - 1))
        start = timestep
        current_index = class_index
    runs.append(format_raw_run(result, start, len(result.raw_indices) - 1))
    return "    ".join(runs)


def format_raw_run(result: RecognitionResult, start: int, end: int) -> str:
    label = result.raw_chars[start]
    avg_confidence = sum(result.raw_confidences[start : end + 1]) / (end - start + 1)
    span = str(start) if start == end else f"{start}-{end}"
    return f"{span} {label} avg {avg_confidence:.3f}"


def save_debug_image(
    source_image: Image.Image,
    result: RecognitionResult,
    output_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font = load_debug_font(16)
    small_font = load_debug_font(14)
    title_font = load_debug_font(22)
    result_font = load_debug_font(20)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    padding = 16
    min_report_width = 1280
    canvas_width = max(min_report_width, source_image.width + padding * 2)

    max_image_width = canvas_width - padding * 2
    image = source_image.convert("RGB")
    if image.width > max_image_width:
        scale = max_image_width / image.width
        image = image.resize((max_image_width, max(1, round(image.height * scale))), Image.Resampling.BICUBIC)

    table_width = canvas_width - padding * 2
    position_width = 56
    char_width = 96
    timestep_width = 96
    confidence_width = 118
    candidates_width = table_width - position_width - char_width - timestep_width - confidence_width
    column_widths = [position_width, char_width, timestep_width, confidence_width, candidates_width]
    column_titles = ["#", "answer", "time", "conf", "ordered candidates"]

    rows = result.decoded_symbols
    line_height = text_height(probe, font) + 12
    row_gap = 3
    row_heights: list[int] = []
    for _ in rows:
        row_heights.append(line_height)
    if not rows:
        row_heights.append(line_height)
    table_height = line_height + sum(row_heights) + row_gap * max(0, len(row_heights) - 1)

    info_lines = [
        f"source: {metadata.get('source', '-')}",
        f"checkpoint: {metadata.get('checkpoint', '-')}",
        f"device: {metadata.get('device', '-')}",
        f"input tensor shape: {result.input_shape}",
        f"logits shape: {result.logits_shape}",
        f"timesteps: {len(result.raw_indices)}",
        f"decoded symbols: {len(result.decoded_symbols)}",
    ]
    if "expected_text" in metadata:
        info_lines.append(f"expected text: {metadata['expected_text']!r}")

    expected_text = metadata.get("expected_text")
    result_lines = wrapped_lines(probe, f"result: {result.text!r}", result_font, table_width)
    expected_lines = wrapped_lines(probe, f"expected: {expected_text!r}", result_font, table_width) if expected_text is not None else []
    result_block_height = (
        len(result_lines) * (text_height(probe, result_font) + 6)
        + len(expected_lines) * (text_height(probe, result_font) + 6)
        + 8
    )
    info_height = len(info_lines) * (text_height(probe, small_font) + 5)
    raw_summary = raw_timestep_summary(result)
    raw_lines = wrapped_lines(probe, f"raw CTC runs: {raw_summary}", small_font, table_width)
    raw_height = len(raw_lines) * (text_height(probe, small_font) + 4)

    report_height = (
        padding
        + text_height(probe, title_font) + 14
        + result_block_height + 12
        + info_height + 14
        + text_height(probe, font) + 8
        + table_height + 14
        + raw_height
        + padding
    )
    canvas_height = padding + image.height + report_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(246, 246, 246))
    draw = ImageDraw.Draw(canvas)

    image_x = (canvas_width - image.width) // 2
    canvas.paste(image, (image_x, padding))
    y = padding + image.height + padding

    draw.text((padding, y), "OCR inference debug", fill=(20, 20, 20), font=title_font)
    y += text_height(draw, title_font) + 14

    result_fill = (20, 90, 40)
    expected_fill = (120, 70, 20)
    if expected_text is not None and expected_text != result.text:
        result_fill = (150, 30, 30)
    for line in result_lines:
        draw.text((padding, y), line, fill=result_fill, font=result_font)
        y += text_height(draw, result_font) + 6
    for line in expected_lines:
        draw.text((padding, y), line, fill=expected_fill, font=result_font)
        y += text_height(draw, result_font) + 6
    y += 8

    for line in info_lines:
        draw.text((padding, y), line, fill=(55, 55, 55), font=small_font)
        y += text_height(draw, small_font) + 5
    y += 10

    draw.text(
        (padding, y),
        f"decoded symbols in output order; each row contains top-{metadata.get('debug_top_k', '-')} candidates sorted by confidence",
        fill=(20, 20, 20),
        font=font,
    )
    y += text_height(draw, font) + 8

    x = padding
    header_y = y
    for title, width in zip(column_titles, column_widths):
        draw.rectangle((x, header_y, x + width, header_y + line_height), fill=(220, 226, 235), outline=(150, 155, 165))
        draw.text((x + 8, header_y + 6), title, fill=(20, 20, 20), font=font)
        x += width
    y += line_height

    if not rows:
        x = padding
        empty_row = ["-", "<empty>", "-", "-", "no decoded non-blank symbols"]
        row_height = row_heights[0]
        for cell, width in zip(empty_row, column_widths):
            draw.rectangle((x, y, x + width, y + row_height), fill=(255, 255, 255), outline=(190, 190, 190))
            draw.text((x + 8, y + 6), cell, fill=(20, 20, 20), font=font)
            x += width
        y += row_height

    for row_index, item in enumerate(rows):
        row_height = row_heights[row_index]
        x = padding
        fill = (255, 255, 255) if row_index % 2 == 0 else (248, 250, 252)
        candidates_text = format_candidate_row(item.candidates)
        cells = [
            str(row_index + 1),
            display_char(item.char),
            "-" if item.timestep < 0 else str(item.timestep),
            "-" if item.timestep < 0 else f"{item.confidence:.4f}",
            candidates_text,
        ]
        for cell_index, (cell, width) in enumerate(zip(cells, column_widths)):
            draw.rectangle((x, y, x + width, y + row_height), fill=fill, outline=(190, 190, 190))
            if cell_index == 4:
                draw.text((x + 6, y + 6), cell, fill=(20, 20, 20), font=font)
            else:
                draw.text((x + 8, y + 6), cell, fill=(20, 20, 20), font=font)
            x += width
        y += row_height + row_gap

    y += 14
    draw_wrapped_text(draw, (padding, y), f"raw CTC runs: {raw_summary}", small_font, (55, 55, 55), table_width)
    canvas.save(output_path)


class TextRecognizer:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.alphabet = self.checkpoint["alphabet"]
        self.idx_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

        model_config = self.checkpoint.get("model_config", {})
        checkpoint_config = self.checkpoint.get("config", {})
        self.in_channels = int(model_config.get("in_channels", 3))
        self.num_classes = int(model_config.get("num_classes", len(self.alphabet) + 1))
        self.blank_idx = int(model_config.get("blank_idx", len(self.alphabet)))
        self.space_char = checkpoint_config.get("space_char", " ")
        self.space_idx = self.alphabet.index(self.space_char) if self.space_char in self.alphabet else None
        self.image_height = int(checkpoint_config.get("image_height", 48))

        self.model = FullyConvTextRecognizer(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        epoch = self.checkpoint.get("epoch", "?")
        loss = self.checkpoint.get("loss")
        loss_text = f", loss: {loss:.8f}" if isinstance(loss, float) else ""
        print(f"Using device: {self.device}")
        print(f"Model loaded from epoch {epoch}{loss_text}")
        print(f"Alphabet size: {len(self.alphabet)}")
        print(f"Blank index: {self.blank_idx}")

    def class_label(self, index: int) -> str:
        if index == self.blank_idx:
            return BLANK_SYMBOL
        return display_char(self.idx_to_char.get(index, f"<{index}>"))

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB" if self.in_channels == 3 else "L")

        if image.height != self.image_height:
            new_width = max(1, round(image.width * self.image_height / image.height))
            image = image.resize((new_width, self.image_height), Image.Resampling.BICUBIC)

        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.in_channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)

        return tensor.unsqueeze(0).to(self.device)

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            return self.preprocess_pil(image)

    def decode_predictions(self, logits: torch.Tensor) -> tuple[str, list[int]]:
        result = self.analyze_logits(logits, input_shape=())
        return result.text, result.raw_indices

    def analyze_logits(self, logits: torch.Tensor, input_shape: tuple[int, ...], top_k: int = 8) -> RecognitionResult:
        probs = torch.softmax(logits, dim=1)
        confidences, pred_ids = probs.max(dim=1)
        collapsed, lengths = decode_greedy_batch_tensor(pred_ids)
        top_k = max(1, min(int(top_k), probs.size(1)))
        top_confidences, top_indices = probs.topk(top_k, dim=1)

        raw_indices = pred_ids[0].detach().cpu().tolist()
        raw_confidences = confidences[0].detach().cpu().tolist()
        raw_chars = [self.class_label(idx) for idx in raw_indices]
        top_candidates_by_timestep: list[list[ClassConfidence]] = []
        for timestep in range(pred_ids.size(1)):
            timestep_candidates: list[ClassConfidence] = []
            for rank in range(top_k):
                class_index = int(top_indices[0, rank, timestep].detach().cpu().item())
                confidence = float(top_confidences[0, rank, timestep].detach().cpu().item())
                timestep_candidates.append(
                    ClassConfidence(
                        label=self.class_label(class_index),
                        confidence=confidence,
                        class_index=class_index,
                    )
                )
            top_candidates_by_timestep.append(timestep_candidates)

        decoded_symbols: list[DecodedSymbol] = []
        keep = torch.ones_like(pred_ids[0], dtype=torch.bool)
        if keep.numel() > 1:
            keep[1:] = pred_ids[0, 1:] != pred_ids[0, :-1]

        for timestep in keep.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
            class_index = raw_indices[timestep]
            if class_index == self.blank_idx:
                continue
            char = self.idx_to_char.get(class_index)
            if char is None:
                continue
            decoded_symbols.append(
                DecodedSymbol(
                    char=char,
                    confidence=float(raw_confidences[timestep]),
                    timestep=int(timestep),
                    class_index=int(class_index),
                    candidates=top_candidates_by_timestep[timestep],
                )
            )

        text = "".join(
            self.idx_to_char[idx]
            for idx in collapsed[0, : lengths[0]].detach().cpu().tolist()
            if idx != self.blank_idx and idx in self.idx_to_char
        )
        return RecognitionResult(
            text=text,
            raw_indices=raw_indices,
            raw_confidences=[float(confidence) for confidence in raw_confidences],
            raw_chars=raw_chars,
            decoded_symbols=decoded_symbols,
            top_candidates_by_timestep=top_candidates_by_timestep,
            input_shape=input_shape,
            logits_shape=tuple(logits.shape),
        )

    @torch.no_grad()
    def recognize_tensor_debug(self, image_tensor: torch.Tensor, top_k: int = 8) -> RecognitionResult:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return self.analyze_logits(logits, input_shape=tuple(image_tensor.shape), top_k=top_k)

    @torch.no_grad()
    def recognize_tensor(self, image_tensor: torch.Tensor) -> tuple[str, list[int]]:
        result = self.recognize_tensor_debug(image_tensor)
        return result.text, result.raw_indices

    def recognize(self, image_path: str | Path) -> tuple[str, list[int]]:
        return self.recognize_tensor(self.preprocess_image(image_path))

    def recognize_image_debug(self, image_path: str | Path, top_k: int = 8) -> RecognitionResult:
        return self.recognize_tensor_debug(self.preprocess_image(image_path), top_k=top_k)


def load_dataset_config(
    config_path: str | Path | None,
    checkpoint_config: dict | None = None,
) -> SingleLineDatasetConfig:
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found: {path}")
        with path.open("r") as file:
            return SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), path)

    if checkpoint_config:
        return SingleLineDatasetConfig.model_validate(checkpoint_config)

    default_path = Path(DEFAULT_CONFIG)
    with default_path.open("r") as file:
        return SingleLineDatasetConfig.model_validate_with_paths(yaml.safe_load(file), default_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FCN OCR inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--image", help="Path to an image file for recognition.")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset config for --sample-index mode. Defaults to checkpoint config, then example config.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Recognize a generated synthetic sample instead of --image.",
    )
    parser.add_argument(
        "--save-sample",
        default="temp.png",
        help="Where to save the generated sample image in --sample-index mode.",
    )
    parser.add_argument("--device", default=None, help="Device to use: cuda or cpu.")
    parser.add_argument("--show-raw", action="store_true", help="Print raw timestep predictions.")
    parser.add_argument(
        "--debug-image",
        default=None,
        help="Optional path to save an annotated inference debug image.",
    )
    parser.add_argument(
        "--debug-top-k",
        type=int,
        default=8,
        help="Number of class-confidence candidates to show per decoded symbol in --debug-image.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    recognizer = TextRecognizer(str(checkpoint_path), args.device)

    if args.image:
        with Image.open(args.image) as image_file:
            source_image = image_file.convert("RGB")
        result = recognizer.recognize_image_debug(args.image, top_k=args.debug_top_k)
        print(f"Image: {args.image}")
        debug_metadata = {
            "source": str(args.image),
            "checkpoint": str(checkpoint_path),
            "device": str(recognizer.device),
            "debug_top_k": args.debug_top_k,
        }
    else:
        sample_index = args.sample_index if args.sample_index is not None else 0
        dataset_config = load_dataset_config(args.config, recognizer.checkpoint.get("config"))
        dataset_config = dataset_config.model_copy(
            update={
                "alphabet": recognizer.alphabet,
                "channels": recognizer.in_channels,
                "image_height": recognizer.image_height,
            }
        )
        dataset = SingleLineDataset(dataset_config)

        rng = random.Random((dataset_config.seed or 0) + sample_index)
        sample = dataset.generate_sample(rng)
        source_image = tensor_to_pil(sample.image)
        source_image.save(args.save_sample)

        result = recognizer.recognize_tensor_debug(sample.image, top_k=args.debug_top_k)
        print(f"Synthetic sample index: {sample_index}")
        print(f"Saved sample image: {args.save_sample}")
        print(f"Expected text: '{sample.text}'")
        debug_metadata = {
            "source": f"synthetic sample index {sample_index}",
            "checkpoint": str(checkpoint_path),
            "device": str(recognizer.device),
            "expected_text": sample.text,
            "debug_top_k": args.debug_top_k,
        }

    print(f"Recognized text: '{result.text}'")

    if args.debug_image:
        save_debug_image(source_image, result, args.debug_image, debug_metadata)
        print(f"Saved debug image: {args.debug_image}")

    if args.show_raw:
        print(f"Raw indices: {result.raw_indices}")
        print(f"Raw chars: {result.raw_chars}")
        print(f"Raw confidences: {[round(confidence, 6) for confidence in result.raw_confidences]}")


if __name__ == "__main__":
    main()
