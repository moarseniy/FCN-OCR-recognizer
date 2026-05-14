from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from model import FullyConvTextRecognizer, decode_greedy_batch_tensor


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


def resize_debug_image(image: Image.Image, max_width: int) -> Image.Image:
    image = image.convert("RGB")
    if image.width <= max_width:
        return image

    scale = max_width / image.width
    return image.resize((max_width, max(1, round(image.height * scale))), Image.Resampling.BICUBIC)


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
    network_input_image: Image.Image | None = None,
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
    canvas_width = max(
        min_report_width,
        source_image.width + padding * 2,
        (network_input_image.width + padding * 2) if network_input_image is not None else 0,
    )

    max_image_width = canvas_width - padding * 2
    image = resize_debug_image(source_image, max_image_width)
    input_image = resize_debug_image(network_input_image, max_image_width) if network_input_image is not None else None

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
    if "scale_x" in metadata:
        info_lines.append(f"scale_x: {float(metadata['scale_x']):+.4f}")
    if "y_pad" in metadata:
        info_lines.append(f"y_pad: {float(metadata['y_pad']):+.4f}")
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
    image_title_height = text_height(probe, small_font) + 6
    images_height = image_title_height + image.height
    if input_image is not None:
        images_height += padding + image_title_height + input_image.height

    canvas_height = padding + images_height + report_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(246, 246, 246))
    draw = ImageDraw.Draw(canvas)

    y = padding
    draw.text(
        (padding, y),
        f"original image ({source_image.width}x{source_image.height})",
        fill=(55, 55, 55),
        font=small_font,
    )
    y += image_title_height
    image_x = (canvas_width - image.width) // 2
    canvas.paste(image, (image_x, y))
    y += image.height

    if input_image is not None:
        y += padding
        draw.text(
            (padding, y),
            f"network input image ({network_input_image.width}x{network_input_image.height})",
            fill=(55, 55, 55),
            font=small_font,
        )
        y += image_title_height
        input_x = (canvas_width - input_image.width) // 2
        canvas.paste(input_image, (input_x, y))
        y += input_image.height

    y += padding

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
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        verbose: bool = False,
        scale_x: float = 0.0,
        y_pad: float = 0.0,
    ):
        if scale_x <= -0.95:
            raise ValueError("scale_x must be > -0.95")
        if y_pad <= -0.95:
            raise ValueError("y_pad must be > -0.95")

        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.scale_x = float(scale_x)
        self.y_pad = float(y_pad)

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
        self.preprocess_fill = int(checkpoint_config.get("background", 255))

        self.model = FullyConvTextRecognizer(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        if verbose:
            self.print_summary()

    def print_summary(self) -> None:
        epoch = self.checkpoint.get("epoch", "?")
        loss = self.checkpoint.get("loss")
        loss_text = f", loss: {loss:.8f}" if isinstance(loss, float) else ""
        print(f"Using device: {self.device}")
        print(f"Model loaded from epoch {epoch}{loss_text}")
        print(f"Alphabet size: {len(self.alphabet)}")
        print(f"Blank index: {self.blank_idx}")
        print(f"Preprocess scale_x: {self.scale_x:+.4f}")
        print(f"Preprocess y_pad:   {self.y_pad:+.4f}")

    def class_label(self, index: int) -> str:
        if index == self.blank_idx:
            return BLANK_SYMBOL
        return display_char(self.idx_to_char.get(index, f"<{index}>"))

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess_pil_3d(image).unsqueeze(0)

    def _preprocess_pil_3d(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB" if self.in_channels == 3 else "L")
        image = self._apply_y_pad(image)

        if image.height != self.image_height:
            new_width = max(1, round(image.width * self.image_height / image.height))
            image = image.resize((new_width, self.image_height), Image.Resampling.BICUBIC)

        image = self._apply_scale_x(image)

        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.in_channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)

        return tensor.to(self.device)

    def _apply_y_pad(self, image: Image.Image) -> Image.Image:
        if self.y_pad == 0.0:
            return image

        delta = int(round(image.height * abs(self.y_pad)))
        if delta <= 0:
            return image

        top = delta // 2
        bottom = delta - top
        if self.y_pad > 0.0:
            return ImageOps.expand(image, border=(0, top, 0, bottom), fill=self._pil_fill_value(image.mode))

        if delta >= image.height:
            delta = image.height - 1
            top = delta // 2
            bottom = delta - top
        return image.crop((0, top, image.width, image.height - bottom))

    def _apply_scale_x(self, image: Image.Image) -> Image.Image:
        if self.scale_x == 0.0:
            return image

        factor = 1.0 + self.scale_x
        new_width = max(1, round(image.width * factor))
        if new_width == image.width:
            return image
        return image.resize((new_width, image.height), Image.Resampling.BICUBIC)

    def _pil_fill_value(self, mode: str) -> int | tuple[int, int, int]:
        fill = max(0, min(255, self.preprocess_fill))
        if mode == "RGB":
            return (fill, fill, fill)
        return fill

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            return self.preprocess_pil(image)

    def decode_predictions(self, logits: torch.Tensor) -> tuple[str, list[int]]:
        pred_ids = logits.argmax(dim=1)
        return self.decode_pred_ids_batch(pred_ids)[0]

    def decode_pred_ids_batch(
        self,
        pred_ids: torch.Tensor,
        input_lengths: list[int] | torch.Tensor | None = None,
    ) -> list[tuple[str, list[int]]]:
        decoded: list[tuple[str, list[int]]] = []
        if input_lengths is None:
            lengths = [pred_ids.size(1)] * pred_ids.size(0)
        elif isinstance(input_lengths, torch.Tensor):
            lengths = [int(length) for length in input_lengths.detach().cpu().tolist()]
        else:
            lengths = [int(length) for length in input_lengths]

        for row, length in zip(pred_ids, lengths):
            raw_ids = row[: max(0, length)].detach().cpu().tolist()
            collapsed_ids: list[int] = []
            previous_id: int | None = None
            for class_index in raw_ids:
                if class_index != previous_id:
                    collapsed_ids.append(class_index)
                previous_id = class_index

            text = "".join(
                self.idx_to_char[class_index]
                for class_index in collapsed_ids
                if class_index != self.blank_idx and class_index in self.idx_to_char
            )
            decoded.append((text, raw_ids))
        return decoded

    def output_width_for_input_width(self, width: int) -> int:
        output_width = int(width)
        for module in self.model.modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue

            kernel = module.kernel_size[1]
            stride = module.stride[1]
            padding = module.padding[1]
            dilation = module.dilation[1]
            output_width = (output_width + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
        return output_width

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
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return self.decode_predictions(logits)

    def recognize(self, image_path: str | Path) -> tuple[str, list[int]]:
        return self.recognize_tensor(self.preprocess_image(image_path))

    def recognize_image_debug(self, image_path: str | Path, top_k: int = 8) -> RecognitionResult:
        return self.recognize_tensor_debug(self.preprocess_image(image_path), top_k=top_k)

    def recognize_paths(self, image_paths: Iterable[str | Path], top_k: int = 8) -> list[tuple[Path, RecognitionResult]]:
        results: list[tuple[Path, RecognitionResult]] = []
        for image_path in image_paths:
            path = Path(image_path)
            results.append((path, self.recognize_image_debug(path, top_k=top_k)))
        return results

    @torch.no_grad()
    def recognize_paths_text(
        self,
        image_paths: Iterable[str | Path],
        batch_size: int = 32,
    ) -> list[tuple[Path, str]]:
        paths = [Path(image_path) for image_path in image_paths]
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        results: list[tuple[Path, str]] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors: list[torch.Tensor] = []
            output_lengths: list[int] = []
            max_width = 0

            for path in batch_paths:
                with Image.open(path) as image:
                    tensor = self._preprocess_pil_3d(image)
                tensors.append(tensor)
                max_width = max(max_width, tensor.size(2))
                output_lengths.append(self.output_width_for_input_width(tensor.size(2)))

            if not tensors:
                continue

            batch = torch.ones(
                (len(tensors), self.in_channels, self.image_height, max_width),
                dtype=tensors[0].dtype,
                device=self.device,
            )
            for batch_index, tensor in enumerate(tensors):
                batch[batch_index, :, :, : tensor.size(2)] = tensor

            logits = self.model(batch)
            pred_ids = logits.argmax(dim=1)
            decoded = self.decode_pred_ids_batch(pred_ids, output_lengths)
            results.extend((path, text) for path, (text, _) in zip(batch_paths, decoded))

        return results
