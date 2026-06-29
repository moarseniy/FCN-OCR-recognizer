from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .results import ClassConfidence, CutDecodingResult, RecognitionResult, VerticalSegmentationResult, display_char


DEFAULT_DEBUG_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


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


def format_confidence_pair(candidate: ClassConfidence) -> str:
    return f"{candidate.label} {candidate.confidence:.3f}"


def format_candidate_row(candidates: list[ClassConfidence]) -> str:
    return "    ".join(format_confidence_pair(candidate) for candidate in candidates)


def format_glyph_width(symbol) -> str:
    if symbol.glyph_width_ratio is None:
        return "-"
    value = f"r={symbol.glyph_width_ratio:.3f}"
    if symbol.glyph_width_score is not None:
        value += f" p={symbol.glyph_width_score:+.3f}"
    return value


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


def segmentation_runs_summary(result: VerticalSegmentationResult) -> str:
    cut_runs = [run for run in result.runs if run.label == 1]
    if not cut_runs:
        return "no cuts"
    return "    ".join(
        f"{run.start} cut={run.score:.3f}"
        for run in cut_runs
    )


def segmentation_x_for_timestep(timestep: float, image_width: int, timesteps: int) -> int:
    if timesteps <= 0:
        return 0
    return max(0, min(image_width - 1, int(round(timestep * image_width / timesteps))))


def draw_segmentation_lines(image: Image.Image, result: VerticalSegmentationResult) -> Image.Image:
    output = image.convert("RGB")
    timesteps = len(result.raw_indices)
    if timesteps <= 0:
        return output

    draw = ImageDraw.Draw(output)

    for run in result.runs:
        if run.label != 1:
            continue

        center = segmentation_x_for_timestep((run.start + run.end + 1) * 0.5, output.width, timesteps)
        draw.line((center, 0, center, output.height - 1), fill=(255, 0, 0), width=1)

    return output


def render_segmentation_panel(
    image: Image.Image,
    result: VerticalSegmentationResult,
) -> Image.Image:
    image = image.convert("RGB")
    image_with_lines = draw_segmentation_lines(image, result)
    font = load_debug_font(14)
    small_font = load_debug_font(12)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    padding = 10
    band_height = 34
    text_line_height = text_height(probe, small_font) + 4
    title_height = text_height(probe, font) + 6
    summary_lines = wrapped_lines(
        probe,
        f"cuts: {segmentation_runs_summary(result)}",
        small_font,
        max(120, image.width - padding * 2),
    )
    panel_height = padding + title_height + image_with_lines.height + 8 + band_height + 8 + text_line_height * len(summary_lines) + padding
    panel = Image.new("RGB", (image.width + padding * 2, panel_height), color=(246, 246, 246))
    draw = ImageDraw.Draw(panel)

    y = padding
    title = (
        f"vertical segmentator input; logits {result.logits_shape}; "
        f"T={len(result.raw_indices)}; "
        f"cut threshold={result.cut_threshold:.3f}"
    )
    draw.text((padding, y), title, fill=(55, 55, 55), font=font)
    y += title_height
    panel.paste(image_with_lines, (padding, y))
    y += image_with_lines.height + 8

    band = Image.new("RGB", (image.width, band_height), color=(235, 235, 235))
    band_draw = ImageDraw.Draw(band)
    timesteps = max(1, len(result.raw_indices))
    for x in range(image.width):
        timestep = min(timesteps - 1, int((x + 0.5) * timesteps / max(1, image.width)))
        cut_score = result.cut_scores[timestep] if result.cut_scores else 0.0
        label = result.raw_indices[timestep] if result.raw_indices else 0
        if label == 1:
            color = (
                255,
                int(round(220 - 140 * cut_score)),
                int(round(210 - 150 * cut_score)),
            )
        else:
            color = (
                int(round(235 - 70 * cut_score)),
                int(round(245 - 40 * cut_score)),
                235,
            )
        band_draw.line((x, 0, x, band_height), fill=color)

    for run in result.runs:
        if run.label != 1:
            continue
        left = int(round(run.start * image.width / timesteps))
        right = int(round((run.end + 1) * image.width / timesteps))
        right = max(left + 1, right)
        band_draw.rectangle((left, 0, right, band_height - 1), outline=(180, 20, 20), width=1)
        center = segmentation_x_for_timestep((run.start + run.end + 1) * 0.5, image.width, timesteps)
        band_draw.line((center, 0, center, band_height - 1), fill=(255, 0, 0), width=1)

    draw.rectangle((padding - 1, y - 1, padding + image.width, y + band_height), outline=(160, 160, 160))
    panel.paste(band, (padding, y))
    y += band_height + 8

    for line in summary_lines:
        draw.text((padding, y), line, fill=(55, 55, 55), font=small_font)
        y += text_line_height
    return panel


def _append_final_input(
    items: list[tuple[str, Image.Image]],
    image: Image.Image | None,
) -> None:
    if image is None:
        return
    if any("final network input" in title.lower() for title, _ in items):
        return
    items.append(("final network input", image))


def _render_stage_column(
    title: str,
    items: list[tuple[str, Image.Image]],
    enabled: bool,
    width: int,
) -> Image.Image:
    padding = 12
    font = load_debug_font(14)
    title_font = load_debug_font(19)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_height = text_height(probe, title_font) + 18
    label_height = text_height(probe, font) + 7
    max_image_width = width - padding * 2
    prepared = [
        (label, image.size, resize_debug_image(image, max_image_width))
        for label, image in items
    ]
    content_height = sum(
        label_height + image.height + padding
        for _, _, image in prepared
    )
    skipped_height = 46 if not enabled else 0
    height = max(180, title_height + padding + content_height + skipped_height)
    panel = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(panel)
    header_fill = (220, 226, 235) if enabled else (232, 232, 232)
    draw.rectangle((0, 0, width - 1, title_height), fill=header_fill)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(170, 175, 182), width=1)
    draw.text((padding, 8), title, fill=(24, 28, 34), font=title_font)

    y = title_height + padding
    for label, original_size, image in prepared:
        draw.text(
            (padding, y),
            f"{label} ({original_size[0]}x{original_size[1]})",
            fill=(55, 55, 55),
            font=font,
        )
        y += label_height
        panel.paste(image, (padding + (max_image_width - image.width) // 2, y))
        y += image.height + padding

    if not enabled:
        draw.rectangle(
            (padding, y, width - padding - 1, y + 32),
            fill=(242, 242, 242),
            outline=(185, 185, 185),
        )
        draw.text((padding + 10, y + 7), "SKIPPED: section is absent or disabled", fill=(105, 105, 105), font=font)
    return panel


def render_pipeline_panel(
    source_image: Image.Image,
    baseline_output_image: Image.Image | None,
    baseline_preprocess_images: list[tuple[str, Image.Image]] | None,
    baseline_enabled: bool,
    segmentator_input_image: Image.Image | None,
    segmentator_preprocess_images: list[tuple[str, Image.Image]] | None,
    segmentation_result: VerticalSegmentationResult | None,
    segmentator_enabled: bool,
    network_input_image: Image.Image | None,
    preprocess_images: list[tuple[str, Image.Image]] | None,
    ocr_enabled: bool,
) -> Image.Image:
    column_width = 560
    gap = 12
    outer_padding = 16
    title_font = load_debug_font(23)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    heading_height = text_height(probe, title_font) + 18

    baseline_items = [("source image", source_image)]
    baseline_items.extend(baseline_preprocess_images or [])
    if baseline_enabled and baseline_output_image is not None:
        baseline_items.append(("shared baseline output", baseline_output_image))

    segmentator_items = list(segmentator_preprocess_images or [])
    _append_final_input(segmentator_items, segmentator_input_image)
    if segmentation_result is not None and segmentator_input_image is not None:
        segmentator_items.append(
            (
                "cut projection and detected vertical lines",
                render_segmentation_panel(segmentator_input_image, segmentation_result),
            )
        )

    ocr_items = list(preprocess_images or [])
    _append_final_input(ocr_items, network_input_image)

    panels = [
        _render_stage_column("1. BASELINE DETECTION", baseline_items, baseline_enabled, column_width),
        _render_stage_column("2. VERTICAL SEGMENTATION", segmentator_items, segmentator_enabled, column_width),
        _render_stage_column("3. OCR", ocr_items, ocr_enabled, column_width),
    ]
    content_height = max(panel.height for panel in panels)
    width = outer_padding * 2 + column_width * 3 + gap * 2
    canvas = Image.new(
        "RGB",
        (width, outer_padding + heading_height + content_height + outer_padding),
        color=(246, 246, 246),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((outer_padding, outer_padding), "Inference pipeline", fill=(20, 20, 20), font=title_font)
    x = outer_padding
    y = outer_padding + heading_height
    for panel in panels:
        canvas.paste(panel, (x, y))
        x += column_width + gap
    return canvas


def _flatten_metadata(metadata: dict[str, Any]) -> list[str]:
    priority = (
        "source",
        "inference_config",
        "device",
        "baseline_status",
        "checkpoint",
        "segmentator_checkpoint",
        "expected_text",
    )
    lines: list[str] = []
    seen: set[str] = set()

    def append_value(key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                append_value(f"{key}.{child_key}", child_value)
            return
        lines.append(f"{key}: {value}")

    for key in priority:
        if key in metadata:
            append_value(key, metadata[key])
            seen.add(key)
    for key in sorted(metadata):
        if key not in seen:
            append_value(key, metadata[key])
    return lines


def _table_block_height(row_count: int, font: ImageFont.ImageFont) -> int:
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    row_height = text_height(probe, font) + 12
    return text_height(probe, font) + 10 + row_height * (max(1, row_count) + 1) + 16


def _draw_table(
    draw: ImageDraw.ImageDraw,
    y: int,
    width: int,
    padding: int,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    empty_message: str,
    font: ImageFont.ImageFont,
) -> int:
    draw.text((padding, y), title, fill=(20, 20, 20), font=font)
    y += text_height(draw, font) + 10
    table_width = width - padding * 2
    fixed_widths = [56, 96, 180, 118]
    column_widths = fixed_widths + [table_width - sum(fixed_widths)]
    row_height = text_height(draw, font) + 12

    x = padding
    for header, column_width in zip(headers, column_widths):
        draw.rectangle(
            (x, y, x + column_width, y + row_height),
            fill=(220, 226, 235),
            outline=(150, 155, 165),
        )
        draw.text((x + 7, y + 6), header, fill=(20, 20, 20), font=font)
        x += column_width
    y += row_height

    table_rows = rows or [["-", "<empty>", "-", "-", empty_message]]
    for row_index, row in enumerate(table_rows):
        x = padding
        fill = (255, 255, 255) if row_index % 2 == 0 else (248, 250, 252)
        for value, column_width in zip(row, column_widths):
            draw.rectangle(
                (x, y, x + column_width, y + row_height),
                fill=fill,
                outline=(190, 190, 190),
            )
            draw.text((x + 7, y + 6), value, fill=(20, 20, 20), font=font)
            x += column_width
        y += row_height
    return y + 16


def save_debug_image(
    source_image: Image.Image,
    result: RecognitionResult | None,
    output_path: str | Path,
    metadata: dict[str, Any],
    network_input_image: Image.Image | None = None,
    preprocess_images: list[tuple[str, Image.Image]] | None = None,
    segmentation_result: VerticalSegmentationResult | None = None,
    segmentator_input_image: Image.Image | None = None,
    cut_decoding_result: CutDecodingResult | None = None,
    baseline_output_image: Image.Image | None = None,
    baseline_preprocess_images: list[tuple[str, Image.Image]] | None = None,
    segmentator_preprocess_images: list[tuple[str, Image.Image]] | None = None,
    baseline_enabled: bool = True,
    segmentator_enabled: bool = True,
    ocr_enabled: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_panel = render_pipeline_panel(
        source_image=source_image,
        baseline_output_image=baseline_output_image,
        baseline_preprocess_images=baseline_preprocess_images,
        baseline_enabled=baseline_enabled,
        segmentator_input_image=segmentator_input_image,
        segmentator_preprocess_images=segmentator_preprocess_images,
        segmentation_result=segmentation_result,
        segmentator_enabled=segmentator_enabled,
        network_input_image=network_input_image,
        preprocess_images=preprocess_images,
        ocr_enabled=ocr_enabled,
    )

    width = pipeline_panel.width
    padding = 16
    table_width = width - padding * 2
    font = load_debug_font(16)
    small_font = load_debug_font(14)
    title_font = load_debug_font(22)
    result_font = load_debug_font(20)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    result_texts: list[tuple[str, tuple[int, int, int]]] = []
    expected_text = metadata.get("expected_text")
    if result is not None:
        result_color = (20, 90, 40)
        if expected_text is not None and expected_text != result.text:
            result_color = (150, 30, 30)
        result_texts.append((f"raw OCR: {result.text!r}", result_color))
    if cut_decoding_result is not None:
        cut_color = (30, 80, 120)
        if expected_text is not None and expected_text != cut_decoding_result.text:
            cut_color = (150, 30, 30)
        result_texts.append((
            f"final OCR with cuts/{cut_decoding_result.decode_method}: {cut_decoding_result.text!r}",
            cut_color,
        ))
    if expected_text is not None:
        result_texts.append((f"expected: {expected_text!r}", (120, 70, 20)))
    if not result_texts:
        result_texts.append(("No OCR output: OCR stage was skipped", (105, 105, 105)))

    result_lines: list[tuple[str, tuple[int, int, int]]] = []
    for text, color in result_texts:
        result_lines.extend(
            (line, color)
            for line in wrapped_lines(probe, text, result_font, table_width)
        )

    metadata_lines: list[str] = []
    for line in _flatten_metadata(metadata):
        metadata_lines.extend(wrapped_lines(probe, line, small_font, table_width))

    if result is not None:
        metadata_lines.extend(
            [
                f"ocr input tensor shape: {result.input_shape}",
                f"ocr logits shape: {result.logits_shape}",
                f"ocr timesteps: {len(result.raw_indices)}",
                f"ocr decoded symbols: {len(result.decoded_symbols)}",
            ]
        )
    if segmentation_result is not None:
        metadata_lines.extend(
            [
                f"segmentator input tensor shape: {segmentation_result.input_shape}",
                f"segmentator logits shape: {segmentation_result.logits_shape}",
                f"segmentator timesteps: {len(segmentation_result.raw_indices)}",
                f"segmentator cuts: {len(segmentation_result.cut_positions or [])}",
                f"segmentator threshold: {segmentation_result.cut_threshold:.4f}",
                f"segmentator min/max width: {segmentation_result.cut_min_width}/{segmentation_result.cut_max_width}",
            ]
        )

    result_line_height = text_height(probe, result_font) + 6
    metadata_line_height = text_height(probe, small_font) + 5
    report_height = (
        padding
        + text_height(probe, title_font)
        + 14
        + len(result_lines) * result_line_height
        + 12
        + len(metadata_lines) * metadata_line_height
        + 12
    )
    if result is not None:
        report_height += _table_block_height(len(result.decoded_symbols), font)
        raw_lines = wrapped_lines(
            probe,
            f"raw OCR runs: {raw_timestep_summary(result)}",
            small_font,
            table_width,
        )
        report_height += len(raw_lines) * (text_height(probe, small_font) + 4) + 14
    else:
        raw_lines = []
    if cut_decoding_result is not None:
        report_height += _table_block_height(len(cut_decoding_result.symbols), font)

    report = Image.new("RGB", (width, report_height), color=(246, 246, 246))
    draw = ImageDraw.Draw(report)
    y = padding
    draw.text((padding, y), "Inference details", fill=(20, 20, 20), font=title_font)
    y += text_height(draw, title_font) + 14
    for line, color in result_lines:
        draw.text((padding, y), line, fill=color, font=result_font)
        y += result_line_height
    y += 8
    for line in metadata_lines:
        draw.text((padding, y), line, fill=(55, 55, 55), font=small_font)
        y += metadata_line_height
    y += 8

    if result is not None:
        ocr_rows = [
            [
                str(index + 1),
                display_char(symbol.char),
                "-" if symbol.timestep < 0 else str(symbol.timestep),
                "-" if symbol.timestep < 0 else f"{symbol.confidence:.4f}",
                format_candidate_row(symbol.candidates),
            ]
            for index, symbol in enumerate(result.decoded_symbols)
        ]
        y = _draw_table(
            draw,
            y,
            width,
            padding,
            f"Raw OCR symbols; top-{metadata.get('debug_top_k', '-')} candidates in confidence order",
            ["#", "answer", "time", "conf", "ordered candidates"],
            ocr_rows,
            "no decoded symbols",
            font,
        )

    if cut_decoding_result is not None:
        cut_rows: list[list[str]] = []
        for index, symbol in enumerate(cut_decoding_result.symbols):
            span = f"{symbol.start}-{symbol.end - 1}" if symbol.end > symbol.start else "-"
            if symbol.score_start is not None and symbol.score_end is not None:
                score_span = (
                    f"{symbol.score_start}-{symbol.score_end - 1}"
                    if symbol.score_end > symbol.score_start
                    else "-"
                )
                if score_span != span:
                    span = f"{span} / score {score_span}"
            cut_rows.append(
                [
                    str(index + 1),
                    display_char(symbol.char),
                    span,
                    f"{symbol.confidence:.4f}",
                    format_glyph_width(symbol),
                    format_candidate_row(symbol.candidates),
                ]
            )
        y = _draw_table(
            draw,
            y,
            width,
            padding,
            f"Final OCR symbols ({cut_decoding_result.decode_method}); one class per selected cut interval",
            ["#", "answer", "ocr span", "conf", "glyph width", "ordered candidates"],
            cut_rows,
            "no intervals decoded from segmentator cuts",
            font,
        )

    for line in raw_lines:
        draw.text((padding, y), line, fill=(55, 55, 55), font=small_font)
        y += text_height(draw, small_font) + 4

    canvas = Image.new(
        "RGB",
        (width, pipeline_panel.height + report.height),
        color=(246, 246, 246),
    )
    canvas.paste(pipeline_panel, (0, 0))
    canvas.paste(report, (0, pipeline_panel.height))
    canvas.save(output_path)
