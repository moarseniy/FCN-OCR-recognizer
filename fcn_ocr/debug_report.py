from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .results import ClassConfidence, RecognitionResult, VerticalSegmentationResult, display_char


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
    gap_runs = [run for run in result.runs if run.label == 1]
    if not gap_runs:
        return "no cuts" if result.mode == "cut_projection" else "no gap runs"
    if result.mode == "cut_projection":
        return "    ".join(
            f"{run.start} cut={run.gap_probability:.3f}"
            for run in gap_runs
        )
    return "    ".join(
        f"{run.start}-{run.end} gap={run.gap_probability:.3f} conf={run.confidence:.3f}"
        for run in gap_runs
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

    overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    line_width = max(2, round(output.width / 260))

    for run in result.runs:
        if run.label != 1:
            continue

        left = segmentation_x_for_timestep(float(run.start), output.width, timesteps)
        right = segmentation_x_for_timestep(float(run.end + 1), output.width, timesteps)
        if right <= left:
            right = min(output.width - 1, left + 1)
        center = segmentation_x_for_timestep((run.start + run.end + 1) * 0.5, output.width, timesteps)

        overlay_draw.rectangle((left, 0, right, output.height), fill=(255, 30, 30, 42))
        overlay_draw.line((center, 0, center, output.height), fill=(0, 0, 0, 230), width=line_width + 2)
        overlay_draw.line((center, 0, center, output.height), fill=(255, 0, 0, 255), width=line_width)

    return Image.alpha_composite(output.convert("RGBA"), overlay).convert("RGB")


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
        f"{'cuts' if result.mode == 'cut_projection' else 'gap runs'}: {segmentation_runs_summary(result)}",
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
        f"{'cut' if result.mode == 'cut_projection' else 'gap'} threshold={result.gap_threshold:.3f}"
    )
    if result.mode == "cut_projection" and result.cut_postprocess:
        title += f"; postprocess={result.cut_postprocess}"
    draw.text((padding, y), title, fill=(55, 55, 55), font=font)
    y += title_height
    panel.paste(image_with_lines, (padding, y))
    y += image_with_lines.height + 8

    band = Image.new("RGB", (image.width, band_height), color=(235, 235, 235))
    band_draw = ImageDraw.Draw(band)
    timesteps = max(1, len(result.raw_indices))
    for x in range(image.width):
        timestep = min(timesteps - 1, int((x + 0.5) * timesteps / max(1, image.width)))
        gap_probability = result.gap_probabilities[timestep] if result.gap_probabilities else 0.0
        label = result.raw_indices[timestep] if result.raw_indices else 0
        if label == 1:
            color = (
                255,
                int(round(220 - 140 * gap_probability)),
                int(round(210 - 150 * gap_probability)),
            )
        else:
            color = (
                int(round(235 - 70 * gap_probability)),
                int(round(245 - 40 * gap_probability)),
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
        band_draw.line((center, 0, center, band_height), fill=(255, 0, 0), width=2)

    draw.rectangle((padding - 1, y - 1, padding + image.width, y + band_height), outline=(160, 160, 160))
    panel.paste(band, (padding, y))
    y += band_height + 8

    for line in summary_lines:
        draw.text((padding, y), line, fill=(55, 55, 55), font=small_font)
        y += text_line_height
    return panel


def save_debug_image(
    source_image: Image.Image,
    result: RecognitionResult,
    output_path: str | Path,
    metadata: dict[str, Any],
    network_input_image: Image.Image | None = None,
    preprocess_images: list[tuple[str, Image.Image]] | None = None,
    segmentation_result: VerticalSegmentationResult | None = None,
    segmentator_input_image: Image.Image | None = None,
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
    raw_segmentation_panel = None
    if segmentation_result is not None:
        segmentation_base_image = segmentator_input_image or network_input_image
        if segmentation_base_image is not None:
            raw_segmentation_panel = render_segmentation_panel(segmentation_base_image, segmentation_result)

    canvas_width = max(
        min_report_width,
        source_image.width + padding * 2,
        (network_input_image.width + padding * 2) if network_input_image is not None else 0,
        (raw_segmentation_panel.width + padding * 2) if raw_segmentation_panel is not None else 0,
    )

    max_image_width = canvas_width - padding * 2
    image = resize_debug_image(source_image, max_image_width)
    input_image = resize_debug_image(network_input_image, max_image_width) if network_input_image is not None else None
    debug_preprocess_images = [
        (title, resize_debug_image(debug_image, max_image_width))
        for title, debug_image in (preprocess_images or [])
    ]
    segmentation_panel = (
        resize_debug_image(raw_segmentation_panel, max_image_width)
        if raw_segmentation_panel is not None
        else None
    )

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
    if "baseline_crop" in metadata:
        info_lines.append(f"baseline crop: {metadata['baseline_crop']}")
    if metadata.get("baseline_status"):
        info_lines.append(f"baseline status: {metadata['baseline_status']}")
    if metadata.get("baseline_angle_degrees") is not None:
        info_lines.append(f"baseline angle: {float(metadata['baseline_angle_degrees']):+.3f} deg")
    if metadata.get("baseline_crop_box") is not None:
        info_lines.append(f"baseline crop box: {metadata['baseline_crop_box']}")
    if metadata.get("baseline_text_height") is not None:
        info_lines.append(f"baseline text height: {metadata['baseline_text_height']}")
    if "expected_text" in metadata:
        info_lines.append(f"expected text: {metadata['expected_text']!r}")
    if segmentation_result is not None:
        info_lines.append(f"segmentator checkpoint: {metadata.get('segmentator_checkpoint', '-')}")
        info_lines.append(f"segmentator mode: {segmentation_result.mode}")
        info_lines.append(f"segmentator input tensor shape: {segmentation_result.input_shape}")
        info_lines.append(f"segmentator logits shape: {segmentation_result.logits_shape}")
        info_lines.append(f"segmentator timesteps: {len(segmentation_result.raw_indices)}")
        if segmentation_result.mode == "cut_projection":
            info_lines.append(f"segmentator cuts: {len(segmentation_result.cut_positions or [])}")
            info_lines.append(f"segmentator candidates: {len(segmentation_result.candidate_cut_positions or [])}")
            info_lines.append(f"segmentator cut threshold: {segmentation_result.gap_threshold:.3f}")
            info_lines.append(f"segmentator peak min distance: {segmentation_result.peak_min_distance}")
            info_lines.append(f"segmentator cut postprocess: {segmentation_result.cut_postprocess}")
            info_lines.append(f"segmentator cut candidate threshold: {segmentation_result.cut_candidate_threshold:.3f}")
            info_lines.append(f"segmentator cut min width: {segmentation_result.cut_min_width}")
            info_lines.append(f"segmentator cut max width: {segmentation_result.cut_max_width}")
            info_lines.append(f"segmentator cut smooth radius: {segmentation_result.cut_smooth_radius}")
        else:
            info_lines.append(f"segmentator gap runs: {sum(1 for run in segmentation_result.runs if run.label == 1)}")
            info_lines.append(f"segmentator gap threshold: {segmentation_result.gap_threshold:.3f}")
            info_lines.append(f"segmentator min gap width: {segmentation_result.min_gap_width}")
            info_lines.append(f"segmentator merge gap width: {segmentation_result.merge_gap_width}")

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
    for _, debug_image in debug_preprocess_images:
        images_height += padding + image_title_height + debug_image.height
    if input_image is not None:
        images_height += padding + image_title_height + input_image.height
    if segmentation_panel is not None:
        images_height += padding + image_title_height + segmentation_panel.height

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

    for title, debug_image in debug_preprocess_images:
        y += padding
        draw.text(
            (padding, y),
            f"{title} ({debug_image.width}x{debug_image.height})",
            fill=(55, 55, 55),
            font=small_font,
        )
        y += image_title_height
        debug_x = (canvas_width - debug_image.width) // 2
        canvas.paste(debug_image, (debug_x, y))
        y += debug_image.height

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

    if segmentation_panel is not None:
        y += padding
        draw.text(
            (padding, y),
            f"vertical segmentator ({segmentation_panel.width}x{segmentation_panel.height})",
            fill=(55, 55, 55),
            font=small_font,
        )
        y += image_title_height
        segmentator_x = (canvas_width - segmentation_panel.width) // 2
        canvas.paste(segmentation_panel, (segmentator_x, y))
        y += segmentation_panel.height

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
