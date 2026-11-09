from __future__ import annotations

from PIL import Image

from fcn_ocr.debug_report import draw_cut_comparison_overlay, render_pipeline_panel
from fcn_ocr.results import CutDecodingResult
from tests.helpers import make_segmentation


def test_cut_comparison_overlay_distinguishes_raw_final_and_shared_cuts() -> None:
    image = Image.new("RGB", (16, 4), color=(255, 255, 255))

    overlay = draw_cut_comparison_overlay(
        image,
        segmenter_cuts=[1, 4],
        decoded_cuts=[4, 6],
        timesteps=8,
    )

    assert overlay.getpixel((3, 1)) == (220, 45, 45)
    assert overlay.getpixel((9, 1)) == (230, 170, 0)
    assert overlay.getpixel((13, 1)) == (25, 145, 55)


def test_pipeline_adds_final_cut_comparison_only_when_decoder_ran() -> None:
    image = Image.new("RGB", (16, 8), color=(255, 255, 255))
    segmentation = make_segmentation([1, 4, 6], width=8)
    common = {
        "source_image": image,
        "baseline_output_image": None,
        "baseline_preprocess_images": None,
        "baseline_enabled": False,
        "vertical_segmentation_input_image": image,
        "vertical_segmentation_preprocess_images": None,
        "segmentation_result": segmentation,
        "vertical_segmentation_enabled": True,
        "network_input_image": None,
        "preprocess_images": None,
        "ocr_enabled": False,
    }

    without_decoder = render_pipeline_panel(**common)
    with_decoder = render_pipeline_panel(
        **common,
        cut_decoding_result=CutDecodingResult(
            text="AB",
            symbols=[],
            cuts=[1, 4, 6],
            boundaries=[1, 4, 6],
            input_width=8,
            ocr_width=8,
            vertical_segmentation_width=8,
            decode_method="dp",
            path_score=0.9,
        ),
    )

    assert with_decoder.height > without_decoder.height
