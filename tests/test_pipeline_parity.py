from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from fcn_ocr.inference_config import InferenceConfig
from fcn_ocr.pipeline import OCRPipeline
from fcn_ocr.results import CutDecodingResult, PreprocessDebug, RecognitionResult

from tests.helpers import make_segmentation


class _FakeRecognizer:
    device = torch.device("cpu")
    architecture = "fake_ocr"

    @staticmethod
    def prepare_baseline_image(image: Image.Image, collect_debug: bool = False):
        return image.convert("RGB"), PreprocessDebug(metadata={}, images=[])

    @staticmethod
    def preprocess_pil_after_baseline_with_source_x(image: Image.Image):
        tensor = torch.zeros((3, 8, 8), dtype=torch.float32)
        source_x = np.broadcast_to(np.arange(8, dtype=np.float32), (8, 8)).copy()
        return tensor, source_x

    @staticmethod
    def output_width_for_input_width(width: int) -> int:
        return width

    @staticmethod
    def logits_from_tensor(image_tensor: torch.Tensor):
        batch_size = 1 if image_tensor.dim() == 3 else image_tensor.size(0)
        logits = torch.zeros((batch_size, 3, image_tensor.shape[-1]))
        return logits, tuple(image_tensor.shape)

    def recognize_tensor_debug_with_logits(
        self, image_tensor: torch.Tensor, top_k: int = 8
    ):
        logits, input_shape = self.logits_from_tensor(image_tensor)
        recognition = RecognitionResult(
            text="RAW",
            raw_indices=[],
            raw_confidences=[],
            raw_chars=[],
            decoded_symbols=[],
            top_candidates_by_timestep=[],
            input_shape=input_shape,
            logits_shape=tuple(logits.shape),
        )
        return recognition, logits

    @staticmethod
    def decode_fcn_ocr_with_cuts(logits, segmentation_result, **kwargs):
        return CutDecodingResult(
            text="AB",
            symbols=[],
            cuts=[0, 4, 7],
            boundaries=[0, 4, 7],
            input_width=8,
            ocr_width=8,
            segmentator_width=8,
            decode_method="cells",
        )


class _FakeSegmentator:
    device = torch.device("cpu")
    architecture = "fake_segmentator"

    @staticmethod
    def preprocess_pil_after_baseline_with_source_x(image: Image.Image):
        tensor = torch.zeros((3, 8, 8), dtype=torch.float32)
        source_x = np.broadcast_to(np.arange(8, dtype=np.float32), (8, 8)).copy()
        return tensor, source_x

    @staticmethod
    def output_width_for_input_width(width: int) -> int:
        return width

    @staticmethod
    def logits_from_tensor(image_tensor: torch.Tensor):
        batch_size = 1 if image_tensor.dim() == 3 else image_tensor.size(0)
        logits = torch.zeros((batch_size, 1, image_tensor.shape[-1]))
        return logits, tuple(image_tensor.shape)

    @staticmethod
    def analyze_segmentation_logits(logits: torch.Tensor, input_shape: tuple[int, ...]):
        return make_segmentation([0, 4, 7], width=8)

    def segment_tensor_debug(self, image_tensor: torch.Tensor):
        logits, input_shape = self.logits_from_tensor(image_tensor)
        return self.analyze_segmentation_logits(logits, input_shape)


def _pipeline_with_fakes(recognizer, segmentator) -> OCRPipeline:
    config = InferenceConfig.model_validate(
        {
            "device": "cpu",
            "fcn_ocr": {
                "checkpoint": "unused_ocr.pth",
                "decode": {"enabled": True, "method": "cells"},
            },
            "vertical_segmentation": {"checkpoint": "unused_segmentator.pth"},
        }
    )
    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.config = config
    pipeline.decode = config.fcn_ocr.decode
    pipeline.recognizer = recognizer
    pipeline.segmentator = segmentator
    pipeline.baseline_processor = None
    return pipeline


@pytest.mark.parametrize("removed_section", ["ocr", "segmentator", "baseline"])
def test_inference_config_rejects_removed_section_names(
    removed_section: str,
) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        InferenceConfig.model_validate(
            {
                removed_section: {"checkpoint": "removed.pth"},
                "fcn_ocr": {"checkpoint": "current.pth"},
            }
        )


def test_pipeline_and_evaluation_path_return_the_same_decoded_text(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    recognizer = _FakeRecognizer()
    segmentator = _FakeSegmentator()

    with Image.open(image_path) as image:
        pipeline_text = (
            _pipeline_with_fakes(recognizer, segmentator).recognize_pil(image).text
        )

    path_results, _ = _pipeline_with_fakes(
        recognizer,
        segmentator,
    ).recognize_paths_text([image_path], batch_size=1)

    assert path_results[0].error == ""
    assert pipeline_text == path_results[0].text == "AB"
