from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .inference_config import InferenceConfig
from .recognizer import TextRecognizer
from .results import (
    CutDecodingResult,
    PreprocessDebug,
    RecognitionResult,
    VerticalSegmentationResult,
)
from .segmentator import VerticalSegmentator


@dataclass(frozen=True)
class OCRPipelineResult:
    recognition: RecognitionResult
    ocr_logits: torch.Tensor
    ocr_input: torch.Tensor
    ocr_preprocess_debug: PreprocessDebug
    baseline_image: Image.Image
    segmentation: VerticalSegmentationResult | None = None
    segmentator_input: torch.Tensor | None = None
    segmentator_preprocess_debug: PreprocessDebug | None = None
    cut_decoding: CutDecodingResult | None = None

    @property
    def text(self) -> str:
        if self.cut_decoding is not None:
            return self.cut_decoding.text
        return self.recognition.text


class OCRPipeline:
    def __init__(
        self,
        config: InferenceConfig | str | Path,
        verbose: bool = False,
    ) -> None:
        self.config = InferenceConfig.load(config) if isinstance(config, (str, Path)) else config
        baseline = self.config.baseline
        ocr_preprocess = self.config.ocr.preprocessing
        self.recognizer = TextRecognizer(
            self.config.ocr.checkpoint,
            device=self.config.device,
            verbose=verbose,
            scale_x=ocr_preprocess.scale_x,
            y_pad=ocr_preprocess.y_pad,
            x_pad=ocr_preprocess.x_pad,
            baseline_crop=baseline.enabled,
            baseline_deskew=baseline.deskew,
            baseline_max_angle=baseline.max_angle,
            baseline_strict_lines=baseline.strict_lines,
            baseline_line_pad=baseline.line_pad,
            baseline_line_pad_px=baseline.line_pad_px,
            baseline_detector_checkpoint=(
                baseline.detector_checkpoint if baseline.enabled else None
            ),
            baseline_detector_threshold=baseline.detector_threshold,
        )

        self.segmentator: VerticalSegmentator | None = None
        if self.config.segmentator.checkpoint is not None:
            segmentator = self.config.segmentator
            preprocess = segmentator.preprocessing
            self.segmentator = VerticalSegmentator(
                segmentator.checkpoint,
                device=self.config.device,
                verbose=verbose,
                scale_x=preprocess.scale_x,
                y_pad=preprocess.y_pad,
                x_pad=preprocess.x_pad,
                baseline_crop=False,
                baseline_detector_checkpoint=None,
                cut_threshold=segmentator.cut_threshold,
                peak_min_distance=segmentator.peak_min_distance,
                cut_postprocess=segmentator.cut_postprocess,
                cut_min_width=segmentator.cut_min_width,
                cut_max_width=segmentator.cut_max_width,
                cut_candidate_threshold=segmentator.cut_candidate_threshold,
                cut_smooth_radius=segmentator.cut_smooth_radius,
            )

    def recognize_path(
        self,
        image_path: str | Path,
        collect_debug: bool = False,
    ) -> OCRPipelineResult:
        with Image.open(image_path) as image:
            return self.recognize_pil(image, collect_debug=collect_debug)

    def recognize_pil(
        self,
        image: Image.Image,
        collect_debug: bool = False,
    ) -> OCRPipelineResult:
        source = image.convert("RGB")
        baseline_image, baseline_debug = self.recognizer.prepare_baseline_image(
            source,
            collect_debug=collect_debug,
        )

        ocr_input, ocr_source_x = self.recognizer.preprocess_pil_after_baseline_with_source_x(
            baseline_image
        )
        if collect_debug:
            _, ocr_debug = self.recognizer.preprocess_pil_after_baseline_debug(
                baseline_image
            )
        else:
            ocr_debug = PreprocessDebug(metadata={}, images=[])
        ocr_debug = PreprocessDebug(
            metadata={**ocr_debug.metadata, **baseline_debug.metadata, "baseline_shared": True},
            images=[*baseline_debug.images, *ocr_debug.images],
        )
        recognition, ocr_logits = self.recognizer.recognize_tensor_debug_with_logits(
            ocr_input,
            top_k=self.config.debug.top_k,
        )

        if self.segmentator is None:
            return OCRPipelineResult(
                recognition=recognition,
                ocr_logits=ocr_logits,
                ocr_input=ocr_input,
                ocr_preprocess_debug=ocr_debug,
                baseline_image=baseline_image,
            )

        segmentator_input, segmentator_source_x = (
            self.segmentator.preprocess_pil_after_baseline_with_source_x(
                baseline_image
            )
        )
        if collect_debug:
            _, segmentator_debug = (
                self.segmentator.preprocess_pil_after_baseline_debug(baseline_image)
            )
        else:
            segmentator_debug = PreprocessDebug(metadata={}, images=[])
        segmentation = self.segmentator.segment_tensor_debug(segmentator_input)

        cut_decoding = None
        if self.config.decode.enabled:
            cut_decoding = self.recognizer.decode_legacy_with_cuts(
                ocr_logits,
                segmentation,
                input_width=int(ocr_input.shape[-1]),
                top_k=self.config.decode.top_k,
                center_fraction=self.config.decode.center_fraction,
                min_score_width=self.config.decode.min_score_width,
                ocr_source_x=ocr_source_x,
                segmentator_source_x=segmentator_source_x,
            )

        return OCRPipelineResult(
            recognition=recognition,
            ocr_logits=ocr_logits,
            ocr_input=ocr_input,
            ocr_preprocess_debug=ocr_debug,
            baseline_image=baseline_image,
            segmentation=segmentation,
            segmentator_input=segmentator_input,
            segmentator_preprocess_debug=segmentator_debug,
            cut_decoding=cut_decoding,
        )
