from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch

from .baseline_detector import BaselineDetector
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
    recognition: RecognitionResult | None
    ocr_logits: torch.Tensor | None
    ocr_input: torch.Tensor | None
    ocr_preprocess_debug: PreprocessDebug | None
    baseline_image: Image.Image
    baseline_preprocess_debug: PreprocessDebug
    segmentation: VerticalSegmentationResult | None = None
    segmentator_input: torch.Tensor | None = None
    segmentator_preprocess_debug: PreprocessDebug | None = None
    cut_decoding: CutDecodingResult | None = None

    @property
    def text(self) -> str:
        if self.cut_decoding is not None:
            return self.cut_decoding.text
        if self.recognition is not None:
            return self.recognition.text
        return ""


class OCRPipeline:
    def __init__(
        self,
        config: InferenceConfig | str | Path,
        verbose: bool = False,
    ) -> None:
        self.config = InferenceConfig.load(config) if isinstance(config, (str, Path)) else config
        self.decode = self.config.ocr.decode if self.config.ocr is not None else None
        self.recognizer: TextRecognizer | None = None
        if self.config.ocr is not None:
            ocr = self.config.ocr
            preprocess = ocr.preprocessing
            self.recognizer = TextRecognizer(
                ocr.checkpoint,
                device=self.config.device,
                verbose=verbose,
                scale_x=preprocess.scale_x,
                y_pad=preprocess.y_pad,
                x_pad=preprocess.x_pad,
                baseline_crop=False,
                baseline_detector_checkpoint=None,
            )

        self.segmentator: VerticalSegmentator | None = None
        if self.config.segmentator is not None:
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
                cut_min_width=segmentator.cut_min_width,
                cut_max_width=segmentator.cut_max_width,
                cut_smooth_radius=segmentator.cut_smooth_radius,
            )

        self.baseline_processor: BaselineDetector | None = None
        baseline = self.config.baseline
        if baseline is not None and baseline.enabled:
            if baseline.detector_checkpoint is None:
                raise ValueError("Enabled baseline stage requires detector_checkpoint")
            self.baseline_processor = BaselineDetector(
                baseline.detector_checkpoint,
                device=self.config.device,
                threshold=baseline.detector_threshold,
                deskew=baseline.deskew,
                max_angle=baseline.max_angle,
                strict_lines=baseline.strict_lines,
                line_pad=baseline.line_pad,
                line_pad_px=baseline.line_pad_px,
            )
            if verbose:
                self.baseline_processor.print_summary()

        if verbose:
            self._print_pipeline_summary()

    def _print_pipeline_summary(self) -> None:
        baseline = self.config.baseline
        print("\nInference pipeline:")
        if baseline is None or not baseline.enabled:
            print("  Shared pipeline baseline: disabled")
        else:
            print(
                "  Shared pipeline baseline: enabled "
                f"(neural detector {baseline.detector_checkpoint})"
            )
            print(
                "    runs once before OCR/segmentator; "
                f"deskew={baseline.deskew}, strict_lines={baseline.strict_lines}, "
                f"line_pad={baseline.line_pad:.3f}, line_pad_px={baseline.line_pad_px:.1f}"
            )
        print(
            "  OCR stage: "
            f"{'enabled; receives shared baseline output' if self.recognizer is not None else 'disabled'}"
        )
        print(
            "  Vertical segmentator stage: "
            f"{'enabled; receives shared baseline output' if self.segmentator is not None else 'disabled'}"
        )
        if self.decode is not None and self.decode.enabled:
            print(f"  Decode with segmentator: enabled ({self.decode.method})")
        else:
            print("  Decode with segmentator: disabled")

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
        if self.baseline_processor is not None:
            baseline_image, baseline_debug = self.baseline_processor.prepare_baseline_image(
                source,
                collect_debug=collect_debug,
            )
        else:
            baseline_image = source
            baseline_debug = PreprocessDebug(
                metadata={"baseline_crop": False, "baseline_status": "skipped"},
                images=[],
            )

        recognition = None
        ocr_logits = None
        ocr_input = None
        ocr_source_x = None
        ocr_debug = None
        if self.recognizer is not None:
            ocr_input, ocr_source_x = self.recognizer.preprocess_pil_after_baseline_with_source_x(
                baseline_image
            )
            if collect_debug:
                _, ocr_debug = self.recognizer.preprocess_pil_after_baseline_debug(
                    baseline_image
                )
            else:
                ocr_debug = PreprocessDebug(metadata={}, images=[])
            recognition, ocr_logits = self.recognizer.recognize_tensor_debug_with_logits(
                ocr_input,
                top_k=self.config.debug.top_k,
            )

        segmentation = None
        segmentator_input = None
        segmentator_source_x = None
        segmentator_debug = None
        if self.segmentator is not None:
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
        if self.decode is not None and self.decode.enabled:
            if (
                self.recognizer is None
                or ocr_logits is None
                or ocr_input is None
                or ocr_source_x is None
                or segmentation is None
                or segmentator_source_x is None
            ):
                raise RuntimeError("decode stage requires completed OCR and segmentator stages")
            decode_kwargs = {
                "input_width": int(ocr_input.shape[-1]),
                "top_k": self.decode.top_k,
                "center_fraction": self.decode.center_fraction,
                "min_score_width": self.decode.min_score_width,
                "ocr_source_x": ocr_source_x,
                "segmentator_source_x": segmentator_source_x,
            }
            if self.decode.method == "dp":
                cut_decoding = self.recognizer.decode_legacy_with_cuts_dp(
                    ocr_logits,
                    segmentation,
                    cut_weight=self.decode.cut_weight,
                    ocr_weight=self.decode.ocr_weight,
                    width_weight=self.decode.width_weight,
                    skip_cut_penalty=self.decode.skip_cut_penalty,
                    **decode_kwargs,
                )
            else:
                cut_decoding = self.recognizer.decode_legacy_with_cuts(
                    ocr_logits,
                    segmentation,
                    **decode_kwargs,
                )

        return OCRPipelineResult(
            recognition=recognition,
            ocr_logits=ocr_logits,
            ocr_input=ocr_input,
            ocr_preprocess_debug=ocr_debug,
            baseline_image=baseline_image,
            baseline_preprocess_debug=baseline_debug,
            segmentation=segmentation,
            segmentator_input=segmentator_input,
            segmentator_preprocess_debug=segmentator_debug,
            cut_decoding=cut_decoding,
        )
