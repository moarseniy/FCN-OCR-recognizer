from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
from .vertical_segmenter import VerticalSegmenter


@dataclass(frozen=True)
class FCNPipelineResult:
    recognition: RecognitionResult | None
    ocr_logits: torch.Tensor | None
    ocr_input: torch.Tensor | None
    ocr_preprocess_debug: PreprocessDebug | None
    baseline_image: Image.Image
    baseline_preprocess_debug: PreprocessDebug
    segmentation: VerticalSegmentationResult | None = None
    vertical_segmentation_input: torch.Tensor | None = None
    vertical_segmentation_preprocess_debug: PreprocessDebug | None = None
    cut_decoding: CutDecodingResult | None = None

    @property
    def text(self) -> str:
        if self.cut_decoding is not None:
            return self.cut_decoding.text
        if self.recognition is not None:
            return self.recognition.text
        return ""


@dataclass(frozen=True)
class FCNPipelinePathResult:
    path: Path
    text: str
    error: str = ""


class FCNPipeline:
    def __init__(
        self,
        config: InferenceConfig | str | Path,
        verbose: bool = False,
    ) -> None:
        self.config = InferenceConfig.load(config) if isinstance(config, (str, Path)) else config
        self.decode = self.config.fcn_ocr.decode if self.config.fcn_ocr is not None else None
        self.recognizer: TextRecognizer | None = None
        if self.config.fcn_ocr is not None:
            ocr = self.config.fcn_ocr
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

        self.vertical_segmenter: VerticalSegmenter | None = None
        if self.config.vertical_segmentation is not None:
            vertical_segmentation = self.config.vertical_segmentation
            preprocess = vertical_segmentation.preprocessing
            self.vertical_segmenter = VerticalSegmenter(
                vertical_segmentation.checkpoint,
                device=self.config.device,
                verbose=verbose,
                scale_x=preprocess.scale_x,
                y_pad=preprocess.y_pad,
                x_pad=preprocess.x_pad,
                baseline_crop=False,
                baseline_detector_checkpoint=None,
                cut_threshold=vertical_segmentation.cut_threshold,
                cut_min_width=vertical_segmentation.cut_min_width,
                cut_max_width=vertical_segmentation.cut_max_width,
                cut_smooth_radius=vertical_segmentation.cut_smooth_radius,
            )

        self.baseline_processor: BaselineDetector | None = None
        baseline = self.config.baseline_detection
        if baseline is not None and baseline.enabled:
            if baseline.detector_checkpoint is None:
                raise ValueError(
                    "Enabled baseline_detection section requires detector_checkpoint"
                )
            self.baseline_processor = BaselineDetector(
                baseline.detector_checkpoint,
                device=self.config.device,
                threshold=baseline.detector_threshold,
                deskew=baseline.deskew,
                max_angle=baseline.max_angle,
                line_pad=baseline.line_pad,
                line_pad_px=baseline.line_pad_px,
            )
            if verbose:
                self.baseline_processor.print_summary()

        if verbose:
            self._print_pipeline_summary()

    def _print_pipeline_summary(self) -> None:
        baseline = self.config.baseline_detection
        print("\nInference pipeline:")
        if baseline is None or not baseline.enabled:
            print("  Shared pipeline baseline: disabled")
        else:
            print(
                "  Shared pipeline baseline: enabled "
                f"(neural detector {baseline.detector_checkpoint})"
            )
            print(
                "    runs once before fcn_ocr/vertical_segmentation; "
                f"deskew={baseline.deskew}, "
                f"line_pad={baseline.line_pad:.3f}, line_pad_px={baseline.line_pad_px:.1f}"
            )
        print(
            "  FCN OCR stage: "
            f"{'enabled; receives shared baseline output' if self.recognizer is not None else 'disabled'}"
        )
        print(
            "  Vertical segmentation stage: "
            f"{'enabled; receives shared baseline output' if self.vertical_segmenter is not None else 'disabled'}"
        )
        if self.decode is not None and self.decode.enabled:
            print(f"  Decode with vertical segmentation: enabled ({self.decode.method})")
            if self.decode.glyph_width_prior.enabled:
                print(
                    "  Glyph width prior: "
                    f"enabled weight={self.decode.glyph_width_prior.weight:g}"
                )
        else:
            print("  Decode with vertical segmentation: disabled")

    def recognize_path(
        self,
        image_path: str | Path,
        collect_debug: bool = False,
    ) -> FCNPipelineResult:
        with Image.open(image_path) as image:
            return self.recognize_pil(image, collect_debug=collect_debug)

    def recognize_pil(
        self,
        image: Image.Image,
        collect_debug: bool = False,
    ) -> FCNPipelineResult:
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
        vertical_segmentation_input = None
        vertical_segmentation_source_x = None
        vertical_segmentation_debug = None
        if self.vertical_segmenter is not None:
            vertical_segmentation_input, vertical_segmentation_source_x = (
                self.vertical_segmenter.preprocess_pil_after_baseline_with_source_x(
                    baseline_image
                )
            )
            if collect_debug:
                _, vertical_segmentation_debug = (
                    self.vertical_segmenter.preprocess_pil_after_baseline_debug(baseline_image)
                )
            else:
                vertical_segmentation_debug = PreprocessDebug(metadata={}, images=[])
            segmentation = self.vertical_segmenter.segment_tensor_debug(
                vertical_segmentation_input
            )

        cut_decoding = None
        if self.decode is not None and self.decode.enabled:
            if (
                self.recognizer is None
                or ocr_logits is None
                or ocr_input is None
                or ocr_source_x is None
                or segmentation is None
                or vertical_segmentation_source_x is None
            ):
                raise RuntimeError(
                    "decode stage requires completed fcn_ocr and vertical_segmentation stages"
                )
            cut_decoding = self._decode_with_vertical_segmentation(
                ocr_logits,
                segmentation,
                ocr_input=ocr_input,
                ocr_source_x=ocr_source_x,
                vertical_segmentation_source_x=vertical_segmentation_source_x,
            )

        return FCNPipelineResult(
            recognition=recognition,
            ocr_logits=ocr_logits,
            ocr_input=ocr_input,
            ocr_preprocess_debug=ocr_debug,
            baseline_image=baseline_image,
            baseline_preprocess_debug=baseline_debug,
            segmentation=segmentation,
            vertical_segmentation_input=vertical_segmentation_input,
            vertical_segmentation_preprocess_debug=vertical_segmentation_debug,
            cut_decoding=cut_decoding,
        )

    @torch.no_grad()
    def recognize_paths_text(
        self,
        image_paths: Sequence[str | Path],
        batch_size: int = 1,
        log_every: int = 0,
    ) -> tuple[list[FCNPipelinePathResult], dict[str, float | int]]:
        if self.recognizer is None:
            raise ValueError(
                "FCNPipeline text recognition requires an fcn_ocr section"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        paths = [Path(path) for path in image_paths]
        results: dict[int, FCNPipelinePathResult] = {}
        prepared: list[dict[str, Any]] = []
        for index, path in enumerate(paths):
            try:
                with Image.open(path) as image_file:
                    source = image_file.convert("RGB")
                if self.baseline_processor is not None:
                    baseline_image, _ = self.baseline_processor.prepare_baseline_image(
                        source,
                        collect_debug=False,
                    )
                else:
                    baseline_image = source

                ocr_input, ocr_source_x = (
                    self.recognizer.preprocess_pil_after_baseline_with_source_x(
                        baseline_image
                    )
                )
                ocr_output_width = self.recognizer.output_width_for_input_width(
                    int(ocr_input.shape[-1])
                )
                if ocr_output_width < 1:
                    raise ValueError(
                        "OCR preprocessing produced an input that is too narrow for "
                        f"{self.recognizer.architecture}: input={tuple(ocr_input.shape)}"
                    )

                item: dict[str, Any] = {
                    "index": index,
                    "path": path,
                    "ocr_input": ocr_input,
                    "ocr_source_x": ocr_source_x,
                    "ocr_output_width": ocr_output_width,
                }
                if self.vertical_segmenter is not None:
                    vertical_segmentation_input, vertical_segmentation_source_x = (
                        self.vertical_segmenter.preprocess_pil_after_baseline_with_source_x(
                            baseline_image
                        )
                    )
                    vertical_segmentation_output_width = (
                        self.vertical_segmenter.output_width_for_input_width(
                            int(vertical_segmentation_input.shape[-1])
                        )
                    )
                    if vertical_segmentation_output_width < 1:
                        raise ValueError(
                            "Vertical segmentation preprocessing produced an input that is too narrow for "
                            f"{self.vertical_segmenter.architecture}: "
                            f"input={tuple(vertical_segmentation_input.shape)}"
                        )
                    item.update(
                        {
                            "vertical_segmentation_input": vertical_segmentation_input,
                            "vertical_segmentation_source_x": vertical_segmentation_source_x,
                            "vertical_segmentation_output_width": vertical_segmentation_output_width,
                        }
                    )
                prepared.append(item)
            except Exception as error:
                results[index] = FCNPipelinePathResult(
                    path=path,
                    text="",
                    error=repr(error),
                )

        gpu_batches = 0
        gpu_batch_items = 0
        padded_width_total = 0
        useful_width_total = 0
        max_gpu_batch_size = 0
        processed = len(results)
        for width_batch in self._make_width_aware_batches(
            prepared,
            max_batch_size=batch_size,
        ):
            try:
                ocr_batch = self._pad_inference_batch(
                    [item["ocr_input"] for item in width_batch],
                    device=self.recognizer.device,
                )
                ocr_logits, _ = self.recognizer.logits_from_tensor(ocr_batch)

                vertical_segmentation_logits = None
                vertical_segmentation_batch = None
                if self.vertical_segmenter is not None:
                    vertical_segmentation_batch = self._pad_inference_batch(
                        [item["vertical_segmentation_input"] for item in width_batch],
                        device=self.vertical_segmenter.device,
                    )
                    vertical_segmentation_logits, _ = self.vertical_segmenter.logits_from_tensor(
                        vertical_segmentation_batch
                    )

                gpu_batches += 1
                gpu_batch_items += len(width_batch)
                max_gpu_batch_size = max(max_gpu_batch_size, len(width_batch))
                for item in width_batch:
                    useful_width_total += int(item["ocr_input"].shape[-1])
                    if self.vertical_segmenter is not None:
                        useful_width_total += int(item["vertical_segmentation_input"].shape[-1])
                padded_width_total += len(width_batch) * int(ocr_batch.shape[-1])
                if vertical_segmentation_batch is not None:
                    padded_width_total += len(width_batch) * int(
                        vertical_segmentation_batch.shape[-1]
                    )

                for batch_index, item in enumerate(width_batch):
                    index = int(item["index"])
                    path = item["path"]
                    try:
                        sample_ocr_logits = ocr_logits[
                            batch_index : batch_index + 1,
                            :,
                            : int(item["ocr_output_width"]),
                        ]
                        if self.decode is not None and self.decode.enabled:
                            if self.vertical_segmenter is None or vertical_segmentation_logits is None:
                                raise RuntimeError(
                                    "decode stage requires a vertical_segmentation section"
                                )
                            vertical_segmentation_input = item["vertical_segmentation_input"]
                            sample_vertical_segmentation_logits = vertical_segmentation_logits[
                                batch_index : batch_index + 1,
                                :,
                                : int(item["vertical_segmentation_output_width"]),
                            ]
                            segmentation = self.vertical_segmenter.analyze_segmentation_logits(
                                sample_vertical_segmentation_logits,
                                input_shape=(1, *tuple(vertical_segmentation_input.shape)),
                            )
                            decoded = self._decode_with_vertical_segmentation(
                                sample_ocr_logits,
                                segmentation,
                                ocr_input=item["ocr_input"],
                                ocr_source_x=item["ocr_source_x"],
                                vertical_segmentation_source_x=item["vertical_segmentation_source_x"],
                            )
                            text = decoded.text
                        else:
                            text, _ = self.recognizer.decode_predictions(
                                sample_ocr_logits
                            )
                        results[index] = FCNPipelinePathResult(
                            path=path,
                            text=text.strip(),
                        )
                    except Exception as error:
                        results[index] = FCNPipelinePathResult(
                            path=path,
                            text="",
                            error=repr(error),
                        )
            except Exception as error:
                for item in width_batch:
                    index = int(item["index"])
                    results[index] = FCNPipelinePathResult(
                        path=item["path"],
                        text="",
                        error=f"batch_error={error!r}",
                    )

            processed += len(width_batch)
            if log_every > 0 and (
                processed == len(paths) or processed % log_every == 0
            ):
                print(f"Recognized {processed}/{len(paths)} images")

        ordered = [results[index] for index in range(len(paths))]
        padding_efficiency = (
            useful_width_total / padded_width_total
            if padded_width_total > 0
            else 1.0
        )
        return ordered, {
            "gpu_batches": gpu_batches,
            "average_gpu_batch_size": (
                gpu_batch_items / gpu_batches if gpu_batches > 0 else 0.0
            ),
            "max_gpu_batch_size": max_gpu_batch_size,
            "padding_efficiency": padding_efficiency,
        }

    def _decode_with_vertical_segmentation(
        self,
        ocr_logits: torch.Tensor,
        segmentation: VerticalSegmentationResult,
        *,
        ocr_input: torch.Tensor,
        ocr_source_x,
        vertical_segmentation_source_x,
    ) -> CutDecodingResult:
        if self.recognizer is None or self.decode is None or not self.decode.enabled:
            raise RuntimeError("vertical segmentation decode is not enabled")
        decode_kwargs = {
            "input_width": int(ocr_input.shape[-1]),
            "input_height": int(ocr_input.shape[-2]),
            "top_k": self.decode.top_k,
            "center_fraction": self.decode.center_fraction,
            "min_score_width": self.decode.min_score_width,
            "ocr_source_x": ocr_source_x,
            "vertical_segmentation_source_x": vertical_segmentation_source_x,
            "glyph_width_prior": self.decode.glyph_width_prior.model_dump(),
        }
        if self.decode.method == "dp":
            return self.recognizer.decode_fcn_ocr_with_cuts_dp(
                ocr_logits,
                segmentation,
                cut_weight=self.decode.cut_weight,
                ocr_weight=self.decode.ocr_weight,
                width_weight=self.decode.width_weight,
                skip_cut_penalty=self.decode.skip_cut_penalty,
                **decode_kwargs,
            )
        return self.recognizer.decode_fcn_ocr_with_cuts(
            ocr_logits,
            segmentation,
            **decode_kwargs,
        )

    @staticmethod
    def _make_width_aware_batches(
        prepared: list[dict[str, Any]],
        max_batch_size: int,
        max_width_ratio: float = 1.35,
    ) -> list[list[dict[str, Any]]]:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_width_ratio < 1.0:
            raise ValueError("max_width_ratio must be >= 1")

        def stage_widths(item: dict[str, Any]) -> tuple[int, int | None]:
            vertical_segmentation_width = (
                int(item["vertical_segmentation_input"].shape[-1])
                if "vertical_segmentation_input" in item
                else None
            )
            return int(item["ocr_input"].shape[-1]), vertical_segmentation_width

        ordered = sorted(prepared, key=lambda item: stage_widths(item)[0])
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        min_ocr_width = max_ocr_width = 0
        min_vertical_segmentation_width = max_vertical_segmentation_width = 0
        for item in ordered:
            ocr_width, vertical_segmentation_width = stage_widths(item)
            if not current:
                current = [item]
                min_ocr_width = max_ocr_width = ocr_width
                if vertical_segmentation_width is not None:
                    min_vertical_segmentation_width = max_vertical_segmentation_width = vertical_segmentation_width
                continue

            next_min_ocr = min(min_ocr_width, ocr_width)
            next_max_ocr = max(max_ocr_width, ocr_width)
            ocr_compatible = (
                next_max_ocr / max(1, next_min_ocr) <= max_width_ratio
            )
            vertical_segmentation_compatible = True
            if vertical_segmentation_width is not None:
                next_min_vertical_segmentation = min(
                    min_vertical_segmentation_width,
                    vertical_segmentation_width,
                )
                next_max_vertical_segmentation = max(
                    max_vertical_segmentation_width,
                    vertical_segmentation_width,
                )
                vertical_segmentation_compatible = (
                    next_max_vertical_segmentation
                    / max(1, next_min_vertical_segmentation)
                    <= max_width_ratio
                )
            if (
                len(current) >= max_batch_size
                or not ocr_compatible
                or not vertical_segmentation_compatible
            ):
                batches.append(current)
                current = [item]
                min_ocr_width = max_ocr_width = ocr_width
                if vertical_segmentation_width is not None:
                    min_vertical_segmentation_width = max_vertical_segmentation_width = vertical_segmentation_width
                continue
            current.append(item)
            min_ocr_width = next_min_ocr
            max_ocr_width = next_max_ocr
            if vertical_segmentation_width is not None:
                min_vertical_segmentation_width = next_min_vertical_segmentation
                max_vertical_segmentation_width = next_max_vertical_segmentation
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _pad_inference_batch(
        tensors: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if not tensors:
            raise ValueError("Cannot build an empty inference batch")
        channels = int(tensors[0].shape[0])
        height = int(tensors[0].shape[1])
        max_width = max(int(tensor.shape[-1]) for tensor in tensors)
        batch = torch.ones(
            (len(tensors), channels, height, max_width),
            dtype=tensors[0].dtype,
            device=device,
        )
        for batch_index, tensor in enumerate(tensors):
            if tuple(tensor.shape[:2]) != (channels, height):
                raise ValueError(
                    "All tensors in an inference batch must have equal channels/height; "
                    f"expected {(channels, height)}, got {tuple(tensor.shape[:2])}"
                )
            batch[batch_index, :, :, : tensor.shape[-1]] = tensor.to(device)
        return batch
