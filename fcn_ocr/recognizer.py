from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
except ImportError:  # pragma: no cover - optional until baseline crop is enabled
    cv2 = None

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from fcn_architectures import create_model, normalize_architecture_name
from .results import (
    ClassConfidence,
    CutDecodedSymbol,
    CutDecodingResult,
    DecodedSymbol,
    PreprocessDebug,
    RecognitionResult,
    VerticalSegmentationResult,
    display_char,
)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().float().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image[0]

    if image.shape[0] == 1:
        array = (image[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class TextRecognizer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        verbose: bool = False,
        scale_x: float = 0.0,
        y_pad: float = 0.0,
        x_pad: float = 0.0,
        baseline_crop: bool = False,
        baseline_top_pad: float = 0.12,
        baseline_bottom_pad: float = 0.18,
        baseline_deskew: bool = True,
        baseline_max_angle: float = 12.0,
        baseline_strict_lines: bool = True,
        baseline_line_pad: float = 0.08,
        baseline_line_pad_px: float = 0.0,
        baseline_detector_checkpoint: str | Path | None = None,
        baseline_detector_threshold: float = 0.35,
        baseline_rectify: str = "lines",
        baseline_curve_smooth_radius: int = 8,
        baseline_curve_min_coverage: float = 0.25,
    ):
        if scale_x <= -0.95:
            raise ValueError("scale_x must be > -0.95")
        if y_pad <= -0.95:
            raise ValueError("y_pad must be > -0.95")
        if x_pad < 0.0:
            raise ValueError("x_pad must be >= 0")
        if baseline_top_pad < 0.0:
            raise ValueError("baseline_top_pad must be >= 0")
        if baseline_bottom_pad < 0.0:
            raise ValueError("baseline_bottom_pad must be >= 0")
        if baseline_line_pad < 0.0:
            raise ValueError("baseline_line_pad must be >= 0")
        if baseline_line_pad_px < 0.0:
            raise ValueError("baseline_line_pad_px must be >= 0")
        if baseline_max_angle <= 0.0:
            raise ValueError("baseline_max_angle must be > 0")
        if not 0.0 < baseline_detector_threshold < 1.0:
            raise ValueError("baseline_detector_threshold must be between 0 and 1")
        baseline_rectify = baseline_rectify.lower()
        if baseline_rectify == "line":
            baseline_rectify = "lines"
        if baseline_rectify not in {"lines", "curved"}:
            raise ValueError("baseline_rectify must be 'lines' or 'curved'")
        if baseline_curve_smooth_radius < 0:
            raise ValueError("baseline_curve_smooth_radius must be >= 0")
        if not 0.0 <= baseline_curve_min_coverage <= 1.0:
            raise ValueError("baseline_curve_min_coverage must be between 0 and 1")

        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.scale_x = float(scale_x)
        self.y_pad = float(y_pad)
        self.x_pad = float(x_pad)
        self.baseline_crop = bool(baseline_crop)
        self.baseline_top_pad = float(baseline_top_pad)
        self.baseline_bottom_pad = float(baseline_bottom_pad)
        self.baseline_deskew = bool(baseline_deskew)
        self.baseline_max_angle = float(baseline_max_angle)
        self.baseline_strict_lines = bool(baseline_strict_lines)
        self.baseline_line_pad = float(baseline_line_pad)
        self.baseline_line_pad_px = float(baseline_line_pad_px)
        self.baseline_detector_checkpoint = Path(baseline_detector_checkpoint) if baseline_detector_checkpoint else None
        self.baseline_detector_threshold = float(baseline_detector_threshold)
        self.baseline_rectify = baseline_rectify
        self.baseline_curve_smooth_radius = int(baseline_curve_smooth_radius)
        self.baseline_curve_min_coverage = float(baseline_curve_min_coverage)
        self.baseline_detector_model: torch.nn.Module | None = None
        self.baseline_detector_in_channels = 1
        self.baseline_detector_image_height = 0
        self.baseline_detector_architecture = ""

        self.alphabet = self.checkpoint["alphabet"]
        self.idx_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

        model_config = self.checkpoint.get("model_config", {})
        checkpoint_config = self.checkpoint.get("config", {})
        self.architecture = normalize_architecture_name(
            model_config.get("architecture", checkpoint_config.get("architecture", "legacy_fcn"))
        )
        self.architecture_params = dict(
            model_config.get(
                "architecture_params",
                checkpoint_config.get("architecture_params", {}),
            )
            or {}
        )
        self.in_channels = int(model_config.get("in_channels", 3))
        self.num_classes = int(model_config.get("num_classes", len(self.alphabet)))
        self.loss_mode = str(model_config.get("loss_mode", checkpoint_config.get("loss_mode", "legacy_logreg"))).lower()
        self.space_char = checkpoint_config.get("space_char", " ")
        self.space_idx = self.alphabet.index(self.space_char) if self.space_char in self.alphabet else None
        self.image_height = int(checkpoint_config.get("image_height", 48))
        self.preprocess_fill = int(checkpoint_config.get("background", 255))
        self.legacy_crop_left = int(checkpoint_config.get("legacy_crop_left", 0))
        self.legacy_crop_right = int(checkpoint_config.get("legacy_crop_right", 0))

        self.model = create_model(
            self.architecture,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            **self.architecture_params,
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        if self.baseline_detector_checkpoint is not None:
            self._load_baseline_detector()

        if verbose:
            self.print_summary()

    def _load_baseline_detector(self) -> None:
        if self.baseline_detector_checkpoint is None:
            return
        if not self.baseline_detector_checkpoint.exists():
            raise FileNotFoundError(f"Baseline detector checkpoint not found: {self.baseline_detector_checkpoint}")

        checkpoint = torch.load(self.baseline_detector_checkpoint, map_location=self.device)
        model_config = checkpoint.get("model_config", {})
        checkpoint_config = checkpoint.get("config", {})
        target_format = str(model_config.get("target_format", checkpoint_config.get("target_format", ""))).lower()
        loss_mode = str(model_config.get("loss_mode", checkpoint_config.get("loss_mode", ""))).lower()
        architecture = normalize_architecture_name(
            model_config.get("architecture", checkpoint_config.get("architecture", "baseline_detector_fcn"))
        )
        architecture_params = dict(
            model_config.get(
                "architecture_params",
                checkpoint_config.get("architecture_params", {}),
            )
            or {}
        )
        in_channels = int(model_config.get("in_channels", checkpoint_config.get("channels", 1)))
        num_classes = int(model_config.get("num_classes", 2))
        if target_format != "baseline_heatmap" and loss_mode != "baseline_heatmap":
            raise ValueError(
                "Baseline detector checkpoint must be trained with loss_mode=baseline_heatmap; "
                f"got loss_mode={loss_mode!r}, target_format={target_format!r}"
            )
        if num_classes != 2:
            raise ValueError(f"Baseline detector checkpoint must have num_classes=2, got {num_classes}")

        model = create_model(
            architecture,
            in_channels=in_channels,
            num_classes=num_classes,
            **architecture_params,
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.baseline_detector_model = model
        self.baseline_detector_in_channels = in_channels
        self.baseline_detector_image_height = int(checkpoint_config.get("image_height", 0) or 0)
        self.baseline_detector_architecture = architecture

    def print_summary(self) -> None:
        epoch = self.checkpoint.get("epoch", "?")
        loss = self.checkpoint.get("loss")
        loss_text = f", loss: {loss:.8f}" if isinstance(loss, float) else ""
        print(f"Using device: {self.device}")
        print(f"Model loaded from epoch {epoch}{loss_text}")
        print(f"Architecture: {self.architecture}")
        if self.architecture_params:
            print(f"Architecture params: {self.architecture_params}")
        print(f"Alphabet size: {len(self.alphabet)}")
        print(f"Loss mode: {self.loss_mode}")
        if self.loss_mode in {"legacy", "legacy_logreg"}:
            print(f"Legacy crop: [{self.legacy_crop_left}, -{self.legacy_crop_right}]")
        print(f"Preprocess scale_x: {self.scale_x:+.4f}")
        print(f"Preprocess y_pad:   {self.y_pad:+.4f}")
        print(f"Preprocess x_pad:   {self.x_pad:.4f}")
        print(f"Baseline crop:      {self.baseline_crop}")
        if self.baseline_crop:
            print(
                f"  top_pad={self.baseline_top_pad:.3f}, "
                f"bottom_pad={self.baseline_bottom_pad:.3f}, "
                f"deskew={self.baseline_deskew}, max_angle={self.baseline_max_angle:.2f}, "
                f"strict_lines={self.baseline_strict_lines}, line_pad={self.baseline_line_pad:.3f}, "
                f"line_pad_px={self.baseline_line_pad_px:.1f}, rectify={self.baseline_rectify}"
            )
            if self.baseline_detector_model is not None:
                print(
                    "  neural_detector="
                    f"{self.baseline_detector_checkpoint} "
                    f"threshold={self.baseline_detector_threshold:.3f} "
                    f"architecture={self.baseline_detector_architecture} "
                    f"curve_smooth={self.baseline_curve_smooth_radius} "
                    f"curve_min_coverage={self.baseline_curve_min_coverage:.3f}"
                )

    def class_label(self, index: int) -> str:
        return display_char(self.idx_to_char.get(index, f"<{index}>"))

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess_pil_3d(image).unsqueeze(0)

    def preprocess_pil_debug(self, image: Image.Image) -> tuple[torch.Tensor, PreprocessDebug]:
        tensor, debug = self._preprocess_pil_3d_with_debug(image, collect_debug=True)
        return tensor.unsqueeze(0), debug

    def _preprocess_pil_3d(self, image: Image.Image) -> torch.Tensor:
        tensor, _ = self._preprocess_pil_3d_with_debug(image, collect_debug=False)
        return tensor

    def _preprocess_pil_3d_with_debug(
        self,
        image: Image.Image,
        collect_debug: bool,
    ) -> tuple[torch.Tensor, PreprocessDebug]:
        debug_metadata: dict[str, Any] = {
            "baseline_crop": self.baseline_crop,
            "baseline_strict_lines": self.baseline_strict_lines,
            "baseline_line_pad": self.baseline_line_pad,
            "baseline_line_pad_px": self.baseline_line_pad_px,
            "baseline_detector_checkpoint": str(self.baseline_detector_checkpoint) if self.baseline_detector_checkpoint else None,
            "baseline_detector_threshold": self.baseline_detector_threshold,
            "baseline_rectify": self.baseline_rectify,
            "baseline_curve_smooth_radius": self.baseline_curve_smooth_radius,
            "baseline_curve_min_coverage": self.baseline_curve_min_coverage,
            "x_pad": self.x_pad,
            "x_pad_mode": "border_median_original",
        }
        debug_images: list[tuple[str, Image.Image]] = []
        image = image.convert("RGB" if self.in_channels == 3 else "L")
        self._append_preprocess_debug_image(
            debug_images,
            collect_debug,
            "preprocess 00 input converted",
            image,
        )

        if self.baseline_crop:
            image, baseline_debug = self._apply_baseline_crop(image, collect_debug=collect_debug)
            debug_metadata.update(baseline_debug.metadata)
            debug_images.extend(baseline_debug.images)
            baseline_step_title = (
                "preprocess 01 after curved baseline rectification"
                if baseline_debug.metadata.get("baseline_status") == "ok_curved"
                else "preprocess 01 after baseline crop"
            )
            self._append_preprocess_debug_image(
                debug_images,
                collect_debug,
                baseline_step_title,
                image,
            )

        image = self._apply_x_pad(image)
        if collect_debug and self.x_pad > 0.0:
            debug_images.append(("preprocess 02 x-pad border median", image.copy()))

        before_y_pad_size = image.size
        image = self._apply_y_pad(image)
        if image.size != before_y_pad_size:
            self._append_preprocess_debug_image(
                debug_images,
                collect_debug,
                "preprocess 03 after y-pad/crop",
                image,
            )

        if image.height != self.image_height:
            new_width = max(1, round(image.width * self.image_height / image.height))
            image = image.resize((new_width, self.image_height), Image.Resampling.BICUBIC)
            self._append_preprocess_debug_image(
                debug_images,
                collect_debug,
                "preprocess 04 resize to network height",
                image,
            )

        before_scale_x_size = image.size
        image = self._apply_scale_x(image)
        if image.size != before_scale_x_size:
            self._append_preprocess_debug_image(
                debug_images,
                collect_debug,
                "preprocess 05 after scale-x",
                image,
            )
        self._append_preprocess_debug_image(
            debug_images,
            collect_debug,
            "preprocess 99 final network input",
            image,
        )

        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.in_channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)

        return tensor.to(self.device), PreprocessDebug(metadata=debug_metadata, images=debug_images)

    @staticmethod
    def _append_preprocess_debug_image(
        debug_images: list[tuple[str, Image.Image]],
        collect_debug: bool,
        title: str,
        image: Image.Image,
    ) -> None:
        if collect_debug:
            debug_images.append((title, image.copy()))

    def _apply_y_pad(self, image: Image.Image) -> Image.Image:
        if self.y_pad == 0.0:
            return image

        delta = int(round(image.height * abs(self.y_pad)))
        if delta <= 0:
            return image

        top = delta // 2
        bottom = delta - top
        if self.y_pad > 0.0:
            return ImageOps.expand(
                image,
                border=(0, top, 0, bottom),
                fill=self._background_fill_value(image),
            )

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

    def _apply_x_pad(self, image: Image.Image) -> Image.Image:
        if self.x_pad == 0.0:
            return image

        delta = int(round(image.width * self.x_pad))
        if delta <= 0:
            return image

        array = np.asarray(image)
        left_fill, right_fill = self._side_background_values(array)
        if array.ndim == 2:
            padded = np.empty((array.shape[0], array.shape[1] + delta * 2), dtype=array.dtype)
            padded[:, :delta] = left_fill
            padded[:, delta : delta + array.shape[1]] = array
            padded[:, delta + array.shape[1] :] = right_fill
        elif array.ndim == 3:
            padded = np.empty((array.shape[0], array.shape[1] + delta * 2, array.shape[2]), dtype=array.dtype)
            padded[:, :delta, :] = left_fill
            padded[:, delta : delta + array.shape[1], :] = array
            padded[:, delta + array.shape[1] :, :] = right_fill
        else:
            raise ValueError(f"Unsupported image array shape for x_pad: {array.shape}")
        return Image.fromarray(padded, mode=image.mode)

    @staticmethod
    def _side_background_values(array: np.ndarray) -> tuple[np.ndarray | int, np.ndarray | int]:
        width = int(array.shape[1])
        band_width = max(1, min(width, max(3, int(round(width * 0.04)))))
        left_band = array[:, :band_width]
        right_band = array[:, width - band_width :]

        if array.ndim == 2:
            return (
                np.asarray(np.median(left_band), dtype=array.dtype),
                np.asarray(np.median(right_band), dtype=array.dtype),
            )

        return (
            np.asarray(np.median(left_band.reshape(-1, array.shape[2]), axis=0), dtype=array.dtype),
            np.asarray(np.median(right_band.reshape(-1, array.shape[2]), axis=0), dtype=array.dtype),
        )

    def _pil_fill_value(self, mode: str) -> int | tuple[int, int, int]:
        fill = max(0, min(255, self.preprocess_fill))
        if mode == "RGB":
            return (fill, fill, fill)
        return fill

    def _background_fill_value(self, image: Image.Image) -> int | tuple[int, int, int]:
        array = np.asarray(image)
        if array.size == 0:
            return self._pil_fill_value(image.mode)

        if array.ndim == 2:
            border = np.concatenate((array[0, :], array[-1, :], array[:, 0], array[:, -1]))
            return int(np.median(border))

        if array.ndim == 3 and array.shape[2] >= 3:
            border = np.concatenate(
                (
                    array[0, :, :],
                    array[-1, :, :],
                    array[:, 0, :],
                    array[:, -1, :],
                ),
                axis=0,
            )
            values = np.median(border[:, :3], axis=0).round().astype(np.uint8).tolist()
            return tuple(int(value) for value in values[:3])

        return self._pil_fill_value(image.mode)

    def _apply_baseline_crop(self, image: Image.Image, collect_debug: bool) -> tuple[Image.Image, PreprocessDebug]:
        if cv2 is None and self.baseline_detector_model is None:
            raise RuntimeError("opencv-python is required for baseline_crop inference preprocessing")

        debug_images: list[tuple[str, Image.Image]] = []
        curve_metadata: dict[str, Any] = {}
        if self.baseline_rectify == "curved":
            if self.baseline_detector_model is None:
                curve_metadata["baseline_curve_status"] = "curved_requires_neural_detector_fallback_lines"
            else:
                curved_detection = self._detect_baseline_curves(image)
                curve_metadata = {
                    "baseline_curve_status": curved_detection["status"],
                    "baseline_curve_smooth_radius": self.baseline_curve_smooth_radius,
                    "baseline_curve_min_coverage": self.baseline_curve_min_coverage,
                }
                if collect_debug:
                    if "heatmaps" in curved_detection:
                        debug_images.append(
                            (
                                "baseline curved heatmap",
                                self._draw_baseline_heatmap_debug(image, curved_detection["heatmaps"]),
                            )
                        )
                    if curved_detection["ok"]:
                        debug_images.append(
                            (
                                "baseline curved lines",
                                self._draw_baseline_curves_debug(image, curved_detection, curved_detection["crop_box"]),
                            )
                        )
                    else:
                        debug_images.append(("baseline curved mask failed", Image.fromarray(curved_detection["cleaned_mask"])))
                if curved_detection["ok"]:
                    rectified = self._rectify_baseline_curves(image, curved_detection)
                    if collect_debug:
                        debug_images.append(("baseline curved rectified", rectified))
                    metadata = {
                        "baseline_status": "ok_curved",
                        "baseline_strict_lines": self.baseline_strict_lines,
                        "baseline_line_pad": self.baseline_line_pad,
                        "baseline_line_pad_px": self.baseline_line_pad_px,
                        "baseline_angle_degrees": float(curved_detection["angle_degrees"]),
                        "baseline_residual_angle_degrees": 0.0,
                        "baseline_crop_box": tuple(int(value) for value in curved_detection["crop_box"]),
                        "baseline_text_bbox": tuple(int(value) for value in curved_detection["text_bbox"]),
                        "baseline_text_height": int(curved_detection["text_height"]),
                        "baseline_foreground_pixels": int(curved_detection["foreground_pixels"]),
                        "baseline_confidence": float(curved_detection["confidence"]),
                        "baseline_bottom_confidence": float(curved_detection["bottom_confidence"]),
                        "baseline_inlier_ratio": float(curved_detection["inlier_ratio"]),
                        "baseline_profile_coverage": float(curved_detection["profile_coverage"]),
                        "baseline_residual_mad": float(curved_detection["residual_mad"]),
                        "baseline_residual_rmse": float(curved_detection["residual_rmse"]),
                        "baseline_candidate_count": 1,
                        "baseline_method": curved_detection.get("method", "unknown"),
                        "baseline_mask": curved_detection.get("mask_name", "unknown"),
                        "topline_angle_degrees": float(curved_detection["topline_angle_degrees"]),
                        "topline_confidence": float(curved_detection["topline_confidence"]),
                        "topline_method": curved_detection.get("topline_method", "unknown"),
                        "topline_inlier_ratio": float(curved_detection.get("topline_inlier_ratio", 0.0)),
                        "topline_profile_coverage": float(curved_detection.get("topline_profile_coverage", 0.0)),
                        "topline_residual_mad": float(curved_detection.get("topline_residual_mad", 0.0)),
                        "baseline_curve_height_median": float(curved_detection["curve_height_median"]),
                        "baseline_curve_height_min": float(curved_detection["curve_height_min"]),
                        "baseline_curve_height_max": float(curved_detection["curve_height_max"]),
                        "baseline_curve_center_smooth_radius": int(curved_detection["curve_center_smooth_radius"]),
                        "baseline_curve_pad_px": float(curved_detection["curve_pad_px"]),
                        "baseline_curve_width": int(curved_detection["curve_width"]),
                        "baseline_curve_coverage": float(curved_detection["curve_coverage"]),
                        "baseline_curve_output_size": (int(rectified.width), int(rectified.height)),
                    }
                    metadata.update(curve_metadata)
                    return rectified, PreprocessDebug(metadata=metadata, images=debug_images)
                curve_metadata["baseline_curve_fallback"] = "lines"

        first = self._detect_baseline(image)
        if not first["ok"]:
            if collect_debug:
                debug_images.append(("baseline mask", Image.fromarray(first["cleaned_mask"])))
            metadata = {
                "baseline_status": first["status"],
                "baseline_strict_lines": self.baseline_strict_lines,
                "baseline_line_pad": self.baseline_line_pad,
                "baseline_line_pad_px": self.baseline_line_pad_px,
                "baseline_foreground_pixels": int(first["foreground_pixels"]),
            }
            for source_key, target_key in (
                ("angle_degrees", "baseline_angle_degrees"),
                ("confidence", "baseline_confidence"),
                ("inlier_ratio", "baseline_inlier_ratio"),
                ("profile_coverage", "baseline_profile_coverage"),
                ("residual_mad", "baseline_residual_mad"),
                ("baseline_angle_degrees", "baseline_angle_degrees"),
                ("baseline_confidence", "baseline_confidence"),
                ("baseline_inlier_ratio", "baseline_inlier_ratio"),
                ("baseline_profile_coverage", "baseline_profile_coverage"),
                ("baseline_residual_mad", "baseline_residual_mad"),
                ("candidate_count", "baseline_candidate_count"),
                ("method", "baseline_method"),
                ("mask_name", "baseline_mask"),
            ):
                if source_key in first:
                    metadata[target_key] = first[source_key]
            metadata.update(curve_metadata)
            return image, PreprocessDebug(
                metadata=metadata,
                images=debug_images,
            )

        working_image = image
        detection = first
        status = "ok"
        original_angle = float(first["angle_degrees"])

        if self.baseline_deskew and abs(original_angle) >= 0.25:
            if collect_debug:
                debug_images.append(("baseline on original", self._draw_baseline_overlay(image, first)))
                debug_images.append(("baseline lines original", self._draw_baseline_lines_debug(image, first)))
            rotated = image.rotate(
                original_angle,
                expand=True,
                resample=Image.Resampling.BICUBIC,
                fillcolor=self._background_fill_value(image),
            )
            second = self._detect_baseline(rotated)
            if second["ok"]:
                working_image = rotated
                detection = second
                status = "ok_deskewed"
            elif self.baseline_strict_lines:
                metadata = {
                    "baseline_status": f"strict_lines_rotated_detection_failed_after_{second['status']}",
                    "baseline_strict_lines": self.baseline_strict_lines,
                    "baseline_line_pad": self.baseline_line_pad,
                    "baseline_line_pad_px": self.baseline_line_pad_px,
                    "baseline_angle_degrees": original_angle,
                    "baseline_foreground_pixels": int(second.get("foreground_pixels", first["foreground_pixels"])),
                }
                metadata.update(curve_metadata)
                if collect_debug:
                    debug_images.append(("baseline rotated detection failed", rotated))
                    debug_images.append(("baseline rotated cleaned mask", Image.fromarray(second["cleaned_mask"])))
                return image, PreprocessDebug(metadata=metadata, images=debug_images)
            else:
                status = f"ok_without_deskew_after_{second['status']}"

        cropped = self._crop_with_fill(working_image, detection["crop_box"])
        if collect_debug:
            debug_images.append(
                (
                    "baseline detected lines",
                    self._draw_baseline_lines_debug(working_image, detection, detection["crop_box"]),
                )
            )
            overlay = self._draw_baseline_overlay(working_image, detection, detection["crop_box"])
            debug_images.append(("baseline crop overlay", overlay))
            debug_images.append(("baseline cleaned mask", Image.fromarray(detection["cleaned_mask"])))
            debug_images.append(("baseline cropped image", cropped))

        metadata = {
            "baseline_status": status,
            "baseline_strict_lines": self.baseline_strict_lines,
            "baseline_line_pad": self.baseline_line_pad,
            "baseline_line_pad_px": self.baseline_line_pad_px,
            "baseline_angle_degrees": original_angle,
            "baseline_residual_angle_degrees": float(detection["angle_degrees"]),
            "baseline_crop_box": tuple(int(value) for value in detection["crop_box"]),
            "baseline_text_bbox": tuple(int(value) for value in detection["text_bbox"]),
            "baseline_text_height": int(detection["text_height"]),
            "baseline_foreground_pixels": int(detection["foreground_pixels"]),
            "baseline_confidence": float(detection["confidence"]),
            "baseline_bottom_confidence": float(detection.get("bottom_confidence", detection["confidence"])),
            "baseline_inlier_ratio": float(detection["inlier_ratio"]),
            "baseline_profile_coverage": float(detection["profile_coverage"]),
            "baseline_residual_mad": float(detection["residual_mad"]),
            "baseline_residual_rmse": float(detection["residual_rmse"]),
            "baseline_candidate_count": int(detection.get("candidate_count", 0)),
            "baseline_method": detection.get("method", "unknown"),
            "baseline_mask": detection.get("mask_name", "unknown"),
        }
        if detection.get("topline_detected"):
            metadata.update(
                {
                    "topline_angle_degrees": float(detection["topline_angle_degrees"]),
                    "topline_confidence": float(detection["topline_confidence"]),
                    "topline_method": detection.get("topline_method", "unknown"),
                    "topline_inlier_ratio": float(detection.get("topline_inlier_ratio", 0.0)),
                    "topline_profile_coverage": float(detection.get("topline_profile_coverage", 0.0)),
                    "topline_residual_mad": float(detection.get("topline_residual_mad", 0.0)),
                }
            )
        if "rejected_baseline_angle_degrees" in detection:
            metadata["baseline_rejected_angle_degrees"] = float(detection["rejected_baseline_angle_degrees"])
        if "rejected_baseline_confidence" in detection:
            metadata["baseline_rejected_confidence"] = float(detection["rejected_baseline_confidence"])
        metadata.update(curve_metadata)
        return cropped, PreprocessDebug(metadata=metadata, images=debug_images)

    def _detect_baseline(self, image: Image.Image) -> dict[str, Any]:
        if self.baseline_detector_model is not None:
            return self._detect_baseline_neural(image)
        return self._detect_baseline_heuristic(image)

    def _detect_baseline_neural(self, image: Image.Image) -> dict[str, Any]:
        if self.baseline_detector_model is None:
            return self._detect_baseline_heuristic(image)

        heatmaps, cleaned_mask, foreground_pixels, scale_x, scale_y = self._baseline_detector_heatmaps(image)
        top_line = self._line_from_baseline_heatmap(heatmaps[0], "neural_top")
        bottom_line = self._line_from_baseline_heatmap(heatmaps[1], "neural_bottom")
        if top_line is None or bottom_line is None:
            return {
                "ok": False,
                "status": "neural_baseline_fit_failed",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_heatmap",
                "mask_name": "baseline_detector",
            }

        top_line = self._scale_baseline_line(top_line, scale_x=scale_x, scale_y=scale_y)
        bottom_line = self._scale_baseline_line(bottom_line, scale_x=scale_x, scale_y=scale_y)
        x_mid = max(0.0, (image.width - 1) * 0.5)
        top_mid = float(top_line["slope"]) * x_mid + float(top_line["intercept"])
        bottom_mid = float(bottom_line["slope"]) * x_mid + float(bottom_line["intercept"])
        if top_mid >= bottom_mid:
            return {
                "ok": False,
                "status": "neural_baseline_lines_reversed",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_heatmap",
                "mask_name": "baseline_detector",
                "topline_confidence": float(top_line["confidence"]),
                "baseline_confidence": float(bottom_line["confidence"]),
            }

        xs = np.concatenate((top_line["profile_x"], bottom_line["profile_x"]))
        ys = np.concatenate((top_line["profile_y"], bottom_line["profile_y"]))
        if xs.size == 0 or ys.size == 0:
            return {
                "ok": False,
                "status": "neural_baseline_empty_profiles",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_heatmap",
                "mask_name": "baseline_detector",
            }

        paired_crop = self._paired_baseline_crop_box(
            top_slope=float(top_line["slope"]),
            top_intercept=float(top_line["intercept"]),
            bottom_slope=float(bottom_line["slope"]),
            bottom_intercept=float(bottom_line["intercept"]),
            xs=xs,
            ys=ys,
            image_width=image.width,
        )
        if paired_crop is None:
            return {
                "ok": False,
                "status": "neural_baseline_crop_failed",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_heatmap",
                "mask_name": "baseline_detector",
                "topline_confidence": float(top_line["confidence"]),
                "baseline_confidence": float(bottom_line["confidence"]),
            }
        crop_box, text_height = paired_crop
        confidence = min(float(top_line["confidence"]), float(bottom_line["confidence"]))
        angle_degrees = math.degrees(math.atan(float(bottom_line["slope"])))
        top_y = top_line["slope"] * xs + top_line["intercept"]
        bottom_y = bottom_line["slope"] * xs + bottom_line["intercept"]
        text_bbox = (
            max(0, int(math.floor(float(xs.min())))),
            max(0, int(math.floor(float(min(top_y.min(), ys.min()))))),
            min(image.width, int(math.ceil(float(xs.max()) + 1.0))),
            min(image.height, int(math.ceil(float(max(bottom_y.max(), ys.max())) + 1.0))),
        )
        return {
            "ok": True,
            "status": "ok",
            "mask_name": "baseline_detector",
            "method": "neural_heatmap",
            "cleaned_mask": cleaned_mask,
            "foreground_pixels": foreground_pixels,
            "slope": float(bottom_line["slope"]),
            "intercept": float(bottom_line["intercept"]),
            "angle_degrees": float(angle_degrees),
            "confidence": float(confidence),
            "bottom_confidence": float(bottom_line["confidence"]),
            "inlier_ratio": float(bottom_line["inlier_ratio"]),
            "profile_coverage": float(bottom_line["profile_coverage"]),
            "residual_mad": float(bottom_line["residual_mad"]),
            "residual_rmse": float(bottom_line["residual_rmse"]),
            "profile_x": bottom_line["profile_x"],
            "profile_y": bottom_line["profile_y"],
            "inlier_mask": bottom_line["inlier_mask"],
            "crop_box": crop_box,
            "text_bbox": text_bbox,
            "text_height": int(text_height),
            "bottom_slope": float(bottom_line["slope"]),
            "bottom_intercept": float(bottom_line["intercept"]),
            "bottom_angle_degrees": float(angle_degrees),
            "topline_detected": True,
            **self._topline_metadata(top_line),
        }

    def _baseline_detector_heatmaps(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray, int, float, float]:
        if self.baseline_detector_model is None:
            raise RuntimeError("baseline detector model is not loaded")

        tensor, scale_x, scale_y, detector_size = self._baseline_detector_input(image)
        with torch.no_grad():
            logits = self.baseline_detector_model(tensor)
            if logits.dim() != 4 or logits.size(1) != 2:
                raise ValueError(
                    "Baseline detector must output logits shaped (B, 2, H, W), "
                    f"got {tuple(logits.shape)}"
                )
            probs = torch.sigmoid(logits[:, :2])
            if probs.shape[-2:] != (detector_size[1], detector_size[0]):
                probs = torch.nn.functional.interpolate(
                    probs,
                    size=(detector_size[1], detector_size[0]),
                    mode="bilinear",
                    align_corners=False,
                )

        heatmaps = probs[0].detach().cpu().numpy()
        cleaned_mask = self._baseline_heatmap_mask(heatmaps, image.size)
        foreground_pixels = int(np.count_nonzero(cleaned_mask))
        return heatmaps, cleaned_mask, foreground_pixels, float(scale_x), float(scale_y)

    def _detect_baseline_curves(self, image: Image.Image) -> dict[str, Any]:
        if self.baseline_detector_model is None:
            return {
                "ok": False,
                "status": "curved_requires_neural_detector",
                "cleaned_mask": np.zeros((image.height, image.width), dtype=np.uint8),
                "foreground_pixels": 0,
                "method": "neural_curve",
                "mask_name": "baseline_detector",
            }

        heatmaps, cleaned_mask, foreground_pixels, scale_x, scale_y = self._baseline_detector_heatmaps(image)
        top_curve = self._curve_from_baseline_heatmap(heatmaps[0], "neural_curve_top")
        bottom_curve = self._curve_from_baseline_heatmap(heatmaps[1], "neural_curve_bottom")
        if top_curve is None or bottom_curve is None:
            return {
                "ok": False,
                "status": "curved_baseline_fit_failed",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_curve",
                "mask_name": "baseline_detector",
            }

        top_curve = self._scale_baseline_curve(top_curve, scale_x=scale_x, scale_y=scale_y)
        bottom_curve = self._scale_baseline_curve(bottom_curve, scale_x=scale_x, scale_y=scale_y)
        x_left = max(float(top_curve["curve_x"][0]), float(bottom_curve["curve_x"][0]))
        x_right = min(float(top_curve["curve_x"][-1]), float(bottom_curve["curve_x"][-1]))
        if x_right - x_left < max(8.0, image.width * 0.08):
            return {
                "ok": False,
                "status": "curved_baseline_overlap_too_small",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_curve",
                "mask_name": "baseline_detector",
            }

        curve_x = np.arange(int(math.ceil(x_left)), int(math.floor(x_right)) + 1, dtype=np.float64)
        if curve_x.size < 8:
            return {
                "ok": False,
                "status": "curved_baseline_too_narrow",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_curve",
                "mask_name": "baseline_detector",
            }

        top_y = np.interp(curve_x, top_curve["curve_x"], top_curve["curve_y"])
        bottom_y = np.interp(curve_x, bottom_curve["curve_x"], bottom_curve["curve_y"])
        top_scores = np.interp(curve_x, top_curve["curve_x"], top_curve["curve_scores"])
        bottom_scores = np.interp(curve_x, bottom_curve["curve_x"], bottom_curve["curve_scores"])
        heights = bottom_y - top_y
        good = heights > 2.0
        if int(np.count_nonzero(good)) < max(8, int(round(curve_x.size * 0.70))):
            return {
                "ok": False,
                "status": "curved_baseline_lines_reversed",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_curve",
                "mask_name": "baseline_detector",
                "topline_confidence": float(top_curve["confidence"]),
                "baseline_confidence": float(bottom_curve["confidence"]),
            }
        if not bool(np.all(good)):
            top_y = np.interp(curve_x, curve_x[good], top_y[good])
            bottom_y = np.interp(curve_x, curve_x[good], bottom_y[good])
            heights = bottom_y - top_y

        line_height = float(np.median(heights))
        if line_height <= 2.0:
            return {
                "ok": False,
                "status": "curved_baseline_height_too_small",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_curve",
                "mask_name": "baseline_detector",
            }

        raw_top_y = top_y.copy()
        raw_bottom_y = bottom_y.copy()
        raw_center_y = (raw_top_y + raw_bottom_y) * 0.5
        center_scores = np.minimum(top_scores, bottom_scores)
        center_slope, center_intercept = self._fit_line_from_curve(curve_x, raw_center_y, center_scores)
        center_trend = center_slope * curve_x + center_intercept
        center_residual = raw_center_y - center_trend
        center_smooth_radius = max(self.baseline_curve_smooth_radius, int(round(curve_x.size * 0.015)))
        center_y = center_trend + self._smooth_curve_1d(center_residual, center_smooth_radius)
        top_y = center_y - line_height * 0.5
        bottom_y = center_y + line_height * 0.5

        confidence = float(min(np.mean(top_scores), np.mean(bottom_scores)))
        bottom_slope, bottom_intercept = self._fit_line_from_curve(curve_x, bottom_y, bottom_scores)
        top_slope, top_intercept = self._fit_line_from_curve(curve_x, top_y, top_scores)
        angle_degrees = math.degrees(math.atan(bottom_slope))
        pad_px = max(0.0, line_height * self.baseline_line_pad + self.baseline_line_pad_px)
        crop_box = (
            max(0, int(math.floor(float(curve_x.min())))),
            int(math.floor(float(top_y.min()) - pad_px)),
            min(image.width, int(math.ceil(float(curve_x.max()) + 1.0))),
            int(math.ceil(float(bottom_y.max()) + 1.0 + pad_px)),
        )
        text_bbox = (
            max(0, int(math.floor(float(curve_x.min())))),
            max(0, int(math.floor(float(top_y.min())))),
            min(image.width, int(math.ceil(float(curve_x.max()) + 1.0))),
            min(image.height, int(math.ceil(float(bottom_y.max()) + 1.0))),
        )
        return {
            "ok": True,
            "status": "ok_curved",
            "mask_name": "baseline_detector",
            "method": "neural_curve",
            "cleaned_mask": cleaned_mask,
            "heatmaps": heatmaps,
            "foreground_pixels": foreground_pixels,
            "slope": float(bottom_slope),
            "intercept": float(bottom_intercept),
            "angle_degrees": float(angle_degrees),
            "confidence": confidence,
            "bottom_confidence": float(bottom_curve["confidence"]),
            "inlier_ratio": 1.0,
            "profile_coverage": float(bottom_curve["profile_coverage"]),
            "residual_mad": 0.0,
            "residual_rmse": 0.0,
            "profile_x": bottom_curve["profile_x"],
            "profile_y": bottom_curve["profile_y"],
            "inlier_mask": np.ones(bottom_curve["profile_x"].shape, dtype=bool),
            "crop_box": crop_box,
            "text_bbox": text_bbox,
            "text_height": int(round(line_height)),
            "bottom_slope": float(bottom_slope),
            "bottom_intercept": float(bottom_intercept),
            "bottom_angle_degrees": float(angle_degrees),
            "topline_detected": True,
            "topline_slope": float(top_slope),
            "topline_intercept": float(top_intercept),
            "topline_angle_degrees": float(math.degrees(math.atan(top_slope))),
            "topline_confidence": float(top_curve["confidence"]),
            "topline_inlier_ratio": 1.0,
            "topline_profile_coverage": float(top_curve["profile_coverage"]),
            "topline_residual_mad": 0.0,
            "topline_residual_rmse": 0.0,
            "topline_method": "neural_curve",
            "topline_profile_x": top_curve["profile_x"],
            "topline_profile_y": top_curve["profile_y"],
            "topline_inlier_mask": np.ones(top_curve["profile_x"].shape, dtype=bool),
            "curve_x": curve_x,
            "top_curve_y": top_y,
            "bottom_curve_y": bottom_y,
            "raw_top_curve_y": raw_top_y,
            "raw_bottom_curve_y": raw_bottom_y,
            "curve_center_y": center_y,
            "top_curve_scores": top_scores,
            "bottom_curve_scores": bottom_scores,
            "curve_height_median": line_height,
            "curve_height_min": float(np.min(heights)),
            "curve_height_max": float(np.max(heights)),
            "curve_center_smooth_radius": int(center_smooth_radius),
            "curve_pad_px": float(pad_px),
            "curve_width": int(curve_x.size),
            "curve_coverage": float(min(top_curve["profile_coverage"], bottom_curve["profile_coverage"])),
        }

    def _curve_from_baseline_heatmap(self, heatmap: np.ndarray, method: str) -> dict[str, Any] | None:
        if heatmap.ndim != 2 or heatmap.size == 0:
            return None
        height, width = heatmap.shape
        scores = heatmap.max(axis=0).astype(np.float64)
        y_positions = heatmap.argmax(axis=0).astype(np.float64)
        keep = scores >= self.baseline_detector_threshold
        min_points = max(8, int(round(width * self.baseline_curve_min_coverage)))
        if int(np.count_nonzero(keep)) < min_points:
            return None

        profile_x = np.flatnonzero(keep).astype(np.float64)
        profile_y = y_positions[keep].astype(np.float64)
        profile_scores = np.maximum(scores[keep].astype(np.float64), 1e-3)
        profile_coverage = float(profile_x.size) / max(1.0, float(width))
        x_start = int(profile_x.min())
        x_end = int(profile_x.max())
        if x_end - x_start + 1 < min_points:
            return None

        curve_x = np.arange(x_start, x_end + 1, dtype=np.float64)
        raw_curve_y = np.interp(curve_x, profile_x, profile_y)
        curve_scores = np.interp(curve_x, profile_x, profile_scores)
        curve_y = self._smooth_curve_1d(raw_curve_y, self.baseline_curve_smooth_radius)
        curve_y = np.clip(curve_y, 0.0, float(height - 1))
        return {
            "method": method,
            "profile_x": profile_x,
            "profile_y": profile_y,
            "profile_scores": profile_scores,
            "profile_coverage": profile_coverage,
            "curve_x": curve_x,
            "raw_curve_y": raw_curve_y,
            "curve_y": curve_y,
            "curve_scores": curve_scores,
            "confidence": float(np.mean(profile_scores)),
        }

    @staticmethod
    def _fit_line_from_curve(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        if xs.size < 2:
            return 0.0, float(ys[0]) if ys.size else 0.0
        safe_weights = np.maximum(weights.astype(np.float64), 1e-3)
        slope, intercept = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), deg=1, w=safe_weights)
        return float(slope), float(intercept)

    @staticmethod
    def _scale_baseline_curve(curve: dict[str, Any], scale_x: float, scale_y: float) -> dict[str, Any]:
        scale_x = max(scale_x, 1e-6)
        scale_y = max(scale_y, 1e-6)
        scaled = dict(curve)
        for key in ("profile_x", "curve_x"):
            scaled[key] = np.asarray(curve[key], dtype=np.float64) / scale_x
        for key in ("profile_y", "raw_curve_y", "curve_y"):
            scaled[key] = np.asarray(curve[key], dtype=np.float64) / scale_y
        scaled["profile_scores"] = np.asarray(curve["profile_scores"], dtype=np.float64)
        scaled["curve_scores"] = np.asarray(curve["curve_scores"], dtype=np.float64)
        return scaled

    @staticmethod
    def _smooth_curve_1d(values: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0 or values.size <= 2:
            return values.astype(np.float64)
        smoothed = TextRecognizer._median_smooth_1d(values.astype(np.float64), radius=radius)
        kernel_size = radius * 2 + 1
        padded = np.pad(smoothed, (radius, radius), mode="edge")
        kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
        return np.convolve(padded, kernel, mode="valid")

    def _baseline_detector_input(self, image: Image.Image) -> tuple[torch.Tensor, float, float, tuple[int, int]]:
        mode = "RGB" if self.baseline_detector_in_channels == 3 else "L"
        detector_image = image.convert(mode)
        if self.baseline_detector_image_height > 0 and detector_image.height != self.baseline_detector_image_height:
            new_width = max(1, round(detector_image.width * self.baseline_detector_image_height / detector_image.height))
            detector_image = detector_image.resize((new_width, self.baseline_detector_image_height), Image.Resampling.BICUBIC)

        scale_x = detector_image.width / max(1.0, float(image.width))
        scale_y = detector_image.height / max(1.0, float(image.height))
        array = np.asarray(detector_image, dtype=np.float32) / 255.0
        if self.baseline_detector_in_channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), float(scale_x), float(scale_y), detector_image.size

    def _line_from_baseline_heatmap(self, heatmap: np.ndarray, method: str) -> dict[str, Any] | None:
        if heatmap.ndim != 2 or heatmap.size == 0:
            return None
        height, width = heatmap.shape
        scores = heatmap.max(axis=0)
        y_positions = heatmap.argmax(axis=0).astype(np.float64)
        keep = scores >= self.baseline_detector_threshold
        min_points = max(6, int(round(width * 0.08)))
        if int(np.count_nonzero(keep)) < min_points:
            return None

        profile_x = np.flatnonzero(keep).astype(np.float64)
        profile_y = y_positions[keep].astype(np.float64)
        profile_weights = np.maximum(scores[keep].astype(np.float64), 1e-3)
        profile_coverage = float(profile_x.size) / max(1.0, float(width))
        line = self._fit_baseline_line(
            profile_x,
            profile_y,
            profile_weights,
            image_height=height,
            text_width=width,
            profile_coverage=profile_coverage,
        )
        if line is None:
            return None

        mean_score = float(np.mean(profile_weights))
        line["confidence"] = float(0.55 * float(line["confidence"]) + 0.45 * mean_score)
        line.update(
            {
                "method": method,
                "profile_x": profile_x,
                "profile_y": profile_y,
                "profile_coverage": profile_coverage,
            }
        )
        return line

    @staticmethod
    def _scale_baseline_line(line: dict[str, Any], scale_x: float, scale_y: float) -> dict[str, Any]:
        scale_x = max(scale_x, 1e-6)
        scale_y = max(scale_y, 1e-6)
        scaled = dict(line)
        scaled["slope"] = float(line["slope"]) * scale_x / scale_y
        scaled["intercept"] = float(line["intercept"]) / scale_y
        scaled["profile_x"] = np.asarray(line["profile_x"], dtype=np.float64) / scale_x
        scaled["profile_y"] = np.asarray(line["profile_y"], dtype=np.float64) / scale_y
        scaled["inlier_mask"] = np.asarray(line["inlier_mask"], dtype=bool)
        return scaled

    def _baseline_heatmap_mask(self, heatmaps: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
        combined = np.max(heatmaps, axis=0)
        combined = (combined >= self.baseline_detector_threshold).astype(np.uint8) * 255
        mask_image = Image.fromarray(combined, mode="L").resize(output_size, Image.Resampling.BILINEAR)
        return np.asarray(mask_image, dtype=np.uint8)

    def _detect_baseline_heuristic(self, image: Image.Image) -> dict[str, Any]:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        detections: list[dict[str, Any]] = []
        bbox_fallbacks: list[dict[str, Any]] = []
        best_cleaned_mask: np.ndarray | None = None
        max_foreground_pixels = 0

        for mask_name, raw_mask in self._make_text_mask_candidates(gray):
            cleaned_mask = self._clean_text_mask(raw_mask)
            foreground_pixels = int(np.count_nonzero(cleaned_mask))
            if foreground_pixels > max_foreground_pixels:
                max_foreground_pixels = foreground_pixels
                best_cleaned_mask = cleaned_mask
            if foreground_pixels < max(4, int(round(gray.size * 0.00005))):
                continue

            bounds = self._mask_bounds(cleaned_mask)
            if bounds is None:
                continue
            x_min, x_max, y_min, y_max, xs, ys = bounds
            if not self.baseline_strict_lines:
                bbox_fallbacks.append(
                    self._bbox_baseline_result(
                        mask_name=mask_name,
                        cleaned_mask=cleaned_mask,
                        foreground_pixels=foreground_pixels,
                        xs=xs,
                        ys=ys,
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        status="bbox_fallback",
                        candidate_count=0,
                    )
                )
            detections.extend(
                self._baseline_detections_from_mask(
                    mask_name=mask_name,
                    cleaned_mask=cleaned_mask,
                    foreground_pixels=foreground_pixels,
                    xs=xs,
                    ys=ys,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    image_height=image.height,
                    image_width=image.width,
                )
            )

        candidate_count = len(detections)
        if detections:
            best = max(detections, key=lambda item: float(item["confidence"]))
            best["candidate_count"] = candidate_count
            angle_degrees = float(best["angle_degrees"])
            if abs(angle_degrees) <= self.baseline_max_angle and float(best["confidence"]) >= 0.24:
                best["ok"] = True
                best["status"] = "ok"
                return best

            if self.baseline_strict_lines:
                best["ok"] = False
                best["status"] = (
                    "strict_lines_angle_rejected"
                    if abs(angle_degrees) > self.baseline_max_angle
                    else "strict_lines_low_confidence"
                )
                return best

            if bbox_fallbacks:
                fallback = max(bbox_fallbacks, key=lambda item: int(item["foreground_pixels"]))
                fallback["status"] = (
                    "baseline_angle_rejected_bbox_fallback"
                    if abs(angle_degrees) > self.baseline_max_angle
                    else "baseline_low_confidence_bbox_fallback"
                )
                fallback["candidate_count"] = candidate_count
                fallback["rejected_baseline_angle_degrees"] = angle_degrees
                fallback["rejected_baseline_confidence"] = float(best["confidence"])
                return fallback

        if bbox_fallbacks:
            fallback = max(bbox_fallbacks, key=lambda item: int(item["foreground_pixels"]))
            fallback["status"] = "baseline_fit_failed_bbox_fallback"
            fallback["candidate_count"] = candidate_count
            return fallback

        if best_cleaned_mask is None:
            best_cleaned_mask = np.zeros_like(gray, dtype=np.uint8)
        if max_foreground_pixels < max(4, int(round(gray.size * 0.00005))):
            return {
                "ok": False,
                "status": "not_enough_foreground",
                "cleaned_mask": best_cleaned_mask,
                "foreground_pixels": max_foreground_pixels,
            }

        return {
            "ok": False,
            "status": "baseline_fit_failed",
            "cleaned_mask": best_cleaned_mask,
            "foreground_pixels": max_foreground_pixels,
            "candidate_count": candidate_count,
        }

    @staticmethod
    def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int, np.ndarray, np.ndarray] | None:
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            return None
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        return x_min, x_max, y_min, y_max, xs, ys

    def _baseline_detections_from_mask(
        self,
        mask_name: str,
        cleaned_mask: np.ndarray,
        foreground_pixels: int,
        xs: np.ndarray,
        ys: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        image_height: int,
        image_width: int,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        text_width = x_max - x_min + 1
        min_points = max(6, int(round(text_width * 0.08)))

        profile_specs = (
            ("lower_q80", 0.80, 2),
            ("lower_q88", 0.88, 2),
            ("lower_q94", 0.94, 3),
            ("lower_edge", 1.00, 3),
        )
        for method, quantile, smooth_radius in profile_specs:
            profile_x, profile_y, profile_weights = self._baseline_profile_points(
                cleaned_mask,
                x_min,
                x_max,
                quantile=quantile,
                smooth_radius=smooth_radius,
            )
            profile_coverage = float(profile_x.size) / max(1.0, float(text_width))
            if profile_x.size < min_points:
                continue
            line = self._fit_baseline_line(
                profile_x,
                profile_y,
                profile_weights,
                image_height=image_height,
                text_width=text_width,
                profile_coverage=profile_coverage,
            )
            if line is None:
                continue
            detection = self._build_baseline_result(
                mask_name=mask_name,
                method=method,
                cleaned_mask=cleaned_mask,
                foreground_pixels=foreground_pixels,
                xs=xs,
                ys=ys,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                image_width=image_width,
                profile_x=profile_x,
                profile_y=profile_y,
                profile_coverage=profile_coverage,
                line=line,
            )
            if detection is not None:
                detections.append(detection)

        component_x, component_y, component_weights, component_coverage = self._component_bottom_points(
            cleaned_mask,
            x_min=x_min,
            x_max=x_max,
        )
        if component_x.size >= max(4, int(round(text_width * 0.015))):
            line = self._fit_baseline_line(
                component_x,
                component_y,
                component_weights,
                image_height=image_height,
                text_width=text_width,
                profile_coverage=component_coverage,
            )
            if line is not None:
                detection = self._build_baseline_result(
                    mask_name=mask_name,
                    method="component_bottoms",
                    cleaned_mask=cleaned_mask,
                    foreground_pixels=foreground_pixels,
                    xs=xs,
                    ys=ys,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    image_width=image_width,
                    profile_x=component_x,
                    profile_y=component_y,
                    profile_coverage=component_coverage,
                    line=line,
                )
                if detection is not None:
                    detections.append(detection)

        return detections

    def _build_baseline_result(
        self,
        mask_name: str,
        method: str,
        cleaned_mask: np.ndarray,
        foreground_pixels: int,
        xs: np.ndarray,
        ys: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        image_width: int,
        profile_x: np.ndarray,
        profile_y: np.ndarray,
        profile_coverage: float,
        line: dict[str, Any],
    ) -> dict[str, Any] | None:
        slope = float(line["slope"])
        intercept = float(line["intercept"])
        angle_degrees = math.degrees(math.atan(slope))
        bottom_confidence = float(line["confidence"])
        top_line = self._detect_top_text_line(
            cleaned_mask,
            x_min=x_min,
            x_max=x_max,
            image_height=cleaned_mask.shape[0],
        )
        if top_line is None:
            if self.baseline_strict_lines:
                return None
            crop_box, text_height = self._baseline_crop_box(
                slope=slope,
                intercept=intercept,
                xs=xs,
                ys=ys,
                image_width=image_width,
            )
        else:
            paired_crop = self._paired_baseline_crop_box(
                top_slope=float(top_line["slope"]),
                top_intercept=float(top_line["intercept"]),
                bottom_slope=slope,
                bottom_intercept=intercept,
                xs=xs,
                ys=ys,
                image_width=image_width,
            )
            if paired_crop is None:
                return None
            crop_box, text_height = paired_crop
        top_confidence = float(top_line["confidence"]) if top_line is not None else None
        confidence = min(bottom_confidence, top_confidence) if self.baseline_strict_lines and top_confidence is not None else bottom_confidence
        return {
            "ok": True,
            "status": "candidate",
            "mask_name": mask_name,
            "method": method,
            "cleaned_mask": cleaned_mask,
            "foreground_pixels": foreground_pixels,
            "slope": slope,
            "intercept": intercept,
            "angle_degrees": float(angle_degrees),
            "confidence": float(confidence),
            "bottom_confidence": bottom_confidence,
            "inlier_ratio": float(line["inlier_ratio"]),
            "profile_coverage": float(profile_coverage),
            "residual_mad": float(line["residual_mad"]),
            "residual_rmse": float(line["residual_rmse"]),
            "profile_x": profile_x,
            "profile_y": profile_y,
            "inlier_mask": line["inlier_mask"],
            "crop_box": crop_box,
            "text_bbox": (x_min, y_min, x_max + 1, y_max + 1),
            "text_height": int(text_height),
            "bottom_slope": slope,
            "bottom_intercept": intercept,
            "bottom_angle_degrees": float(angle_degrees),
            "topline_detected": top_line is not None,
            **self._topline_metadata(top_line),
        }

    def _bbox_baseline_result(
        self,
        mask_name: str,
        cleaned_mask: np.ndarray,
        foreground_pixels: int,
        xs: np.ndarray,
        ys: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        status: str,
        candidate_count: int,
    ) -> dict[str, Any]:
        baseline_y = float(np.quantile(ys.astype(np.float64), 0.88))
        topline_y = float(np.quantile(ys.astype(np.float64), 0.02))
        profile_x = np.asarray([x_min, x_max], dtype=np.float64)
        profile_y = np.asarray([baseline_y, baseline_y], dtype=np.float64)
        inlier_mask = np.ones(profile_x.shape, dtype=bool)
        crop_box, text_height = self._paired_baseline_crop_box(
            top_slope=0.0,
            top_intercept=topline_y,
            bottom_slope=0.0,
            bottom_intercept=baseline_y,
            xs=xs,
            ys=ys,
            image_width=cleaned_mask.shape[1],
        )
        text_width = max(1, x_max - x_min + 1)
        return {
            "ok": True,
            "status": status,
            "mask_name": mask_name,
            "method": "bbox_fallback",
            "candidate_count": int(candidate_count),
            "cleaned_mask": cleaned_mask,
            "foreground_pixels": foreground_pixels,
            "slope": 0.0,
            "intercept": baseline_y,
            "angle_degrees": 0.0,
            "confidence": 0.18,
            "inlier_ratio": 1.0,
            "profile_coverage": min(1.0, float(text_width) / max(1.0, float(cleaned_mask.shape[1]))),
            "residual_mad": 0.0,
            "residual_rmse": 0.0,
            "profile_x": profile_x,
            "profile_y": profile_y,
            "inlier_mask": inlier_mask,
            "crop_box": crop_box,
            "text_bbox": (x_min, y_min, x_max + 1, y_max + 1),
            "text_height": int(text_height),
            "bottom_slope": 0.0,
            "bottom_intercept": baseline_y,
            "bottom_angle_degrees": 0.0,
            "topline_detected": True,
            "topline_slope": 0.0,
            "topline_intercept": topline_y,
            "topline_angle_degrees": 0.0,
            "topline_confidence": 0.18,
            "topline_inlier_ratio": 1.0,
            "topline_profile_coverage": min(1.0, float(text_width) / max(1.0, float(cleaned_mask.shape[1]))),
            "topline_residual_mad": 0.0,
            "topline_residual_rmse": 0.0,
            "topline_method": "bbox_fallback",
            "topline_profile_x": profile_x,
            "topline_profile_y": np.asarray([topline_y, topline_y], dtype=np.float64),
            "topline_inlier_mask": inlier_mask,
        }

    @staticmethod
    def _topline_metadata(top_line: dict[str, Any] | None) -> dict[str, Any]:
        if top_line is None:
            return {}
        slope = float(top_line["slope"])
        return {
            "topline_slope": slope,
            "topline_intercept": float(top_line["intercept"]),
            "topline_angle_degrees": float(math.degrees(math.atan(slope))),
            "topline_confidence": float(top_line["confidence"]),
            "topline_inlier_ratio": float(top_line["inlier_ratio"]),
            "topline_profile_coverage": float(top_line["profile_coverage"]),
            "topline_residual_mad": float(top_line["residual_mad"]),
            "topline_residual_rmse": float(top_line["residual_rmse"]),
            "topline_method": top_line.get("method", "unknown"),
            "topline_profile_x": top_line["profile_x"],
            "topline_profile_y": top_line["profile_y"],
            "topline_inlier_mask": top_line["inlier_mask"],
        }

    def _detect_top_text_line(
        self,
        mask: np.ndarray,
        x_min: int,
        x_max: int,
        image_height: int,
    ) -> dict[str, Any] | None:
        text_width = x_max - x_min + 1
        min_points = max(6, int(round(text_width * 0.08)))
        candidates: list[dict[str, Any]] = []

        profile_specs = (
            ("upper_edge", 0.00, 3),
            ("upper_q04", 0.04, 2),
            ("upper_q08", 0.08, 2),
            ("upper_q14", 0.14, 3),
        )
        for method, quantile, smooth_radius in profile_specs:
            profile_x, profile_y, profile_weights = self._baseline_profile_points(
                mask,
                x_min,
                x_max,
                quantile=quantile,
                smooth_radius=smooth_radius,
            )
            profile_coverage = float(profile_x.size) / max(1.0, float(text_width))
            if profile_x.size < min_points:
                continue
            line = self._fit_baseline_line(
                profile_x,
                profile_y,
                profile_weights,
                image_height=image_height,
                text_width=text_width,
                profile_coverage=profile_coverage,
            )
            if line is None:
                continue
            line.update(
                {
                    "method": method,
                    "profile_x": profile_x,
                    "profile_y": profile_y,
                    "profile_coverage": profile_coverage,
                }
            )
            candidates.append(line)

        if not candidates:
            return None
        best = max(candidates, key=lambda item: float(item["confidence"]))
        if float(best["confidence"]) < 0.18:
            return None
        return best

    @staticmethod
    def _component_bottom_points(
        mask: np.ndarray,
        x_min: int,
        x_max: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        components, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        height, width = mask.shape
        min_area = max(2, int(round(height * width * 0.00004)))
        text_width = max(1, x_max - x_min + 1)
        xs: list[float] = []
        ys: list[float] = []
        weights: list[float] = []
        covered_width = 0.0

        for label in range(1, components):
            x, y, component_width, component_height, area = stats[label]
            if area < min_area or component_height < 2:
                continue
            if component_width > width * 0.45 and component_height <= max(2, height * 0.08):
                continue
            xs.append(float(centroids[label][0]))
            ys.append(float(y + component_height - 1))
            weights.append(float(min(8.0, math.sqrt(float(area)))))
            covered_width += min(float(component_width), float(text_width))

        if not xs:
            empty = np.asarray([], dtype=np.float64)
            return empty, empty, empty, 0.0

        order = np.argsort(np.asarray(xs, dtype=np.float64))
        component_x = np.asarray(xs, dtype=np.float64)[order]
        component_y = np.asarray(ys, dtype=np.float64)[order]
        component_weights = np.asarray(weights, dtype=np.float64)[order]
        if component_y.size >= 5:
            component_y = TextRecognizer._median_smooth_1d(component_y, radius=1)
        coverage = min(1.0, covered_width / max(1.0, float(text_width)))
        return component_x, component_y, component_weights, coverage

    def _make_text_mask_candidates(self, gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
        if cv2 is None:
            return [("simple", self._make_text_mask(gray))]

        candidates: list[tuple[str, np.ndarray]] = []
        background_is_bright = self._background_is_bright(gray)
        threshold_type = cv2.THRESH_BINARY_INV if background_is_bright else cv2.THRESH_BINARY

        _, otsu = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
        candidates.append(("otsu", self._normalize_text_mask(otsu)))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, clahe_otsu = cv2.threshold(clahe, 0, 255, threshold_type | cv2.THRESH_OTSU)
        candidates.append(("clahe_otsu", self._normalize_text_mask(clahe_otsu)))

        min_dim = max(3, min(gray.shape))
        block_size = int(max(15, min(61, (min_dim // 2) | 1)))
        if block_size % 2 == 0:
            block_size += 1
        if block_size < min_dim or min_dim >= 15:
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                threshold_type,
                block_size,
                7,
            )
            candidates.append(("adaptive", self._normalize_text_mask(adaptive)))

        # A light black-hat/top-hat candidate helps when background is uneven.
        kernel_width = max(9, min(45, int(round(gray.shape[1] * 0.08)) | 1))
        kernel_height = max(3, min(15, int(round(gray.shape[0] * 0.18)) | 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        if background_is_bright:
            enhanced = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        else:
            enhanced = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, enhanced_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        candidates.append(("morph_contrast", self._normalize_text_mask(enhanced_mask)))

        output: list[tuple[str, np.ndarray]] = []
        seen: set[bytes] = set()
        for name, mask in candidates:
            key = np.packbits(mask > 0).tobytes()
            if key in seen:
                continue
            seen.add(key)
            output.append((name, mask.astype(np.uint8)))
        return output

    def _make_text_mask(self, gray: np.ndarray) -> np.ndarray:
        background_is_bright = self._background_is_bright(gray)
        threshold_type = cv2.THRESH_BINARY_INV if background_is_bright else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
        return self._normalize_text_mask(mask)

    @staticmethod
    def _background_is_bright(gray: np.ndarray) -> bool:
        border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
        return float(np.median(border)) >= 128.0

    @staticmethod
    def _normalize_text_mask(mask: np.ndarray) -> np.ndarray:
        mask = ((mask > 0).astype(np.uint8) * 255)
        foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if foreground_ratio > 0.45:
            mask = 255 - mask
        return mask.astype(np.uint8)

    def _clean_text_mask(self, mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        min_area = max(3, int(round(height * width * 0.00005)))
        thin_height = max(2, int(round(height * 0.08)))
        long_width = max(8, int(round(width * 0.35)))
        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)

        for label in range(1, components):
            x, y, component_width, component_height, area = stats[label]
            if area < min_area:
                continue
            aspect = component_width / max(1, component_height)
            is_long_thin_line = (
                component_width >= long_width
                and component_height <= thin_height
                and aspect >= 8.0
            )
            if is_long_thin_line:
                continue
            cleaned[labels == label] = 255

        if np.count_nonzero(cleaned) == 0:
            return mask
        if min(height, width) >= 4:
            kernel = np.ones((2, 2), dtype=np.uint8)
            opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            if np.count_nonzero(opened) >= np.count_nonzero(cleaned) * 0.35:
                cleaned = opened
        return cleaned

    @staticmethod
    def _baseline_profile_points(
        mask: np.ndarray,
        x_min: int,
        x_max: int,
        quantile: float = 0.88,
        smooth_radius: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        profile_x: list[int] = []
        profile_y: list[float] = []
        profile_weights: list[float] = []
        quantile = max(0.0, min(1.0, float(quantile)))

        for x in range(x_min, x_max + 1):
            column_y = np.flatnonzero(mask[:, x])
            if column_y.size == 0:
                continue
            profile_x.append(x)
            if quantile >= 1.0:
                profile_y.append(float(column_y.max()))
            else:
                profile_y.append(float(np.quantile(column_y.astype(np.float64), quantile)))
            profile_weights.append(float(np.sqrt(column_y.size)))

        if not profile_x:
            empty = np.asarray([], dtype=np.float64)
            return empty, empty, empty

        xs = np.asarray(profile_x, dtype=np.float64)
        ys = np.asarray(profile_y, dtype=np.float64)
        weights = np.asarray(profile_weights, dtype=np.float64)
        if ys.size >= 5 and smooth_radius > 0:
            ys = TextRecognizer._median_smooth_1d(ys, radius=smooth_radius)
        return xs, ys, weights

    @staticmethod
    def _median_smooth_1d(values: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0 or values.size <= 2:
            return values

        output = np.empty_like(values, dtype=np.float64)
        for index in range(values.size):
            left = max(0, index - radius)
            right = min(values.size, index + radius + 1)
            output[index] = float(np.median(values[left:right]))
        return output

    def _fit_baseline_line(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        weights: np.ndarray,
        image_height: int,
        text_width: int,
        profile_coverage: float,
    ) -> dict[str, Any] | None:
        if xs.size < 2:
            return None

        work_x = xs.astype(np.float64)
        work_y = ys.astype(np.float64)
        work_weights = weights.astype(np.float64)
        low = np.quantile(work_y, 0.05)
        high = np.quantile(work_y, 0.98)
        keep = (work_y >= low) & (work_y <= high)
        if int(keep.sum()) >= 2:
            work_x = work_x[keep]
            work_y = work_y[keep]
            work_weights = work_weights[keep]

        if work_x.size < 2:
            return None

        line = self._ransac_baseline_line(work_x, work_y, image_height, text_width)
        if line is None:
            slope, intercept = np.polyfit(work_x, work_y, deg=1, w=work_weights)
        else:
            slope, intercept = line

        inlier_mask = np.ones(work_x.shape, dtype=bool)
        for _ in range(5):
            predicted = slope * work_x + intercept
            residuals = np.abs(work_y - predicted)
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            tolerance = max(2.0, image_height * 0.045, median + mad * 2.8)
            next_keep = residuals <= tolerance
            if int(next_keep.sum()) < max(2, int(round(work_x.size * 0.40))):
                break
            if bool(np.all(next_keep)):
                inlier_mask = next_keep
                break
            inlier_mask = next_keep
            work_x = work_x[next_keep]
            work_y = work_y[next_keep]
            work_weights = work_weights[next_keep]
            slope, intercept = np.polyfit(work_x, work_y, deg=1, w=work_weights)

        if work_x.size < 2:
            return None

        slope, intercept = np.polyfit(work_x, work_y, deg=1, w=work_weights)
        final_residuals = np.abs(work_y - (slope * work_x + intercept))
        residual_mad = float(np.median(final_residuals))
        residual_rmse = float(np.sqrt(np.mean(np.square(final_residuals))))
        original_residuals = np.abs(ys - (slope * xs + intercept))
        tolerance = max(2.0, image_height * 0.055, residual_mad * 2.5)
        original_inlier_mask = original_residuals <= tolerance
        inlier_ratio = float(np.count_nonzero(original_inlier_mask)) / max(1.0, float(xs.size))

        coverage_score = min(1.0, profile_coverage / 0.55)
        inlier_score = max(0.0, min(1.0, (inlier_ratio - 0.25) / 0.65))
        residual_score = max(0.0, min(1.0, 1.0 - residual_mad / max(2.0, image_height * 0.14)))
        confidence = 0.25 * coverage_score + 0.45 * inlier_score + 0.30 * residual_score

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "confidence": float(confidence),
            "inlier_ratio": float(inlier_ratio),
            "residual_mad": residual_mad,
            "residual_rmse": residual_rmse,
            "inlier_mask": original_inlier_mask,
        }

    @staticmethod
    def _ransac_baseline_line(
        xs: np.ndarray,
        ys: np.ndarray,
        image_height: int,
        text_width: int,
    ) -> tuple[float, float] | None:
        if xs.size < 2:
            return None

        sample_count = min(80, int(xs.size))
        sample_indices = np.unique(np.linspace(0, xs.size - 1, sample_count, dtype=np.int64))
        if sample_indices.size < 2:
            return None

        min_dx = max(3.0, float(text_width) * 0.12)
        tolerance = max(2.0, float(image_height) * 0.055)
        best_score: tuple[int, float, float] | None = None
        best_line: tuple[float, float] | None = None

        for left_pos, left_index in enumerate(sample_indices[:-1]):
            x1 = float(xs[left_index])
            y1 = float(ys[left_index])
            for right_index in sample_indices[left_pos + 1:]:
                x2 = float(xs[right_index])
                if abs(x2 - x1) < min_dx:
                    continue
                y2 = float(ys[right_index])
                slope = (y2 - y1) / (x2 - x1)
                if abs(math.degrees(math.atan(slope))) > 25.0:
                    continue
                intercept = y1 - slope * x1
                residuals = np.abs(ys - (slope * xs + intercept))
                inliers = int(np.count_nonzero(residuals <= tolerance))
                if inliers < 2:
                    continue
                median_residual = float(np.median(residuals[residuals <= tolerance]))
                score = (inliers, -median_residual, -abs(slope))
                if best_score is None or score > best_score:
                    best_score = score
                    best_line = (float(slope), float(intercept))

        return best_line


    def _baseline_crop_box(
        self,
        slope: float,
        intercept: float,
        xs: np.ndarray,
        ys: np.ndarray,
        image_width: int,
    ) -> tuple[tuple[int, int, int, int], int]:
        text_top = float(np.quantile(ys, 0.02))
        text_bottom = float(np.quantile(ys, 0.98))
        text_height = max(4.0, text_bottom - text_top + 1.0)
        x_min = int(xs.min())
        x_max = int(xs.max())
        baseline_xs = np.arange(x_min, x_max + 1, dtype=np.float64)
        baseline_ys = slope * baseline_xs + intercept
        baseline_center = float(np.median(baseline_ys))
        above_baseline = max(4.0, baseline_center - text_top)
        if above_baseline < text_height * 0.35:
            above_baseline = max(4.0, text_height * 0.85)

        top_margin = max(1.0, above_baseline * self.baseline_top_pad)
        bottom_margin = max(1.0, above_baseline * self.baseline_bottom_pad)
        top = int(math.floor(min(text_top, float(baseline_ys.min()) - above_baseline) - top_margin))
        bottom = int(math.ceil(max(text_bottom + 1.0, float(baseline_ys.max())) + bottom_margin))
        if bottom <= top:
            bottom = top + max(4, int(round(text_height)))

        return (0, top, image_width, bottom), int(round(text_height))

    def _paired_baseline_crop_box(
        self,
        top_slope: float,
        top_intercept: float,
        bottom_slope: float,
        bottom_intercept: float,
        xs: np.ndarray,
        ys: np.ndarray,
        image_width: int,
    ) -> tuple[tuple[int, int, int, int], int] | None:
        x_min = int(xs.min())
        x_max = int(xs.max())
        line_xs = np.arange(x_min, x_max + 1, dtype=np.float64)
        top_ys = top_slope * line_xs + top_intercept
        bottom_ys = bottom_slope * line_xs + bottom_intercept

        text_top = float(np.quantile(ys, 0.02))
        text_bottom = float(np.quantile(ys, 0.98))
        line_height = float(np.median(bottom_ys - top_ys))
        bbox_height = text_bottom - text_top + 1.0
        text_height = max(4.0, line_height, bbox_height)

        if line_height <= 2.0 or float(np.median(top_ys)) >= float(np.median(bottom_ys)):
            if self.baseline_strict_lines:
                return None
            return self._baseline_crop_box(
                slope=bottom_slope,
                intercept=bottom_intercept,
                xs=xs,
                ys=ys,
                image_width=image_width,
            )

        if self.baseline_strict_lines:
            margin_reference = max(line_height, bbox_height)
            margin = max(0.0, margin_reference * self.baseline_line_pad + self.baseline_line_pad_px)
            top = int(math.floor(float(top_ys.min()) - margin))
            bottom = int(math.ceil(float(bottom_ys.max()) + 1.0 + margin))
            if bottom <= top:
                return None
            return (0, top, image_width, bottom), max(1, int(round(bottom - top)))

        top_margin = max(1.0, text_height * self.baseline_top_pad)
        bottom_margin = max(1.0, text_height * self.baseline_bottom_pad)
        top = int(math.floor(min(text_top, float(top_ys.min())) - top_margin))
        bottom = int(math.ceil(max(text_bottom + 1.0, float(bottom_ys.max()) + 1.0) + bottom_margin))
        if bottom <= top:
            bottom = top + max(4, int(round(text_height)))
        return (0, top, image_width, bottom), int(round(text_height))

    def _rectify_baseline_curves(self, image: Image.Image, detection: dict[str, Any]) -> Image.Image:
        curve_x = np.asarray(detection["curve_x"], dtype=np.float64)
        top_y = np.asarray(detection["top_curve_y"], dtype=np.float64)
        bottom_y = np.asarray(detection["bottom_curve_y"], dtype=np.float64)
        if curve_x.size < 2 or top_y.size != curve_x.size or bottom_y.size != curve_x.size:
            raise ValueError("curved baseline detection contains inconsistent curve arrays")

        heights = np.maximum(bottom_y - top_y, 2.0)
        median_height = float(np.median(heights))
        pad_px = float(detection.get("curve_pad_px", median_height * self.baseline_line_pad + self.baseline_line_pad_px))
        output_height = max(4, int(round(median_height + 2.0 * pad_px)))
        output_width = max(1, int(curve_x.size))

        y_out = np.linspace(0.0, 1.0, output_height, dtype=np.float64)[:, None]
        x_map = np.broadcast_to(curve_x[None, :], (output_height, output_width))
        y_top = top_y[None, :] - pad_px
        y_bottom = bottom_y[None, :] + pad_px
        y_map = y_top + y_out * (y_bottom - y_top)

        return self._remap_image_bilinear(image, x_map=x_map, y_map=y_map)

    def _remap_image_bilinear(self, image: Image.Image, x_map: np.ndarray, y_map: np.ndarray) -> Image.Image:
        source = np.asarray(image)
        if source.ndim not in {2, 3}:
            raise ValueError(f"unsupported image array shape for remap: {source.shape}")

        src_height, src_width = source.shape[:2]
        out_height, out_width = x_map.shape
        fill = self._background_fill_value(image)
        if source.ndim == 2:
            output = np.full((out_height, out_width), int(fill), dtype=np.float64)
        else:
            fill_tuple = fill if isinstance(fill, tuple) else (int(fill), int(fill), int(fill))
            output = np.empty((out_height, out_width, source.shape[2]), dtype=np.float64)
            output[:, :] = np.asarray(fill_tuple[: source.shape[2]], dtype=np.float64)

        valid = (
            (x_map >= 0.0)
            & (x_map <= float(src_width - 1))
            & (y_map >= 0.0)
            & (y_map <= float(src_height - 1))
        )
        if not bool(np.any(valid)):
            return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode=image.mode)

        x = x_map[valid]
        y = y_map[valid]
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, src_width - 1)
        y1 = np.clip(y0 + 1, 0, src_height - 1)
        wx = x - x0.astype(np.float64)
        wy = y - y0.astype(np.float64)

        if source.ndim == 2:
            top = source[y0, x0].astype(np.float64) * (1.0 - wx) + source[y0, x1].astype(np.float64) * wx
            bottom = source[y1, x0].astype(np.float64) * (1.0 - wx) + source[y1, x1].astype(np.float64) * wx
            output[valid] = top * (1.0 - wy) + bottom * wy
        else:
            wx = wx[:, None]
            wy = wy[:, None]
            top = source[y0, x0, :].astype(np.float64) * (1.0 - wx) + source[y0, x1, :].astype(np.float64) * wx
            bottom = source[y1, x0, :].astype(np.float64) * (1.0 - wx) + source[y1, x1, :].astype(np.float64) * wx
            output[valid] = top * (1.0 - wy) + bottom * wy

        return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode=image.mode)

    def _draw_baseline_overlay(
        self,
        image: Image.Image,
        detection: dict[str, Any],
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        output = image.convert("RGB")
        draw = ImageDraw.Draw(output)
        line_width = max(1, int(round(image.height / 96)))
        self._draw_textline(
            draw,
            image_width=image.width,
            slope=float(detection["slope"]),
            intercept=float(detection["intercept"]),
            color=(230, 30, 30),
            width=line_width,
        )
        if detection.get("topline_detected"):
            self._draw_textline(
                draw,
                image_width=image.width,
                slope=float(detection["topline_slope"]),
                intercept=float(detection["topline_intercept"]),
                color=(40, 110, 240),
                width=line_width,
            )
        profile_x = detection.get("profile_x")
        profile_y = detection.get("profile_y")
        inlier_mask = detection.get("inlier_mask")
        if profile_x is not None and profile_y is not None:
            radius = max(1, line_width)
            step = max(1, int(math.ceil(len(profile_x) / 500)))
            for index in range(0, len(profile_x), step):
                x = float(profile_x[index])
                y = float(profile_y[index])
                is_inlier = bool(inlier_mask[index]) if inlier_mask is not None and index < len(inlier_mask) else False
                color = (20, 150, 70) if is_inlier else (40, 110, 220)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        top_profile_x = detection.get("topline_profile_x")
        top_profile_y = detection.get("topline_profile_y")
        top_inlier_mask = detection.get("topline_inlier_mask")
        if top_profile_x is not None and top_profile_y is not None:
            radius = max(1, line_width)
            step = max(1, int(math.ceil(len(top_profile_x) / 500)))
            for index in range(0, len(top_profile_x), step):
                x = float(top_profile_x[index])
                y = float(top_profile_y[index])
                is_inlier = bool(top_inlier_mask[index]) if top_inlier_mask is not None and index < len(top_inlier_mask) else False
                color = (70, 190, 230) if is_inlier else (80, 120, 240)
                draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
        if crop_box is not None:
            draw.rectangle(crop_box, outline=(20, 150, 60), width=line_width)
        if "text_bbox" in detection:
            draw.rectangle(detection["text_bbox"], outline=(80, 120, 240), width=max(1, line_width // 2))
        return output

    def _draw_baseline_lines_debug(
        self,
        image: Image.Image,
        detection: dict[str, Any],
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        output = image.convert("RGB")
        draw = ImageDraw.Draw(output)
        line_width = max(1, int(round(image.height / 180)))

        if crop_box is not None:
            overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            left, top, right, bottom = crop_box
            visible_top = max(0, top)
            visible_bottom = min(output.height, bottom)
            if visible_top > 0:
                overlay_draw.rectangle((0, 0, output.width, visible_top), fill=(0, 0, 0, 55))
            if visible_bottom < output.height:
                overlay_draw.rectangle((0, visible_bottom, output.width, output.height), fill=(0, 0, 0, 55))
            output = Image.alpha_composite(output.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(output)
            draw.rectangle(crop_box, outline=(30, 190, 70), width=max(2, line_width // 2))

        if detection.get("topline_detected"):
            self._draw_textline(
                draw,
                image_width=image.width,
                slope=float(detection["topline_slope"]),
                intercept=float(detection["topline_intercept"]),
                color=(0, 190, 255),
                width=line_width,
            )

        self._draw_textline(
            draw,
            image_width=image.width,
            slope=float(detection["slope"]),
            intercept=float(detection["intercept"]),
            color=(255, 45, 45),
            width=line_width,
        )

        return output

    def _draw_baseline_curves_debug(
        self,
        image: Image.Image,
        detection: dict[str, Any],
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        output = image.convert("RGB")
        line_width = max(1, int(round(image.height / 180)))

        if crop_box is not None:
            overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            _, top, _, bottom = crop_box
            visible_top = max(0, top)
            visible_bottom = min(output.height, bottom)
            if visible_top > 0:
                overlay_draw.rectangle((0, 0, output.width, visible_top), fill=(0, 0, 0, 55))
            if visible_bottom < output.height:
                overlay_draw.rectangle((0, visible_bottom, output.width, output.height), fill=(0, 0, 0, 55))
            output = Image.alpha_composite(output.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(output)
        if crop_box is not None:
            draw.rectangle(crop_box, outline=(30, 190, 70), width=max(2, line_width // 2))

        curve_x = np.asarray(detection["curve_x"], dtype=np.float64)
        self._draw_curve(
            draw,
            curve_x,
            np.asarray(detection["top_curve_y"], dtype=np.float64),
            color=(0, 190, 255),
            width=line_width,
        )
        self._draw_curve(
            draw,
            curve_x,
            np.asarray(detection["bottom_curve_y"], dtype=np.float64),
            color=(255, 45, 45),
            width=line_width,
        )
        return output

    def _draw_baseline_heatmap_debug(self, image: Image.Image, heatmaps: np.ndarray) -> Image.Image:
        if heatmaps.ndim != 3 or heatmaps.shape[0] < 2:
            return image.convert("RGB")

        top = Image.fromarray(np.clip(heatmaps[0] * 255.0, 0, 255).astype(np.uint8), mode="L")
        bottom = Image.fromarray(np.clip(heatmaps[1] * 255.0, 0, 255).astype(np.uint8), mode="L")
        top = top.resize(image.size, Image.Resampling.BILINEAR)
        bottom = bottom.resize(image.size, Image.Resampling.BILINEAR)
        top_array = np.asarray(top, dtype=np.float32) / 255.0
        bottom_array = np.asarray(bottom, dtype=np.float32) / 255.0
        alpha = np.maximum(top_array, bottom_array)

        overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        overlay[..., 0] = np.clip(bottom_array * 255.0, 0, 255).astype(np.uint8)
        overlay[..., 1] = np.clip(top_array * 190.0, 0, 255).astype(np.uint8)
        overlay[..., 2] = np.clip(top_array * 255.0, 0, 255).astype(np.uint8)
        overlay[..., 3] = np.clip(alpha * 155.0, 0, 180).astype(np.uint8)

        base = image.convert("RGBA")
        heatmap_overlay = Image.fromarray(overlay, mode="RGBA")
        return Image.alpha_composite(base, heatmap_overlay).convert("RGB")

    @staticmethod
    def _draw_textline(
        draw: ImageDraw.ImageDraw,
        image_width: int,
        slope: float,
        intercept: float,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        x0 = 0
        x1 = max(0, image_width - 1)
        y0 = slope * x0 + intercept
        y1 = slope * x1 + intercept
        draw.line((x0, y0, x1, y1), fill=color, width=width)

    @staticmethod
    def _draw_curve(
        draw: ImageDraw.ImageDraw,
        xs: np.ndarray,
        ys: np.ndarray,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        if xs.size < 2 or ys.size != xs.size:
            return
        step = max(1, int(math.ceil(xs.size / 1000)))
        points = [(float(xs[index]), float(ys[index])) for index in range(0, xs.size, step)]
        if points[-1] != (float(xs[-1]), float(ys[-1])):
            points.append((float(xs[-1]), float(ys[-1])))
        draw.line(points, fill=color, width=width)

    def _crop_with_fill(self, image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
        left, top, right, bottom = box
        width = max(1, right - left)
        height = max(1, bottom - top)
        output = Image.new(image.mode, (width, height), self._background_fill_value(image))
        source_box = (
            max(0, left),
            max(0, top),
            min(image.width, right),
            min(image.height, bottom),
        )
        if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
            return output
        paste_xy = (source_box[0] - left, source_box[1] - top)
        output.paste(image.crop(source_box), paste_xy)
        return output

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            return self.preprocess_pil(image)

    def preprocess_image_debug(self, image_path: str | Path) -> tuple[torch.Tensor, PreprocessDebug]:
        with Image.open(image_path) as image:
            return self.preprocess_pil_debug(image)

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
                if class_index in self.idx_to_char
            )
            decoded.append((text, raw_ids))
        return decoded

    def output_width_for_input_width(self, width: int) -> int:
        if hasattr(self.model, "output_width_for_input_width"):
            return int(self.model.output_width_for_input_width(width))

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

        text = "".join(symbol.char for symbol in decoded_symbols)
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

    @staticmethod
    def _segmentation_cut_positions(segmentation_result: VerticalSegmentationResult) -> list[int]:
        return [int(position) for position in segmentation_result.cut_positions or []]

    @staticmethod
    def _map_cut_to_boundary(cut_position: int, source_width: int, target_width: int) -> int:
        if source_width <= 0 or target_width <= 1:
            return 0
        boundary = int(round((float(cut_position) + 0.5) * float(target_width) / float(source_width)))
        return max(1, min(target_width - 1, boundary))

    @staticmethod
    def _map_boundary_to_source(boundary: int, source_width: int, target_width: int) -> int:
        if source_width <= 0 or target_width <= 0:
            return 0
        mapped = int(round(float(boundary) * float(source_width) / float(target_width)))
        return max(0, min(source_width, mapped))

    def _map_input_boundary_to_ocr(self, boundary: int, input_width: int, ocr_width: int) -> int:
        if input_width <= 0 or ocr_width <= 0:
            return 0
        left = min(max(0, self.legacy_crop_left), max(0, input_width - 1))
        right = max(left + 1, input_width - max(0, self.legacy_crop_right))
        mapped = int(round((float(boundary) - float(left)) * float(ocr_width) / float(right - left)))
        return max(0, min(ocr_width, mapped))

    def _map_ocr_boundary_to_input(self, boundary: int, input_width: int, ocr_width: int) -> int:
        if input_width <= 0 or ocr_width <= 0:
            return 0
        left = min(max(0, self.legacy_crop_left), max(0, input_width - 1))
        right = max(left + 1, input_width - max(0, self.legacy_crop_right))
        mapped = int(round(float(left) + float(boundary) * float(right - left) / float(ocr_width)))
        return max(0, min(input_width, mapped))

    def text_x_bounds_from_tensor(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        mask = self._foreground_mask_from_tensor(image_tensor)
        height, width = mask.shape
        if width <= 0:
            return {"ok": False, "status": "empty_width", "left": 0, "right": 0, "confidence": 0.0}

        column_counts = np.count_nonzero(mask, axis=0)
        min_column_pixels = max(1, int(round(height * 0.025)))
        active = column_counts >= min_column_pixels
        active = self._close_boolean_holes(active, max_hole=max(1, int(round(width * 0.01))))
        runs = self._boolean_runs(active)
        runs = [
            (start, end) for start, end in runs
            if end - start >= max(2, int(round(width * 0.015)))
        ]
        if not runs:
            return {
                "ok": False,
                "status": "no_foreground_columns",
                "left": 0,
                "right": width,
                "confidence": 0.0,
            }

        left = min(start for start, _ in runs)
        right = max(end for _, end in runs)
        foreground_columns_ratio = float(np.count_nonzero(active[left:right])) / max(1.0, float(right - left))
        foreground_pixels = int(np.count_nonzero(mask[:, left:right]))
        confidence = min(
            1.0,
            foreground_columns_ratio * 0.75
            + min(1.0, foreground_pixels / max(1.0, height * (right - left) * 0.12)) * 0.25,
        )
        if right - left < max(2, int(round(width * 0.02))):
            return {
                "ok": False,
                "status": "too_narrow_foreground",
                "left": left,
                "right": right,
                "confidence": confidence,
            }
        return {
            "ok": True,
            "status": "ok",
            "left": left,
            "right": right,
            "confidence": confidence,
            "foreground_columns_ratio": foreground_columns_ratio,
            "foreground_pixels": foreground_pixels,
        }

    def _foreground_mask_from_tensor(self, image_tensor: torch.Tensor) -> np.ndarray:
        image = image_tensor.detach().cpu().float().clamp(0.0, 1.0)
        if image.dim() == 4:
            image = image[0]
        if image.dim() != 3:
            raise ValueError(f"Expected image tensor with shape (C,H,W) or (1,C,H,W), got {tuple(image_tensor.shape)}")

        if image.size(0) == 1:
            gray = image[0].numpy()
        else:
            gray = image.mean(dim=0).numpy()
        gray_u8 = (gray * 255.0).astype(np.uint8)
        height, width = gray_u8.shape
        if width <= 0:
            return np.zeros((height, 0), dtype=np.uint8)

        border = np.concatenate((gray_u8[0, :], gray_u8[-1, :], gray_u8[:, 0], gray_u8[:, -1]))
        background_is_bright = float(np.median(border)) >= 128.0
        if cv2 is not None:
            threshold_type = cv2.THRESH_BINARY_INV if background_is_bright else cv2.THRESH_BINARY
            _, mask = cv2.threshold(gray_u8, 0, 255, threshold_type | cv2.THRESH_OTSU)
        else:
            threshold = float(np.mean(border))
            mask = gray_u8 < threshold if background_is_bright else gray_u8 > threshold
            mask = (mask.astype(np.uint8) * 255)
        return mask.astype(np.uint8)

    @staticmethod
    def _close_boolean_holes(values: np.ndarray, max_hole: int) -> np.ndarray:
        output = values.astype(bool).copy()
        if max_hole <= 0 or output.size == 0:
            return output

        runs = TextRecognizer._boolean_runs(output)
        for (_, prev_end), (next_start, _) in zip(runs, runs[1:]):
            if next_start - prev_end <= max_hole:
                output[prev_end:next_start] = True
        return output

    @staticmethod
    def _boolean_runs(values: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(values.astype(bool).tolist()):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, int(values.size)))
        return runs

    def _edge_interval_has_foreground(
        self,
        mask: np.ndarray,
        start: int,
        end: int,
        input_width: int,
        ocr_width: int,
        min_ink_ratio: float,
        min_pixel_density: float,
    ) -> bool:
        left = self._map_ocr_boundary_to_input(start, input_width, ocr_width)
        right = self._map_ocr_boundary_to_input(end, input_width, ocr_width)
        if right <= left:
            return False

        region = mask[:, left:right]
        if region.size == 0:
            return False
        min_column_pixels = max(1, int(round(region.shape[0] * 0.025)))
        ink_columns = np.count_nonzero(np.count_nonzero(region, axis=0) >= min_column_pixels)
        ink_ratio = float(ink_columns) / max(1.0, float(region.shape[1]))
        pixel_density = float(np.count_nonzero(region)) / float(region.size)
        return ink_ratio >= min_ink_ratio or pixel_density >= min_pixel_density

    @staticmethod
    def _typical_cut_width(cuts: list[int], left_boundary: int, right_boundary: int) -> float:
        widths = [
            right - left
            for left, right in zip(cuts, cuts[1:])
            if right > left
        ]
        if not widths and right_boundary > left_boundary:
            widths = [right_boundary - left_boundary]
        if not widths:
            return 0.0
        return float(np.median(np.asarray(widths, dtype=np.float32)))

    @classmethod
    def _promote_edge_cuts_to_bounds(
        cls,
        mapped_cuts: list[int],
        left_boundary: int,
        right_boundary: int,
        mode: str,
        max_edge_ratio: float,
        min_edge_width: int,
    ) -> tuple[int, int, list[int]]:
        mode = mode.lower()
        if mode not in {"off", "auto", "on"}:
            raise ValueError("boundary_cuts mode must be 'off', 'auto', or 'on'")

        cuts = [
            cut for cut in sorted(set(mapped_cuts))
            if left_boundary < cut < right_boundary
        ]
        if len(cuts) < 2 or mode == "off":
            return left_boundary, right_boundary, cuts

        if mode == "on":
            return cuts[0], cuts[-1], cuts[1:-1]

        typical_width = cls._typical_cut_width(cuts, left_boundary, right_boundary)
        if typical_width <= 0.0:
            return left_boundary, right_boundary, cuts
        if max_edge_ratio < 0.0:
            raise ValueError("boundary_cut_max_edge_ratio must be non-negative")

        threshold = max(float(min_edge_width), typical_width * float(max_edge_ratio))
        left_edge_width = cuts[0] - left_boundary
        right_edge_width = right_boundary - cuts[-1]

        # In auto mode we promote both edges only when both outer intervals look
        # much smaller than a normal character interval. This catches outputs
        # like |A|B|C| without deleting a real narrow first/last glyph by itself.
        if left_edge_width <= threshold and right_edge_width <= threshold:
            left_boundary = cuts[0]
            right_boundary = cuts[-1]
            cuts = cuts[1:-1]

        cuts = [
            cut for cut in cuts
            if left_boundary < cut < right_boundary
        ]
        return left_boundary, right_boundary, cuts

    def decode_legacy_with_cuts(
        self,
        logits: torch.Tensor,
        segmentation_result: VerticalSegmentationResult,
        input_width: int | None = None,
        top_k: int = 8,
        text_x_bounds: tuple[int, int] | None = None,
        input_tensor: torch.Tensor | None = None,
        trim_empty_edges: bool = True,
        edge_min_ink_ratio: float = 0.035,
        edge_min_pixel_density: float = 0.003,
        edge_min_width: int = 2,
        boundary_cuts: str = "auto",
        boundary_cut_max_edge_ratio: float = 0.45,
    ) -> CutDecodingResult:
        if self.loss_mode not in {"legacy", "legacy_logreg"}:
            raise ValueError(
                "legacy+cuts decoding expects a legacy OCR checkpoint; "
                f"got loss_mode={self.loss_mode!r}"
            )
        if logits.dim() != 3 or logits.size(0) != 1:
            raise ValueError(f"legacy+cuts decoding expects logits shape (1, C, T), got {tuple(logits.shape)}")

        probs = torch.softmax(logits, dim=1)[0]
        ocr_width = int(probs.size(1))
        segmentator_width = len(segmentation_result.raw_indices)
        input_width = int(input_width if input_width is not None else segmentation_result.input_shape[-1])
        if ocr_width <= 0:
            return CutDecodingResult(
                text="",
                symbols=[],
                cuts=[],
                boundaries=[],
                input_width=input_width,
                ocr_width=ocr_width,
                segmentator_width=segmentator_width,
            )

        raw_cuts = [
            position for position in self._segmentation_cut_positions(segmentation_result)
            if 0 <= position < max(0, segmentator_width)
        ]
        mapped_cuts = []
        for position in raw_cuts:
            if segmentator_width > 0:
                input_position = int(round((float(position) + 0.5) * float(input_width) / float(segmentator_width)))
            else:
                input_position = 0
            mapped_cuts.append(self._map_input_boundary_to_ocr(input_position, input_width, ocr_width))
        left_boundary = 0
        right_boundary = ocr_width
        if text_x_bounds is not None:
            text_left, text_right = text_x_bounds
            left_boundary = self._map_input_boundary_to_ocr(int(text_left), input_width, ocr_width)
            right_boundary = self._map_input_boundary_to_ocr(int(text_right), input_width, ocr_width)
            if right_boundary <= left_boundary:
                left_boundary = 0
                right_boundary = ocr_width

        left_boundary, right_boundary, mapped_cuts = self._promote_edge_cuts_to_bounds(
            mapped_cuts,
            left_boundary,
            right_boundary,
            mode=boundary_cuts,
            max_edge_ratio=boundary_cut_max_edge_ratio,
            min_edge_width=edge_min_width,
        )
        boundaries = [left_boundary, *sorted(set(mapped_cuts)), right_boundary]
        intervals = [
            (start, end) for start, end in zip(boundaries, boundaries[1:])
            if end > start
        ]
        if trim_empty_edges and input_tensor is not None and intervals:
            mask = self._foreground_mask_from_tensor(input_tensor)
            while intervals and not self._edge_interval_has_foreground(
                mask,
                intervals[0][0],
                intervals[0][1],
                input_width,
                ocr_width,
                edge_min_ink_ratio,
                edge_min_pixel_density,
            ):
                intervals.pop(0)
            while intervals and not self._edge_interval_has_foreground(
                mask,
                intervals[-1][0],
                intervals[-1][1],
                input_width,
                ocr_width,
                edge_min_ink_ratio,
                edge_min_pixel_density,
            ):
                intervals.pop()
            if intervals:
                boundaries = [intervals[0][0], *[end for _, end in intervals]]
            else:
                boundaries = []
        intervals = self._merge_narrow_edge_intervals(intervals, min_width=edge_min_width)
        if intervals:
            boundaries = [intervals[0][0], *[end for _, end in intervals]]
        else:
            boundaries = []
        top_k = max(1, min(int(top_k), probs.size(0)))

        symbols: list[CutDecodedSymbol] = []
        for start, end in intervals:
            if end <= start:
                continue
            scores = probs[:, start:end].mean(dim=1)
            top_confidences, top_indices = scores.topk(top_k)
            class_index = int(top_indices[0].detach().cpu().item())
            char = self.idx_to_char.get(class_index)
            if char is None:
                continue

            candidates: list[ClassConfidence] = []
            for rank in range(top_k):
                candidate_index = int(top_indices[rank].detach().cpu().item())
                candidates.append(
                    ClassConfidence(
                        label=self.class_label(candidate_index),
                        confidence=float(top_confidences[rank].detach().cpu().item()),
                        class_index=candidate_index,
                    )
                )

            symbols.append(
                CutDecodedSymbol(
                    char=char,
                    confidence=float(scores[class_index].detach().cpu().item()),
                    class_index=class_index,
                    start=int(start),
                    end=int(end),
                    source_start=self._map_boundary_to_source(start, segmentator_width, ocr_width),
                    source_end=self._map_boundary_to_source(end, segmentator_width, ocr_width),
                    candidates=candidates,
                )
            )

        return CutDecodingResult(
            text="".join(symbol.char for symbol in symbols),
            symbols=symbols,
            cuts=raw_cuts,
            boundaries=boundaries,
            input_width=input_width,
            ocr_width=ocr_width,
            segmentator_width=segmentator_width,
        )

    @staticmethod
    def _merge_narrow_edge_intervals(
        intervals: list[tuple[int, int]],
        min_width: int,
    ) -> list[tuple[int, int]]:
        if min_width <= 1 or len(intervals) <= 1:
            return intervals

        output = list(intervals)
        while len(output) > 1 and output[0][1] - output[0][0] < min_width:
            output[1] = (output[0][0], output[1][1])
            output.pop(0)
        while len(output) > 1 and output[-1][1] - output[-1][0] < min_width:
            output[-2] = (output[-2][0], output[-1][1])
            output.pop()
        return output

    @torch.no_grad()
    def logits_from_tensor(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return logits, tuple(image_tensor.shape)

    @torch.no_grad()
    def recognize_tensor_debug_with_logits(
        self,
        image_tensor: torch.Tensor,
        top_k: int = 8,
    ) -> tuple[RecognitionResult, torch.Tensor]:
        logits, input_shape = self.logits_from_tensor(image_tensor)
        return self.analyze_logits(logits, input_shape=input_shape, top_k=top_k), logits

    @torch.no_grad()
    def recognize_tensor_debug(self, image_tensor: torch.Tensor, top_k: int = 8) -> RecognitionResult:
        logits, input_shape = self.logits_from_tensor(image_tensor)
        return self.analyze_logits(logits, input_shape=input_shape, top_k=top_k)

    @torch.no_grad()
    def recognize_tensor(self, image_tensor: torch.Tensor) -> tuple[str, list[int]]:
        logits, _ = self.logits_from_tensor(image_tensor)
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
