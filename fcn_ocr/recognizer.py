from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

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
        baseline_deskew: bool = True,
        baseline_max_angle: float = 12.0,
        baseline_strict_lines: bool = True,
        baseline_line_pad: float = 0.08,
        baseline_line_pad_px: float = 0.0,
        baseline_detector_checkpoint: str | Path | None = None,
        baseline_detector_threshold: float = 0.35,
    ):
        if scale_x <= -0.95:
            raise ValueError("scale_x must be > -0.95")
        if y_pad <= -0.95:
            raise ValueError("y_pad must be > -0.95")
        if x_pad < 0.0:
            raise ValueError("x_pad must be >= 0")
        if baseline_line_pad < 0.0:
            raise ValueError("baseline_line_pad must be >= 0")
        if baseline_line_pad_px < 0.0:
            raise ValueError("baseline_line_pad_px must be >= 0")
        if baseline_max_angle <= 0.0:
            raise ValueError("baseline_max_angle must be > 0")
        if not 0.0 < baseline_detector_threshold < 1.0:
            raise ValueError("baseline_detector_threshold must be between 0 and 1")

        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.scale_x = float(scale_x)
        self.y_pad = float(y_pad)
        self.x_pad = float(x_pad)
        self.baseline_crop = bool(baseline_crop)
        self.baseline_deskew = bool(baseline_deskew)
        self.baseline_max_angle = float(baseline_max_angle)
        self.baseline_strict_lines = bool(baseline_strict_lines)
        self.baseline_line_pad = float(baseline_line_pad)
        self.baseline_line_pad_px = float(baseline_line_pad_px)
        self.baseline_detector_checkpoint = Path(baseline_detector_checkpoint) if baseline_detector_checkpoint else None
        self.baseline_detector_threshold = float(baseline_detector_threshold)
        self.baseline_detector_model: torch.nn.Module | None = None
        self.baseline_detector_in_channels = 1
        self.baseline_detector_image_height = 0
        self.baseline_detector_architecture = ""

        if self.baseline_crop and self.baseline_detector_checkpoint is None:
            raise ValueError("baseline_crop requires baseline_detector_checkpoint")

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
        print(f"Model-internal baseline crop: {self.baseline_crop}")
        if self.baseline_crop:
            print(
                f"  deskew={self.baseline_deskew}, max_angle={self.baseline_max_angle:.2f}, "
                f"strict_lines={self.baseline_strict_lines}, line_pad={self.baseline_line_pad:.3f}, "
                f"line_pad_px={self.baseline_line_pad_px:.1f}"
            )
            if self.baseline_detector_model is not None:
                print(
                    "  neural_detector="
                    f"{self.baseline_detector_checkpoint} "
                    f"threshold={self.baseline_detector_threshold:.3f} "
                    f"architecture={self.baseline_detector_architecture}"
                )

    def class_label(self, index: int) -> str:
        return display_char(self.idx_to_char.get(index, f"<{index}>"))

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess_pil_3d(image).unsqueeze(0)

    def preprocess_pil_debug(self, image: Image.Image) -> tuple[torch.Tensor, PreprocessDebug]:
        tensor, debug = self._preprocess_pil_3d_with_debug(image, collect_debug=True)
        return tensor.unsqueeze(0), debug

    def prepare_baseline_image(
        self,
        image: Image.Image,
        collect_debug: bool = False,
    ) -> tuple[Image.Image, PreprocessDebug]:
        source = image.convert("RGB")
        if not self.baseline_crop:
            return source, PreprocessDebug(
                metadata={
                    "baseline_crop": False,
                    "baseline_status": "disabled",
                },
                images=[],
            )
        prepared, debug = self._apply_baseline_crop(source, collect_debug=collect_debug)
        return prepared, PreprocessDebug(
            metadata={"baseline_crop": True, **debug.metadata},
            images=debug.images,
        )

    def preprocess_pil_after_baseline(self, image: Image.Image) -> torch.Tensor:
        tensor, _ = self._preprocess_pil_3d_with_debug(
            image,
            collect_debug=False,
            apply_baseline=False,
        )
        return tensor.unsqueeze(0)

    def preprocess_pil_after_baseline_debug(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, PreprocessDebug]:
        tensor, debug = self._preprocess_pil_3d_with_debug(
            image,
            collect_debug=True,
            apply_baseline=False,
        )
        return tensor.unsqueeze(0), debug

    def preprocess_pil_with_source_x(self, image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
        return self._preprocess_pil_with_source_x(image, apply_baseline=True)

    def preprocess_pil_after_baseline_with_source_x(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, np.ndarray]:
        return self._preprocess_pil_with_source_x(image, apply_baseline=False)

    def _preprocess_pil_with_source_x(
        self,
        image: Image.Image,
        apply_baseline: bool,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess an image and retain the source-image X represented by every
        final network-input pixel. This is used for exact geometric evaluation.
        """
        image = image.convert("RGB" if self.in_channels == 3 else "L")
        source_x = np.broadcast_to(
            np.arange(image.width, dtype=np.float32)[None, :],
            (image.height, image.width),
        ).copy()

        if apply_baseline and self.baseline_crop:
            image, source_x = self._apply_baseline_crop_with_source_x(image, source_x)

        before_x_pad_width = image.width
        image = self._apply_x_pad(image)
        if image.width != before_x_pad_width:
            delta = (image.width - before_x_pad_width) // 2
            source_x = np.pad(source_x, ((0, 0), (delta, delta)), mode="edge")

        source_x = self._apply_y_pad_to_float_map(source_x)
        image = self._apply_y_pad(image)

        if image.height != self.image_height:
            new_width = max(1, round(image.width * self.image_height / image.height))
            image = image.resize((new_width, self.image_height), Image.Resampling.BICUBIC)
            source_x = self._resize_float_map(source_x, image.size)

        before_scale_width = image.width
        image = self._apply_scale_x(image)
        if image.width != before_scale_width:
            source_x = self._resize_float_map(source_x, image.size)

        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.in_channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor.to(self.device), source_x

    def _preprocess_pil_3d(self, image: Image.Image) -> torch.Tensor:
        tensor, _ = self._preprocess_pil_3d_with_debug(image, collect_debug=False)
        return tensor

    def _preprocess_pil_3d_with_debug(
        self,
        image: Image.Image,
        collect_debug: bool,
        apply_baseline: bool = True,
    ) -> tuple[torch.Tensor, PreprocessDebug]:
        baseline_enabled = bool(apply_baseline and self.baseline_crop)
        debug_metadata: dict[str, Any] = {
            "baseline_crop": baseline_enabled,
            "baseline_strict_lines": self.baseline_strict_lines,
            "baseline_line_pad": self.baseline_line_pad,
            "baseline_line_pad_px": self.baseline_line_pad_px,
            "baseline_detector_checkpoint": str(self.baseline_detector_checkpoint) if self.baseline_detector_checkpoint else None,
            "baseline_detector_threshold": self.baseline_detector_threshold,
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

        if baseline_enabled:
            image, baseline_debug = self._apply_baseline_crop(image, collect_debug=collect_debug)
            debug_metadata.update(baseline_debug.metadata)
            debug_images.extend(baseline_debug.images)
            self._append_preprocess_debug_image(
                debug_images,
                collect_debug,
                "preprocess 01 after baseline crop",
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
        debug_images: list[tuple[str, Image.Image]] = []
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
                ("bottom_angle_degrees", "baseline_bottom_angle_degrees"),
                ("topline_angle_degrees", "baseline_top_angle_degrees"),
                ("baseline_pair_angle_difference", "baseline_pair_angle_difference"),
                ("baseline_pair_angle_max_difference", "baseline_pair_angle_max_difference"),
                ("baseline_top_angle_weight", "baseline_top_angle_weight"),
                ("baseline_bottom_angle_weight", "baseline_bottom_angle_weight"),
                ("baseline_angle_method", "baseline_angle_method"),
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
            "baseline_top_angle_degrees": float(first.get("topline_angle_degrees", original_angle)),
            "baseline_bottom_angle_degrees": float(first.get("bottom_angle_degrees", original_angle)),
            "baseline_pair_angle_difference": float(first.get("baseline_pair_angle_difference", 0.0)),
            "baseline_pair_angle_max_difference": float(
                first.get("baseline_pair_angle_max_difference", self._baseline_pair_angle_max_difference())
            ),
            "baseline_top_angle_weight": float(first.get("baseline_top_angle_weight", 0.0)),
            "baseline_bottom_angle_weight": float(first.get("baseline_bottom_angle_weight", 1.0)),
            "baseline_angle_method": first.get("baseline_angle_method", "bottom_only"),
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
        return cropped, PreprocessDebug(metadata=metadata, images=debug_images)

    def _apply_baseline_crop_with_source_x(
        self,
        image: Image.Image,
        source_x: np.ndarray,
    ) -> tuple[Image.Image, np.ndarray]:
        first = self._detect_baseline(image)
        if not first["ok"]:
            return image, source_x

        working_image = image
        working_source_x = source_x
        detection = first
        original_angle = float(first["angle_degrees"])
        if self.baseline_deskew and abs(original_angle) >= 0.25:
            rotated = image.rotate(
                original_angle,
                expand=True,
                resample=Image.Resampling.BICUBIC,
                fillcolor=self._background_fill_value(image),
            )
            rotated_source_x = np.asarray(
                Image.fromarray(source_x, mode="F").rotate(
                    original_angle,
                    expand=True,
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=-1.0,
                ),
                dtype=np.float32,
            )
            second = self._detect_baseline(rotated)
            if second["ok"]:
                working_image = rotated
                working_source_x = rotated_source_x
                detection = second
            elif self.baseline_strict_lines:
                return image, source_x

        cropped = self._crop_with_fill(working_image, detection["crop_box"])
        cropped_source_x = self._crop_float_map_with_fill(
            working_source_x,
            detection["crop_box"],
            fill=-1.0,
        )
        return cropped, cropped_source_x

    @staticmethod
    def _resize_float_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        if values.shape == (size[1], size[0]):
            return values.astype(np.float32, copy=False)
        return np.asarray(
            Image.fromarray(values.astype(np.float32), mode="F").resize(
                size,
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

    def _apply_y_pad_to_float_map(self, values: np.ndarray) -> np.ndarray:
        if self.y_pad == 0.0:
            return values
        delta = int(round(values.shape[0] * abs(self.y_pad)))
        if delta <= 0:
            return values
        top = delta // 2
        bottom = delta - top
        if self.y_pad > 0.0:
            return np.pad(
                values,
                ((top, bottom), (0, 0)),
                mode="constant",
                constant_values=-1.0,
            )
        if delta >= values.shape[0]:
            delta = values.shape[0] - 1
            top = delta // 2
            bottom = delta - top
        return values[top : values.shape[0] - bottom]

    @staticmethod
    def _crop_float_map_with_fill(
        values: np.ndarray,
        box: tuple[int, int, int, int],
        fill: float,
    ) -> np.ndarray:
        left, top, right, bottom = box
        width = max(1, right - left)
        height = max(1, bottom - top)
        output = np.full((height, width), float(fill), dtype=np.float32)
        source_box = (
            max(0, left),
            max(0, top),
            min(values.shape[1], right),
            min(values.shape[0], bottom),
        )
        if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
            return output
        paste_x = source_box[0] - left
        paste_y = source_box[1] - top
        source = values[source_box[1] : source_box[3], source_box[0] : source_box[2]]
        output[paste_y : paste_y + source.shape[0], paste_x : paste_x + source.shape[1]] = source
        return output

    def _detect_baseline(self, image: Image.Image) -> dict[str, Any]:
        if self.baseline_detector_model is None:
            raise RuntimeError("baseline detector model is not loaded")
        return self._detect_baseline_neural(image)

    def _detect_baseline_neural(self, image: Image.Image) -> dict[str, Any]:
        if self.baseline_detector_model is None:
            raise RuntimeError("baseline detector model is not loaded")

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
        angle = self._combined_baseline_angle(top_line, bottom_line)
        if self.baseline_strict_lines and not angle["baseline_pair_angle_consistent"]:
            return {
                "ok": False,
                "status": "neural_baseline_angle_mismatch",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "method": "neural_heatmap",
                "mask_name": "baseline_detector",
                "topline_confidence": float(top_line["confidence"]),
                "baseline_confidence": float(bottom_line["confidence"]),
                **angle,
                **self._topline_metadata(top_line),
            }
        angle_degrees = float(angle["angle_degrees"])
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
            "bottom_angle_degrees": float(angle["bottom_angle_degrees"]),
            "topline_detected": True,
            **angle,
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

    def _combined_baseline_angle(
        self,
        top_line: dict[str, Any] | None,
        bottom_line: dict[str, Any],
    ) -> dict[str, Any]:
        bottom_angle = math.degrees(math.atan(float(bottom_line["slope"])))
        if top_line is None:
            return {
                "angle_degrees": float(bottom_angle),
                "bottom_angle_degrees": float(bottom_angle),
                "baseline_pair_angle_difference": 0.0,
                "baseline_pair_angle_max_difference": self._baseline_pair_angle_max_difference(),
                "baseline_pair_angle_consistent": True,
                "baseline_top_angle_weight": 0.0,
                "baseline_bottom_angle_weight": 1.0,
                "baseline_angle_method": "bottom_only",
            }

        top_angle = math.degrees(math.atan(float(top_line["slope"])))
        top_weight = self._baseline_line_angle_weight(top_line)
        bottom_weight = self._baseline_line_angle_weight(bottom_line)
        weight_sum = top_weight + bottom_weight
        if weight_sum <= 1e-8:
            top_weight = bottom_weight = 0.5
            weight_sum = 1.0
        top_weight /= weight_sum
        bottom_weight /= weight_sum
        angle = top_angle * top_weight + bottom_angle * bottom_weight
        difference = abs(top_angle - bottom_angle)
        max_difference = self._baseline_pair_angle_max_difference()
        return {
            "angle_degrees": float(angle),
            "bottom_angle_degrees": float(bottom_angle),
            "baseline_pair_angle_difference": float(difference),
            "baseline_pair_angle_max_difference": float(max_difference),
            "baseline_pair_angle_consistent": bool(difference <= max_difference),
            "baseline_top_angle_weight": float(top_weight),
            "baseline_bottom_angle_weight": float(bottom_weight),
            "baseline_angle_method": "confidence_coverage_weighted_pair",
        }

    @staticmethod
    def _baseline_line_angle_weight(line: dict[str, Any]) -> float:
        confidence = max(0.0, float(line.get("confidence", 0.0)))
        coverage = max(0.0, float(line.get("profile_coverage", 0.0)))
        return max(1e-6, confidence * coverage)

    def _baseline_pair_angle_max_difference(self) -> float:
        return max(2.0, min(6.0, self.baseline_max_angle * 0.5))


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
                break
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

        margin = max(0.0, above_baseline * self.baseline_line_pad + self.baseline_line_pad_px)
        top = int(math.floor(min(text_top, float(baseline_ys.min()) - above_baseline) - margin))
        bottom = int(math.ceil(max(text_bottom + 1.0, float(baseline_ys.max())) + margin))
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

        margin = max(0.0, text_height * self.baseline_line_pad + self.baseline_line_pad_px)
        top = int(math.floor(min(text_top, float(top_ys.min())) - margin))
        bottom = int(math.ceil(max(text_bottom + 1.0, float(bottom_ys.max()) + 1.0) + margin))
        if bottom <= top:
            bottom = top + max(4, int(round(text_height)))
        return (0, top, image_width, bottom), int(round(text_height))

    def _draw_baseline_overlay(
        self,
        image: Image.Image,
        detection: dict[str, Any],
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        output = image.convert("RGB")
        draw = ImageDraw.Draw(output)
        self._draw_textline(
            draw,
            image_width=image.width,
            slope=float(detection["slope"]),
            intercept=float(detection["intercept"]),
            color=(230, 30, 30),
        )
        if detection.get("topline_detected"):
            self._draw_textline(
                draw,
                image_width=image.width,
                slope=float(detection["topline_slope"]),
                intercept=float(detection["topline_intercept"]),
                color=(40, 110, 240),
            )
        profile_x = detection.get("profile_x")
        profile_y = detection.get("profile_y")
        inlier_mask = detection.get("inlier_mask")
        if profile_x is not None and profile_y is not None:
            radius = 1
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
            radius = 1
            step = max(1, int(math.ceil(len(top_profile_x) / 500)))
            for index in range(0, len(top_profile_x), step):
                x = float(top_profile_x[index])
                y = float(top_profile_y[index])
                is_inlier = bool(top_inlier_mask[index]) if top_inlier_mask is not None and index < len(top_inlier_mask) else False
                color = (70, 190, 230) if is_inlier else (80, 120, 240)
                draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
        if crop_box is not None:
            draw.rectangle(crop_box, outline=(20, 150, 60), width=1)
        if "text_bbox" in detection:
            draw.rectangle(detection["text_bbox"], outline=(80, 120, 240), width=1)
        return output

    def _draw_baseline_lines_debug(
        self,
        image: Image.Image,
        detection: dict[str, Any],
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        output = image.convert("RGB")
        draw = ImageDraw.Draw(output)

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
            draw.rectangle(crop_box, outline=(30, 190, 70), width=1)

        if detection.get("topline_detected"):
            self._draw_textline(
                draw,
                image_width=image.width,
                slope=float(detection["topline_slope"]),
                intercept=float(detection["topline_intercept"]),
                color=(0, 190, 255),
            )

        self._draw_textline(
            draw,
            image_width=image.width,
            slope=float(detection["slope"]),
            intercept=float(detection["intercept"]),
            color=(255, 45, 45),
        )

        return output

    @staticmethod
    def _draw_textline(
        draw: ImageDraw.ImageDraw,
        image_width: int,
        slope: float,
        intercept: float,
        color: tuple[int, int, int],
    ) -> None:
        x0 = 0
        x1 = max(0, image_width - 1)
        y0 = slope * x0 + intercept
        y1 = slope * x1 + intercept
        draw.line((x0, y0, x1, y1), fill=color, width=1)

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

    def _map_input_boundary_to_ocr(self, boundary: float, input_width: int, ocr_width: int) -> int:
        if input_width <= 0 or ocr_width <= 0:
            return 0
        left = min(max(0, self.legacy_crop_left), max(0, input_width - 1))
        right = max(left + 1, input_width - max(0, self.legacy_crop_right))
        mapped = int(round((float(boundary) - float(left)) * float(ocr_width) / float(right - left)))
        return max(0, min(ocr_width, mapped))

    @staticmethod
    def _source_x_profile(source_x: np.ndarray) -> np.ndarray:
        if source_x.ndim != 2 or source_x.shape[1] == 0:
            raise ValueError("source_x map must have shape (H, W) with W > 0")
        profile = np.full(source_x.shape[1], np.nan, dtype=np.float64)
        for column in range(source_x.shape[1]):
            values = source_x[:, column]
            valid = values[values >= 0.0]
            if valid.size:
                profile[column] = float(np.median(valid))
        return profile

    @classmethod
    def _source_x_for_timestep(
        cls,
        position: int,
        output_width: int,
        source_x: np.ndarray,
    ) -> float | None:
        if output_width <= 0:
            return None
        profile = cls._source_x_profile(source_x)
        input_position = (float(position) + 0.5) * float(profile.size) / float(output_width) - 0.5
        column = max(0, min(profile.size - 1, int(round(input_position))))
        value = profile[column]
        return float(value) if np.isfinite(value) else None

    @classmethod
    def _input_x_for_source_x(
        cls,
        source_position: float,
        source_x: np.ndarray,
        edge: str | None = None,
    ) -> float | None:
        profile = cls._source_x_profile(source_x)
        valid_indices = np.flatnonzero(np.isfinite(profile))
        if valid_indices.size == 0:
            return None
        distances = np.abs(profile[valid_indices] - float(source_position))
        best_distance = float(distances.min())
        best = valid_indices[np.isclose(distances, best_distance, rtol=0.0, atol=1e-6)]
        if edge == "left":
            column = int(best.min())
        elif edge == "right":
            column = int(best.max())
        else:
            column = int(round(float(np.mean(best))))
        return float(column) + 0.5

    @staticmethod
    def _central_decode_span(
        start: int,
        end: int,
        center_fraction: float,
        min_width: int,
    ) -> tuple[int, int]:
        if end <= start:
            return start, end
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError("center_fraction must be in (0, 1]")
        if min_width < 1:
            raise ValueError("min_width must be >= 1")

        width = end - start
        if center_fraction >= 1.0 or width <= 1:
            return start, end

        score_width = int(round(float(width) * center_fraction))
        score_width = max(1, min(width, max(int(min_width), score_width)))
        center = (float(start) + float(end)) * 0.5
        score_start = int(round(center - float(score_width) * 0.5))
        score_start = max(start, min(end - score_width, score_start))
        score_end = score_start + score_width
        return int(score_start), int(score_end)

    def _map_segmentator_cuts_to_ocr_boundaries(
        self,
        raw_cuts: list[int],
        *,
        segmentator_width: int,
        input_width: int,
        ocr_width: int,
        ocr_source_x: np.ndarray | None,
        segmentator_source_x: np.ndarray | None,
    ) -> list[int]:
        boundaries = []
        use_coordinate_maps = ocr_source_x is not None and segmentator_source_x is not None
        for cut_index, position in enumerate(raw_cuts):
            input_position: float
            if use_coordinate_maps:
                source_position = self._source_x_for_timestep(
                    position,
                    segmentator_width,
                    segmentator_source_x,
                )
                edge = "left" if cut_index == 0 else "right" if cut_index == len(raw_cuts) - 1 else None
                mapped_input = (
                    self._input_x_for_source_x(source_position, ocr_source_x, edge=edge)
                    if source_position is not None
                    else None
                )
                if mapped_input is not None:
                    input_position = mapped_input
                else:
                    input_position = (
                        (float(position) + 0.5) * float(input_width) / float(segmentator_width)
                        if segmentator_width > 0
                        else 0.0
                    )
            elif segmentator_width > 0:
                input_position = int(round((float(position) + 0.5) * float(input_width) / float(segmentator_width)))
            else:
                input_position = 0
            boundaries.append(self._map_input_boundary_to_ocr(input_position, input_width, ocr_width))
        return boundaries

    @staticmethod
    def _candidate_cut_positions_from_scores(
        segmentation_result: VerticalSegmentationResult,
    ) -> list[int]:
        scores = segmentation_result.cut_scores
        if not scores:
            return []

        threshold = float(segmentation_result.cut_threshold)
        candidates = {
            int(position)
            for position in (segmentation_result.cut_positions or [])
            if 0 <= int(position) < len(scores)
        }
        last_index = len(scores) - 1
        for index, score in enumerate(scores):
            if float(score) < threshold:
                continue
            left = scores[index - 1] if index > 0 else float("-inf")
            right = scores[index + 1] if index < last_index else float("-inf")
            if float(score) >= float(left) and float(score) >= float(right):
                candidates.add(index)
        return sorted(candidates)

    @staticmethod
    def _width_prior_score(width: int, min_width: int, max_width: int) -> float:
        if max_width <= min_width:
            return 0.0
        midpoint = 0.5 * float(min_width + max_width)
        half_range = max(1.0, 0.5 * float(max_width - min_width))
        return max(0.0, 1.0 - abs(float(width) - midpoint) / half_range)

    @staticmethod
    def _safe_cut_score(scores: list[float], position: int) -> float:
        if not scores:
            return 0.0
        position = max(0, min(len(scores) - 1, int(position)))
        return float(scores[position])

    @staticmethod
    def _glyph_prior_field(config: dict[str, Any] | Any | None, name: str, default: Any) -> Any:
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    def _glyph_width_prior_is_active(self, config: dict[str, Any] | Any | None) -> bool:
        if config is None:
            return False
        ranges = self._glyph_prior_field(config, "ranges", {})
        weight = float(self._glyph_prior_field(config, "weight", 0.0) or 0.0)
        return bool(self._glyph_prior_field(config, "enabled", False)) and weight > 0.0 and bool(ranges)

    def _glyph_width_bounds_for_char(
        self,
        char: str,
        config: dict[str, Any] | Any | None,
    ) -> tuple[float, float] | None:
        ranges = self._glyph_prior_field(config, "ranges", {}) or {}
        fallback = None
        for group, bounds in ranges.items():
            if group == "~":
                fallback = bounds
                continue
            if char in str(group):
                low, high = bounds
                return float(low), float(high)
        if fallback is None:
            return None
        low, high = fallback
        return float(low), float(high)

    def _glyph_width_prior_adjustment(
        self,
        class_index: int,
        cell_width: float,
        input_height: int,
        config: dict[str, Any] | Any | None,
    ) -> tuple[float, float | None]:
        if not self._glyph_width_prior_is_active(config):
            return 0.0, None
        char = self.idx_to_char.get(int(class_index))
        if char is None:
            return 0.0, None
        bounds = self._glyph_width_bounds_for_char(char, config)
        if bounds is None:
            return 0.0, None

        low, high = bounds
        denominator = max(1.0, float(input_height))
        ratio = max(0.0, float(cell_width)) / denominator
        if low <= ratio <= high:
            return 0.0, ratio

        span = max(1e-6, high - low)
        distance = (low - ratio) / span if ratio < low else (ratio - high) / span
        penalty = min(4.0, distance * distance)
        weight = float(self._glyph_prior_field(config, "weight", 0.0) or 0.0)
        return -weight * penalty, ratio

    @staticmethod
    def _cell_width_in_input_pixels(
        start: int,
        end: int,
        *,
        input_width: int,
        ocr_width: int,
        source_start: int,
        source_end: int,
        segmentator_width: int,
    ) -> float:
        if end > start and ocr_width > 0:
            return max(0.0, float(end - start) * float(input_width) / float(ocr_width))
        if segmentator_width > 0:
            return max(0.0, float(source_end - source_start) * float(input_width) / float(segmentator_width))
        return 0.0

    def _rank_class_scores(
        self,
        class_scores: torch.Tensor,
        *,
        cell_width: float,
        input_height: int,
        glyph_width_prior: dict[str, Any] | Any | None,
        top_k: int,
        ocr_weight: float = 1.0,
    ) -> tuple[int, float, float, float | None, list[ClassConfidence]]:
        raw_scores = class_scores.detach().cpu().float().tolist()
        prior_active = self._glyph_width_prior_is_active(glyph_width_prior)
        ranked: list[tuple[float, float, int, float, float | None]] = []
        for class_index, raw_score in enumerate(raw_scores):
            if class_index not in self.idx_to_char:
                continue
            prior_score, ratio = self._glyph_width_prior_adjustment(
                class_index,
                cell_width,
                input_height,
                glyph_width_prior,
            )
            total_score = float(ocr_weight) * float(raw_score) + prior_score
            ranked.append((total_score, float(raw_score), class_index, prior_score, ratio))

        if not ranked:
            raise ValueError("No OCR classes are available for decoding")
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_k = max(1, min(int(top_k), len(ranked)))
        candidates = [
            ClassConfidence(
                label=self.class_label(class_index),
                confidence=raw_score,
                class_index=class_index,
                score=total_score if prior_active else None,
            )
            for total_score, raw_score, class_index, _, _ in ranked[:top_k]
        ]
        total_score, raw_score, class_index, prior_score, ratio = ranked[0]
        return class_index, raw_score, prior_score, ratio, candidates

    def decode_legacy_with_cuts(
        self,
        logits: torch.Tensor,
        segmentation_result: VerticalSegmentationResult,
        input_width: int | None = None,
        input_height: int | None = None,
        top_k: int = 8,
        center_fraction: float = 0.6,
        min_score_width: int = 1,
        ocr_source_x: np.ndarray | None = None,
        segmentator_source_x: np.ndarray | None = None,
        glyph_width_prior: dict[str, Any] | Any | None = None,
    ) -> CutDecodingResult:
        if self.loss_mode not in {"legacy", "legacy_logreg"}:
            raise ValueError(
                "legacy+cuts decoding expects a legacy OCR checkpoint; "
                f"got loss_mode={self.loss_mode!r}"
            )
        if logits.dim() != 3 or logits.size(0) != 1:
            raise ValueError(f"legacy+cuts decoding expects logits shape (1, C, T), got {tuple(logits.shape)}")
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError("center_fraction must be in (0, 1]")
        if min_score_width < 1:
            raise ValueError("min_score_width must be >= 1")

        probs = torch.softmax(logits, dim=1)[0]
        ocr_width = int(probs.size(1))
        segmentator_width = len(segmentation_result.raw_indices)
        input_width = int(input_width if input_width is not None else segmentation_result.input_shape[-1])
        input_height = int(input_height if input_height is not None else segmentation_result.input_shape[-2])
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

        raw_cuts = sorted({
            position for position in self._segmentation_cut_positions(segmentation_result)
            if 0 <= position < max(0, segmentator_width)
        })
        # Cut lines are explicit cell boundaries: only consecutive pairs are decoded.
        boundaries = self._map_segmentator_cuts_to_ocr_boundaries(
            raw_cuts,
            segmentator_width=segmentator_width,
            input_width=input_width,
            ocr_width=ocr_width,
            ocr_source_x=ocr_source_x,
            segmentator_source_x=segmentator_source_x,
        )
        use_coordinate_maps = ocr_source_x is not None and segmentator_source_x is not None
        intervals = list(zip(boundaries, boundaries[1:]))
        source_intervals = list(zip(raw_cuts, raw_cuts[1:]))
        top_k = max(1, min(int(top_k), probs.size(0)))

        symbols: list[CutDecodedSymbol] = []
        for (start, end), (source_start, source_end) in zip(intervals, source_intervals):
            if end > start:
                score_start, score_end = self._central_decode_span(
                    start,
                    end,
                    center_fraction=center_fraction,
                    min_width=min_score_width,
                )
            else:
                input_center: float
                if use_coordinate_maps:
                    left_source = self._source_x_for_timestep(
                        source_start,
                        segmentator_width,
                        segmentator_source_x,
                    )
                    right_source = self._source_x_for_timestep(
                        source_end,
                        segmentator_width,
                        segmentator_source_x,
                    )
                    mapped_center = None
                    if left_source is not None and right_source is not None:
                        mapped_center = self._input_x_for_source_x(
                            (left_source + right_source) * 0.5,
                            ocr_source_x,
                        )
                    if mapped_center is not None:
                        input_center = mapped_center
                    else:
                        source_center = (float(source_start) + float(source_end) + 1.0) * 0.5
                        input_center = source_center * float(input_width) / max(1.0, float(segmentator_width))
                else:
                    source_center = (float(source_start) + float(source_end) + 1.0) * 0.5
                    input_center = source_center * float(input_width) / max(1.0, float(segmentator_width))
                score_start = self._map_input_boundary_to_ocr(input_center, input_width, ocr_width)
                score_start = max(0, min(ocr_width - 1, score_start))
                score_end = score_start + 1
            if score_end <= score_start:
                continue
            scores = probs[:, score_start:score_end].mean(dim=1)
            cell_width = self._cell_width_in_input_pixels(
                start,
                end,
                input_width=input_width,
                ocr_width=ocr_width,
                source_start=source_start,
                source_end=source_end,
                segmentator_width=segmentator_width,
            )
            class_index, raw_score, glyph_width_score, glyph_width_ratio, candidates = self._rank_class_scores(
                scores,
                cell_width=cell_width,
                input_height=input_height,
                glyph_width_prior=glyph_width_prior,
                top_k=top_k,
            )
            char = self.idx_to_char.get(class_index)
            if char is None:
                raise ValueError(
                    f"OCR class index {class_index} is not present in the checkpoint alphabet"
                )

            symbols.append(
                CutDecodedSymbol(
                    char=char,
                    confidence=raw_score,
                    class_index=class_index,
                    start=int(start),
                    end=int(end),
                    source_start=int(source_start),
                    source_end=int(source_end),
                    candidates=candidates,
                    score_start=int(score_start),
                    score_end=int(score_end),
                    glyph_width_ratio=glyph_width_ratio,
                    glyph_width_score=glyph_width_score,
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
            decode_method="cells",
        )

    def decode_legacy_with_cuts_dp(
        self,
        logits: torch.Tensor,
        segmentation_result: VerticalSegmentationResult,
        input_width: int | None = None,
        input_height: int | None = None,
        top_k: int = 8,
        center_fraction: float = 0.6,
        min_score_width: int = 1,
        ocr_source_x: np.ndarray | None = None,
        segmentator_source_x: np.ndarray | None = None,
        cut_weight: float = 1.0,
        ocr_weight: float = 1.0,
        width_weight: float = 0.05,
        skip_cut_penalty: float = 0.35,
        glyph_width_prior: dict[str, Any] | Any | None = None,
    ) -> CutDecodingResult:
        if self.loss_mode not in {"legacy", "legacy_logreg"}:
            raise ValueError(
                "legacy+cuts DP decoding expects a legacy OCR checkpoint; "
                f"got loss_mode={self.loss_mode!r}"
            )
        if logits.dim() != 3 or logits.size(0) != 1:
            raise ValueError(f"legacy+cuts DP decoding expects logits shape (1, C, T), got {tuple(logits.shape)}")
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError("center_fraction must be in (0, 1]")
        if min_score_width < 1:
            raise ValueError("min_score_width must be >= 1")
        if (
            cut_weight < 0.0
            or ocr_weight < 0.0
            or width_weight < 0.0
            or skip_cut_penalty < 0.0
        ):
            raise ValueError("DP decode weights must be non-negative")

        probs = torch.softmax(logits, dim=1)[0]
        ocr_width = int(probs.size(1))
        segmentator_width = len(segmentation_result.raw_indices)
        input_width = int(input_width if input_width is not None else segmentation_result.input_shape[-1])
        input_height = int(input_height if input_height is not None else segmentation_result.input_shape[-2])
        if ocr_width <= 0 or segmentator_width <= 0:
            return CutDecodingResult(
                text="",
                symbols=[],
                cuts=[],
                boundaries=[],
                input_width=input_width,
                ocr_width=ocr_width,
                segmentator_width=segmentator_width,
                decode_method="dp",
                path_score=None,
            )

        candidate_cuts = self._candidate_cut_positions_from_scores(segmentation_result)
        if len(candidate_cuts) < 2:
            fallback = self.decode_legacy_with_cuts(
                logits,
                segmentation_result,
                input_width=input_width,
                input_height=input_height,
                top_k=top_k,
                center_fraction=center_fraction,
                min_score_width=min_score_width,
                ocr_source_x=ocr_source_x,
                segmentator_source_x=segmentator_source_x,
                glyph_width_prior=glyph_width_prior,
            )
            return CutDecodingResult(
                text=fallback.text,
                symbols=fallback.symbols,
                cuts=fallback.cuts,
                boundaries=fallback.boundaries,
                input_width=fallback.input_width,
                ocr_width=fallback.ocr_width,
                segmentator_width=fallback.segmentator_width,
                decode_method="dp_fallback_cells",
                path_score=fallback.path_score,
            )

        boundaries = self._map_segmentator_cuts_to_ocr_boundaries(
            candidate_cuts,
            segmentator_width=segmentator_width,
            input_width=input_width,
            ocr_width=ocr_width,
            ocr_source_x=ocr_source_x,
            segmentator_source_x=segmentator_source_x,
        )
        boundary_by_index = {
            index: boundary for index, boundary in enumerate(boundaries)
        }
        top_k = max(1, min(int(top_k), probs.size(0)))
        min_width = max(1, int(segmentation_result.cut_min_width or 1))
        max_width = max(0, int(segmentation_result.cut_max_width or 0))

        if max_width > 0:
            start_window = max_width
            end_window = max_width
            start_indices = [
                index for index, cut in enumerate(candidate_cuts)
                if cut <= start_window
            ]
            end_indices = [
                index for index, cut in enumerate(candidate_cuts)
                if cut >= segmentator_width - 1 - end_window
            ]
        else:
            start_indices = [0]
            end_indices = [len(candidate_cuts) - 1]
        if not start_indices:
            start_indices = [0]
        if not end_indices:
            end_indices = [len(candidate_cuts) - 1]
        end_index_set = set(end_indices)

        edge_cache: dict[tuple[int, int], tuple[float, CutDecodedSymbol]] = {}

        def score_edge(left_index: int, right_index: int) -> tuple[float, CutDecodedSymbol] | None:
            key = (left_index, right_index)
            if key in edge_cache:
                return edge_cache[key]
            source_start = int(candidate_cuts[left_index])
            source_end = int(candidate_cuts[right_index])
            width = source_end - source_start
            if width < min_width:
                return None
            if max_width > 0 and width > max_width:
                return None

            start = int(boundary_by_index[left_index])
            end = int(boundary_by_index[right_index])
            if end > start:
                score_start, score_end = self._central_decode_span(
                    start,
                    end,
                    center_fraction=center_fraction,
                    min_width=min_score_width,
                )
            else:
                center = max(0, min(ocr_width - 1, int(round((start + end) * 0.5))))
                score_start, score_end = center, center + 1
            if score_end <= score_start:
                return None

            class_scores = probs[:, score_start:score_end].mean(dim=1)
            cell_width = self._cell_width_in_input_pixels(
                start,
                end,
                input_width=input_width,
                ocr_width=ocr_width,
                source_start=source_start,
                source_end=source_end,
                segmentator_width=segmentator_width,
            )
            class_index, ocr_score, glyph_width_score, glyph_width_ratio, candidates = self._rank_class_scores(
                class_scores,
                cell_width=cell_width,
                input_height=input_height,
                glyph_width_prior=glyph_width_prior,
                top_k=top_k,
                ocr_weight=ocr_weight,
            )
            char = self.idx_to_char.get(class_index)
            if char is None:
                raise ValueError(
                    f"OCR class index {class_index} is not present in the checkpoint alphabet"
                )

            left_cut_score = self._safe_cut_score(segmentation_result.cut_scores, source_start)
            right_cut_score = self._safe_cut_score(segmentation_result.cut_scores, source_end)
            cut_score = 0.5 * (left_cut_score + right_cut_score)
            width_score = self._width_prior_score(width, min_width, max_width)
            skipped_cut_count = max(0, right_index - left_index - 1)
            edge_score = (
                float(cut_weight) * cut_score
                + float(ocr_weight) * ocr_score
                + float(width_weight) * width_score
                + float(glyph_width_score)
                - float(skip_cut_penalty) * float(skipped_cut_count)
            )
            symbol = CutDecodedSymbol(
                char=char,
                confidence=ocr_score,
                class_index=class_index,
                start=start,
                end=end,
                source_start=source_start,
                source_end=source_end,
                candidates=candidates,
                score_start=int(score_start),
                score_end=int(score_end),
                glyph_width_ratio=glyph_width_ratio,
                glyph_width_score=glyph_width_score,
            )
            edge_cache[key] = (edge_score, symbol)
            return edge_cache[key]

        states: dict[tuple[int, int], tuple[float, tuple[int, int] | None, CutDecodedSymbol | None]] = {
            (index, 0): (0.0, None, None) for index in start_indices
        }
        candidate_count = len(candidate_cuts)
        for right_index in range(candidate_count):
            for left_index in range(right_index):
                scored = score_edge(left_index, right_index)
                if scored is None:
                    continue
                edge_score, symbol = scored
                previous_states = [
                    (count, value)
                    for (state_index, count), value in states.items()
                    if state_index == left_index
                ]
                for count, (previous_score, _, _) in previous_states:
                    next_key = (right_index, count + 1)
                    next_score = previous_score + edge_score
                    if next_key not in states or next_score > states[next_key][0]:
                        states[next_key] = (
                            next_score,
                            (left_index, count),
                            symbol,
                        )

        final_items = [
            (key, value)
            for key, value in states.items()
            if key[0] in end_index_set and key[1] > 0
        ]
        if not final_items:
            fallback = self.decode_legacy_with_cuts(
                logits,
                segmentation_result,
                input_width=input_width,
                input_height=input_height,
                top_k=top_k,
                center_fraction=center_fraction,
                min_score_width=min_score_width,
                ocr_source_x=ocr_source_x,
                segmentator_source_x=segmentator_source_x,
                glyph_width_prior=glyph_width_prior,
            )
            return CutDecodingResult(
                text=fallback.text,
                symbols=fallback.symbols,
                cuts=fallback.cuts,
                boundaries=fallback.boundaries,
                input_width=fallback.input_width,
                ocr_width=fallback.ocr_width,
                segmentator_width=fallback.segmentator_width,
                decode_method="dp_fallback_cells",
                path_score=fallback.path_score,
            )

        best_key, (best_score, _, _) = max(
            final_items,
            key=lambda item: (item[1][0], item[0][1]),
        )
        path_score = best_score / float(best_key[1])
        symbols_reversed: list[CutDecodedSymbol] = []
        cut_indices_reversed: list[int] = [best_key[0]]
        current_key: tuple[int, int] | None = best_key
        while current_key is not None:
            _, previous_key, symbol = states[current_key]
            if symbol is not None:
                symbols_reversed.append(symbol)
            if previous_key is not None:
                cut_indices_reversed.append(previous_key[0])
            current_key = previous_key
        symbols = list(reversed(symbols_reversed))
        path_cut_indices = list(reversed(cut_indices_reversed))
        path_cuts = [candidate_cuts[index] for index in path_cut_indices]
        path_boundaries = [boundaries[index] for index in path_cut_indices]

        return CutDecodingResult(
            text="".join(symbol.char for symbol in symbols),
            symbols=symbols,
            cuts=path_cuts,
            boundaries=path_boundaries,
            input_width=input_width,
            ocr_width=ocr_width,
            segmentator_width=segmentator_width,
            decode_method="dp",
            path_score=float(path_score),
        )

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
