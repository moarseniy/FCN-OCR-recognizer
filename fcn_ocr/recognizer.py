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
        if baseline_max_angle <= 0.0:
            raise ValueError("baseline_max_angle must be > 0")

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

        if verbose:
            self.print_summary()

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
                f"deskew={self.baseline_deskew}, max_angle={self.baseline_max_angle:.2f}"
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
            "x_pad": self.x_pad,
            "x_pad_mode": "border_median_original",
        }
        debug_images: list[tuple[str, Image.Image]] = []
        image = image.convert("RGB" if self.in_channels == 3 else "L")

        if self.baseline_crop:
            image, baseline_debug = self._apply_baseline_crop(image, collect_debug=collect_debug)
            debug_metadata.update(baseline_debug.metadata)
            debug_images.extend(baseline_debug.images)

        image = self._apply_x_pad(image)
        if collect_debug and self.x_pad > 0.0:
            debug_images.append(("x-pad border median from original geometry", image))

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

        return tensor.to(self.device), PreprocessDebug(metadata=debug_metadata, images=debug_images)

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

    def _apply_baseline_crop(self, image: Image.Image, collect_debug: bool) -> tuple[Image.Image, PreprocessDebug]:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for baseline_crop inference preprocessing")

        debug_images: list[tuple[str, Image.Image]] = []
        first = self._detect_baseline(image)
        if not first["ok"]:
            if collect_debug:
                debug_images.append(("baseline mask", Image.fromarray(first["cleaned_mask"])))
            metadata = {
                "baseline_status": first["status"],
                "baseline_foreground_pixels": int(first["foreground_pixels"]),
            }
            for source_key, target_key in (
                ("baseline_angle_degrees", "baseline_angle_degrees"),
                ("baseline_confidence", "baseline_confidence"),
                ("baseline_inlier_ratio", "baseline_inlier_ratio"),
                ("baseline_profile_coverage", "baseline_profile_coverage"),
                ("baseline_residual_mad", "baseline_residual_mad"),
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
            rotated = image.rotate(
                original_angle,
                expand=True,
                resample=Image.Resampling.BICUBIC,
                fillcolor=self._pil_fill_value(image.mode),
            )
            second = self._detect_baseline(rotated)
            if second["ok"]:
                working_image = rotated
                detection = second
                status = "ok_deskewed"
            else:
                status = f"ok_without_deskew_after_{second['status']}"

        cropped = self._crop_with_fill(working_image, detection["crop_box"])
        if collect_debug:
            overlay = self._draw_baseline_overlay(working_image, detection, detection["crop_box"])
            debug_images.append(("baseline crop overlay", overlay))
            debug_images.append(("baseline cleaned mask", Image.fromarray(detection["cleaned_mask"])))
            debug_images.append(("baseline cropped image", cropped))

        metadata = {
            "baseline_status": status,
            "baseline_angle_degrees": original_angle,
            "baseline_residual_angle_degrees": float(detection["angle_degrees"]),
            "baseline_crop_box": tuple(int(value) for value in detection["crop_box"]),
            "baseline_text_bbox": tuple(int(value) for value in detection["text_bbox"]),
            "baseline_text_height": int(detection["text_height"]),
            "baseline_foreground_pixels": int(detection["foreground_pixels"]),
            "baseline_confidence": float(detection["confidence"]),
            "baseline_inlier_ratio": float(detection["inlier_ratio"]),
            "baseline_profile_coverage": float(detection["profile_coverage"]),
            "baseline_residual_mad": float(detection["residual_mad"]),
            "baseline_residual_rmse": float(detection["residual_rmse"]),
        }
        return cropped, PreprocessDebug(metadata=metadata, images=debug_images)

    def _detect_baseline(self, image: Image.Image) -> dict[str, Any]:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        mask = self._make_text_mask(gray)
        cleaned_mask = self._clean_text_mask(mask)
        foreground_pixels = int(np.count_nonzero(cleaned_mask))
        if foreground_pixels < max(4, int(round(gray.size * 0.00005))):
            return {
                "ok": False,
                "status": "not_enough_foreground",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
            }

        ys, xs = np.nonzero(cleaned_mask)
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        profile_x, profile_y, profile_weights = self._baseline_profile_points(cleaned_mask, x_min, x_max)

        text_width = x_max - x_min + 1
        profile_coverage = float(profile_x.size) / max(1.0, float(text_width))
        if profile_x.size < max(6, int(round(text_width * 0.08))):
            return {
                "ok": False,
                "status": "not_enough_baseline_columns",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "baseline_profile_coverage": profile_coverage,
            }

        line = self._fit_baseline_line(
            profile_x,
            profile_y,
            profile_weights,
            image_height=image.height,
            text_width=text_width,
            profile_coverage=profile_coverage,
        )
        if line is None:
            return {
                "ok": False,
                "status": "baseline_fit_failed",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "baseline_profile_coverage": profile_coverage,
            }

        slope = float(line["slope"])
        intercept = float(line["intercept"])
        angle_degrees = math.degrees(math.atan(float(slope)))
        if abs(angle_degrees) > self.baseline_max_angle:
            return {
                "ok": False,
                "status": "baseline_angle_rejected",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "baseline_angle_degrees": float(angle_degrees),
                "baseline_confidence": float(line["confidence"]),
                "baseline_profile_coverage": profile_coverage,
            }
        if float(line["confidence"]) < 0.28:
            return {
                "ok": False,
                "status": "baseline_low_confidence",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
                "baseline_angle_degrees": float(angle_degrees),
                "baseline_confidence": float(line["confidence"]),
                "baseline_inlier_ratio": float(line["inlier_ratio"]),
                "baseline_profile_coverage": profile_coverage,
                "baseline_residual_mad": float(line["residual_mad"]),
            }

        crop_box, text_height = self._baseline_crop_box(
            slope=float(slope),
            intercept=float(intercept),
            xs=xs,
            ys=ys,
            image_width=image.width,
        )
        return {
            "ok": True,
            "status": "ok",
            "cleaned_mask": cleaned_mask,
            "foreground_pixels": foreground_pixels,
            "slope": float(slope),
            "intercept": float(intercept),
            "angle_degrees": float(angle_degrees),
            "confidence": float(line["confidence"]),
            "inlier_ratio": float(line["inlier_ratio"]),
            "profile_coverage": profile_coverage,
            "residual_mad": float(line["residual_mad"]),
            "residual_rmse": float(line["residual_rmse"]),
            "profile_x": profile_x,
            "profile_y": profile_y,
            "inlier_mask": line["inlier_mask"],
            "crop_box": crop_box,
            "text_bbox": (x_min, y_min, x_max + 1, y_max + 1),
            "text_height": int(text_height),
        }

    def _make_text_mask(self, gray: np.ndarray) -> np.ndarray:
        border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
        background_is_bright = float(np.median(border)) >= 128.0
        threshold_type = cv2.THRESH_BINARY_INV if background_is_bright else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        profile_x: list[int] = []
        profile_y: list[float] = []
        profile_weights: list[float] = []

        for x in range(x_min, x_max + 1):
            column_y = np.flatnonzero(mask[:, x])
            if column_y.size == 0:
                continue
            profile_x.append(x)
            profile_y.append(float(np.quantile(column_y.astype(np.float64), 0.88)))
            profile_weights.append(float(np.sqrt(column_y.size)))

        if not profile_x:
            empty = np.asarray([], dtype=np.float64)
            return empty, empty, empty

        xs = np.asarray(profile_x, dtype=np.float64)
        ys = np.asarray(profile_y, dtype=np.float64)
        weights = np.asarray(profile_weights, dtype=np.float64)
        if ys.size >= 5:
            ys = TextRecognizer._median_smooth_1d(ys, radius=2)
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

    def _draw_baseline_overlay(
        self,
        image: Image.Image,
        detection: dict[str, Any],
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        output = image.convert("RGB")
        draw = ImageDraw.Draw(output)
        line_width = max(1, int(round(image.height / 96)))
        x0 = 0
        x1 = max(0, image.width - 1)
        y0 = float(detection["slope"]) * x0 + float(detection["intercept"])
        y1 = float(detection["slope"]) * x1 + float(detection["intercept"])
        draw.line((x0, y0, x1, y1), fill=(230, 30, 30), width=line_width)
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
        if crop_box is not None:
            draw.rectangle(crop_box, outline=(20, 150, 60), width=line_width)
        if "text_bbox" in detection:
            draw.rectangle(detection["text_bbox"], outline=(80, 120, 240), width=max(1, line_width // 2))
        return output

    def _crop_with_fill(self, image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
        left, top, right, bottom = box
        width = max(1, right - left)
        height = max(1, bottom - top)
        output = Image.new(image.mode, (width, height), self._pil_fill_value(image.mode))
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
        if segmentation_result.cut_positions is not None:
            return [int(position) for position in segmentation_result.cut_positions]

        cuts: list[int] = []
        for run in segmentation_result.runs:
            if run.label != 1:
                continue
            cuts.append(int(round((run.start + run.end + 1) * 0.5)))
        return cuts

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
        active = self._close_boolean_gaps(active, max_gap=max(1, int(round(width * 0.01))))
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
    def _close_boolean_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
        output = values.astype(bool).copy()
        if max_gap <= 0 or output.size == 0:
            return output

        runs = TextRecognizer._boolean_runs(output)
        for (_, prev_end), (next_start, _) in zip(runs, runs[1:]):
            if next_start - prev_end <= max_gap:
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
        max_gap_ratio: float,
        min_gap_width: int,
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
        if max_gap_ratio < 0.0:
            raise ValueError("boundary_cut_max_gap_ratio must be non-negative")

        threshold = max(float(min_gap_width), typical_width * float(max_gap_ratio))
        left_gap = cuts[0] - left_boundary
        right_gap = right_boundary - cuts[-1]

        # In auto mode we promote both edges only when both outer intervals look
        # much smaller than a normal character interval. This catches outputs
        # like |A|B|C| without deleting a real narrow first/last glyph by itself.
        if left_gap <= threshold and right_gap <= threshold:
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
        boundary_cut_max_gap_ratio: float = 0.45,
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
            max_gap_ratio=boundary_cut_max_gap_ratio,
            min_gap_width=edge_min_width,
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
