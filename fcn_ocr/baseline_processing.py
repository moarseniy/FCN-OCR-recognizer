from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from .checkpoint import load_fcn_checkpoint
from .results import PreprocessDebug


class NeuralBaselineMixin:
    def _load_baseline_detector(self) -> None:
        if self.baseline_detector_checkpoint is None:
            return
        if not self.baseline_detector_checkpoint.exists():
            raise FileNotFoundError(
                f"Baseline detector checkpoint not found: {self.baseline_detector_checkpoint}"
            )

        loaded = load_fcn_checkpoint(self.baseline_detector_checkpoint, self.device)
        if loaded.loss_mode != "baseline_heatmap":
            raise ValueError(
                "Baseline detector checkpoint must be trained with loss_mode=baseline_heatmap; "
                f"got loss_mode={loaded.loss_mode!r}"
            )

        self.baseline_detector_model = loaded.model
        self.baseline_detector_in_channels = loaded.in_channels
        self.baseline_detector_image_height = int(
            loaded.training_config["image_height"]
        )
        self.baseline_detector_architecture = loaded.architecture

    def _apply_baseline_crop(
        self, image: Image.Image, collect_debug: bool
    ) -> tuple[Image.Image, PreprocessDebug]:
        debug_images: list[tuple[str, Image.Image]] = []
        first = self._detect_baseline(image)
        if not first["ok"]:
            if collect_debug:
                debug_images.append(
                    ("baseline mask", Image.fromarray(first["cleaned_mask"]))
                )
            metadata = {
                "baseline_status": first["status"],
                "baseline_line_pad": self.baseline_line_pad,
                "baseline_line_pad_px": self.baseline_line_pad_px,
                "baseline_foreground_pixels": int(first["foreground_pixels"]),
            }
            for source_key, target_key in (
                ("angle_degrees", "baseline_angle_degrees"),
                ("bottom_angle_degrees", "baseline_bottom_angle_degrees"),
                ("topline_angle_degrees", "baseline_top_angle_degrees"),
                ("baseline_pair_angle_difference", "baseline_pair_angle_difference"),
                (
                    "baseline_pair_angle_max_difference",
                    "baseline_pair_angle_max_difference",
                ),
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
                debug_images.append(
                    ("baseline on original", self._draw_baseline_overlay(image, first))
                )
                debug_images.append(
                    (
                        "baseline lines original",
                        self._draw_baseline_lines_debug(image, first),
                    )
                )
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
            else:
                metadata = {
                    "baseline_status": f"rotated_detection_failed_after_{second['status']}",
                    "baseline_line_pad": self.baseline_line_pad,
                    "baseline_line_pad_px": self.baseline_line_pad_px,
                    "baseline_angle_degrees": original_angle,
                    "baseline_foreground_pixels": int(
                        second.get("foreground_pixels", first["foreground_pixels"])
                    ),
                }
                if collect_debug:
                    debug_images.append(("baseline rotated detection failed", rotated))
                    debug_images.append(
                        (
                            "baseline rotated cleaned mask",
                            Image.fromarray(second["cleaned_mask"]),
                        )
                    )
                return image, PreprocessDebug(metadata=metadata, images=debug_images)

        cropped = self._crop_with_fill(working_image, detection["crop_box"])
        if collect_debug:
            debug_images.append(
                (
                    "baseline detected lines",
                    self._draw_baseline_lines_debug(
                        working_image, detection, detection["crop_box"]
                    ),
                )
            )
            overlay = self._draw_baseline_overlay(
                working_image, detection, detection["crop_box"]
            )
            debug_images.append(("baseline crop overlay", overlay))
            debug_images.append(
                ("baseline cleaned mask", Image.fromarray(detection["cleaned_mask"]))
            )
            debug_images.append(("baseline cropped image", cropped))

        metadata = {
            "baseline_status": status,
            "baseline_line_pad": self.baseline_line_pad,
            "baseline_line_pad_px": self.baseline_line_pad_px,
            "baseline_angle_degrees": original_angle,
            "baseline_residual_angle_degrees": float(detection["angle_degrees"]),
            "baseline_top_angle_degrees": float(
                first.get("topline_angle_degrees", original_angle)
            ),
            "baseline_bottom_angle_degrees": float(
                first.get("bottom_angle_degrees", original_angle)
            ),
            "baseline_pair_angle_difference": float(
                first.get("baseline_pair_angle_difference", 0.0)
            ),
            "baseline_pair_angle_max_difference": float(
                first.get(
                    "baseline_pair_angle_max_difference",
                    self._baseline_pair_angle_max_difference(),
                )
            ),
            "baseline_top_angle_weight": float(
                first.get("baseline_top_angle_weight", 0.0)
            ),
            "baseline_bottom_angle_weight": float(
                first.get("baseline_bottom_angle_weight", 1.0)
            ),
            "baseline_angle_method": first.get("baseline_angle_method", "bottom_only"),
            "baseline_crop_box": tuple(int(value) for value in detection["crop_box"]),
            "baseline_text_bbox": tuple(int(value) for value in detection["text_bbox"]),
            "baseline_text_height": int(detection["text_height"]),
            "baseline_foreground_pixels": int(detection["foreground_pixels"]),
            "baseline_confidence": float(detection["confidence"]),
            "baseline_bottom_confidence": float(
                detection.get("bottom_confidence", detection["confidence"])
            ),
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
                    "topline_inlier_ratio": float(
                        detection.get("topline_inlier_ratio", 0.0)
                    ),
                    "topline_profile_coverage": float(
                        detection.get("topline_profile_coverage", 0.0)
                    ),
                    "topline_residual_mad": float(
                        detection.get("topline_residual_mad", 0.0)
                    ),
                }
            )
        if "rejected_baseline_angle_degrees" in detection:
            metadata["baseline_rejected_angle_degrees"] = float(
                detection["rejected_baseline_angle_degrees"]
            )
        if "rejected_baseline_confidence" in detection:
            metadata["baseline_rejected_confidence"] = float(
                detection["rejected_baseline_confidence"]
            )
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
            else:
                return image, source_x

        cropped = self._crop_with_fill(working_image, detection["crop_box"])
        cropped_source_x = self._crop_float_map_with_fill(
            working_source_x,
            detection["crop_box"],
            fill=-1.0,
        )
        return cropped, cropped_source_x

    def _detect_baseline(self, image: Image.Image) -> dict[str, Any]:
        if self.baseline_detector_model is None:
            raise RuntimeError("baseline detector model is not loaded")
        return self._detect_baseline_neural(image)

    def _detect_baseline_neural(self, image: Image.Image) -> dict[str, Any]:
        if self.baseline_detector_model is None:
            raise RuntimeError("baseline detector model is not loaded")

        heatmaps, cleaned_mask, foreground_pixels, scale_x, scale_y = (
            self._baseline_detector_heatmaps(image)
        )
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
        bottom_line = self._scale_baseline_line(
            bottom_line, scale_x=scale_x, scale_y=scale_y
        )
        x_mid = max(0.0, (image.width - 1) * 0.5)
        top_mid = float(top_line["slope"]) * x_mid + float(top_line["intercept"])
        bottom_mid = float(bottom_line["slope"]) * x_mid + float(
            bottom_line["intercept"]
        )
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
        confidence = min(
            float(top_line["confidence"]), float(bottom_line["confidence"])
        )
        angle = self._combined_baseline_angle(top_line, bottom_line)
        if not angle["baseline_pair_angle_consistent"]:
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
            min(
                image.height, int(math.ceil(float(max(bottom_y.max(), ys.max())) + 1.0))
            ),
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

    def _baseline_detector_input(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, float, float, tuple[int, int]]:
        mode = "RGB" if self.baseline_detector_in_channels == 3 else "L"
        detector_image = image.convert(mode)
        if (
            self.baseline_detector_image_height > 0
            and detector_image.height != self.baseline_detector_image_height
        ):
            new_width = max(
                1,
                round(
                    detector_image.width
                    * self.baseline_detector_image_height
                    / detector_image.height
                ),
            )
            detector_image = detector_image.resize(
                (new_width, self.baseline_detector_image_height),
                Image.Resampling.BICUBIC,
            )

        scale_x = detector_image.width / max(1.0, float(image.width))
        scale_y = detector_image.height / max(1.0, float(image.height))
        array = np.asarray(detector_image, dtype=np.float32) / 255.0
        if self.baseline_detector_in_channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return (
            tensor.to(self.device),
            float(scale_x),
            float(scale_y),
            detector_image.size,
        )

    def _line_from_baseline_heatmap(
        self, heatmap: np.ndarray, method: str
    ) -> dict[str, Any] | None:
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
    def _scale_baseline_line(
        line: dict[str, Any], scale_x: float, scale_y: float
    ) -> dict[str, Any]:
        scale_x = max(scale_x, 1e-6)
        scale_y = max(scale_y, 1e-6)
        scaled = dict(line)
        scaled["slope"] = float(line["slope"]) * scale_x / scale_y
        scaled["intercept"] = float(line["intercept"]) / scale_y
        scaled["profile_x"] = np.asarray(line["profile_x"], dtype=np.float64) / scale_x
        scaled["profile_y"] = np.asarray(line["profile_y"], dtype=np.float64) / scale_y
        scaled["inlier_mask"] = np.asarray(line["inlier_mask"], dtype=bool)
        return scaled

    def _baseline_heatmap_mask(
        self, heatmaps: np.ndarray, output_size: tuple[int, int]
    ) -> np.ndarray:
        combined = np.max(heatmaps, axis=0)
        combined = (combined >= self.baseline_detector_threshold).astype(np.uint8) * 255
        mask_image = Image.fromarray(combined, mode="L").resize(
            output_size, Image.Resampling.BILINEAR
        )
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
        inlier_ratio = float(np.count_nonzero(original_inlier_mask)) / max(
            1.0, float(xs.size)
        )

        coverage_score = min(1.0, profile_coverage / 0.55)
        inlier_score = max(0.0, min(1.0, (inlier_ratio - 0.25) / 0.65))
        residual_score = max(
            0.0, min(1.0, 1.0 - residual_mad / max(2.0, image_height * 0.14))
        )
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
        sample_indices = np.unique(
            np.linspace(0, xs.size - 1, sample_count, dtype=np.int64)
        )
        if sample_indices.size < 2:
            return None

        min_dx = max(3.0, float(text_width) * 0.12)
        tolerance = max(2.0, float(image_height) * 0.055)
        best_score: tuple[int, float, float] | None = None
        best_line: tuple[float, float] | None = None

        for left_pos, left_index in enumerate(sample_indices[:-1]):
            x1 = float(xs[left_index])
            y1 = float(ys[left_index])
            for right_index in sample_indices[left_pos + 1 :]:
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

        if line_height <= 2.0 or float(np.median(top_ys)) >= float(
            np.median(bottom_ys)
        ):
            return None

        margin_reference = max(line_height, bbox_height)
        margin = max(
            0.0, margin_reference * self.baseline_line_pad + self.baseline_line_pad_px
        )
        top = int(math.floor(float(top_ys.min()) - margin))
        bottom = int(math.ceil(float(bottom_ys.max()) + 1.0 + margin))
        if bottom <= top:
            return None
        return (0, top, image_width, bottom), max(1, int(round(bottom - top)))

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
                is_inlier = (
                    bool(inlier_mask[index])
                    if inlier_mask is not None and index < len(inlier_mask)
                    else False
                )
                color = (20, 150, 70) if is_inlier else (40, 110, 220)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=color
                )
        top_profile_x = detection.get("topline_profile_x")
        top_profile_y = detection.get("topline_profile_y")
        top_inlier_mask = detection.get("topline_inlier_mask")
        if top_profile_x is not None and top_profile_y is not None:
            radius = 1
            step = max(1, int(math.ceil(len(top_profile_x) / 500)))
            for index in range(0, len(top_profile_x), step):
                x = float(top_profile_x[index])
                y = float(top_profile_y[index])
                is_inlier = (
                    bool(top_inlier_mask[index])
                    if top_inlier_mask is not None and index < len(top_inlier_mask)
                    else False
                )
                color = (70, 190, 230) if is_inlier else (80, 120, 240)
                draw.rectangle(
                    (x - radius, y - radius, x + radius, y + radius), fill=color
                )
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
                overlay_draw.rectangle(
                    (0, 0, output.width, visible_top), fill=(0, 0, 0, 55)
                )
            if visible_bottom < output.height:
                overlay_draw.rectangle(
                    (0, visible_bottom, output.width, output.height), fill=(0, 0, 0, 55)
                )
            output = Image.alpha_composite(output.convert("RGBA"), overlay).convert(
                "RGB"
            )
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
