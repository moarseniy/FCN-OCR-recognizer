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

from model import FullyConvTextRecognizer, decode_greedy_batch_tensor
from .results import BLANK_SYMBOL, ClassConfidence, DecodedSymbol, PreprocessDebug, RecognitionResult, display_char


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
        self.baseline_crop = bool(baseline_crop)
        self.baseline_top_pad = float(baseline_top_pad)
        self.baseline_bottom_pad = float(baseline_bottom_pad)
        self.baseline_deskew = bool(baseline_deskew)
        self.baseline_max_angle = float(baseline_max_angle)

        self.alphabet = self.checkpoint["alphabet"]
        self.idx_to_char = {idx: char for idx, char in enumerate(self.alphabet)}

        model_config = self.checkpoint.get("model_config", {})
        checkpoint_config = self.checkpoint.get("config", {})
        self.in_channels = int(model_config.get("in_channels", 3))
        self.num_classes = int(model_config.get("num_classes", len(self.alphabet) + 1))
        self.blank_idx = int(model_config.get("blank_idx", len(self.alphabet)))
        self.space_char = checkpoint_config.get("space_char", " ")
        self.space_idx = self.alphabet.index(self.space_char) if self.space_char in self.alphabet else None
        self.image_height = int(checkpoint_config.get("image_height", 48))
        self.preprocess_fill = int(checkpoint_config.get("background", 255))

        self.model = FullyConvTextRecognizer(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
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
        print(f"Alphabet size: {len(self.alphabet)}")
        print(f"Blank index: {self.blank_idx}")
        print(f"Preprocess scale_x: {self.scale_x:+.4f}")
        print(f"Preprocess y_pad:   {self.y_pad:+.4f}")
        print(f"Baseline crop:      {self.baseline_crop}")
        if self.baseline_crop:
            print(
                f"  top_pad={self.baseline_top_pad:.3f}, "
                f"bottom_pad={self.baseline_bottom_pad:.3f}, "
                f"deskew={self.baseline_deskew}, max_angle={self.baseline_max_angle:.2f}"
            )

    def class_label(self, index: int) -> str:
        if index == self.blank_idx:
            return BLANK_SYMBOL
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
        }
        debug_images: list[tuple[str, Image.Image]] = []
        image = image.convert("RGB" if self.in_channels == 3 else "L")

        if self.baseline_crop:
            image, baseline_debug = self._apply_baseline_crop(image, collect_debug=collect_debug)
            debug_metadata.update(baseline_debug.metadata)
            debug_images.extend(baseline_debug.images)

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
            return image, PreprocessDebug(
                metadata={
                    "baseline_status": first["status"],
                    "baseline_foreground_pixels": int(first["foreground_pixels"]),
                },
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

        profile_x: list[int] = []
        profile_y: list[int] = []
        for x in range(x_min, x_max + 1):
            column_y = np.flatnonzero(cleaned_mask[:, x])
            if column_y.size == 0:
                continue
            profile_x.append(x)
            profile_y.append(int(column_y.max()))

        if len(profile_x) < max(6, int(round((x_max - x_min + 1) * 0.08))):
            return {
                "ok": False,
                "status": "not_enough_baseline_columns",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
            }

        line = self._fit_baseline_line(np.asarray(profile_x), np.asarray(profile_y), image.height)
        if line is None:
            return {
                "ok": False,
                "status": "baseline_fit_failed",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
            }

        slope, intercept = line
        angle_degrees = math.degrees(math.atan(float(slope)))
        if abs(angle_degrees) > self.baseline_max_angle:
            return {
                "ok": False,
                "status": "baseline_angle_rejected",
                "cleaned_mask": cleaned_mask,
                "foreground_pixels": foreground_pixels,
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
        return cleaned

    def _fit_baseline_line(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        image_height: int,
    ) -> tuple[float, float] | None:
        if xs.size < 2:
            return None

        work_x = xs.astype(np.float64)
        work_y = ys.astype(np.float64)
        low = np.quantile(work_y, 0.10)
        high = np.quantile(work_y, 0.98)
        keep = (work_y >= low) & (work_y <= high)
        if int(keep.sum()) >= 2:
            work_x = work_x[keep]
            work_y = work_y[keep]

        if work_x.size < 2:
            return None

        for _ in range(4):
            slope, intercept = np.polyfit(work_x, work_y, deg=1)
            predicted = slope * work_x + intercept
            residuals = work_y - predicted
            median = float(np.median(residuals))
            mad = float(np.median(np.abs(residuals - median)))
            tolerance = max(2.0, image_height * 0.04, mad * 3.0)
            next_keep = (residuals >= median - tolerance) & (residuals <= median + tolerance)
            if int(next_keep.sum()) < max(2, int(round(work_x.size * 0.40))):
                break
            if bool(np.all(next_keep)):
                break
            work_x = work_x[next_keep]
            work_y = work_y[next_keep]

        if work_x.size < 2:
            return None

        slope, intercept = np.polyfit(work_x, work_y, deg=1)
        return float(slope), float(intercept)

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
        if crop_box is not None:
            draw.rectangle(crop_box, outline=(20, 150, 60), width=line_width)
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
                if class_index != self.blank_idx and class_index in self.idx_to_char
            )
            decoded.append((text, raw_ids))
        return decoded

    def output_width_for_input_width(self, width: int) -> int:
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
        collapsed, lengths = decode_greedy_batch_tensor(pred_ids)
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
            if class_index == self.blank_idx:
                continue
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

        text = "".join(
            self.idx_to_char[idx]
            for idx in collapsed[0, : lengths[0]].detach().cpu().tolist()
            if idx != self.blank_idx and idx in self.idx_to_char
        )
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

    @torch.no_grad()
    def recognize_tensor_debug(self, image_tensor: torch.Tensor, top_k: int = 8) -> RecognitionResult:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
        return self.analyze_logits(logits, input_shape=tuple(image_tensor.shape), top_k=top_k)

    @torch.no_grad()
    def recognize_tensor(self, image_tensor: torch.Tensor) -> tuple[str, list[int]]:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        logits = self.model(image_tensor)
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
