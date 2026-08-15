from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch

from .results import PreprocessDebug


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().float().clamp(0.0, 1.0)
    if image.dim() == 4:
        image = image[0]

    if image.shape[0] == 1:
        array = (image[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class ImagePreprocessingMixin:
    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess_pil_3d(image).unsqueeze(0)

    def preprocess_pil_debug(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, PreprocessDebug]:
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

    def preprocess_pil_with_source_x(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, np.ndarray]:
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
            image = image.resize(
                (new_width, self.image_height), Image.Resampling.BICUBIC
            )
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
            "baseline_line_pad": self.baseline_line_pad,
            "baseline_line_pad_px": self.baseline_line_pad_px,
            "baseline_detector_checkpoint": str(self.baseline_detector_checkpoint)
            if self.baseline_detector_checkpoint
            else None,
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
            image, baseline_debug = self._apply_baseline_crop(
                image, collect_debug=collect_debug
            )
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
            image = image.resize(
                (new_width, self.image_height), Image.Resampling.BICUBIC
            )
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

        return tensor.to(self.device), PreprocessDebug(
            metadata=debug_metadata, images=debug_images
        )

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
            padded = np.empty(
                (array.shape[0], array.shape[1] + delta * 2), dtype=array.dtype
            )
            padded[:, :delta] = left_fill
            padded[:, delta : delta + array.shape[1]] = array
            padded[:, delta + array.shape[1] :] = right_fill
        elif array.ndim == 3:
            padded = np.empty(
                (array.shape[0], array.shape[1] + delta * 2, array.shape[2]),
                dtype=array.dtype,
            )
            padded[:, :delta, :] = left_fill
            padded[:, delta : delta + array.shape[1], :] = array
            padded[:, delta + array.shape[1] :, :] = right_fill
        else:
            raise ValueError(f"Unsupported image array shape for x_pad: {array.shape}")
        return Image.fromarray(padded, mode=image.mode)

    @staticmethod
    def _side_background_values(
        array: np.ndarray,
    ) -> tuple[np.ndarray | int, np.ndarray | int]:
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
            np.asarray(
                np.median(left_band.reshape(-1, array.shape[2]), axis=0),
                dtype=array.dtype,
            ),
            np.asarray(
                np.median(right_band.reshape(-1, array.shape[2]), axis=0),
                dtype=array.dtype,
            ),
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
            border = np.concatenate(
                (array[0, :], array[-1, :], array[:, 0], array[:, -1])
            )
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
        output[
            paste_y : paste_y + source.shape[0], paste_x : paste_x + source.shape[1]
        ] = source
        return output

    def _crop_with_fill(
        self, image: Image.Image, box: tuple[int, int, int, int]
    ) -> Image.Image:
        left, top, right, bottom = box
        width = max(1, right - left)
        height = max(1, bottom - top)
        output = Image.new(
            image.mode, (width, height), self._background_fill_value(image)
        )
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

    def preprocess_image_debug(
        self, image_path: str | Path
    ) -> tuple[torch.Tensor, PreprocessDebug]:
        with Image.open(image_path) as image:
            return self.preprocess_pil_debug(image)
