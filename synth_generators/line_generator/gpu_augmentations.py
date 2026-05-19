from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .dataset import SUPPORTED_AUGMENTATIONS, SingleLineDatasetConfig
except ImportError:
    from dataset import SUPPORTED_AUGMENTATIONS, SingleLineDatasetConfig


AugmentationParams = dict[str, Any] | None


class GpuTextAugmenter:
    """Torch-only OCR augmentations for BCHW tensors in [0, 1]."""

    def __init__(self, config: SingleLineDatasetConfig):
        self.config = config
        self.probabilities = self._effective_probabilities(config)
        self.params = config.augmentations
        self.last_augmentations: list[list[dict[str, Any]]] = []

    def enabled(self) -> bool:
        return any(probability > 0.0 for probability in self.probabilities.values())

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        augmented, _ = self._augment(images, collect_metadata=False)
        return augmented

    def augment_with_metadata(self, images: torch.Tensor) -> tuple[torch.Tensor, list[list[dict[str, Any]]]]:
        augmented, metadata = self._augment(images, collect_metadata=True)
        return augmented, metadata or [[] for _ in range(images.size(0))]

    def _augment(
        self,
        images: torch.Tensor,
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[list[dict[str, Any]]] | None]:
        metadata: list[list[dict[str, Any]]] | None = [[] for _ in range(images.size(0))] if collect_metadata else None
        if not self.enabled() or images.numel() == 0:
            self.last_augmentations = metadata or []
            return images, metadata

        output = images
        for name in SUPPORTED_AUGMENTATIONS:
            probability = self.probabilities.get(name, 0.0)
            if probability <= 0.0:
                continue

            mask = torch.rand(output.size(0), device=output.device) <= probability
            if not bool(mask.any()):
                continue

            selected_indices = mask.nonzero(as_tuple=False).flatten()
            augmented, params_by_sample = self._apply_one(
                name,
                output[mask],
                self.params.get(name, {}),
                collect_metadata,
            )
            output = output.clone()
            output[mask] = augmented

            if metadata is not None and params_by_sample is not None:
                for sample_index, params in zip(selected_indices.tolist(), params_by_sample):
                    if params is not None:
                        metadata[sample_index].append({"name": name, "params": params})

        output = output.clamp(0.0, 1.0)
        self.last_augmentations = metadata or []
        return output, metadata

    def _apply_one(
        self,
        name: str,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        if name == "cycle_shift":
            return self._cycle_shift(images, params, collect_metadata)
        if name == "preprocess_geometry":
            return self._preprocess_geometry(images, params, collect_metadata)
        if name in {"strong_blur", "gaussian_blur"}:
            default = 1.2 if name == "strong_blur" else self.config.blur_radius
            return self._gaussian_blur(images, params, default, collect_metadata)
        if name == "motion_blur":
            return self._motion_blur(images, params, collect_metadata)
        if name == "scale":
            return self._scale(images, params, collect_metadata)
        if name in {"darkening", "brightness"}:
            default = 0.75 if name == "darkening" else 1.0
            return self._brightness(images, params, default, collect_metadata)
        if name in {"noise", "gaussian_noise"}:
            return self._noise(images, params, collect_metadata)
        if name == "projective":
            return self._projective(images, params, collect_metadata)
        if name == "rotate":
            return self._rotate(images, params, collect_metadata)
        if name == "crop_x":
            return self._crop_x(images, params, collect_metadata)
        if name == "crop_y":
            return self._crop_y(images, params, collect_metadata)
        if name == "random_line":
            return self._random_line(images, params, collect_metadata)
        if name == "morphology":
            return self._morphology(images, params, collect_metadata)
        if name == "unsharp_mask":
            return self._unsharp_mask(images, params, collect_metadata)
        if name == "contrast":
            return self._contrast(images, params, collect_metadata)
        if name == "invert":
            return 1.0 - images, self._repeat_log(images, {}) if collect_metadata else None
        return images, None

    @staticmethod
    def _effective_probabilities(config: SingleLineDatasetConfig) -> dict[str, float]:
        if config.augmentation_probabilities:
            return dict(config.augmentation_probabilities)

        probabilities: dict[str, float] = {}
        if config.max_rotation_degrees:
            probabilities["rotate"] = 1.0
        if config.blur_radius:
            probabilities["gaussian_blur"] = 1.0
        if config.noise_std:
            probabilities["noise"] = 1.0
        return probabilities

    def _cycle_shift(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        max_x = int(params.get("max_x", 0))
        max_y = int(params.get("max_y", 0))
        if max_x <= 0 and max_y <= 0:
            return images, None

        output = images.clone()
        logs: list[AugmentationParams] | None = [] if collect_metadata else None
        for index in range(images.size(0)):
            shift_x = self._randint(-max_x, max_x) if max_x > 0 else 0
            shift_y = self._randint(-max_y, max_y) if max_y > 0 else 0
            if shift_x != 0 or shift_y != 0:
                output[index] = torch.roll(images[index], shifts=(shift_y, shift_x), dims=(-2, -1))
                if logs is not None:
                    logs.append({"shift_x": shift_x, "shift_y": shift_y, "max_x": max_x, "max_y": max_y})
            elif logs is not None:
                logs.append(None)
        return output, logs

    def _preprocess_geometry(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        scale_x = self._sample_tensor_range(params, "scale_x", 0.0, images.size(0), images.device, images.dtype)
        y_pad = self._sample_tensor_range(params, "y_pad", 0.0, images.size(0), images.device, images.dtype)
        scale_x = scale_x.clamp(min=-0.95)
        y_pad = y_pad.clamp(min=-0.95)

        if bool((scale_x == 0.0).all() and (y_pad == 0.0).all()):
            return images, None

        theta = images.new_zeros((images.size(0), 2, 3))
        theta[:, 0, 0] = 1.0 / (1.0 + scale_x)
        theta[:, 1, 1] = 1.0 + y_pad

        logs = None
        if collect_metadata:
            logs = [
                {
                    "scale_x": float(sample_scale_x),
                    "y_pad": float(sample_y_pad),
                    "fillcolor": int(params.get("fillcolor", self.config.background)),
                }
                for sample_scale_x, sample_y_pad in zip(
                    scale_x.detach().cpu().tolist(),
                    y_pad.detach().cpu().tolist(),
                )
            ]
        return self._warp_affine(images, theta, self._fill_value(params)), logs

    def _gaussian_blur(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        default_radius: float,
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        radius = self._sample_range(params, "radius", default_radius)
        if radius <= 0.0:
            return images, None

        sigma = max(float(radius), 0.05)
        size = max(3, int(math.ceil(sigma * 6.0)) | 1)
        coords = torch.arange(size, device=images.device, dtype=images.dtype) - size // 2
        kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        logs = self._repeat_log(images, {"radius": radius, "kernel_size": size}) if collect_metadata else None
        return self._depthwise_conv(images, kernel_2d), logs

    def _motion_blur(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        size = int(round(self._sample_range(params, "size", 5)))
        if size <= 1:
            return images, None
        size = size | 1
        angle = self._sample_range(params, "angle", 0.0)
        kernel = self._motion_kernel(images.device, images.dtype, size, angle)
        logs = self._repeat_log(images, {"size": size, "angle": angle}) if collect_metadata else None
        return self._depthwise_conv(images, kernel), logs

    def _scale(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        factor_default = self._sample_range(params, "factor", 1.0)
        factor_x = self._sample_range(params, "factor_x", factor_default)
        factor_y = self._sample_range(params, "factor_y", factor_default)
        if factor_x <= 0.0 or factor_y <= 0.0 or (factor_x == 1.0 and factor_y == 1.0):
            return images, None
        theta = images.new_zeros((images.size(0), 2, 3))
        theta[:, 0, 0] = 1.0 / factor_x
        theta[:, 1, 1] = 1.0 / factor_y
        logs = self._repeat_log(
            images,
            {
                "factor_x": factor_x,
                "factor_y": factor_y,
                "fillcolor": int(params.get("fillcolor", self.config.background)),
            },
        ) if collect_metadata else None
        return self._warp_affine(images, theta, self._fill_value(params)), logs

    def _rotate(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        max_degrees = float(params.get("max_degrees", self.config.max_rotation_degrees))
        if max_degrees <= 0.0:
            return images, None
        angles = (torch.rand(images.size(0), device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_degrees
        radians = angles * math.pi / 180.0
        cos = torch.cos(radians)
        sin = torch.sin(radians)
        theta = images.new_zeros((images.size(0), 2, 3))
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        logs = None
        if collect_metadata:
            logs = [
                {
                    "angle": float(angle),
                    "max_degrees": max_degrees,
                    "fillcolor": int(params.get("fillcolor", self.config.background)),
                }
                for angle in angles.detach().cpu().tolist()
            ]
        return self._warp_affine(images, theta, self._fill_value(params)), logs

    def _projective(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        max_dx = self._sample_range(params, "max_dx", 4.0)
        max_dy = self._sample_range(params, "max_dy", 2.0)
        if max_dx <= 0.0 and max_dy <= 0.0:
            return images, None
        _, _, height, width = images.shape
        tx = ((torch.rand(images.size(0), device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_dx) / max(width, 1)
        ty = ((torch.rand(images.size(0), device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_dy) / max(height, 1)
        shear_x = (torch.rand(images.size(0), device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_dx / max(width, 1)
        shear_y = (torch.rand(images.size(0), device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_dy / max(height, 1)
        theta = images.new_zeros((images.size(0), 2, 3))
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 1] = shear_x
        theta[:, 1, 0] = shear_y
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        logs = None
        if collect_metadata:
            tx_px = (tx * max(width, 1)).detach().cpu().tolist()
            ty_px = (ty * max(height, 1)).detach().cpu().tolist()
            shear_x_px = (shear_x * max(width, 1)).detach().cpu().tolist()
            shear_y_px = (shear_y * max(height, 1)).detach().cpu().tolist()
            logs = [
                {
                    "max_dx": max_dx,
                    "max_dy": max_dy,
                    "tx_px": float(dx),
                    "ty_px": float(dy),
                    "shear_x_px": float(sx),
                    "shear_y_px": float(sy),
                    "fillcolor": int(params.get("fillcolor", self.config.background)),
                }
                for dx, dy, sx, sy in zip(tx_px, ty_px, shear_x_px, shear_y_px)
            ]
        return self._warp_affine(images, theta, self._fill_value(params)), logs

    def _brightness(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        default: float,
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        factor = self._sample_range(params, "factor", default)
        if factor == 1.0:
            return images, None
        logs = self._repeat_log(images, {"factor": factor}) if collect_metadata else None
        return images * factor, logs

    def _contrast(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        factor = self._sample_range(params, "factor", 1.0)
        if factor == 1.0:
            return images, None
        mean = images.mean(dim=(-2, -1), keepdim=True)
        logs = self._repeat_log(images, {"factor": factor}) if collect_metadata else None
        return (images - mean) * factor + mean, logs

    def _noise(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        if params.get("kind", "gaussian") == "salt_pepper":
            amount = self._sample_range(params, "amount", 0.01)
            if amount <= 0.0:
                return images, None
            mask = torch.rand_like(images[:, :1])
            output = images.clone()
            output = torch.where(mask < amount / 2.0, torch.zeros_like(output), output)
            output = torch.where((mask >= amount / 2.0) & (mask < amount), torch.ones_like(output), output)
            logs = self._repeat_log(images, {"kind": "salt_pepper", "amount": amount}) if collect_metadata else None
            return output, logs

        std = self._sample_range(params, "std", self.config.noise_std) / 255.0
        if std <= 0.0:
            return images, None
        logs = self._repeat_log(images, {"kind": "gaussian", "std": std * 255.0}) if collect_metadata else None
        return images + torch.randn_like(images) * std, logs

    def _crop_x(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        max_left = int(round(self._sample_range(params, "left", float(params.get("max_left", 0)))))
        max_right = int(round(self._sample_range(params, "right", float(params.get("max_right", 0)))))
        if max_left <= 0 and max_right <= 0:
            return images, None
        output = images.clone()
        logs: list[AugmentationParams] | None = [] if collect_metadata else None
        _, _, height, width = images.shape
        for index in range(images.size(0)):
            left = self._randint(0, max(0, max_left))
            right = self._randint(0, max(0, max_right))
            if left + right >= width:
                right = max(0, width - left - 1)
            if left + right > 0:
                cropped = images[index : index + 1, :, :, left : width - right]
                output[index : index + 1] = F.interpolate(
                    cropped,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            if logs is not None:
                if left > 0 or right > 0:
                    logs.append({
                        "crop_left": left,
                        "crop_right": right,
                        "resize_to_width": width,
                    })
                else:
                    logs.append(None)
        return output, logs

    def _crop_y(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        max_top = int(round(self._sample_range(params, "top", float(params.get("max_top", 0)))))
        max_bottom = int(round(self._sample_range(params, "bottom", float(params.get("max_bottom", 0)))))
        if max_top <= 0 and max_bottom <= 0:
            return images, None
        output = images.clone()
        logs: list[AugmentationParams] | None = [] if collect_metadata else None
        _, _, height, width = images.shape
        for index in range(images.size(0)):
            top = self._randint(0, max(0, max_top))
            bottom = self._randint(0, max(0, max_bottom))
            if top + bottom >= height:
                bottom = max(0, height - top - 1)
            if top + bottom > 0:
                cropped = images[index : index + 1, :, top : height - bottom, :]
                output[index : index + 1] = F.interpolate(
                    cropped,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            if logs is not None:
                if top > 0 or bottom > 0:
                    logs.append({
                        "crop_top": top,
                        "crop_bottom": bottom,
                        "resize_to_height": height,
                    })
                else:
                    logs.append(None)
        return output, logs

    def _random_line(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        _, _, height, width = images.shape
        if height <= 0 or width <= 0:
            return images, None

        angle_degrees = self._sample_tensor_range(params, "angle_degrees", 0.0, images.size(0), images.device, images.dtype)
        line_width = self._sample_tensor_range(params, "line_width", 1.0, images.size(0), images.device, images.dtype).clamp(min=0.25)
        alpha = self._sample_tensor_range(params, "alpha", 1.0, images.size(0), images.device, images.dtype).clamp(0.0, 1.0)
        value = self._sample_tensor_range(params, "value", 0.0, images.size(0), images.device, images.dtype).clamp(0.0, 255.0) / 255.0
        y_position = self._sample_tensor_range(params, "y", 0.5, images.size(0), images.device, images.dtype).clamp(0.0, 1.0)

        if bool((alpha <= 0.0).all()):
            return images, None

        x_coords = torch.arange(width, device=images.device, dtype=images.dtype).view(1, 1, width)
        y_coords = torch.arange(height, device=images.device, dtype=images.dtype).view(1, height, 1)
        x_center = (width - 1) * 0.5
        y_center = y_position.view(-1, 1, 1) * max(height - 1, 1)
        slope = torch.tan(angle_degrees.view(-1, 1, 1) * math.pi / 180.0)

        distance = torch.abs((y_coords - y_center) - slope * (x_coords - x_center))
        distance = distance / torch.sqrt(1.0 + slope * slope)
        half_width = (line_width.view(-1, 1, 1) * 0.5).clamp(min=0.125)
        mask = (half_width + 0.5 - distance).clamp(0.0, 1.0).unsqueeze(1)

        blend = mask * alpha.view(-1, 1, 1, 1)
        line_value = value.view(-1, 1, 1, 1)
        output = images * (1.0 - blend) + line_value * blend

        logs = None
        if collect_metadata:
            logs = [
                {
                    "angle_degrees": float(sample_angle),
                    "line_width": float(sample_width),
                    "alpha": float(sample_alpha),
                    "value": float(sample_value),
                    "y": float(sample_y),
                }
                for sample_angle, sample_width, sample_alpha, sample_value, sample_y in zip(
                    angle_degrees.detach().cpu().tolist(),
                    line_width.detach().cpu().tolist(),
                    alpha.detach().cpu().tolist(),
                    (value * 255.0).detach().cpu().tolist(),
                    y_position.detach().cpu().tolist(),
                )
            ]
        return output, logs

    def _morphology(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        size = int(round(self._sample_range(params, "size", 3)))
        if size <= 1:
            return images, None
        size = size | 1
        operation = params.get("operation", "random")
        if operation == "random":
            operation = "dilate" if random.random() < 0.5 else "erode"

        logs = self._repeat_log(images, {"operation": operation, "size": size}) if collect_metadata else None
        if operation == "dilate":
            return -F.max_pool2d(-images, size, stride=1, padding=size // 2), logs
        if operation == "erode":
            return F.max_pool2d(images, size, stride=1, padding=size // 2), logs
        if operation == "open":
            eroded = F.max_pool2d(images, size, stride=1, padding=size // 2)
            return -F.max_pool2d(-eroded, size, stride=1, padding=size // 2), logs
        if operation == "close":
            dilated = -F.max_pool2d(-images, size, stride=1, padding=size // 2)
            return F.max_pool2d(dilated, size, stride=1, padding=size // 2), logs
        return images, None

    def _unsharp_mask(
        self,
        images: torch.Tensor,
        params: dict[str, Any],
        collect_metadata: bool,
    ) -> tuple[torch.Tensor, list[AugmentationParams] | None]:
        radius = self._sample_range(params, "radius", 1.0)
        percent = self._sample_range(params, "percent", 120.0)
        if radius <= 0.0 or percent == 0.0:
            return images, None
        blurred, _ = self._gaussian_blur(images, {"radius": radius}, radius, collect_metadata=False)
        logs = self._repeat_log(images, {"radius": radius, "percent": percent}) if collect_metadata else None
        return images + (images - blurred) * (percent / 100.0), logs

    def _warp_affine(self, images: torch.Tensor, theta: torch.Tensor, fill: float) -> torch.Tensor:
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        warped = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        mask = F.grid_sample(torch.ones_like(images[:, :1]), grid, mode="nearest", padding_mode="zeros", align_corners=False)
        return warped * mask + fill * (1.0 - mask)

    def _depthwise_conv(self, images: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
        channels = images.size(1)
        kernel = kernel_2d.to(dtype=images.dtype).expand(channels, 1, -1, -1)
        padding = kernel_2d.size(0) // 2
        padded = F.pad(images, (padding, padding, padding, padding), mode="replicate")
        return F.conv2d(padded, kernel, groups=channels)

    @staticmethod
    def _motion_kernel(device: torch.device, dtype: torch.dtype, size: int, angle: float) -> torch.Tensor:
        kernel = torch.zeros((size, size), dtype=dtype)
        center = (size - 1) / 2.0
        radians = angle * math.pi / 180.0
        dx = math.cos(radians)
        dy = math.sin(radians)
        for step_idx in range(size):
            step = -center + step_idx
            x = int(round(center + step * dx))
            y = int(round(center + step * dy))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        if float(kernel.sum()) == 0.0:
            kernel[size // 2, :] = 1.0
        return (kernel / kernel.sum()).to(device=device)

    @staticmethod
    def _sample_range(params: dict[str, Any], name: str, default: float) -> float:
        if name in params:
            return float(params[name])

        min_name = f"{name}_min"
        max_name = f"{name}_max"
        if min_name in params or max_name in params:
            low = float(params.get(min_name, default))
            high = float(params.get(max_name, default))
            if high < low:
                low, high = high, low
            return random.uniform(low, high)

        return float(default)

    @staticmethod
    def _sample_tensor_range(
        params: dict[str, Any],
        name: str,
        default: float,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if name in params:
            return torch.full((count,), float(params[name]), device=device, dtype=dtype)

        min_name = f"{name}_min"
        max_name = f"{name}_max"
        if min_name in params or max_name in params:
            low = float(params.get(min_name, default))
            high = float(params.get(max_name, default))
            if high < low:
                low, high = high, low
            return torch.empty((count,), device=device, dtype=dtype).uniform_(low, high)

        return torch.full((count,), float(default), device=device, dtype=dtype)

    @staticmethod
    def _randint(low: int, high: int) -> int:
        if high <= low:
            return low
        return random.randint(low, high)

    def _fill_value(self, params: dict[str, Any]) -> float:
        return float(params.get("fillcolor", self.config.background)) / 255.0

    def _repeat_log(self, images: torch.Tensor, params: dict[str, Any]) -> list[AugmentationParams]:
        return [self._jsonable(params) for _ in range(images.size(0))]

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return cls._jsonable(value.detach().cpu().tolist())
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._jsonable(item) for key, item in value.items()}
        return value
