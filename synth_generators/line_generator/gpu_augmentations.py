from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .dataset import SUPPORTED_AUGMENTATIONS, SingleLineDatasetConfig


class GpuTextAugmenter:
    """Torch-only OCR augmentations for BCHW tensors in [0, 1]."""

    def __init__(self, config: SingleLineDatasetConfig):
        self.config = config
        self.probabilities = self._effective_probabilities(config)
        self.params = config.augmentations

    def enabled(self) -> bool:
        return any(probability > 0.0 for probability in self.probabilities.values())

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if not self.enabled() or images.numel() == 0:
            return images

        output = images
        for name in SUPPORTED_AUGMENTATIONS:
            probability = self.probabilities.get(name, 0.0)
            if probability <= 0.0:
                continue

            mask = torch.rand(output.size(0), device=output.device) <= probability
            if not bool(mask.any()):
                continue

            augmented = self._apply_one(name, output, self.params.get(name, {}))
            output = torch.where(mask[:, None, None, None], augmented, output)

        return output.clamp(0.0, 1.0)

    def _apply_one(self, name: str, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        if name == "cycle_shift":
            return self._cycle_shift(images, params)
        if name in {"strong_blur", "gaussian_blur"}:
            default = 1.2 if name == "strong_blur" else self.config.blur_radius
            return self._gaussian_blur(images, params, default)
        if name == "motion_blur":
            return self._motion_blur(images, params)
        if name == "scale":
            return self._scale(images, params)
        if name in {"darkening", "brightness"}:
            default = 0.75 if name == "darkening" else 1.0
            return self._brightness(images, params, default)
        if name in {"noise", "gaussian_noise"}:
            return self._noise(images, params)
        if name == "projective":
            return self._projective(images, params)
        if name == "rotate":
            return self._rotate(images, params)
        if name == "crop_x":
            return self._crop_x(images, params)
        if name == "crop_y":
            return self._crop_y(images, params)
        if name == "morphology":
            return self._morphology(images, params)
        if name == "unsharp_mask":
            return self._unsharp_mask(images, params)
        if name == "contrast":
            return self._contrast(images, params)
        if name == "invert":
            return 1.0 - images
        return images

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

    def _cycle_shift(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        max_x = int(params.get("max_x", 0))
        max_y = int(params.get("max_y", 0))
        if max_x <= 0 and max_y <= 0:
            return images

        output = images.clone()
        for index in range(images.size(0)):
            shift_x = self._randint(images.device, -max_x, max_x) if max_x > 0 else 0
            shift_y = self._randint(images.device, -max_y, max_y) if max_y > 0 else 0
            output[index] = torch.roll(images[index], shifts=(shift_y, shift_x), dims=(-2, -1))
        return output

    def _gaussian_blur(self, images: torch.Tensor, params: dict[str, Any], default_radius: float) -> torch.Tensor:
        radius = self._sample_range(images.device, params, "radius", default_radius)
        if radius <= 0.0:
            return images

        sigma = max(float(radius), 0.05)
        size = max(3, int(math.ceil(sigma * 6.0)) | 1)
        coords = torch.arange(size, device=images.device, dtype=images.dtype) - size // 2
        kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return self._depthwise_conv(images, kernel_2d)

    def _motion_blur(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        size = int(round(self._sample_range(images.device, params, "size", 5)))
        if size <= 1:
            return images
        size = size | 1
        angle = self._sample_range(images.device, params, "angle", 0.0)
        kernel = self._motion_kernel(images.device, images.dtype, size, angle)
        return self._depthwise_conv(images, kernel)

    def _scale(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        factor_default = self._sample_range(images.device, params, "factor", 1.0)
        factor_x = self._sample_range(images.device, params, "factor_x", factor_default)
        factor_y = self._sample_range(images.device, params, "factor_y", factor_default)
        if factor_x <= 0.0 or factor_y <= 0.0:
            return images
        theta = images.new_zeros((images.size(0), 2, 3))
        theta[:, 0, 0] = 1.0 / factor_x
        theta[:, 1, 1] = 1.0 / factor_y
        return self._warp_affine(images, theta, self._fill_value(params))

    def _rotate(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        max_degrees = float(params.get("max_degrees", self.config.max_rotation_degrees))
        if max_degrees <= 0.0:
            return images
        angles = (torch.rand(images.size(0), device=images.device, dtype=images.dtype) * 2.0 - 1.0) * max_degrees
        radians = angles * math.pi / 180.0
        cos = torch.cos(radians)
        sin = torch.sin(radians)
        theta = images.new_zeros((images.size(0), 2, 3))
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        return self._warp_affine(images, theta, self._fill_value(params))

    def _projective(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        max_dx = self._sample_range(images.device, params, "max_dx", 4.0)
        max_dy = self._sample_range(images.device, params, "max_dy", 2.0)
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
        return self._warp_affine(images, theta, self._fill_value(params))

    def _brightness(self, images: torch.Tensor, params: dict[str, Any], default: float) -> torch.Tensor:
        factor = self._sample_range(images.device, params, "factor", default)
        return images * factor

    def _contrast(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        factor = self._sample_range(images.device, params, "factor", 1.0)
        mean = images.mean(dim=(-2, -1), keepdim=True)
        return (images - mean) * factor + mean

    def _noise(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        if params.get("kind", "gaussian") == "salt_pepper":
            amount = self._sample_range(images.device, params, "amount", 0.01)
            mask = torch.rand_like(images[:, :1])
            output = images.clone()
            output = torch.where(mask < amount / 2.0, torch.zeros_like(output), output)
            output = torch.where((mask >= amount / 2.0) & (mask < amount), torch.ones_like(output), output)
            return output

        std = self._sample_range(images.device, params, "std", self.config.noise_std) / 255.0
        if std <= 0.0:
            return images
        return images + torch.randn_like(images) * std

    def _crop_x(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        max_left = int(round(self._sample_range(images.device, params, "left", float(params.get("max_left", 0)))))
        max_right = int(round(self._sample_range(images.device, params, "right", float(params.get("max_right", 0)))))
        if max_left <= 0 and max_right <= 0:
            return images
        output = images.clone()
        fill = self._fill_value(params)
        for index in range(images.size(0)):
            left = self._randint(images.device, 0, max(0, max_left))
            right = self._randint(images.device, 0, max(0, max_right))
            if left > 0:
                output[index, :, :, :left] = fill
            if right > 0:
                output[index, :, :, images.size(-1) - right :] = fill
        return output

    def _crop_y(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        max_top = int(round(self._sample_range(images.device, params, "top", float(params.get("max_top", 0)))))
        max_bottom = int(round(self._sample_range(images.device, params, "bottom", float(params.get("max_bottom", 0)))))
        if max_top <= 0 and max_bottom <= 0:
            return images
        output = images.clone()
        fill = self._fill_value(params)
        for index in range(images.size(0)):
            top = self._randint(images.device, 0, max(0, max_top))
            bottom = self._randint(images.device, 0, max(0, max_bottom))
            if top > 0:
                output[index, :, :top, :] = fill
            if bottom > 0:
                output[index, :, images.size(-2) - bottom :, :] = fill
        return output

    def _morphology(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        size = int(round(self._sample_range(images.device, params, "size", 3)))
        if size <= 1:
            return images
        size = size | 1
        operation = params.get("operation", "random")
        if operation == "random":
            operation = "dilate" if bool(torch.rand((), device=images.device) < 0.5) else "erode"

        if operation == "dilate":
            return -F.max_pool2d(-images, size, stride=1, padding=size // 2)
        if operation == "erode":
            return F.max_pool2d(images, size, stride=1, padding=size // 2)
        if operation == "open":
            return -F.max_pool2d(-F.max_pool2d(images, size, stride=1, padding=size // 2), size, stride=1, padding=size // 2)
        if operation == "close":
            return F.max_pool2d(-F.max_pool2d(-images, size, stride=1, padding=size // 2), size, stride=1, padding=size // 2)
        return images

    def _unsharp_mask(self, images: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        radius = self._sample_range(images.device, params, "radius", 1.0)
        percent = self._sample_range(images.device, params, "percent", 120.0)
        blurred = self._gaussian_blur(images, {"radius": radius}, radius)
        return images + (images - blurred) * (percent / 100.0)

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
        kernel = torch.zeros((size, size), device=device, dtype=dtype)
        center = (size - 1) / 2.0
        radians = angle * math.pi / 180.0
        dx = math.cos(radians)
        dy = math.sin(radians)
        for step in torch.linspace(-center, center, size, device=device):
            x = int(round(center + float(step) * dx))
            y = int(round(center + float(step) * dy))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        if float(kernel.sum()) == 0.0:
            kernel[size // 2, :] = 1.0
        return kernel / kernel.sum()

    @staticmethod
    def _sample_range(device: torch.device, params: dict[str, Any], name: str, default: float) -> float:
        if name in params:
            return float(params[name])

        min_name = f"{name}_min"
        max_name = f"{name}_max"
        if min_name in params or max_name in params:
            low = float(params.get(min_name, default))
            high = float(params.get(max_name, default))
            if high < low:
                low, high = high, low
            sample = torch.rand((), device=device).item()
            return low + (high - low) * sample

        return float(default)

    @staticmethod
    def _randint(device: torch.device, low: int, high: int) -> int:
        if high <= low:
            return low
        return int(torch.randint(low, high + 1, (), device=device).item())

    def _fill_value(self, params: dict[str, Any]) -> float:
        return float(params.get("fillcolor", self.config.background)) / 255.0
