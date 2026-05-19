from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


ARCHITECTURE_NAME = "vertical_segmentator_fcn"


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int] = (3, 3),
    padding: tuple[int, int] = (1, 1),
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=1 - 0.9),
        nn.ReLU(inplace=True),
    )


class VerticalSegmentatorFCN(nn.Module):
    """
    Width-preserving FCN for binary vertical gap segmentation.

    The network reduces only the vertical dimension. Output timestep T is equal
    to the input image width, so binary gap targets can be trained without
    horizontal resampling or legacy crop offsets.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 16,
        temporal_kernel: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be >= 4")
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 3
        c4 = c1 * 4
        c5 = c1 * 6
        temporal_padding = temporal_kernel // 2

        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, c1),
            _conv_bn_relu(c1, c1),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            _conv_bn_relu(c1, c2),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            _conv_bn_relu(c2, c3),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            _conv_bn_relu(c3, c4),
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            _conv_bn_relu(c4, c4),
        )
        self.height_fuse = _conv_bn_relu(c4, c5)
        self.temporal = nn.Sequential(
            _conv_bn_relu(
                c5,
                c5,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_padding),
            ),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            _conv_bn_relu(
                c5,
                c5,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_padding),
            ),
        )
        self.final = nn.Conv2d(c5, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.down1(y)
        y = self.down2(y)
        y = self.down3(y)
        y = self.down4(y)
        y = self.height_fuse(y)
        y = y.mean(dim=2, keepdim=True)
        y = self.temporal(y)
        y = self.final(y)

        if y.size(2) != 1:
            raise RuntimeError(
                "VerticalSegmentatorFCN expects output height 1 before squeezing; "
                f"got output shape {tuple(y.shape)}. Check training image_height."
            )

        return y.squeeze(2)

    @staticmethod
    def output_width_for_input_width(width: int) -> int:
        return int(width)


def create_model(in_channels: int, num_classes: int, **kwargs: Any) -> VerticalSegmentatorFCN:
    return VerticalSegmentatorFCN(
        in_channels=in_channels,
        num_classes=num_classes,
        **dict(kwargs),
    )
