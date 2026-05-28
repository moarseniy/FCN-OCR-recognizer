from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


ARCHITECTURE_NAME = "baseline_detector_fcn"


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    dilation: int = 1,
) -> nn.Sequential:
    padding = dilation * (kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=1 - 0.9),
        nn.ReLU(inplace=True),
    )


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            _conv_bn_relu(channels, channels, dilation=dilation),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels, eps=0.001, momentum=1 - 0.9),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class BaselineDetectorFCN(nn.Module):
    """
    Width/height-preserving FCN for top and bottom text-line heatmaps.

    Output shape is (B, 2, H, W): channel 0 predicts the upper text line,
    channel 1 predicts the lower text line.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 24,
        depth: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("BaselineDetectorFCN expects num_classes=2")
        if base_channels < 8:
            raise ValueError("base_channels must be >= 8")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        dilations = [1, 2, 4, 8]
        blocks = [
            ResidualDilatedBlock(
                int(base_channels),
                dilation=dilations[index % len(dilations)],
                dropout=float(dropout),
            )
            for index in range(int(depth))
        ]
        self.net = nn.Sequential(
            _conv_bn_relu(int(in_channels), int(base_channels), dilation=1),
            _conv_bn_relu(int(base_channels), int(base_channels), dilation=1),
            *blocks,
            _conv_bn_relu(int(base_channels), int(base_channels), kernel_size=1, dilation=1),
        )
        self.final = nn.Conv2d(int(base_channels), 2, kernel_size=1, stride=1, padding=0)
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
        return self.final(self.net(x))

    @staticmethod
    def output_width_for_input_width(width: int) -> int:
        return int(width)


def create_model(in_channels: int, num_classes: int, **kwargs: Any) -> BaselineDetectorFCN:
    return BaselineDetectorFCN(
        in_channels=in_channels,
        num_classes=num_classes,
        **dict(kwargs),
    )
