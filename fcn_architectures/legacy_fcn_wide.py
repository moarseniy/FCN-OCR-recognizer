from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


ARCHITECTURE_NAME = "legacy_fcn_wide"


def _scaled_channels(channels: int, width_multiplier: float) -> int:
    return max(1, int(round(channels * width_multiplier)))


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0),
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=1 - 0.9),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class LegacyFCNWide(nn.Module):
    """
    A drop-in heavier version of legacy_fcn.

    It keeps the exact legacy kernel/stride geometry, so for the same input size
    it has the same output width as legacy_fcn. Only channel counts and optional
    dropout are changed.
    """

    BASE_CHANNELS = (8, 12, 16, 16, 16, 24, 32, 48, 72)

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width_multiplier: float = 1.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if width_multiplier <= 0.0:
            raise ValueError("width_multiplier must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        c0, c1, c2, c3, c4, c41, c42, c5, c6 = [
            _scaled_channels(channels, width_multiplier) for channels in self.BASE_CHANNELS
        ]

        self.conv0 = _conv_bn_relu(in_channels, c0, kernel_size=(4, 3), stride=(1, 1), dropout=dropout)
        self.conv1 = _conv_bn_relu(c0, c1, kernel_size=(4, 2), stride=(2, 1), dropout=dropout)
        self.conv2 = _conv_bn_relu(c1, c2, kernel_size=(3, 4), stride=(1, 2), dropout=dropout)
        self.conv3 = _conv_bn_relu(c2, c3, kernel_size=(3, 3), stride=(1, 1), dropout=dropout)
        self.conv4 = _conv_bn_relu(c3, c4, kernel_size=(3, 3), stride=(2, 1), dropout=dropout)
        self.conv41 = _conv_bn_relu(c4, c41, kernel_size=(3, 3), stride=(1, 1), dropout=dropout)
        self.conv42 = _conv_bn_relu(c41, c42, kernel_size=(3, 3), stride=(1, 1), dropout=dropout)
        self.conv5 = _conv_bn_relu(c42, c5, kernel_size=(3, 2), stride=(1, 1), dropout=dropout)
        self.conv6 = _conv_bn_relu(c5, c6, kernel_size=(2, 2), stride=(1, 1), dropout=dropout)
        self.final = nn.Conv2d(c6, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
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
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv41(y)
        y = self.conv42(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.final(y)

        if y.size(2) != 1:
            raise RuntimeError(
                "LegacyFCNWide expects output height 1 before squeezing; "
                f"got output shape {tuple(y.shape)}. Check training image_height."
            )

        return y.squeeze(2)


def create_model(in_channels: int, num_classes: int, **kwargs: Any) -> LegacyFCNWide:
    return LegacyFCNWide(
        in_channels=in_channels,
        num_classes=num_classes,
        **dict(kwargs),
    )
