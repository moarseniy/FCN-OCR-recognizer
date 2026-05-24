from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


ARCHITECTURE_NAME = "residual_temporal_fcn"


def _conv_norm(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=1 - 0.9),
    )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        dilation: tuple[int, int] = (1, 1),
        dropout: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        padding = (
            dilation[0] * (kernel_size[0] // 2),
            dilation[1] * (kernel_size[1] // 2),
        )
        self.conv1 = _conv_norm(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = _conv_norm(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.projection = (
            _conv_norm(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        y = self.activation(self.conv1(x))
        y = self.dropout(y)
        y = self.conv2(y)
        return self.activation(y + residual)


class ResidualTemporalFCN(nn.Module):
    """
    Width-preserving FCN with residual blocks and temporal context.

    The network downsamples only the vertical dimension, averages height to one
    row, then applies dilated temporal convolutions along X. It can be used for
    OCR (`num_classes=len(alphabet)`) or for cut projection (`num_classes=1`).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 24,
        height_levels: int = 4,
        temporal_blocks: int = 4,
        temporal_kernel: int = 5,
        temporal_dilations: list[int] | tuple[int, ...] | None = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels must be >= 4")
        if height_levels < 1:
            raise ValueError("height_levels must be >= 1")
        if temporal_blocks < 1:
            raise ValueError("temporal_blocks must be >= 1")
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        dilations = tuple(int(value) for value in (temporal_dilations or (1, 2, 4, 1)))
        if not dilations or any(value < 1 for value in dilations):
            raise ValueError("temporal_dilations must contain positive integers")

        self.stem = nn.Sequential(
            ResidualBlock(in_channels, base_channels, dropout=dropout),
            ResidualBlock(base_channels, base_channels, dropout=dropout),
        )

        levels: list[nn.Module] = []
        channels = base_channels
        for level in range(height_levels):
            next_channels = base_channels * min(level + 2, 6)
            levels.extend(
                [
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                    ResidualBlock(channels, next_channels, dropout=dropout),
                ]
            )
            channels = next_channels
        self.height_encoder = nn.Sequential(*levels)

        temporal_channels = channels
        temporal_layers: list[nn.Module] = []
        for block_index in range(temporal_blocks):
            dilation = dilations[block_index % len(dilations)]
            temporal_layers.append(
                ResidualBlock(
                    temporal_channels,
                    temporal_channels,
                    kernel_size=(1, temporal_kernel),
                    dilation=(1, dilation),
                    dropout=dropout,
                )
            )
        self.temporal = nn.Sequential(*temporal_layers)
        self.final = nn.Conv2d(temporal_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
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
        y = self.height_encoder(y)
        y = y.mean(dim=2, keepdim=True)
        y = self.temporal(y)
        y = self.final(y)

        if y.size(2) != 1:
            raise RuntimeError(
                "ResidualTemporalFCN expects output height 1 before squeezing; "
                f"got output shape {tuple(y.shape)}."
            )

        return y.squeeze(2)

    @staticmethod
    def output_width_for_input_width(width: int) -> int:
        return int(width)


def create_model(in_channels: int, num_classes: int, **kwargs: Any) -> ResidualTemporalFCN:
    return ResidualTemporalFCN(
        in_channels=in_channels,
        num_classes=num_classes,
        **dict(kwargs),
    )
