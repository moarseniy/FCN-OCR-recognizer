from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


ARCHITECTURE_NAME = "legacy_fcn"


class LegacyFCN(nn.Module):
    """
    Полносверточный распознаватель строк.
      - in_channels: 1 (grayscale) или 3 (RGB)
      - num_classes: число выходных классов final 1x1 conv
        * CTC: len(alphabet) + 1, последний класс -- blank
        * legacy_logreg: len(alphabet), без blank
      - input height должен быть заранее подобран так, чтобы сеть могла
        свести высоту к 1 (см. stride по высоте в conv1 и т.д.)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=(4, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(8, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=(4, 2), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(12, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=(3, 4), stride=(1, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.conv41 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(24, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.conv42 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(3, 2), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(48, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(72, eps=0.001, momentum=1 - 0.9),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(72, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
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
                "LegacyFCN expects output height 1 before squeezing; "
                f"got output shape {tuple(y.shape)}. Check training image_height."
            )

        return y.squeeze(2)


FullyConvTextRecognizer = LegacyFCN


def create_model(in_channels: int, num_classes: int, **kwargs: Any) -> LegacyFCN:
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise ValueError(f"{ARCHITECTURE_NAME} does not support architecture_params: {unknown}")
    return LegacyFCN(in_channels=in_channels, num_classes=num_classes)
