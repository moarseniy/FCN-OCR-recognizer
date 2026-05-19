
import torch
import torch.nn as nn

def decode_greedy_batch_tensor(pred_ids: torch.Tensor):
    """
    pred_ids: (B, W) LongTensor
    returns:
        collapsed: (B, W) LongTensor
        lengths:   (B,)   LongTensor
    """
    collapsed = torch.zeros_like(pred_ids)
    lengths = torch.zeros(pred_ids.size(0), dtype=torch.long, device=pred_ids.device)

    for batch_idx, row in enumerate(pred_ids):
        if row.numel() == 0:
            continue

        keep = torch.ones_like(row, dtype=torch.bool)
        keep[1:] = row[1:] != row[:-1]
        values = row[keep]
        collapsed[batch_idx, : values.numel()] = values
        lengths[batch_idx] = values.numel()

    return collapsed, lengths

class FullyConvTextRecognizer(nn.Module):
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
        # conv0: f=[4,3], s=[1,1], n=8, bn
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=(4,3), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(8, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # conv1: f=[4,2], s=[2,1], n=12, bn
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=(4,2), stride=(2,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(12, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # conv2: f=[3,4], s=[1,2], n=16, bn
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=(3,4), stride=(1,2), padding=(0,0), bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # conv3: f=[3,3], s=[1,1], n=16, bn
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # conv4: f=[3,3], s=[2,1], n=16, bn
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3,3), stride=(2,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # conv41: f=[3,3], s=[1,1], n=24, bn
        self.conv41 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=(3,3), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(24, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # conv42: f=[3,3], s=[1,1], n=32, bn
        self.conv42 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=(3,3), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # conv5: f=[3,2], s=[1,1], n=48, bn
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=(3,2), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(48, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # conv6: f=[2,2], s=[1,1], n=72, bn
        self.conv6 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=(2,2), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(72, eps=0.001, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        # final: f=[1,1], s=[1,1], n=num_classes (проекция в число классов)
        self.final = nn.Conv2d(72, num_classes, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=True)

        # инициализация весов: небольшая, стандартная Xavier
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  -- переменная ширина W, фиксированная H
        Возвращает:
          logits: (B, C_classes, T)  -- логиты по временным шагам (T — ширина выхода)
        """

        # последовательные проходы
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv41(y)
        y = self.conv42(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.final(y)  # shape (B, num_classes, H_out, W_out)

        if y.size(2) != 1:
            raise RuntimeError(
                "FullyConvTextRecognizer expects output height 1 before squeezing; "
                f"got output shape {tuple(y.shape)}. Check training image_height."
            )

        y = y.squeeze(2)
        logits = y
        return logits
