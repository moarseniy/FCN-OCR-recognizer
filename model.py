
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def transform_back(logits, W_in):
    """
    logits: (B, C, W_out)
    return: (B, C, W_in)
    """
    B, C, W_out = logits.shape
    device = logits.device

    # inverse mapping: x -> t
    t_idx = torch.floor(
        torch.arange(W_in, device=device) * W_out / W_in
    ).long()

    return logits.index_select(dim=2, index=t_idx)

def decode_greedy_batch_tensor(pred_ids: torch.Tensor):
    """
    pred_ids: (B, W) LongTensor
    returns:
        collapsed: (B, W) LongTensor
        lengths:   (B,)   LongTensor
    """
    B, W = pred_ids.shape
    device = pred_ids.device

    # keep[b, t] = True если t == 0 или pred[b,t] != pred[b,t-1]
    keep = torch.ones((B, W), dtype=torch.bool, device=device)
    keep[:, 1:] = pred_ids[:, 1:] != pred_ids[:, :-1]

    # индексы для scatter
    idx = torch.cumsum(keep, dim=1) - 1   # (B, W), starts from 0

    # длины после collapse
    lengths = keep.sum(dim=1)

    # выходной тензор
    collapsed = torch.zeros_like(pred_ids)

    # scatter без циклов
    collapsed.scatter_(1, idx, pred_ids * keep)

    return collapsed, lengths

class FullyConvTextRecognizer(nn.Module):
    """
    Полносверточный распознаватель строк.
      - in_channels: 1 (grayscale) или 3 (RGB)
      - num_classes: число классов (символов)
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

    def *init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform*(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, H, W)  -- переменная ширина W, фиксированная H
        Возвращает:
          logits: (B, T, C_classes)  -- невыпуклые логиты по временным шагам (T — ширина выхода)
          probs:  (B, T, C_classes)  -- softmax probs (можно не вычислять, если нужен только лосс)
        """

        w_in = x.shape[3]

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

        # ожидаем, что H_out == 1 (после страйдов по высоте)
        # if y.size(2) != 1:
        #     # если H_out != 1 — можно сделать adaptive pooling по высоте, но я предпочитаю предупредить
        #     # для простоты: сведём высоту к 1 через mean по высоте
        #     y = y.mean(dim=2, keepdim=True)

        # y = y.squeeze(2)          # (B, num_classes, W_out)
        # y = y.permute(0, 2, 1)    # (B, T=W_out, num_classes)

        y = y.squeeze(2)
        logits = y
        # probs = F.softmax(logits, dim=1)
        # pred_ids = logits.argmax(dim=1)
        # print("PREDS", pred_ids[0], pred_ids.shape)

        # collapsed, lengths = decode_greedy_batch_tensor(pred_ids)
        # print("GREEDY", collapsed, lengths)

        return logits
