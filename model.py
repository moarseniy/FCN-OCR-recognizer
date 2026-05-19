from __future__ import annotations

import torch

from fcn_architectures import FullyConvTextRecognizer, LegacyFCN, available_architectures, create_model


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


__all__ = [
    "FullyConvTextRecognizer",
    "LegacyFCN",
    "available_architectures",
    "create_model",
    "decode_greedy_batch_tensor",
]
