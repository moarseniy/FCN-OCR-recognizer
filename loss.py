import torch
import torch.nn.functional as F


def legacy_dense_symbols_to_labels(
    symbols: torch.Tensor,
    crop_left: int = 6,
    crop_right: int = 5,
) -> torch.Tensor:
    """
    Recreates the label side of the old graph:

      syms -> max_pool2d(kernel=(4, 1), stride=(4, 1), padding=(1, 0))
           -> cropX([crop_left, -crop_right])

    Expected input is a dense symbol target with class indices:
      - (B, W) already reduced over height
      - (B, 1, H, W) or (B, H, W)

    The returned tensor has shape (B, T_labels).
    """
    if crop_left < 0 or crop_right < 0:
        raise ValueError("crop_left and crop_right must be non-negative")

    if symbols.dim() == 2:
        labels = symbols.long()
    else:
        if symbols.dim() == 3:
            symbols = symbols.unsqueeze(1)
        if symbols.dim() != 4:
            raise ValueError(
                "dense symbol targets must have shape (B, W), (B, H, W), or (B, 1, H, W), "
                f"got {tuple(symbols.shape)}"
            )
        if symbols.size(1) != 1:
            raise ValueError(
                "dense symbol targets must contain one channel with class indices; "
                f"got {symbols.size(1)} channels"
            )

        pooled = F.max_pool2d(
            symbols.float(),
            kernel_size=(4, 1),
            stride=(4, 1),
            padding=(1, 0),
        )
        labels = pooled.squeeze(1)
        if labels.dim() == 3:
            labels = labels.max(dim=1).values

    width = labels.size(1)
    right = width - crop_right if crop_right else width
    if crop_left >= right:
        raise ValueError(
            f"legacy symbol crop [{crop_left}, -{crop_right}] is empty for label width {width}"
        )
    return labels[:, crop_left:right].long()


def legacy_binary_gaps_to_labels(
    targets: torch.Tensor,
    crop_left: int = 6,
    crop_right: int = 5,
) -> torch.Tensor:
    """
    Binary vertical-segmentation targets.

    Expected input:
      - (B, W), values 0 for non-gap and 1 for gap columns.
    """
    if crop_left < 0 or crop_right < 0:
        raise ValueError("crop_left and crop_right must be non-negative")
    if targets.dim() != 2:
        raise ValueError(f"binary gap targets must have shape (B, W), got {tuple(targets.shape)}")

    labels = targets.long().clamp(0, 1)
    width = labels.size(1)
    right = width - crop_right if crop_right else width
    if crop_left >= right:
        raise ValueError(f"binary gap crop [{crop_left}, -{crop_right}] is empty for label width {width}")
    return labels[:, crop_left:right]


def _align_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    strict_width: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.dim() != 2:
        raise ValueError(f"labels must have shape (B, T), got {tuple(labels.shape)}")
    if logits.size(0) != labels.size(0):
        raise ValueError(
            f"batch mismatch between logits {tuple(logits.shape)} and labels {tuple(labels.shape)}"
        )

    logits_width = logits.size(2)
    label_width = labels.size(1)
    if logits_width == label_width:
        return logits, labels

    if strict_width:
        raise ValueError(
            f"legacy_logreg width mismatch: logits T={logits_width}, labels T={label_width}"
        )

    if label_width <= 0 or logits_width <= 0:
        raise ValueError(f"legacy_logreg got empty width: logits T={logits_width}, labels T={label_width}")

    positions = (
        (torch.arange(logits_width, device=labels.device, dtype=torch.float32) + 0.5)
        * float(label_width)
        / float(logits_width)
    ).floor().long().clamp(max=label_width - 1)
    return logits, labels[:, positions]


def _crop_projection_targets(
    targets: torch.Tensor,
    crop_left: int = 0,
    crop_right: int = 0,
) -> torch.Tensor:
    if crop_left < 0 or crop_right < 0:
        raise ValueError("crop_left and crop_right must be non-negative")
    if targets.dim() != 2:
        raise ValueError(f"cut projection targets must have shape (B, W), got {tuple(targets.shape)}")

    width = targets.size(1)
    right = width - crop_right if crop_right else width
    if crop_left >= right:
        raise ValueError(f"cut projection crop [{crop_left}, -{crop_right}] is empty for target width {width}")
    return targets[:, crop_left:right].float().clamp(0.0, 1.0)


def _align_logits_and_projection(
    logits: torch.Tensor,
    targets: torch.Tensor,
    strict_width: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() != 3:
        raise ValueError(f"logits must have shape (B, C, T), got {tuple(logits.shape)}")
    if logits.size(1) != 1:
        raise ValueError(f"cut_projection loss expects one output channel, got C={logits.size(1)}")
    if targets.dim() != 2:
        raise ValueError(f"cut projection targets must have shape (B, W), got {tuple(targets.shape)}")
    if logits.size(0) != targets.size(0):
        raise ValueError(
            f"batch mismatch between logits {tuple(logits.shape)} and targets {tuple(targets.shape)}"
        )

    logits_1d = logits[:, 0, :]
    logits_width = logits_1d.size(1)
    target_width = targets.size(1)
    if logits_width == target_width:
        return logits_1d, targets

    if strict_width:
        raise ValueError(
            f"cut_projection width mismatch: logits T={logits_width}, targets W={target_width}"
        )

    if logits_width <= 0 or target_width <= 0:
        raise ValueError(f"cut_projection got empty width: logits T={logits_width}, targets W={target_width}")

    resized_targets = F.interpolate(
        targets.unsqueeze(1),
        size=logits_width,
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    return logits_1d, resized_targets


def cut_projection_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    crop_left: int = 0,
    crop_right: int = 0,
    strict_width: bool = True,
    loss: str = "mse",
    positive_weight: float = 1.0,
) -> torch.Tensor:
    """
    Regression/classification loss for a vertical cut projection.

    logits:  (B, 1, T)
    targets: (B, W), values in [0, 1], where peaks mark correct cut columns.
    """
    if positive_weight < 1.0:
        raise ValueError("positive_weight must be >= 1.0")

    targets = _crop_projection_targets(
        targets.to(device=logits.device),
        crop_left=crop_left,
        crop_right=crop_right,
    )
    logits_1d, targets = _align_logits_and_projection(logits, targets, strict_width=strict_width)
    loss = loss.lower()

    if loss == "mse":
        prediction = torch.sigmoid(logits_1d)
        per_column = F.mse_loss(prediction, targets, reduction="none")
    elif loss == "smooth_l1":
        prediction = torch.sigmoid(logits_1d)
        per_column = F.smooth_l1_loss(prediction, targets, reduction="none")
    elif loss == "bce":
        per_column = F.binary_cross_entropy_with_logits(logits_1d, targets, reduction="none")
    else:
        raise ValueError("cut projection loss must be 'mse', 'smooth_l1', or 'bce'")

    if positive_weight > 1.0:
        weights = torch.ones_like(targets)
        weights = weights + (positive_weight - 1.0) * targets
        per_column = per_column * weights
    return per_column.mean()


def legacy_logreg_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor | None = None,
    target_mode: str = "dense_symbols",
    crop_left: int = 6,
    crop_right: int = 5,
    strict_width: bool = False,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Dense classification loss, analogous to the old final+softmax+logreg branch.

    logits: (B, C, T)

    target_mode:
      - "dense_symbols": targets are old-style symbol maps and are aligned by
        maxpool + cropX([crop_left, -crop_right]).
      - "binary_gaps": targets are 0/1 column labels for vertical gaps.
    """
    target_mode = target_mode.lower()
    if target_mode == "dense_symbols":
        labels = legacy_dense_symbols_to_labels(
            targets.to(device=logits.device),
            crop_left=crop_left,
            crop_right=crop_right,
        )
    elif target_mode == "binary_gaps":
        labels = legacy_binary_gaps_to_labels(
            targets.to(device=logits.device),
            crop_left=crop_left,
            crop_right=crop_right,
        )
    else:
        raise ValueError("target_mode must be 'dense_symbols' or 'binary_gaps'")

    logits, labels = _align_logits_and_labels(logits, labels, strict_width=strict_width)
    return F.cross_entropy(logits, labels.to(device=logits.device), ignore_index=ignore_index)
