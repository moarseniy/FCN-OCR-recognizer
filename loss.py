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


def _align_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    strict_width: bool,
    label_align: str = "majority_bins",
    label_min_majority: float = 0.6,
    ignore_index: int = -100,
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

    label_align = label_align.lower()
    if label_align == "legacy_crop_resample":
        positions = (
            (torch.arange(logits_width, device=labels.device, dtype=torch.float32) + 0.5)
            * float(label_width)
            / float(logits_width)
        ).floor().long().clamp(max=label_width - 1)
        return logits, labels[:, positions]

    if label_align == "majority_bins":
        labels = _align_labels_by_majority_bins(
            labels,
            target_width=logits_width,
            num_classes=logits.size(1),
            min_majority=label_min_majority,
            ignore_index=ignore_index,
        )
        return logits, labels

    raise ValueError("label_align must be 'majority_bins' or 'legacy_crop_resample'")


def _align_labels_by_majority_bins(
    labels: torch.Tensor,
    target_width: int,
    num_classes: int,
    min_majority: float,
    ignore_index: int,
) -> torch.Tensor:
    if not 0.0 <= min_majority <= 1.0:
        raise ValueError("min_majority must be between 0 and 1")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    batch_size, label_width = labels.shape
    aligned = torch.full(
        (batch_size, target_width),
        ignore_index,
        device=labels.device,
        dtype=torch.long,
    )

    for output_x in range(target_width):
        start = (output_x * label_width) // target_width
        end = ((output_x + 1) * label_width + target_width - 1) // target_width
        end = min(label_width, max(start + 1, end))

        segment = labels[:, start:end].long()
        valid = (segment != ignore_index) & (segment >= 0) & (segment < num_classes)
        safe_segment = segment.clamp(min=0, max=num_classes - 1)
        counts = F.one_hot(safe_segment, num_classes=num_classes).to(dtype=torch.float32)
        counts = counts * valid.unsqueeze(-1)
        counts = counts.sum(dim=1)

        top_counts, top_labels = counts.max(dim=1)
        valid_counts = valid.sum(dim=1)
        majority = top_counts / valid_counts.clamp_min(1).to(dtype=torch.float32)
        keep = (valid_counts > 0) & (majority >= min_majority)
        aligned[keep, output_x] = top_labels[keep]

    return aligned


def _weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    space_index: int | None,
    space_weight: float,
) -> torch.Tensor:
    if space_weight <= 0.0:
        raise ValueError("space_weight must be positive")
    if space_index is None or space_weight == 1.0:
        return F.cross_entropy(logits, labels, ignore_index=ignore_index)

    per_position = F.cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    valid = labels != ignore_index
    weights = torch.ones_like(per_position)
    weights = torch.where(labels == space_index, weights * space_weight, weights)
    weights = weights * valid
    denominator = weights.sum().clamp_min(1.0)
    return (per_position * weights).sum() / denominator


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


def baseline_heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    strict_size: bool = True,
    loss: str = "bce",
    positive_weight: float = 4.0,
) -> torch.Tensor:
    """
    Two-channel top/bottom text-line heatmap loss.

    logits:  (B, 2, H, W)
    targets: (B, 2, H, W), values in [0, 1].
    """
    if positive_weight < 1.0:
        raise ValueError("positive_weight must be >= 1.0")
    if logits.dim() != 4:
        raise ValueError(f"baseline_heatmap logits must have shape (B, 2, H, W), got {tuple(logits.shape)}")
    if logits.size(1) != 2:
        raise ValueError(f"baseline_heatmap expects two output channels, got C={logits.size(1)}")
    if targets.dim() != 4 or targets.size(1) != 2:
        raise ValueError(f"baseline_heatmap targets must have shape (B, 2, H, W), got {tuple(targets.shape)}")
    if logits.size(0) != targets.size(0):
        raise ValueError(
            f"batch mismatch between logits {tuple(logits.shape)} and targets {tuple(targets.shape)}"
        )

    targets = targets.to(device=logits.device, dtype=torch.float32).clamp(0.0, 1.0)
    if logits.shape[-2:] != targets.shape[-2:]:
        if strict_size:
            raise ValueError(
                "baseline_heatmap strict_size requires logits and targets to have the same HxW, "
                f"got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
            )
        targets = F.interpolate(targets, size=logits.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    loss = loss.lower()
    if loss == "bce":
        per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    elif loss == "mse":
        per_pixel = F.mse_loss(torch.sigmoid(logits), targets, reduction="none")
    elif loss == "smooth_l1":
        per_pixel = F.smooth_l1_loss(torch.sigmoid(logits), targets, reduction="none")
    else:
        raise ValueError("baseline_heatmap loss must be 'bce', 'mse', or 'smooth_l1'")

    if positive_weight > 1.0:
        weights = 1.0 + (positive_weight - 1.0) * targets
        per_pixel = per_pixel * weights
    return per_pixel.mean()


def legacy_logreg_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor | None = None,
    target_mode: str = "dense_symbols",
    crop_left: int = 6,
    crop_right: int = 5,
    strict_width: bool = False,
    label_align: str = "majority_bins",
    label_min_majority: float = 0.6,
    space_index: int | None = None,
    space_weight: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Dense classification loss, analogous to the old final+softmax+logreg branch.

    logits: (B, C, T)

    target_mode:
      - "dense_symbols": targets are old-style symbol maps and are aligned by
        maxpool + cropX([crop_left, -crop_right]).
    """
    target_mode = target_mode.lower()
    if target_mode == "dense_symbols":
        labels = legacy_dense_symbols_to_labels(
            targets.to(device=logits.device),
            crop_left=crop_left,
            crop_right=crop_right,
        )
    else:
        raise ValueError("target_mode must be 'dense_symbols'")

    logits, labels = _align_logits_and_labels(
        logits,
        labels,
        strict_width=strict_width,
        label_align=label_align,
        label_min_majority=label_min_majority,
        ignore_index=ignore_index,
    )
    labels = labels.to(device=logits.device)
    return _weighted_cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        space_index=space_index,
        space_weight=space_weight,
    )
