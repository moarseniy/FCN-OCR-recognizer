
import torch
import torch.nn.functional as F


def ctc_loss(logits, targets, lengths, blank_idx):
    """
    logits:  (B, C, T)
    targets: (B, Lmax), padded with any non-blank value
    lengths: (B,), real target lengths
    """
    log_probs = F.log_softmax(logits, dim=1).permute(2, 0, 1)
    input_lengths = torch.full(
        size=(logits.size(0),),
        fill_value=logits.size(2),
        dtype=torch.long,
        device=logits.device,
    )
    targets = targets.to(device=logits.device, dtype=torch.long)
    target_lengths = lengths.to(device=logits.device, dtype=torch.long)
    target_positions = torch.arange(targets.size(1), device=logits.device).unsqueeze(0)
    flat_targets = targets[target_positions < target_lengths.unsqueeze(1)]
    return F.ctc_loss(
        log_probs,
        flat_targets,
        input_lengths,
        target_lengths,
        blank=blank_idx,
        reduction="mean",
        zero_infinity=True,
    )


def text_targets_to_dense_labels(
    targets: torch.Tensor,
    lengths: torch.Tensor,
    output_width: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Projects sequence targets to per-output-column labels.

    This is a CTC-free approximation for the legacy dense logreg head when the
    dataset only contains text strings, not old-style dense symbol maps.

    targets:      (B, Lmax)
    lengths:      (B,)
    output_width: output T from the model
    returns:      (B, T)
    """
    if targets.dim() != 2:
        raise ValueError(f"text targets must have shape (B, Lmax), got {tuple(targets.shape)}")
    if output_width <= 0:
        raise ValueError("output_width must be positive")

    targets = targets.long()
    lengths = lengths.to(device=targets.device, dtype=torch.long)
    labels = torch.full(
        (targets.size(0), output_width),
        fill_value=ignore_index,
        dtype=torch.long,
        device=targets.device,
    )
    time_positions = torch.arange(output_width, device=targets.device)

    for batch_idx, target_length_tensor in enumerate(lengths):
        target_length = int(target_length_tensor.item())
        if target_length <= 0:
            continue
        if target_length > output_width:
            raise ValueError(
                f"legacy_logreg/uniform_text cannot align target length {target_length} "
                f"to model output width {output_width}"
            )

        char_positions = torch.div(
            time_positions * target_length,
            output_width,
            rounding_mode="floor",
        ).clamp(max=target_length - 1)
        labels[batch_idx] = targets[batch_idx, char_positions]

    return labels


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


def legacy_logreg_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor | None = None,
    target_mode: str = "uniform_text",
    crop_left: int = 6,
    crop_right: int = 5,
    strict_width: bool = False,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    CTC-free dense classification loss, analogous to the old final+softmax+logreg
    branch.

    logits: (B, C, T)

    target_mode:
      - "uniform_text": current text targets are projected uniformly over T.
      - "dense_symbols": targets are old-style symbol maps and are aligned by
        maxpool + cropX([crop_left, -crop_right]).
      - "binary_gaps": targets are 0/1 column labels for vertical gaps.
    """
    target_mode = target_mode.lower()
    if target_mode == "uniform_text":
        if lengths is None:
            raise ValueError("lengths are required for legacy_logreg target_mode=uniform_text")
        labels = text_targets_to_dense_labels(
            targets.to(device=logits.device),
            lengths.to(device=logits.device),
            output_width=logits.size(2),
            ignore_index=ignore_index,
        )
    elif target_mode == "dense_symbols":
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
        raise ValueError("target_mode must be 'uniform_text', 'dense_symbols', or 'binary_gaps'")

    logits, labels = _align_logits_and_labels(logits, labels, strict_width=strict_width)
    return F.cross_entropy(logits, labels.to(device=logits.device), ignore_index=ignore_index)
