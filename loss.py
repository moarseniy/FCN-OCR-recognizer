
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
