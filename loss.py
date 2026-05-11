
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

def logreg_loss(logits, targets):
    """
    logits:  (B, T, C)
    targets: (B, Lmax)
    """
    return F.cross_entropy(logits, targets)

def simple_logreg_loss(logits, targets, lengths):
    """
    logits:  (B, T, C)
    targets: (B, Lmax)
    lengths: (B,)
    """
    print(logits.shape, targets.shape)
    B, T, C = logits.shape
    device = logits.device

    t_idx = torch.arange(T, device=device).unsqueeze(0)
    mask = t_idx < lengths.unsqueeze(1)

    logits_flat = logits[mask]

    target_flat = torch.cat(
        [targets[i, :lengths[i]] for i in range(B)],
        dim=0
    )

    return F.cross_entropy(logits_flat, target_flat)

def hard_logreg_loss(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Простая реализация 'logreg' — сумма отрицательных логарифмов правдоподобия
    по первым lengths[t] временным шагам для каждого примера.
    logits: (B, T, C)
    targets: (B, L_max)  -- int64
    lengths: (B,)        -- int64, <= T
    """
    B, T, C = logits.shape
    device = logits.device
    lengths = lengths.to(device)
    # сформируем маску (B, T)
    idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    mask = idxs < lengths.unsqueeze(1)  # True для позиций, по которым учитываем лосс

    # flatten по валидным позициям
    logits_flat = logits[mask]            # (sum(lengths), C)
    # подготовим targets_flat: для каждого i берем первые lengths[i] элементов
    targets_flat = []
    for i in range(B):
        li = lengths[i].item()
        if li > 0:
            targets_flat.append(targets[i, :li])
    if len(targets_flat) == 0:
        return torch.tensor(0., device=device, requires_grad=True)
    targets_flat = torch.cat(targets_flat, dim=0)  # (sum(lengths),)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
    return loss
