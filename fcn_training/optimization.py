from __future__ import annotations

import torch

from .config import TrainingConfig


def create_optimizer(model: torch.nn.Module, config: TrainingConfig):
    parameters = model.parameters()
    if config.optimizer == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        if config.sgd_nesterov and config.sgd_momentum <= 0.0:
            raise ValueError("sgd_nesterov requires sgd_momentum > 0")
        return torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            nesterov=config.sgd_nesterov,
        )
    if config.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=config.lr,
            alpha=config.rmsprop_alpha,
            eps=config.rmsprop_eps,
            weight_decay=config.weight_decay,
            momentum=config.rmsprop_momentum,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def print_optimizer_summary(config: TrainingConfig) -> None:
    print("Optimizer: ", config.optimizer)
    print(f"  lr={config.lr:g} weight_decay={config.weight_decay:g}")
    if config.optimizer in {"adam", "adamw"}:
        print(
            f"  betas=({config.adam_beta1:g}, {config.adam_beta2:g}) "
            f"eps={config.adam_eps:g}"
        )
    elif config.optimizer == "sgd":
        print(f"  momentum={config.sgd_momentum:g} nesterov={config.sgd_nesterov}")
    elif config.optimizer == "rmsprop":
        print(
            f"  alpha={config.rmsprop_alpha:g} momentum={config.rmsprop_momentum:g} "
            f"eps={config.rmsprop_eps:g}"
        )


def create_scheduler(optimizer, config: TrainingConfig):
    if config.scheduler == "none":
        return None
    if config.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            threshold=config.scheduler_threshold,
            cooldown=config.scheduler_cooldown,
            min_lr=config.scheduler_min_lr,
        )
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_t_max or config.epochs,
            eta_min=config.scheduler_eta_min,
        )
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def step_scheduler(
    scheduler, config: TrainingConfig, val_loss: float, optimizer
) -> tuple[float, float]:
    old_lr = current_lr(optimizer)
    if scheduler is None:
        return old_lr, old_lr
    if config.scheduler == "reduce_on_plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()
    return old_lr, current_lr(optimizer)


__all__ = [
    "create_optimizer",
    "create_scheduler",
    "current_lr",
    "print_optimizer_summary",
    "step_scheduler",
]
