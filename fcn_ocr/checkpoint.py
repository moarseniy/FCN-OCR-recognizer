from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fcn_architectures import create_model, normalize_architecture_name


SUPPORTED_LOSS_MODES = frozenset({"fcn_ocr", "cut_projection", "baseline_heatmap"})


@dataclass(frozen=True)
class LoadedFCNCheckpoint:
    path: Path
    payload: dict[str, Any]
    training_config: dict[str, Any]
    model: nn.Module
    alphabet: str
    architecture: str
    architecture_params: dict[str, Any]
    in_channels: int
    num_classes: int
    loss_mode: str
    target_format: str


def load_fcn_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> LoadedFCNCheckpoint:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint must contain a mapping, got {type(payload).__name__}"
        )

    alphabet = payload["alphabet"]
    model_config = payload["model_config"]
    training_config = payload["config"]
    if not isinstance(alphabet, str) or not alphabet:
        raise ValueError("Checkpoint alphabet must be a non-empty string")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint model_config must be a mapping")
    if not isinstance(training_config, dict):
        raise ValueError("Checkpoint config must be a mapping")

    architecture = normalize_architecture_name(model_config["architecture"])
    architecture_params = dict(model_config["architecture_params"])
    in_channels = int(model_config["in_channels"])
    num_classes = int(model_config["num_classes"])
    loss_mode = str(model_config["loss_mode"]).lower()
    target_format = str(model_config["target_format"]).lower()

    if loss_mode not in SUPPORTED_LOSS_MODES:
        raise ValueError(f"Unsupported checkpoint loss_mode: {loss_mode!r}")
    if target_format != loss_mode:
        raise ValueError(
            "Checkpoint target_format must match loss_mode exactly; "
            f"got loss_mode={loss_mode!r}, target_format={target_format!r}"
        )

    expected_classes = {
        "fcn_ocr": len(alphabet),
        "cut_projection": 1,
        "baseline_heatmap": 2,
    }[loss_mode]
    if num_classes != expected_classes:
        raise ValueError(
            f"Checkpoint {loss_mode} expects {expected_classes} output classes, "
            f"got {num_classes}"
        )

    model = create_model(
        architecture,
        in_channels=in_channels,
        num_classes=num_classes,
        **architecture_params,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    return LoadedFCNCheckpoint(
        path=path,
        payload=payload,
        training_config=training_config,
        model=model,
        alphabet=alphabet,
        architecture=architecture,
        architecture_params=architecture_params,
        in_channels=in_channels,
        num_classes=num_classes,
        loss_mode=loss_mode,
        target_format=target_format,
    )
