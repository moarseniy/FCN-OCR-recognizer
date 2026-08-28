from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fcn_architectures import create_model, normalize_architecture_name
from fcn_checkpoint_contract import validate_checkpoint_contract


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
    task: str


def load_fcn_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> LoadedFCNCheckpoint:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device)
    model_config = validate_checkpoint_contract(payload)
    alphabet = payload["alphabet"]
    training_config = payload["config"]

    architecture = normalize_architecture_name(model_config.architecture)
    architecture_params = dict(model_config.architecture_params)
    in_channels = model_config.in_channels
    num_classes = model_config.num_classes
    task = model_config.task

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
        task=task,
    )
