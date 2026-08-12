from __future__ import annotations

from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from typing import Any, Callable

import torch.nn as nn


_PACKAGE = "fcn_architectures"
_PACKAGE_DIR = Path(__file__).resolve().parent

def _discover_architectures() -> dict[str, str]:
    modules: dict[str, str] = {}
    for module_info in iter_modules([str(_PACKAGE_DIR)]):
        module_name = module_info.name
        if module_name.startswith("_") or module_name in {"registry"}:
            continue

        module_path = f"{_PACKAGE}.{module_name}"
        module = import_module(module_path)
        if getattr(module, "create_model", None) is None:
            continue

        architecture_name = normalize_architecture_name(module.ARCHITECTURE_NAME)
        modules[architecture_name] = module_path
    return modules


def normalize_architecture_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("architecture must be a non-empty module name")
    return name.strip().lower()


def available_architectures() -> tuple[str, ...]:
    return tuple(sorted(_discover_architectures()))


def _model_factory(name: str) -> Callable[..., nn.Module]:
    normalized = normalize_architecture_name(name)
    module_path = _discover_architectures().get(normalized)
    if module_path is None:
        available = ", ".join(available_architectures())
        raise ValueError(f"Unknown FCN architecture: {name!r}. Available architectures: {available}")

    module = import_module(module_path)
    return module.create_model


def create_model(
    architecture: str,
    in_channels: int,
    num_classes: int,
    **architecture_params: Any,
) -> nn.Module:
    factory = _model_factory(normalize_architecture_name(architecture))
    return factory(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        **dict(architecture_params),
    )
