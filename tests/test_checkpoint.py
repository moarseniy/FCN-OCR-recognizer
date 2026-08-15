from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fcn_architectures import create_model
from fcn_ocr.checkpoint import load_fcn_checkpoint


ALPHABET = " AB"


def _checkpoint_payload(*, target_format: str = "fcn_ocr") -> dict:
    model = create_model("fcn_ocr", in_channels=1, num_classes=len(ALPHABET))
    return {
        "epoch": 1,
        "loss": 0.5,
        "alphabet": ALPHABET,
        "config": {
            "space_char": " ",
            "image_height": 48,
            "background": 255,
            "ocr_crop_left": 6,
            "ocr_crop_right": 5,
        },
        "model_config": {
            "architecture": "fcn_ocr",
            "architecture_params": {},
            "in_channels": 1,
            "num_classes": len(ALPHABET),
            "loss_mode": "fcn_ocr",
            "target_format": target_format,
        },
        "model_state_dict": model.state_dict(),
    }


def _save_checkpoint(path: Path, *, target_format: str = "fcn_ocr") -> Path:
    torch.save(_checkpoint_payload(target_format=target_format), path)
    return path


def test_load_fcn_checkpoint_builds_the_declared_architecture(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path / "model.pth")

    loaded = load_fcn_checkpoint(checkpoint_path, torch.device("cpu"))

    assert loaded.loss_mode == "fcn_ocr"
    assert loaded.target_format == "fcn_ocr"
    assert loaded.architecture == "fcn_ocr"
    assert loaded.alphabet == ALPHABET
    assert not loaded.model.training


def test_load_fcn_checkpoint_rejects_mismatched_target_format(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(
        tmp_path / "model.pth", target_format="cut_projection"
    )

    with pytest.raises(ValueError, match="target_format must match loss_mode"):
        load_fcn_checkpoint(checkpoint_path, torch.device("cpu"))
