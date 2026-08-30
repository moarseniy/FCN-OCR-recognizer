from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fcn_architectures import create_model
from fcn_checkpoint_contract import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from fcn_ocr.baseline_detector import BaselineDetector
from fcn_ocr.checkpoint import load_fcn_checkpoint
from fcn_ocr.recognizer import TextRecognizer
from fcn_ocr.vertical_segmenter import VerticalSegmenter


ALPHABET = " AB"


def _checkpoint_payload() -> dict:
    model = create_model("fcn_ocr", in_channels=1, num_classes=len(ALPHABET))
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "epoch": 1,
        "loss": 0.5,
        "alphabet": ALPHABET,
        "config": {
            "space_char": " ",
            "image_height": 48,
            "background": 255,
            "fcn_ocr_crop_left": 6,
            "fcn_ocr_crop_right": 5,
        },
        "model_config": {
            "architecture": "fcn_ocr",
            "architecture_params": {},
            "in_channels": 1,
            "num_classes": len(ALPHABET),
            "task": "fcn_ocr",
        },
        "model_state_dict": model.state_dict(),
    }


def _save_checkpoint(path: Path) -> Path:
    torch.save(_checkpoint_payload(), path)
    return path


def test_load_fcn_checkpoint_builds_the_declared_architecture(tmp_path: Path) -> None:
    checkpoint_path = _save_checkpoint(tmp_path / "model.pth")

    loaded = load_fcn_checkpoint(checkpoint_path, torch.device("cpu"))

    assert loaded.task == "fcn_ocr"
    assert loaded.architecture == "fcn_ocr"
    assert loaded.alphabet == ALPHABET
    assert not loaded.model.training


def test_load_fcn_checkpoint_rejects_removed_model_config_fields(
    tmp_path: Path,
) -> None:
    payload = _checkpoint_payload()
    payload["model_config"]["loss_mode"] = "fcn_ocr"
    payload["model_config"]["target_format"] = "fcn_ocr"
    checkpoint_path = tmp_path / "model.pth"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_fcn_checkpoint(checkpoint_path, torch.device("cpu"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", None, "Checkpoint format"),
        ("version", 0, "Checkpoint version"),
    ],
)
def test_load_fcn_checkpoint_requires_current_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _checkpoint_payload()
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    checkpoint_path = tmp_path / f"invalid_{field}.pth"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        load_fcn_checkpoint(checkpoint_path, torch.device("cpu"))


def test_vertical_segmentation_checkpoint_does_not_require_ocr_task_fields(
    tmp_path: Path,
) -> None:
    architecture = "vertical_segmentation_fcn"
    model = create_model(architecture, in_channels=1, num_classes=1)
    checkpoint_path = tmp_path / "vertical_segmentation.pth"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "epoch": 1,
            "loss": 0.5,
            "alphabet": ALPHABET,
            "config": {
                "space_char": " ",
                "image_height": 48,
                "background": 255,
            },
            "model_config": {
                "architecture": architecture,
                "architecture_params": {},
                "in_channels": 1,
                "num_classes": 1,
                "task": "vertical_segmentation",
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    vertical_segmentation = VerticalSegmenter(checkpoint_path, device="cpu")

    assert vertical_segmentation.task == "vertical_segmentation"
    assert not isinstance(vertical_segmentation, TextRecognizer)
    assert not hasattr(vertical_segmentation, "decode_predictions")


def test_baseline_detector_is_not_an_ocr_recognizer(tmp_path: Path) -> None:
    architecture = "baseline_detection_fcn"
    model = create_model(architecture, in_channels=1, num_classes=2)
    checkpoint_path = tmp_path / "baseline_detector.pth"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "epoch": 1,
            "loss": 0.5,
            "alphabet": ALPHABET,
            "config": {
                "space_char": " ",
                "image_height": 48,
                "background": 255,
            },
            "model_config": {
                "architecture": architecture,
                "architecture_params": {},
                "in_channels": 1,
                "num_classes": 2,
                "task": "baseline_detection",
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    detector = BaselineDetector(checkpoint_path, device="cpu")

    assert detector.task == "baseline_detection"
    assert not isinstance(detector, TextRecognizer)
    assert not hasattr(detector, "decode_predictions")
