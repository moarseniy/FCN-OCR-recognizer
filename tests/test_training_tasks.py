from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from fcn_synth_generator.dataset import SingleLineDatasetConfig
from fcn_tasks import TASK_NAMES, task_output_channels
from fcn_training import (
    TrainingConfig,
    available_training_tasks,
    effective_training_config_data,
    get_training_task,
)


def _dataset_config() -> SingleLineDatasetConfig:
    return SingleLineDatasetConfig(
        task="fcn_ocr",
        alphabet=" AB",
        image_height=16,
        image_width=32,
        channels=1,
    )


def _training_config(task_name: str, **overrides) -> TrainingConfig:
    values = {
        "architecture": "fcn_ocr",
        "chunks_dir": "/unused",
        "task": task_name,
    }
    values.update(overrides)
    return TrainingConfig.model_validate(values)


def test_registry_exposes_exactly_the_current_training_tasks() -> None:
    assert available_training_tasks() == TASK_NAMES
    assert get_training_task("FCN_OCR").name == "fcn_ocr"
    for removed_name in ("cut_projection", "baseline_heatmap", "removed_task"):
        with pytest.raises(ValueError, match="Unknown training task"):
            get_training_task(removed_name)


@pytest.mark.parametrize(
    ("task_name", "expected_outputs"),
    [
        ("fcn_ocr", 3),
        ("vertical_segmentation", 1),
        ("baseline_detection", 2),
    ],
)
def test_task_owns_output_count(
    task_name: str,
    expected_outputs: int,
) -> None:
    task = get_training_task(task_name)

    assert task.num_outputs(" AB") == expected_outputs
    assert task.num_outputs(" AB") == task_output_channels(task_name, " AB")


@pytest.mark.parametrize("task_name", available_training_tasks())
def test_task_computes_a_finite_differentiable_loss(task_name: str) -> None:
    dataset_config = _dataset_config()
    task = get_training_task(task_name)
    overrides = (
        {"fcn_ocr_crop_left": 0, "fcn_ocr_crop_right": 0}
        if task_name == "fcn_ocr"
        else {}
    )
    config = _training_config(task_name, **overrides)

    if task_name == "fcn_ocr":
        logits = torch.randn(2, 3, 8, requires_grad=True)
        targets = torch.randint(0, 3, (2, 32))
    elif task_name == "vertical_segmentation":
        logits = torch.randn(2, 1, 32, requires_grad=True)
        targets = torch.rand(2, 32)
    else:
        logits = torch.randn(2, 2, 16, 32, requires_grad=True)
        targets = torch.rand(2, 2, 16, 32)

    loss = task.compute_loss(logits, targets, config, dataset_config)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None


@pytest.mark.parametrize("task_name", available_training_tasks())
def test_task_validates_its_model_geometry(task_name: str) -> None:
    dataset_config = _dataset_config()
    task = get_training_task(task_name)
    overrides = (
        {
            "fcn_ocr_crop_left": 0,
            "fcn_ocr_crop_right": 0,
            "fcn_ocr_strict_width": True,
        }
        if task_name == "fcn_ocr"
        else {}
    )
    config = _training_config(task_name, **overrides)
    model = torch.nn.Conv2d(
        dataset_config.channels,
        task.num_outputs(dataset_config.alphabet),
        kernel_size=1,
    )

    task.validate_model(model, config, dataset_config)

    assert task.summary_lines(config, dataset_config)


@pytest.mark.parametrize(
    ("task_name", "foreign_field", "value"),
    [
        ("fcn_ocr", "vertical_segmentation_loss", "mse"),
        ("vertical_segmentation", "baseline_detection_loss", "bce"),
        ("baseline_detection", "fcn_ocr_space_weight", 0.5),
    ],
)
def test_training_config_rejects_fields_owned_by_another_task(
    task_name: str,
    foreign_field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError, match="does not accept config fields"):
        _training_config(task_name, **{foreign_field: value})


@pytest.mark.parametrize(
    ("task_name", "loss_field", "value"),
    [
        (
            "vertical_segmentation",
            "vertical_segmentation_loss",
            "removed_loss",
        ),
        ("baseline_detection", "baseline_detection_loss", "removed_loss"),
    ],
)
def test_task_rejects_an_unknown_loss(
    task_name: str,
    loss_field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError, match=f"{loss_field} must be one of"):
        _training_config(task_name, **{loss_field: value})


@pytest.mark.parametrize(
    ("task_name", "expected_prefix", "foreign_prefixes"),
    [
        (
            "fcn_ocr",
            "fcn_ocr_",
            ("vertical_segmentation_", "baseline_detection_"),
        ),
        (
            "vertical_segmentation",
            "vertical_segmentation_",
            ("fcn_ocr_", "baseline_detection_"),
        ),
        (
            "baseline_detection",
            "baseline_detection_",
            ("fcn_ocr_", "vertical_segmentation_"),
        ),
    ],
)
def test_effective_checkpoint_config_contains_only_selected_task_fields(
    task_name: str,
    expected_prefix: str,
    foreign_prefixes: tuple[str, ...],
) -> None:
    data = effective_training_config_data(
        _training_config(task_name),
        _dataset_config(),
    )

    assert any(key.startswith(expected_prefix) for key in data)
    assert not any(key.startswith(foreign_prefixes) for key in data)
