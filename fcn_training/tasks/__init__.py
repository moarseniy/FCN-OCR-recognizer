from __future__ import annotations

from fcn_tasks import TASK_NAMES

from .base import TrainingTask
from .baseline_detection import BaselineDetectionTrainingTask
from .fcn_ocr import FCNOCRTrainingTask
from .vertical_segmentation import VerticalSegmentationTrainingTask


_TASKS: dict[str, TrainingTask] = {
    task.name: task
    for task in (
        FCNOCRTrainingTask(),
        VerticalSegmentationTrainingTask(),
        BaselineDetectionTrainingTask(),
    )
}
if tuple(_TASKS) != TASK_NAMES:
    raise RuntimeError("Training task registry does not match the shared task vocabulary")


def available_training_tasks() -> tuple[str, ...]:
    return tuple(_TASKS)


def get_training_task(name: str) -> TrainingTask:
    normalized = str(name).strip().lower()
    try:
        return _TASKS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown training task {name!r}; expected one of {available_training_tasks()}"
        ) from exc


def all_training_task_config_fields() -> frozenset[str]:
    return frozenset().union(*(task.config_fields for task in _TASKS.values()))


__all__ = [
    "TrainingTask",
    "all_training_task_config_fields",
    "available_training_tasks",
    "get_training_task",
]
