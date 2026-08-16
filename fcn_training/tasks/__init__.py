from __future__ import annotations

from .base import TrainingTask
from .baselines import BaselineHeatmapTrainingTask
from .cuts import CutProjectionTrainingTask
from .ocr import OcrTrainingTask


_TASKS: dict[str, TrainingTask] = {
    task.name: task
    for task in (
        OcrTrainingTask(),
        CutProjectionTrainingTask(),
        BaselineHeatmapTrainingTask(),
    )
}


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
