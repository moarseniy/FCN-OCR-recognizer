"""Training orchestration contracts for FCN tasks."""

from .tasks import (
    TrainingTask,
    all_training_task_config_fields,
    available_training_tasks,
    get_training_task,
)

__all__ = [
    "TrainingTask",
    "all_training_task_config_fields",
    "available_training_tasks",
    "get_training_task",
]
