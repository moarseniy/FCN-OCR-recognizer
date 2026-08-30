"""Training orchestration contracts for FCN tasks."""

from .tasks import (
    TrainingTask,
    all_training_task_config_fields,
    available_training_tasks,
    get_training_task,
)
from .config import TrainingConfig, load_training_config
from .optimization import (
    create_optimizer,
    create_scheduler,
    current_lr,
    print_optimizer_summary,
    step_scheduler,
)
from .checkpoints import build_checkpoint, save_checkpoint, save_named_checkpoint
from .data import ChunkBatchSampler, RandomFixedBatchSampler, make_data_loader
from .engine import InputPreviewSaver, append_training_log, train_one_epoch, validate
from .dataset import (
    dataset_config_from_chunk_metadata,
    effective_training_config_data,
    load_dataset_from_config,
    resolve_chunks_dir,
    validate_and_log_alphabet,
)
from .experiment import (
    EpochCallback,
    resolve_checkpoint_dir,
    save_experiment_config_snapshots,
)
from .runner import run_training

__all__ = [
    "TrainingTask",
    "ChunkBatchSampler",
    "RandomFixedBatchSampler",
    "InputPreviewSaver",
    "EpochCallback",
    "TrainingConfig",
    "all_training_task_config_fields",
    "available_training_tasks",
    "append_training_log",
    "build_checkpoint",
    "create_optimizer",
    "create_scheduler",
    "current_lr",
    "get_training_task",
    "dataset_config_from_chunk_metadata",
    "effective_training_config_data",
    "load_dataset_from_config",
    "load_training_config",
    "make_data_loader",
    "print_optimizer_summary",
    "save_checkpoint",
    "save_named_checkpoint",
    "resolve_chunks_dir",
    "resolve_checkpoint_dir",
    "save_experiment_config_snapshots",
    "run_training",
    "step_scheduler",
    "train_one_epoch",
    "validate",
    "validate_and_log_alphabet",
]
