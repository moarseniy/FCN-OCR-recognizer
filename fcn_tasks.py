from __future__ import annotations


FCN_OCR_TASK = "fcn_ocr"
VERTICAL_SEGMENTATION_TASK = "vertical_segmentation"
BASELINE_DETECTION_TASK = "baseline_detection"

TASK_NAMES = (
    FCN_OCR_TASK,
    VERTICAL_SEGMENTATION_TASK,
    BASELINE_DETECTION_TASK,
)


def normalize_task_name(value: str) -> str:
    task = str(value).strip().lower()
    if task not in TASK_NAMES:
        raise ValueError(f"task must be one of {TASK_NAMES}")
    return task


def task_output_channels(task: str, alphabet: str) -> int:
    normalized = normalize_task_name(task)
    if normalized == FCN_OCR_TASK:
        return len(alphabet)
    if normalized == VERTICAL_SEGMENTATION_TASK:
        return 1
    return 2


__all__ = [
    "BASELINE_DETECTION_TASK",
    "FCN_OCR_TASK",
    "TASK_NAMES",
    "VERTICAL_SEGMENTATION_TASK",
    "normalize_task_name",
    "task_output_channels",
]
