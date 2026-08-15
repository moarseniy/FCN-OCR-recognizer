from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabelStudioSample:
    task_id: Any
    image_name: str
    image_path: Path
    text: str


def load_json_document(path: str | Path) -> Any:
    document_path = Path(path).expanduser()
    with document_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def label_studio_text(task: dict[str, Any]) -> str:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            text_items = result.get("value", {}).get("text", [])
            if text_items:
                return str(text_items[0]).strip()
    return ""


def label_studio_image_name(task: dict[str, Any]) -> str:
    image_path = task.get("data", {}).get("image", "")
    return Path(image_path).name


def load_label_studio_samples(
    json_path: str | Path,
    images_dir: str | Path,
    limit: int | None,
) -> list[LabelStudioSample]:
    document = load_json_document(json_path)
    return label_studio_samples(document, images_dir, limit)


def label_studio_samples(
    document: Any,
    images_dir: str | Path,
    limit: int | None,
) -> list[LabelStudioSample]:
    if not isinstance(document, list):
        raise ValueError("Label Studio evaluation JSON must contain a list of tasks")
    images_root = Path(images_dir).expanduser()
    samples: list[LabelStudioSample] = []
    for task in document:
        if not isinstance(task, dict):
            raise ValueError("Every Label Studio task must be a mapping")
        image_name = label_studio_image_name(task)
        image_path = images_root / image_name
        if not image_path.is_file():
            continue
        samples.append(
            LabelStudioSample(
                task_id=task.get("id"),
                image_name=image_name,
                image_path=image_path,
                text=label_studio_text(task),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples
