from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SCHEMA_NAME = "fcn_ocr.manual_markup"
SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def discover_images(images_root: Path, recursive: bool = True) -> list[str]:
    root = images_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Images directory does not exist: {root}")
    candidates: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        (
            path.relative_to(root).as_posix()
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda value: value.casefold(),
    )


def new_document(images_root: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "images_root": str(images_root.expanduser().resolve()),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "items": [],
    }


def is_manual_markup(data: Any) -> bool:
    return isinstance(data, dict) and data.get("schema") == SCHEMA_NAME


def load_document(path: Path, images_root: Path | None = None) -> dict[str, Any]:
    path = path.expanduser()
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            document = json.load(file)
        if not is_manual_markup(document):
            raise ValueError(f"Unsupported markup schema in {path}")
        if int(document.get("version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported markup version {document.get('version')}; expected {SCHEMA_VERSION}"
            )
    else:
        if images_root is None:
            raise ValueError("images_root is required when creating a markup file")
        document = new_document(images_root)

    if images_root is not None:
        document["images_root"] = str(images_root.expanduser().resolve())
    document.setdefault("items", [])
    return document


def save_document(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    document["schema"] = SCHEMA_NAME
    document["version"] = SCHEMA_VERSION
    document["updated_at"] = utc_now()
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def item_by_image(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["image"]): item
        for item in document.get("items", [])
        if isinstance(item, dict) and item.get("image")
    }


def ensure_item(
    document: dict[str, Any],
    images_root: Path,
    relative_path: str,
) -> dict[str, Any]:
    relative_path = normalize_relative_path(relative_path)
    existing = item_by_image(document).get(relative_path)
    if existing is not None:
        return existing

    image_path = safe_image_path(images_root, relative_path)
    with Image.open(image_path) as image:
        width, height = image.size
    item = {
        "image": relative_path,
        "width": int(width),
        "height": int(height),
        "cuts": [],
        "baselines": {"top": [], "bottom": []},
        "completed": False,
        "updated_at": utc_now(),
    }
    document.setdefault("items", []).append(item)
    return item


def normalize_relative_path(value: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative image path: {value!r}")
    normalized = path.as_posix().lstrip("/")
    if not normalized:
        raise ValueError("Image path must not be empty")
    return normalized


def safe_image_path(images_root: Path, relative_path: str) -> Path:
    root = images_root.expanduser().resolve()
    candidate = (root / normalize_relative_path(relative_path)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Image path escapes images root: {relative_path!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Image does not exist: {candidate}")
    return candidate


def normalize_item(payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    cuts = sorted(
        {
            round(min(max(float(value), 0.0), max(0.0, float(width - 1))), 4)
            for value in payload.get("cuts", [])
        }
    )
    baselines = payload.get("baselines") or {}
    return {
        "width": int(width),
        "height": int(height),
        "cuts": cuts,
        "baselines": {
            "top": normalize_points(baselines.get("top", []), width, height),
            "bottom": normalize_points(baselines.get("bottom", []), width, height),
        },
        "completed": bool(payload.get("completed", False)),
        "updated_at": utc_now(),
    }


def normalize_points(points: Any, width: int, height: int) -> list[list[float]]:
    normalized: list[list[float]] = []
    if not isinstance(points, list):
        return normalized
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x = round(min(max(float(point[0]), 0.0), max(0.0, float(width - 1))), 4)
        y = round(min(max(float(point[1]), 0.0), max(0.0, float(height - 1))), 4)
        normalized.append([x, y])
    normalized.sort(key=lambda point: (point[0], point[1]))
    return normalized


def annotated_items(
    document: dict[str, Any],
    require_completed: bool = True,
) -> list[dict[str, Any]]:
    items = []
    for item in document.get("items", []):
        if not isinstance(item, dict) or not item.get("image"):
            continue
        if require_completed and not item.get("completed", False):
            continue
        items.append(item)
    return items

