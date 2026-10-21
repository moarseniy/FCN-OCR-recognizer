from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_image(image_ref: str, images_root: Path) -> Path:
    """Resolve a markup image path, including old flat paths after files moved."""
    raw_path = Path(image_ref).expanduser()
    direct_path = raw_path.resolve() if raw_path.is_absolute() else (images_root / raw_path).resolve()

    if direct_path.is_file() and _is_relative_to(direct_path, images_root):
        return direct_path

    # Existing markup may contain only a basename, while images have since been
    # moved into field subdirectories. Recover it only when the name is unique.
    matches = sorted(
        path.resolve()
        for path in images_root.rglob(raw_path.name)
        if path.is_file()
    )
    matches = [path for path in matches if _is_relative_to(path, images_root)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"image not found under images root: {image_ref!r}")
    locations = ", ".join(str(path.relative_to(images_root)) for path in matches)
    raise ValueError(
        f"image path {image_ref!r} is ambiguous; matching files: {locations}"
    )


def split_markup(
    markup_path: Path,
    images_root: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    markup_path = markup_path.expanduser().resolve()
    images_root = images_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not markup_path.is_file():
        raise FileNotFoundError(f"Markup JSON not found: {markup_path}")
    if not images_root.is_dir():
        raise NotADirectoryError(f"Images root is not a directory: {images_root}")

    with markup_path.open("r", encoding="utf-8") as file:
        document: Any = json.load(file)
    if not isinstance(document, dict) or document.get("schema") != "fcn_ocr.manual_markup":
        raise ValueError(f"Unsupported manual markup JSON: {markup_path}")
    items = document.get("items")
    if not isinstance(items, list):
        raise ValueError("Markup JSON must contain an 'items' list")

    groups: dict[str, list[dict[str, Any]]] = {}
    problems: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("image"), str):
            problems.append(f"item #{index}: missing string 'image' path")
            continue

        try:
            image_path = _resolve_image(item["image"], images_root)
        except ValueError as error:
            problems.append(f"item #{index} ({item['image']}): {error}")
            continue

        relative_path = image_path.relative_to(images_root)
        if len(relative_path.parts) < 2:
            problems.append(
                f"item #{index} ({item['image']}): image is not inside a field subdirectory"
            )
            continue

        field = relative_path.parts[0]
        groups.setdefault(field, []).append(
            {**item, "image": relative_path.as_posix()}
        )

    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems)
        raise ValueError(
            "Could not safely map every markup item to an image. "
            "No output files were written:\n" + details
        )
    if not groups:
        raise ValueError("No annotated images were found in the field subdirectories")

    output_paths = [output_dir / f"{field}.json" for field in groups]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output file(s) already exist: {names}. Use --overwrite to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for field, field_items in sorted(groups.items()):
        result = {
            **document,
            "images_root": str(images_root),
            "updated_at": timestamp,
            "items": field_items,
        }
        output_path = output_dir / f"{field}.json"
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(output_path)
        counts[field] = len(field_items)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split FCN OCR manual markup JSON into one JSON per immediate "
            "image subdirectory."
        )
    )
    parser.add_argument("markup", type=Path, help="Source manual_markup.json")
    parser.add_argument(
        "images_root",
        type=Path,
        help="Common directory containing the field subdirectories and images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <markup folder>/<markup name>_by_field)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output JSON files that already exist",
    )
    args = parser.parse_args()

    markup_path = args.markup.expanduser().resolve()
    output_dir = args.output_dir or markup_path.parent / f"{markup_path.stem}_by_field"
    try:
        counts = split_markup(
            markup_path,
            args.images_root,
            output_dir,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, NotADirectoryError, FileExistsError, ValueError) as error:
        parser.error(str(error))

    for field, count in counts.items():
        print(f"{field}: {count} annotations -> {output_dir / (field + '.json')}")


if __name__ == "__main__":
    main()
