from __future__ import annotations

import csv
from pathlib import Path
import shlex
from typing import Any, Iterable, Sequence

from fcn_ocr.inference_config import InferenceConfig


def write_csv_rows(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames) if fieldnames is not None else list(rows[0]) if rows else ["image"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_tsv_row(
    path: str | Path,
    columns: Sequence[str],
    values: Iterable[Any],
) -> None:
    output_path = Path(path)
    is_new = not output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        if is_new:
            file.write("\t".join(columns) + "\n")
        file.write("\t".join(str(value) for value in values) + "\n")


def save_and_print_inference_command(
    config_data: dict[str, Any],
    metrics_output: str | Path,
    image_path: str | Path | None,
) -> Path:
    config = InferenceConfig.model_validate(config_data)
    config_path = Path(metrics_output).expanduser().resolve().with_suffix(".inference.yaml")
    config.save(config_path)
    command = [
        "python",
        "inference.py",
        "--config",
        str(config_path),
        "--image",
        str(image_path) if image_path is not None else "<IMAGE_PATH>",
    ]
    print(f"Inference config saved to:  {config_path}")
    print("\n=== Inference command ===")
    print(shlex.join(command))
    return config_path
