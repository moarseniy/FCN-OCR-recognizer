from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import yaml


def expand_optuna_ranges(
    config_data: dict,
    *,
    valid_fields: set[str],
) -> dict:
    expanded = dict(config_data)
    old_range_fields = sorted(
        field
        for field in expanded
        if field.startswith("optuna_")
        and field.endswith(("_min", "_max"))
        and field in valid_fields
    )
    if old_range_fields:
        raise ValueError(
            "evaluation YAML must use optuna_ranges instead of: "
            + ", ".join(old_range_fields)
        )
    ranges = expanded.pop("optuna_ranges", None)
    if ranges is None:
        return expanded
    if not isinstance(ranges, dict):
        raise ValueError("optuna_ranges must be a YAML mapping")

    for raw_name, bounds in ranges.items():
        name = str(raw_name).strip().lower().replace("-", "_")
        prefix = name if name.startswith("optuna_") else f"optuna_{name}"
        min_field = f"{prefix}_min"
        max_field = f"{prefix}_max"
        if min_field not in valid_fields or max_field not in valid_fields:
            raise ValueError(f"unknown Optuna range: {raw_name}")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"optuna_ranges.{raw_name} must contain [min, max]")
        minimum, maximum = bounds
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
        ):
            raise ValueError(f"optuna_ranges.{raw_name} bounds must be numbers")
        if minimum > maximum:
            raise ValueError(f"optuna_ranges.{raw_name} requires min <= max")
        expanded[min_field] = minimum
        expanded[max_field] = maximum
    return expanded


def parse_args_with_evaluation_config(
    parser: argparse.ArgumentParser,
    *,
    path_fields: Iterable[str] = (),
    required_fields: Iterable[str] = (),
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    known, _ = config_parser.parse_known_args(argv)

    config_path: Path | None = None
    if known.config:
        config_path = Path(known.config).expanduser().resolve()
        if not config_path.exists():
            parser.error(f"evaluation config does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
        if not isinstance(config_data, dict):
            parser.error("evaluation config must contain a YAML mapping")

        valid_fields = {action.dest for action in parser._actions}
        try:
            config_data = expand_optuna_ranges(
                config_data,
                valid_fields=valid_fields,
            )
        except ValueError as error:
            parser.error(str(error))
        unknown_fields = sorted(set(config_data) - valid_fields)
        if unknown_fields:
            parser.error(
                "unknown evaluation config keys: " + ", ".join(unknown_fields)
            )

        config_dir = config_path.parent
        resolved_data = dict(config_data)
        for field in path_fields:
            value = resolved_data.get(field)
            if value is None or value == "":
                continue
            path = Path(str(value)).expanduser()
            resolved_data[field] = str(
                path.resolve() if path.is_absolute() else (config_dir / path).resolve()
            )
        storage = resolved_data.get("optuna_storage")
        if isinstance(storage, str) and storage.startswith("sqlite:///"):
            database = storage.removeprefix("sqlite:///")
            database_path = Path(database).expanduser()
            if not database_path.is_absolute():
                database_path = (config_dir / database_path).resolve()
                resolved_data["optuna_storage"] = f"sqlite:///{database_path}"
        parser.set_defaults(**resolved_data)

    args = parser.parse_args(argv)
    missing = [
        field
        for field in required_fields
        if getattr(args, field, None) is None or getattr(args, field, None) == ""
    ]
    if missing:
        parser.error(
            "missing required values (set them in --config or CLI): "
            + ", ".join(missing)
        )
    args.evaluation_config_path = config_path
    return args
