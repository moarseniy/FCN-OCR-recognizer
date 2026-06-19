from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import yaml


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
