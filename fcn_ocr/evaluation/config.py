from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing an evaluation config",
                node.start_mark,
                f"duplicate key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_evaluation_yaml(file) -> dict:
    data = yaml.load(file, Loader=_UniqueKeyLoader) or {}
    if not isinstance(data, dict):
        raise ValueError("evaluation config must contain a YAML mapping")
    return data


def expand_evaluation_parameters(
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
            "evaluation YAML must use parameters with [min, max] instead of: "
            + ", ".join(old_range_fields)
        )
    if "optuna_ranges" in expanded:
        raise ValueError("evaluation YAML must use parameters instead of optuna_ranges")

    parameters = expanded.pop("parameters", None)
    misplaced_ranges = sorted(
        str(field) for field, value in expanded.items() if isinstance(value, (list, tuple))
    )
    if misplaced_ranges:
        raise ValueError(
            "evaluation ranges must be placed inside parameters: "
            + ", ".join(misplaced_ranges)
        )
    if parameters is None:
        return expanded
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a YAML mapping")

    for raw_name, value in parameters.items():
        name = str(raw_name).strip().lower().replace("-", "_")
        if name not in valid_fields:
            raise ValueError(f"unknown evaluation parameter: {raw_name}")
        if name in expanded:
            raise ValueError(f"parameter {raw_name!r} is also set at the YAML root")

        min_field = f"optuna_{name}_min"
        max_field = f"optuna_{name}_max"
        tune_field = f"optuna_tune_{name}"
        has_numeric_range = min_field in valid_fields and max_field in valid_fields
        has_tune_flag = tune_field in valid_fields

        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError(f"parameters.{raw_name} range must contain [min, max]")
            minimum, maximum = value
            if isinstance(minimum, bool) or isinstance(maximum, bool):
                if [minimum, maximum] != [False, True] or not has_tune_flag:
                    raise ValueError(
                        f"parameters.{raw_name} boolean range must be [false, true] "
                        "and supported by this evaluator"
                    )
                expanded[tune_field] = True
                continue
            if not has_numeric_range:
                raise ValueError(f"parameter {raw_name!r} does not support numeric tuning")
            if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
                raise ValueError(f"parameters.{raw_name} bounds must be numbers")
            if minimum > maximum:
                raise ValueError(f"parameters.{raw_name} requires min <= max")
            expanded[min_field] = minimum
            expanded[max_field] = maximum
            if has_tune_flag:
                expanded[tune_field] = True
            continue

        expanded[name] = value
        if has_tune_flag:
            expanded[tune_field] = False
    return expanded


def evaluation_parameter_modes(config_data: dict) -> dict[str, str]:
    parameters = config_data.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    modes: dict[str, str] = {}
    for raw_name, value in parameters.items():
        name = str(raw_name).strip().lower().replace("-", "_")
        modes[name] = "range" if isinstance(value, (list, tuple)) else "fixed"
    return modes


def evaluation_parameter_range(
    args: argparse.Namespace,
    name: str,
) -> tuple[float | int | None, float | int | None]:
    """Return an active Optuna range, respecting fixed values from YAML."""

    modes = getattr(args, "evaluation_parameter_modes", {})
    if modes.get(name) == "fixed":
        return None, None
    return (
        getattr(args, f"optuna_{name}_min", None),
        getattr(args, f"optuna_{name}_max", None),
    )


def parse_args_with_evaluation_config(
    parser: argparse.ArgumentParser,
    *,
    path_fields: Iterable[str] = (),
    required_fields: Iterable[str] = (),
    parameter_ranges: Mapping[str, tuple[Any, Any]] | None = None,
    tunable_booleans: Iterable[str] = (),
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parameter_ranges = dict(parameter_ranges or {})
    tunable_booleans = set(tunable_booleans)
    internal_defaults: dict[str, Any] = {}
    internal_fields: set[str] = set()
    for name, (minimum, maximum) in parameter_ranges.items():
        min_field = f"optuna_{name}_min"
        max_field = f"optuna_{name}_max"
        internal_fields.update((min_field, max_field))
        internal_defaults[min_field] = minimum
        internal_defaults[max_field] = maximum
    for name in tunable_booleans:
        tune_field = f"optuna_tune_{name}"
        internal_fields.add(tune_field)
        internal_defaults[tune_field] = False
    parser.set_defaults(**internal_defaults)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    known, _ = config_parser.parse_known_args(argv)

    config_path: Path | None = None
    if known.config:
        config_path = Path(known.config).expanduser().resolve()
        if not config_path.exists():
            parser.error(f"evaluation config does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            try:
                config_data = load_evaluation_yaml(file)
            except (ValueError, yaml.YAMLError) as error:
                parser.error(str(error))

        parameter_modes = evaluation_parameter_modes(config_data)
        valid_fields = {action.dest for action in parser._actions} | internal_fields
        try:
            config_data = expand_evaluation_parameters(
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
    args.evaluation_parameter_modes = parameter_modes if config_path is not None else {}
    return args
