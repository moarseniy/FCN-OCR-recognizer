from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings
from typing import Any, Callable

from tqdm import tqdm


def file_contract(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def create_study(
    *,
    direction: str,
    study_name: str | None,
    storage: str | None,
    seed: int = 0,
):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna") from exc
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
        )
    return optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
        sampler=sampler,
    )


def validate_study_contract(study: Any, contract: dict[str, Any]) -> None:
    """Keep a persistent study tied to one comparable evaluation problem."""

    normalized = json.loads(json.dumps(contract, sort_keys=True, default=str))
    existing = study.user_attrs.get("evaluation_contract")
    if existing is None:
        if study.trials:
            raise RuntimeError(
                "The existing Optuna study has no evaluation contract and cannot be "
                "resumed safely. Use a new optuna_study_name or remove optuna_storage."
            )
        study.set_user_attr("evaluation_contract", normalized)
        return
    if existing != normalized:
        raise RuntimeError(
            "The existing Optuna study was created for a different checkpoint, dataset, "
            "metric, or search space. Use a new optuna_study_name or remove optuna_storage."
        )


def suggest_float_or_fixed(
    trial: Any,
    name: str,
    fixed: float | None,
    minimum: float | None,
    maximum: float | None,
) -> float | None:
    if minimum is None and maximum is None:
        return fixed
    if minimum is None or maximum is None:
        raise ValueError(f"{name} tuning requires both min and max")
    return float(trial.suggest_float(name, float(minimum), float(maximum)))


def suggest_int_or_fixed(
    trial: Any,
    name: str,
    fixed: int | None,
    minimum: int | None,
    maximum: int | None,
) -> int | None:
    if minimum is None and maximum is None:
        return fixed
    if minimum is None or maximum is None:
        raise ValueError(f"{name} tuning requires both min and max")
    return int(trial.suggest_int(name, int(minimum), int(maximum)))


def require_float_parameter(
    trial: Any,
    name: str,
    fixed: float | None,
    minimum: float | None,
    maximum: float | None,
) -> float:
    value = suggest_float_or_fixed(trial, name, fixed, minimum, maximum)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be fixed or have an Optuna range")
    return float(value)


def require_int_parameter(
    trial: Any,
    name: str,
    fixed: int | None,
    minimum: int | None,
    maximum: int | None,
) -> int:
    value = suggest_int_or_fixed(trial, name, fixed, minimum, maximum)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be fixed or have an Optuna range")
    return int(value)


def best_or_fixed(best_params: dict[str, Any], name: str, fixed: Any) -> Any:
    return best_params[name] if name in best_params else fixed


def validate_float_range(
    name: str,
    minimum: float | None,
    maximum: float | None,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    if (minimum is None) != (maximum is None):
        raise ValueError(f"{name} tuning requires both min and max")
    if minimum is None or maximum is None:
        return
    if minimum > maximum:
        raise ValueError(f"{name} tuning requires min <= max")
    if lower is not None and minimum < lower:
        raise ValueError(f"{name} tuning minimum must be >= {lower}")
    if upper is not None and maximum > upper:
        raise ValueError(f"{name} tuning maximum must be <= {upper}")


def validate_int_range(
    name: str,
    minimum: int | None,
    maximum: int | None,
    *,
    lower: int,
) -> None:
    if (minimum is None) != (maximum is None):
        raise ValueError(f"{name} tuning requires both min and max")
    if minimum is None or maximum is None:
        return
    if minimum < lower or maximum < minimum:
        raise ValueError(f"{name} bounds must satisfy {lower} <= min <= max")


def optimize_with_progress(
    study: Any,
    objective: Callable[[Any], float],
    *,
    n_trials: int,
    metric_name: str,
    enabled: bool,
) -> None:
    if not enabled or not sys.stderr.isatty():
        study.optimize(objective, n_trials=n_trials)
        return

    with tqdm(
        total=n_trials,
        desc=f"Optuna {metric_name}",
        unit="trial",
        dynamic_ncols=True,
        file=sys.stderr,
    ) as progress:

        def update_progress(current_study: Any, trial: Any) -> None:
            values: dict[str, str] = {}
            if trial.value is not None:
                values["last"] = f"{float(trial.value):.6g}"
            if "x_pad" in trial.params:
                values["x_pad"] = f"{float(trial.params['x_pad']):.5f}"
            try:
                values["best"] = f"{float(current_study.best_value):.6g}"
            except ValueError:
                pass
            if values:
                progress.set_postfix(values, refresh=False)
            progress.update(1)

        study.optimize(objective, n_trials=n_trials, callbacks=[update_progress])
