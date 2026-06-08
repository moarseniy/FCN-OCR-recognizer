from __future__ import annotations

import sys
from typing import Any, Callable

from tqdm import tqdm


def optimize_with_progress(
    study: Any,
    objective: Callable[[Any], float],
    *,
    n_trials: int,
    metric_name: str,
    enabled: bool,
) -> None:
    """Run an Optuna study with an interactive-terminal progress bar."""
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
            try:
                values["best"] = f"{float(current_study.best_value):.6g}"
            except ValueError:
                pass
            if values:
                progress.set_postfix(values, refresh=False)
            progress.update(1)

        study.optimize(objective, n_trials=n_trials, callbacks=[update_progress])
