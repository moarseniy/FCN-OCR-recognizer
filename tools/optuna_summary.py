from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import optuna


METRIC_ATTRS = (
    "length_accuracy",
    "average_abs_length_error",
    "total_abs_length_error",
    "normalized_length_error",
    "cut_precision",
    "cut_recall",
    "cut_f1",
    "cut_mae_px",
    "failure_penalized_normalized_mae",
    "normalized_mae",
    "combined_mae_px",
    "top_mae_px",
    "bottom_mae_px",
    "success_rate",
    "mean_coverage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Optuna studies and best trials stored in SQLite DBs."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("output"),
        help="Directory to scan recursively for Optuna DB files (default: output).",
    )
    parser.add_argument(
        "--all-trials",
        action="store_true",
        help="Include every trial, not only each study's best trial.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    return parser.parse_args()


def _metric_attrs(user_attrs: dict[str, Any]) -> dict[str, Any]:
    return {name: user_attrs[name] for name in METRIC_ATTRS if name in user_attrs}


def _contract_summary(contract: Any) -> dict[str, Any] | None:
    if not isinstance(contract, dict):
        return None
    keys = ("evaluator", "json", "images", "checkpoint", "metric", "limit", "parameters")
    return {key: contract[key] for key in keys if key in contract}


def summarize_study(
    database: Path,
    study_name: str,
    *,
    include_trials: bool,
) -> dict[str, Any]:
    storage = f"sqlite:///{database.resolve()}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.get_trials(deepcopy=False)
    state_counts = Counter(trial.state.name for trial in trials)
    contract = study.user_attrs.get("evaluation_contract")

    try:
        best = study.best_trial
    except ValueError:
        best = None

    report: dict[str, Any] = {
        "database": str(database),
        "study": study.study_name,
        "direction": study.direction.name.lower(),
        "metric": contract.get("metric") if isinstance(contract, dict) else None,
        "trial_count": len(trials),
        "states": dict(sorted(state_counts.items())),
        "evaluation_contract": _contract_summary(contract),
        "best_trial": None,
    }
    if best is not None:
        report["best_trial"] = {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "metrics": _metric_attrs(best.user_attrs),
        }
    if include_trials:
        report["trials"] = [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "metrics": _metric_attrs(trial.user_attrs),
            }
            for trial in trials
        ]
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"DB: {report['database']}")
    print(
        f"  Study: {report['study']} | direction={report['direction']} | "
        f"trials={report['trial_count']} | states={report['states']}"
    )
    contract = report.get("evaluation_contract")
    if contract:
        print(f"  Evaluation: {contract}")

    best = report.get("best_trial")
    if best is None:
        print("  Best trial: none (no completed trials)")
    else:
        print(
            f"  Best trial #{best['number']}: "
            f"{report['metric'] or 'objective'}={best['value']}"
        )
        print(f"    Params:  {best['params']}")
        if best["metrics"]:
            print(f"    Metrics: {best['metrics']}")

    for trial in report.get("trials", []):
        print(
            f"  Trial #{trial['number']} [{trial['state']}] "
            f"value={trial['value']} params={trial['params']}"
        )


def main() -> None:
    args = parse_args()
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Optuna DB directory does not exist: {root}")

    databases = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
    )
    if not databases:
        print(f"No Optuna DB files found under {root}")
        print("Runs without optuna_storage cannot be recovered from SQLite.")
        return

    reports = []
    errors = []
    for database in databases:
        try:
            summaries = optuna.study.get_all_study_summaries(
                storage=f"sqlite:///{database.resolve()}"
            )
            for summary in summaries:
                reports.append(
                    summarize_study(
                        database,
                        summary.study_name,
                        include_trials=args.all_trials,
                    )
                )
        except Exception as error:
            errors.append({"database": str(database), "error": repr(error)})

    if args.format == "json":
        print(json.dumps({"studies": reports, "errors": errors}, indent=2, ensure_ascii=False))
        return

    if not reports and not errors:
        print(f"No Optuna studies found under {root}")
    for report in reports:
        _print_report(report)
    for error in errors:
        print(f"Could not read {error['database']}: {error['error']}")


if __name__ == "__main__":
    main()
