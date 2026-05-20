from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import time
from pathlib import Path
from typing import Any

from fcn_ocr import TextRecognizer


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n < m:
        return levenshtein(b, a)

    previous = list(range(m + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[m]


def char_accuracy(gt: str, pred: str) -> float:
    if not gt:
        return 1.0 if not pred else 0.0

    distance = levenshtein(gt, pred)
    return max(0.0, 1.0 - distance / len(gt))


def exact_match(gt: str, pred: str) -> bool:
    return gt == pred


def get_gt_text(task: dict[str, Any]) -> str:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            text_items = result.get("value", {}).get("text", [])
            if text_items:
                return str(text_items[0]).strip()
    return ""


def get_image_name(task: dict[str, Any]) -> str:
    image_path = task.get("data", {}).get("image", "")
    return Path(image_path).name


def recognize_images(
    recognizer: TextRecognizer,
    jobs: list[tuple[int, Path]],
    batch_size: int,
    log_every: int,
) -> tuple[dict[int, str], dict[int, str]]:
    predictions: dict[int, str] = {}
    errors: dict[int, str] = {}
    processed = 0
    started_at = time.perf_counter()

    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        paths = [path for _, path in batch_jobs]

        try:
            batch_results = recognizer.recognize_paths_text(paths, batch_size=len(paths))
            for (row_index, _), (_, text) in zip(batch_jobs, batch_results):
                predictions[row_index] = text.strip()
                errors[row_index] = ""
        except Exception as batch_error:
            for row_index, path in batch_jobs:
                try:
                    text, _ = recognizer.recognize(path)
                    predictions[row_index] = text.strip()
                    errors[row_index] = ""
                except Exception as image_error:
                    predictions[row_index] = ""
                    errors[row_index] = f"batch_error={batch_error!r}; image_error={image_error!r}"

        processed += len(batch_jobs)
        if log_every > 0 and (processed == len(jobs) or processed % log_every == 0):
            elapsed = max(1e-9, time.perf_counter() - started_at)
            speed = processed / elapsed
            print(f"Recognized {processed}/{len(jobs)} images ({speed:.2f} img/s)")

    return predictions, errors


def build_rows_and_jobs(
    json_path: Path,
    images_dir: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[int, Path]]]:
    with json_path.open("r", encoding="utf-8") as file:
        tasks = json.load(file)

    if limit is not None:
        tasks = tasks[:limit]

    rows: list[dict[str, Any]] = []
    jobs: list[tuple[int, Path]] = []

    for task in tasks:
        image_name = get_image_name(task)
        image_path = images_dir / image_name
        row = {
            "task_id": task.get("id"),
            "image": image_name,
            "gt": get_gt_text(task),
            "pred": "",
            "exact_match": 0,
            "char_accuracy": 0.0,
            "levenshtein": 0,
            "gt_len": 0,
            "pred_len": 0,
            "error": "",
        }

        if image_path.exists():
            jobs.append((len(rows), image_path))
        else:
            row["error"] = f"image_not_found: {image_path}"
        rows.append(row)

    return rows, jobs


def compute_metrics(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    total = 0
    exact_ok = 0
    total_lev = 0
    total_gt_chars = 0
    total_char_acc = 0.0

    for row in rows:
        gt = row["gt"]
        pred = row["pred"]
        lev = levenshtein(gt, pred)
        c_acc = char_accuracy(gt, pred)
        is_exact = exact_match(gt, pred)

        row["exact_match"] = int(is_exact)
        row["char_accuracy"] = round(c_acc, 6)
        row["levenshtein"] = lev
        row["gt_len"] = len(gt)
        row["pred_len"] = len(pred)

        total += 1
        exact_ok += int(is_exact)
        total_lev += lev
        total_gt_chars += len(gt)
        total_char_acc += c_acc

    recognized = sum(1 for row in rows if not row["error"])
    return {
        "total_samples": total,
        "recognized_samples": recognized,
        "exact_matches": exact_ok,
        "line_accuracy": exact_ok / total if total else 0.0,
        "average_char_accuracy": total_char_acc / total if total else 0.0,
        "global_char_accuracy": max(0.0, 1.0 - total_lev / total_gt_chars) if total_gt_chars else 0.0,
        "average_levenshtein": total_lev / total if total else 0.0,
        "total_levenshtein": total_lev,
        "elapsed": elapsed,
        "speed": recognized / elapsed if elapsed > 0 else 0.0,
    }


def write_rows_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "image",
                "gt",
                "pred",
                "exact_match",
                "char_accuracy",
                "levenshtein",
                "gt_len",
                "pred_len",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(metrics: dict[str, Any], output_csv: Path | None = None) -> None:
    print("=== OCR evaluation ===")
    print(f"Total samples:              {metrics['total_samples']}")
    print(f"Recognized samples:         {metrics['recognized_samples']}")
    print(f"Exact line matches:         {metrics['exact_matches']}")
    print(f"Line accuracy:              {metrics['line_accuracy']:.4f}")
    print(f"Average char accuracy:      {metrics['average_char_accuracy']:.4f}")
    print(f"Global char accuracy:       {metrics['global_char_accuracy']:.4f}")
    print(f"Average Levenshtein:        {metrics['average_levenshtein']:.4f}")
    print(f"Total Levenshtein:          {metrics['total_levenshtein']}")
    print(f"Elapsed:                    {metrics['elapsed']:.2f}s")
    print(f"Speed:                      {metrics['speed']:.2f} img/s")
    print(f"scale_x:                    {metrics['scale_x']:+.5f}")
    print(f"y_pad:                      {metrics['y_pad']:+.5f}")
    if "x_pad" in metrics:
        print(f"x_pad:                      {metrics['x_pad']:.5f}")
    if "baseline_crop" in metrics:
        print(f"Baseline crop:              {metrics['baseline_crop']}")
    if output_csv is not None:
        print(f"CSV saved to:               {output_csv}")


def evaluate_prepared(
    base_rows: list[dict[str, Any]],
    jobs: list[tuple[int, Path]],
    checkpoint_path: Path,
    output_csv: Path | None,
    device: str | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    batch_size: int,
    log_every: int,
    verbose: bool,
    baseline_crop: bool = False,
    baseline_top_pad: float = 0.12,
    baseline_bottom_pad: float = 0.18,
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    rows = deepcopy(base_rows)
    started_at = time.perf_counter()
    recognizer = TextRecognizer(
        checkpoint_path,
        device=device,
        verbose=verbose,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
    )
    predictions, errors = recognize_images(recognizer, jobs, batch_size=batch_size, log_every=log_every)
    elapsed = time.perf_counter() - started_at

    for row_index, prediction in predictions.items():
        rows[row_index]["pred"] = prediction
    for row_index, error in errors.items():
        rows[row_index]["error"] = error

    metrics = compute_metrics(rows, elapsed)
    metrics["scale_x"] = float(scale_x)
    metrics["y_pad"] = float(y_pad)
    metrics["x_pad"] = float(x_pad)
    metrics["baseline_crop"] = bool(baseline_crop)

    if output_csv is not None:
        write_rows_csv(rows, output_csv)

    if verbose:
        print_metrics(metrics, output_csv)

    return metrics


def append_trial_log(path: Path, trial_number: int, metrics: dict[str, Any], metric_name: str) -> None:
    is_new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        if is_new_file:
            file.write(
                "trial\tscale_x\ty_pad\tx_pad\tmetric\tline_accuracy\taverage_char_accuracy\t"
                "global_char_accuracy\taverage_levenshtein\ttotal_levenshtein\tspeed\n"
            )
        file.write(
            f"{trial_number}\t{metrics['scale_x']:.8f}\t{metrics['y_pad']:.8f}\t{metrics.get('x_pad', 0.0):.8f}\t"
            f"{metrics[metric_name]:.8f}\t{metrics['line_accuracy']:.8f}\t"
            f"{metrics['average_char_accuracy']:.8f}\t{metrics['global_char_accuracy']:.8f}\t"
            f"{metrics['average_levenshtein']:.8f}\t{metrics['total_levenshtein']}\t"
            f"{metrics['speed']:.6f}\n"
        )


def optimize_preprocess(
    json_path: Path,
    images_dir: Path,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    batch_size: int,
    limit: int | None,
    trials: int,
    scale_x_min: float,
    scale_x_max: float,
    y_pad_min: float,
    y_pad_max: float,
    x_pad: float,
    metric_name: str,
    log_every: int,
    trials_output: Path | None,
    study_name: str | None = None,
    storage: str | None = None,
    baseline_crop: bool = False,
    baseline_top_pad: float = 0.12,
    baseline_bottom_pad: float = 0.18,
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna") from exc
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    if trials < 1:
        raise ValueError("trials must be >= 1")

    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    direction = "minimize" if metric_name in {"average_levenshtein", "total_levenshtein"} else "maximize"
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )

    def objective(trial) -> float:
        scale_x = trial.suggest_float("scale_x", scale_x_min, scale_x_max)
        y_pad = trial.suggest_float("y_pad", y_pad_min, y_pad_max)
        metrics = evaluate_prepared(
            base_rows,
            jobs,
            checkpoint_path=checkpoint_path,
            output_csv=None,
            device=device,
            scale_x=scale_x,
            y_pad=y_pad,
            x_pad=x_pad,
            batch_size=batch_size,
            log_every=0,
            verbose=False,
            baseline_crop=baseline_crop,
            baseline_top_pad=baseline_top_pad,
            baseline_bottom_pad=baseline_bottom_pad,
            baseline_deskew=baseline_deskew,
            baseline_max_angle=baseline_max_angle,
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                trial.set_user_attr(key, value)
        if trials_output is not None:
            append_trial_log(trials_output, trial.number, metrics, metric_name)
        return float(metrics[metric_name])

    print(
        "Optuna preprocess search: "
        f"trials={trials}, metric={metric_name}, "
        f"scale_x=[{scale_x_min}, {scale_x_max}], y_pad=[{y_pad_min}, {y_pad_max}], "
        f"x_pad={x_pad}, baseline_crop={baseline_crop}"
    )
    study.optimize(objective, n_trials=trials)

    best_scale_x = float(study.best_params["scale_x"])
    best_y_pad = float(study.best_params["y_pad"])
    print(
        f"Best Optuna params: scale_x={best_scale_x:+.5f}, "
        f"y_pad={best_y_pad:+.5f}, {metric_name}={study.best_value:.8f}"
    )

    final_metrics = evaluate_prepared(
        base_rows,
        jobs,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        device=device,
        scale_x=best_scale_x,
        y_pad=best_y_pad,
        x_pad=x_pad,
        batch_size=batch_size,
        log_every=log_every,
        verbose=True,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
    )
    final_metrics["optuna_trials"] = trials
    final_metrics["optuna_metric"] = metric_name
    final_metrics["optuna_best_value"] = float(study.best_value)
    return final_metrics


def evaluate(
    json_path: Path,
    images_dir: Path,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    scale_x: float,
    y_pad: float,
    x_pad: float,
    batch_size: int,
    limit: int | None,
    log_every: int,
    verbose: bool = True,
    baseline_crop: bool = False,
    baseline_top_pad: float = 0.12,
    baseline_bottom_pad: float = 0.18,
    baseline_deskew: bool = True,
    baseline_max_angle: float = 12.0,
) -> dict[str, Any]:
    base_rows, jobs = build_rows_and_jobs(json_path, images_dir, limit)
    return evaluate_prepared(
        base_rows,
        jobs,
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        device=device,
        scale_x=scale_x,
        y_pad=y_pad,
        x_pad=x_pad,
        batch_size=batch_size,
        log_every=log_every,
        verbose=verbose,
        baseline_crop=baseline_crop,
        baseline_top_pad=baseline_top_pad,
        baseline_bottom_pad=baseline_bottom_pad,
        baseline_deskew=baseline_deskew,
        baseline_max_angle=baseline_max_angle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FCN OCR on a Label Studio JSON export.")
    parser.add_argument("--json", required=True, help="Path to Label Studio export JSON.")
    parser.add_argument("--images", required=True, help="Folder with images.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--out", default="ocr_metrics.csv", help="Output CSV path.")
    parser.add_argument("--device", default=None, help="Device to use: cuda, cpu, or empty for auto.")
    parser.add_argument("--scale-x", type=float, default=0.0, help="Normalized horizontal inference scale.")
    parser.add_argument("--y-pad", type=float, default=0.0, help="Normalized vertical inference padding/crop.")
    parser.add_argument("--x-pad", type=float, default=0.0, help="Normalized symmetric horizontal inference padding.")
    parser.add_argument("--baseline-crop", action="store_true", help="Use baseline detection/crop before y-pad and resize.")
    parser.add_argument("--baseline-top-pad", type=float, default=0.12)
    parser.add_argument("--baseline-bottom-pad", type=float, default=0.18)
    parser.add_argument("--no-baseline-deskew", action="store_true")
    parser.add_argument("--baseline-max-angle", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to evaluate.")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N recognized images; 0 disables.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="If > 0, tune scale_x and y_pad with Optuna before final evaluation.")
    parser.add_argument("--optuna-scale-x-min", type=float, default=-0.25)
    parser.add_argument("--optuna-scale-x-max", type=float, default=0.25)
    parser.add_argument("--optuna-y-pad-min", type=float, default=-0.25)
    parser.add_argument("--optuna-y-pad-max", type=float, default=0.25)
    parser.add_argument(
        "--optuna-metric",
        default="global_char_accuracy",
        choices=[
            "line_accuracy",
            "average_char_accuracy",
            "global_char_accuracy",
            "average_levenshtein",
            "total_levenshtein",
        ],
    )
    parser.add_argument("--optuna-trials-out", default=None, help="Optional TSV file with Optuna trial metrics.")
    parser.add_argument("--optuna-study-name", default=None)
    parser.add_argument("--optuna-storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///study.db.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.optuna_trials > 0:
        optimize_preprocess(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            trials=args.optuna_trials,
            scale_x_min=args.optuna_scale_x_min,
            scale_x_max=args.optuna_scale_x_max,
            y_pad_min=args.optuna_y_pad_min,
            y_pad_max=args.optuna_y_pad_max,
            x_pad=args.x_pad,
            metric_name=args.optuna_metric,
            log_every=args.log_every,
            trials_output=Path(args.optuna_trials_out) if args.optuna_trials_out else None,
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            baseline_crop=args.baseline_crop,
            baseline_top_pad=args.baseline_top_pad,
            baseline_bottom_pad=args.baseline_bottom_pad,
            baseline_deskew=not args.no_baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
        )
    else:
        evaluate(
            json_path=Path(args.json),
            images_dir=Path(args.images),
            checkpoint_path=Path(args.checkpoint),
            output_csv=Path(args.out),
            device=args.device,
            scale_x=args.scale_x,
            y_pad=args.y_pad,
            x_pad=args.x_pad,
            batch_size=args.batch_size,
            limit=args.limit,
            log_every=args.log_every,
            baseline_crop=args.baseline_crop,
            baseline_top_pad=args.baseline_top_pad,
            baseline_bottom_pad=args.baseline_bottom_pad,
            baseline_deskew=not args.no_baseline_deskew,
            baseline_max_angle=args.baseline_max_angle,
        )


if __name__ == "__main__":
    main()
