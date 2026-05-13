from __future__ import annotations

import argparse
import csv
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


def evaluate(
    json_path: Path,
    images_dir: Path,
    checkpoint_path: Path,
    output_csv: Path,
    device: str | None,
    scale_x: float,
    y_pad: float,
    batch_size: int,
    limit: int | None,
    log_every: int,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

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

    started_at = time.perf_counter()
    recognizer = TextRecognizer(
        checkpoint_path,
        device=device,
        verbose=True,
        scale_x=scale_x,
        y_pad=y_pad,
    )
    predictions, errors = recognize_images(recognizer, jobs, batch_size=batch_size, log_every=log_every)
    elapsed = time.perf_counter() - started_at

    for row_index, prediction in predictions.items():
        rows[row_index]["pred"] = prediction
    for row_index, error in errors.items():
        rows[row_index]["error"] = error

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

    line_accuracy = exact_ok / total if total else 0.0
    avg_char_accuracy = total_char_acc / total if total else 0.0
    normalized_char_accuracy = max(0.0, 1.0 - total_lev / total_gt_chars) if total_gt_chars else 0.0
    avg_levenshtein = total_lev / total if total else 0.0
    recognized = sum(1 for row in rows if not row["error"])
    speed = recognized / elapsed if elapsed > 0 else 0.0

    print("=== OCR evaluation ===")
    print(f"Total samples:              {total}")
    print(f"Recognized samples:         {recognized}")
    print(f"Exact line matches:         {exact_ok}")
    print(f"Line accuracy:              {line_accuracy:.4f}")
    print(f"Average char accuracy:      {avg_char_accuracy:.4f}")
    print(f"Global char accuracy:       {normalized_char_accuracy:.4f}")
    print(f"Average Levenshtein:        {avg_levenshtein:.4f}")
    print(f"Total Levenshtein:          {total_lev}")
    print(f"Elapsed:                    {elapsed:.2f}s")
    print(f"Speed:                      {speed:.2f} img/s")
    print(f"CSV saved to:               {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FCN OCR on a Label Studio JSON export.")
    parser.add_argument("--json", required=True, help="Path to Label Studio export JSON.")
    parser.add_argument("--images", required=True, help="Folder with images.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--out", default="ocr_metrics.csv", help="Output CSV path.")
    parser.add_argument("--device", default=None, help="Device to use: cuda, cpu, or empty for auto.")
    parser.add_argument("--scale-x", type=float, default=0.0, help="Normalized horizontal inference scale.")
    parser.add_argument("--y-pad", type=float, default=0.0, help="Normalized vertical inference padding/crop.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to evaluate.")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N recognized images; 0 disables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        json_path=Path(args.json),
        images_dir=Path(args.images),
        checkpoint_path=Path(args.checkpoint),
        output_csv=Path(args.out),
        device=args.device,
        scale_x=args.scale_x,
        y_pad=args.y_pad,
        batch_size=args.batch_size,
        limit=args.limit,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
