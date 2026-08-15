from __future__ import annotations

from typing import Any

from .geometry import match_sorted_points


OCR_RESULT_FIELDS = (
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
)

CUT_RESULT_FIELDS = (
    "task_id",
    "image",
    "gt",
    "gt_len",
    "pred_len",
    "cut_count",
    "length_error",
    "abs_length_error",
    "cuts",
    "gt_cuts",
    "pred_cuts",
    "matched_cuts",
    "false_positive_cuts",
    "false_negative_cuts",
    "cut_mae_px",
    "error",
)


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        return levenshtein(right, left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[len(right)]


def char_accuracy(expected: str, predicted: str) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    return max(0.0, 1.0 - levenshtein(expected, predicted) / len(expected))


def compute_ocr_metrics(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    exact_matches = 0
    total_distance = 0
    total_expected_chars = 0
    total_char_accuracy = 0.0

    for row in rows:
        expected = row["gt"]
        predicted = row["pred"]
        distance = levenshtein(expected, predicted)
        sample_accuracy = char_accuracy(expected, predicted)
        is_exact = expected == predicted

        row["exact_match"] = int(is_exact)
        row["char_accuracy"] = round(sample_accuracy, 6)
        row["levenshtein"] = distance
        row["gt_len"] = len(expected)
        row["pred_len"] = len(predicted)

        exact_matches += int(is_exact)
        total_distance += distance
        total_expected_chars += len(expected)
        total_char_accuracy += sample_accuracy

    total = len(rows)
    recognized = sum(1 for row in rows if not row["error"])
    return {
        "total_samples": total,
        "recognized_samples": recognized,
        "exact_matches": exact_matches,
        "line_accuracy": exact_matches / total if total else 0.0,
        "average_char_accuracy": total_char_accuracy / total if total else 0.0,
        "global_char_accuracy": (
            max(0.0, 1.0 - total_distance / total_expected_chars)
            if total_expected_chars
            else 0.0
        ),
        "average_levenshtein": total_distance / total if total else 0.0,
        "total_levenshtein": total_distance,
        "elapsed": elapsed,
        "speed": recognized / elapsed if elapsed > 0 else 0.0,
    }


def compute_cut_metrics(
    rows: list[dict[str, Any]],
    elapsed: float,
    cut_tolerance_px: float,
) -> dict[str, Any]:
    evaluated = sum(1 for row in rows if not row["error"])
    exact = 0
    total_abs_error = 0
    total_signed_error = 0
    total_gt_len = 0
    expected_cuts = 0
    predicted_cuts = 0
    matched_cuts = 0
    total_cut_error = 0.0
    manual_samples = 0

    for row in rows:
        if row["error"]:
            row["length_error"] = -row["gt_len"]
            row["abs_length_error"] = abs(row["length_error"])
            continue

        row["length_error"] = row["pred_len"] - row["gt_len"]
        row["abs_length_error"] = abs(row["length_error"])
        exact += int(row["length_error"] == 0)
        total_abs_error += row["abs_length_error"]
        total_signed_error += row["length_error"]
        total_gt_len += row["gt_len"]

        gt_cuts = [float(value) for value in row.get("gt_cuts", [])]
        if gt_cuts:
            manual_samples += 1
            pred_cuts = [float(value) for value in row.get("pred_cuts", [])]
            matches = match_sorted_points(gt_cuts, pred_cuts, cut_tolerance_px)
            row["matched_cuts"] = len(matches)
            row["false_positive_cuts"] = len(pred_cuts) - len(matches)
            row["false_negative_cuts"] = len(gt_cuts) - len(matches)
            row["cut_mae_px"] = (
                sum(match.error for match in matches) / len(matches) if matches else 0.0
            )
            expected_cuts += len(gt_cuts)
            predicted_cuts += len(pred_cuts)
            matched_cuts += len(matches)
            total_cut_error += sum(match.error for match in matches)

    cut_precision = matched_cuts / predicted_cuts if predicted_cuts else 0.0
    cut_recall = matched_cuts / expected_cuts if expected_cuts else 0.0
    cut_f1 = (
        2.0 * cut_precision * cut_recall / (cut_precision + cut_recall)
        if cut_precision + cut_recall > 0.0
        else 0.0
    )
    total = len(rows)
    return {
        "total_samples": total,
        "evaluated_samples": evaluated,
        "exact_length_matches": exact,
        "length_accuracy": exact / evaluated if evaluated else 0.0,
        "average_abs_length_error": total_abs_error / evaluated if evaluated else 0.0,
        "total_abs_length_error": total_abs_error,
        "average_signed_length_error": total_signed_error / evaluated if evaluated else 0.0,
        "normalized_length_error": total_abs_error / total_gt_len if total_gt_len else 0.0,
        "manual_cut_samples": manual_samples,
        "expected_cuts": expected_cuts,
        "predicted_cuts": predicted_cuts,
        "matched_cuts": matched_cuts,
        "cut_precision": cut_precision,
        "cut_recall": cut_recall,
        "cut_f1": cut_f1,
        "cut_mae_px": total_cut_error / matched_cuts if matched_cuts else 0.0,
        "cut_tolerance_px": float(cut_tolerance_px),
        "elapsed": elapsed,
        "speed": evaluated / elapsed if elapsed > 0 else 0.0,
    }
