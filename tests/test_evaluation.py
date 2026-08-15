from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fcn_ocr.evaluation.config import expand_evaluation_parameters, load_evaluation_yaml
from fcn_ocr.evaluation.metrics import compute_cut_metrics, compute_ocr_metrics
from fcn_ocr.evaluation.optuna import (
    require_float_parameter,
    select_compatible_trial,
)
from fcn_ocr.evaluation.samples import load_label_studio_samples


def _label_studio_task(task_id: int, image: str, text: str) -> dict:
    return {
        "id": task_id,
        "data": {"image": f"/data/local-files/?d=dataset/{image}"},
        "annotations": [{"result": [{"value": {"text": [text]}}]}],
    }


def test_label_studio_limit_counts_only_available_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "present.png").write_bytes(b"image contents are not read here")
    json_path = tmp_path / "labels.json"
    json_path.write_text(
        json.dumps(
            [
                _label_studio_task(1, "missing.png", "MISSING"),
                _label_studio_task(2, "present.png", " ABC "),
            ]
        ),
        encoding="utf-8",
    )

    samples = load_label_studio_samples(json_path, images_dir, limit=1)

    assert [(sample.task_id, sample.image_name, sample.text) for sample in samples] == [
        (2, "present.png", "ABC")
    ]


def test_ocr_metrics_preserve_average_and_global_accuracy_semantics() -> None:
    rows = [
        {"gt": "AB", "pred": "A", "error": ""},
        {"gt": "CDEF", "pred": "CDEF", "error": ""},
    ]

    metrics = compute_ocr_metrics(rows, elapsed=2.0)

    assert metrics["line_accuracy"] == pytest.approx(0.5)
    assert metrics["average_char_accuracy"] == pytest.approx(0.75)
    assert metrics["global_char_accuracy"] == pytest.approx(5 / 6)
    assert metrics["speed"] == pytest.approx(1.0)
    assert rows[0]["levenshtein"] == 1


def test_cut_metrics_match_manual_lines_with_tolerance() -> None:
    rows = [
        {
            "gt_len": 2,
            "pred_len": 2,
            "gt_cuts": [1.0, 5.0, 9.0],
            "pred_cuts": [1.5, 5.5, 12.0],
            "error": "",
        }
    ]

    metrics = compute_cut_metrics(rows, elapsed=1.0, cut_tolerance_px=1.0)

    assert metrics["matched_cuts"] == 2
    assert metrics["cut_precision"] == pytest.approx(2 / 3)
    assert metrics["cut_recall"] == pytest.approx(2 / 3)
    assert metrics["cut_mae_px"] == pytest.approx(0.5)


def test_evaluation_parameters_expand_fixed_and_range_values() -> None:
    expanded = expand_evaluation_parameters(
        {"parameters": {"x_pad": [0.0, 0.1], "cut_min_width": 5}},
        valid_fields={
            "x_pad",
            "optuna_x_pad_min",
            "optuna_x_pad_max",
            "cut_min_width",
            "optuna_cut_min_width_min",
            "optuna_cut_min_width_max",
        },
    )

    assert expanded == {
        "optuna_x_pad_min": 0.0,
        "optuna_x_pad_max": 0.1,
        "cut_min_width": 5,
    }


def test_evaluation_yaml_rejects_duplicate_keys() -> None:
    with pytest.raises(Exception, match="duplicate key"):
        load_evaluation_yaml(io.StringIO("threshold: 0.2\nthreshold: 0.4\n"))


class _Trial:
    def __init__(self, value: float, params: dict[str, object]) -> None:
        self.value = value
        self.params = params
        self.state = SimpleNamespace(is_finished=lambda: True)

    def suggest_float(self, name: str, minimum: float, maximum: float) -> float:
        assert name == "x_pad"
        assert minimum == 0.0
        assert maximum == 0.1
        return 0.04


def test_optuna_range_is_suggested_while_fixed_value_is_preserved() -> None:
    trial = _Trial(0.0, {})

    assert require_float_parameter(trial, "x_pad", 0.01, 0.0, 0.1) == pytest.approx(0.04)
    assert require_float_parameter(trial, "x_pad", 0.01, None, None) == pytest.approx(0.01)


def test_optuna_ignores_stale_trial_with_incompatible_fixed_parameters() -> None:
    stale = _Trial(0.99, {"scale_x": -0.5, "x_pad": 0.2})
    current = _Trial(0.90, {"scale_x": -0.2, "x_pad": 0.01})
    study = SimpleNamespace(trials=[stale, current])

    selected = select_compatible_trial(
        study,
        direction="maximize",
        required_params={"scale_x"},
        fixed_params={"x_pad": 0.01},
    )

    assert selected is current
