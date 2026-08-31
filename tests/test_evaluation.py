from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import torch

from fcn_ocr import VerticalSegmenter
from fcn_ocr.evaluation.config import (
    evaluation_parameter_range,
    expand_evaluation_parameters,
    load_evaluation_yaml,
)
from fcn_ocr.evaluation.images import RGBImageCache
from fcn_ocr.evaluation import fcn_ocr_runner
from fcn_ocr.evaluation import vertical_segmentation_runner
from fcn_ocr.evaluation.metrics import compute_cut_metrics, compute_ocr_metrics
from fcn_ocr.evaluation.optuna import (
    require_float_parameter,
    validate_study_contract,
)
from fcn_ocr.evaluation.samples import load_label_studio_samples
from fcn_ocr.evaluation.vertical_segmentation_runner import (
    SegmentationInference,
    postprocess_segment_inferences,
)


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


def test_fixed_evaluation_parameter_disables_its_optuna_range() -> None:
    args = SimpleNamespace(
        evaluation_parameter_modes={"x_pad": "fixed", "scale_x": "range"},
        optuna_x_pad_min=0.0,
        optuna_x_pad_max=0.1,
        optuna_scale_x_min=-0.2,
        optuna_scale_x_max=0.2,
    )

    assert evaluation_parameter_range(args, "x_pad") == (None, None)
    assert evaluation_parameter_range(args, "scale_x") == (-0.2, 0.2)


def test_rgb_image_cache_reuses_decoded_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("L", (8, 4), color=127).save(image_path)
    cache = RGBImageCache(max_megabytes=1.0)

    first = cache.load(image_path)
    second = cache.load(image_path)

    assert first is second
    assert first.mode == "RGB"
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 1


class _Study:
    def __init__(self, *, attrs: dict | None = None, trials: list | None = None) -> None:
        self.user_attrs = attrs or {}
        self.trials = trials or []

    def set_user_attr(self, name: str, value: object) -> None:
        self.user_attrs[name] = value


def test_persistent_optuna_study_rejects_a_different_evaluation_contract() -> None:
    study = _Study()
    validate_study_contract(study, {"checkpoint": "a.pth", "parameters": {"x": [0, 1]}})

    validate_study_contract(study, {"checkpoint": "a.pth", "parameters": {"x": [0, 1]}})
    with pytest.raises(RuntimeError, match="different checkpoint, dataset, metric, or search space"):
        validate_study_contract(
            study,
            {"checkpoint": "b.pth", "parameters": {"x": [0, 1]}},
        )


def test_vertical_segmentation_logits_can_be_reused_for_postprocessing_trials() -> None:
    segmenter = object.__new__(VerticalSegmenter)
    segmenter.cut_threshold = 0.9
    segmenter.cut_min_width = 1
    segmenter.cut_max_width = 0
    segmenter.cut_smooth_radius = 0
    inference = SegmentationInference(
        logits=torch.tensor([[[-10.0, 4.0, -10.0, 3.0, -10.0]]]),
        input_shape=(1, 1, 2, 5),
        output_length=5,
        source_x_map=np.tile(np.arange(5, dtype=np.float64), (2, 1)),
    )

    loose, loose_errors = postprocess_segment_inferences(segmenter, {0: inference})
    segmenter.cut_threshold = 0.99
    strict, strict_errors = postprocess_segment_inferences(segmenter, {0: inference})

    assert loose_errors == {0: ""}
    assert strict_errors == {0: ""}
    assert loose[0]["cut_count"] == 2
    assert strict[0]["cut_count"] == 0


def test_ocr_pipeline_pool_loads_models_once_per_study(monkeypatch) -> None:
    created: list[object] = []
    configured: list[tuple[object, object]] = []

    class FakePipeline:
        def __init__(self, config, *, verbose: bool) -> None:
            self.config = config
            self.verbose = verbose
            created.append(self)

    monkeypatch.setattr(fcn_ocr_runner, "FCNPipeline", FakePipeline)
    monkeypatch.setattr(
        fcn_ocr_runner,
        "configure_evaluation_pipeline",
        lambda pipeline, config: configured.append((pipeline, config)),
    )
    pool = fcn_ocr_runner.OCRPipelinePool()
    first_config = object()
    second_config = object()

    first = pool.acquire(first_config, verbose=False)
    second = pool.acquire(second_config, verbose=False)

    assert first is second
    assert pool.loads == 1
    assert created == [first]
    assert configured == [(first, second_config)]


def test_segment_batch_does_not_treat_empty_error_strings_as_failures(monkeypatch) -> None:
    prediction = {7: {"pred_len": 1}}
    monkeypatch.setattr(
        vertical_segmentation_runner,
        "infer_segment_batch",
        lambda *args, **kwargs: {7: object()},
    )
    monkeypatch.setattr(
        vertical_segmentation_runner,
        "postprocess_segment_inferences",
        lambda *args, **kwargs: (prediction, {7: ""}),
    )

    result = vertical_segmentation_runner.segment_batch(object(), [(7, Path("sample.png"))])

    assert result == prediction
