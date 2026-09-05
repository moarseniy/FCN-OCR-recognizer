from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch

from fcn_synth_generator.dataset import SingleLineDataset, SingleLineDatasetConfig
from fcn_synth_generator.generate_dataset import (
    build_metadata,
    generate_chunk_worker,
    generate_chunks_sequential,
    worker_config_data,
)


def make_dataset(**overrides) -> SingleLineDataset:
    data = dict(
        task="fcn_ocr",
        alphabet=" ABCDEFG",
        channels=1,
        image_height=16,
        image_width=32,
        crop_stride=32,
    )
    data.update(overrides)
    config = SingleLineDatasetConfig(**data)
    dataset = SingleLineDataset.__new__(SingleLineDataset)
    dataset.config = config
    dataset.alphabet = config.alphabet
    dataset.char_to_index = {char: i for i, char in enumerate(config.alphabet)}
    dataset.crop_statistics = Counter()
    return dataset


def reference_candidates(dataset, spans, ink_spans, line_width):
    cfg = dataset.config
    candidates, touching = [], []
    for left in range(line_width - cfg.image_width + 1):
        cropped = dataset._crop_spans(spans, ink_spans, left, left + cfg.image_width)
        if cropped is None:
            continue
        logical, ink = cropped
        if not cfg.min_crop_text_length <= len(logical) <= cfg.max_crop_text_length:
            continue
        gaps = [
            b[1] - a[2]
            for a, b in zip(ink, ink[1:])
            if a[0] != cfg.space_char and b[0] != cfg.space_char
        ]
        min_gap = min(gaps, default=float("inf"))
        if cfg.ink_spacing_enabled and min_gap < cfg.ink_spacing_min_char_gap_px:
            continue
        candidates.append(left)
        touching.append(cfg.ink_spacing_enabled and min_gap <= cfg.ink_spacing_touch_gap_px)
    return candidates, touching


@pytest.mark.parametrize("task", ["fcn_ocr", "vertical_segmentation", "baseline_detection"])
def test_planned_crops_recover_line_instead_of_only_a_grid_survivor(task):
    dataset = make_dataset(task=task)
    spans = [(char, 20.0 + i * 24, 44.0 + i * 24) for i, char in enumerate("ABCDE")]
    pixels = np.tile(np.arange(160, dtype=np.uint8), (16, 1))
    source = Image.fromarray(pixels)

    grid_survivors = [
        x for x in range(0, 129, 32)
        if dataset._crop_spans(spans, spans, x, x + 32) is not None
    ]
    assert grid_survivors == [64]
    samples = dataset._slice_line_image(source, spans, spans, 3, 12, None, random.Random(7))

    assert [sample.text for sample in samples] == list("ABCDE")
    origins = [sample.crop_left for sample in samples]
    assert origins == sorted(set(origins))
    assert dataset.crop_statistics["shifted_crops"] == 4
    for nominal, sample in zip(range(0, 129, 32), samples):
        assert abs(sample.crop_left - nominal) <= 16
        assert sample.source_width == 160
        assert sample.image.shape == (1, 16, 32)
        expected_pixels = pixels[:, sample.crop_left:sample.crop_left + 32]
        np.testing.assert_array_equal((sample.image[0] * 255).round().numpy(), expected_pixels)
        cropped, ink = dataset._crop_spans(spans, spans, sample.crop_left, sample.crop_left + 32)
        if task == "fcn_ocr":
            expected = dataset._encode_fcn_ocr_targets(cropped, 32, ink_spans=ink)
        elif task == "vertical_segmentation":
            expected = dataset._encode_vertical_segmentation_target(ink, 32)
        else:
            expected = dataset._encode_baseline_detection_target(
                3, 12, 16, 32,
                x_start=min(start for _, start, _ in cropped),
                x_end=max(end for _, _, end in cropped),
            )
        assert torch.equal(sample.target, expected)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("visible,fragment", [(0.75, 0.25), (1.0, 0.0), (0.5, 0.5)])
@pytest.mark.parametrize("spacing", [False, True])
def test_candidate_mask_matches_exhaustive_label_and_ink_checks(seed, visible, fragment, spacing):
    rng = random.Random(seed)
    dataset = make_dataset(
        edge_char_min_visible_ratio=visible,
        edge_fragment_max_visible_ratio=fragment,
        min_crop_text_length=2,
        max_crop_text_length=5,
        ink_spacing_enabled=spacing,
        ink_spacing_min_char_gap_px=-0.3,
    )
    spans, ink_spans = [], []
    x = -4.25
    for char in " AAB  C D EFG ":
        width = rng.uniform(2, 18)
        spans.append((char, x, x + width))
        ink_spans.append((char, x + rng.uniform(-1, 3), x + width - rng.uniform(0, 3)))
        x += width + rng.uniform(-0.5, 2)
    line_width = int(x) + 32

    expected, expected_touching = reference_candidates(dataset, spans, ink_spans, line_width)
    positions, touching = dataset._crop_candidate_positions(spans, ink_spans, line_width)

    assert positions.tolist() == expected
    assert touching.tolist() == expected_touching


def test_touch_acceptance_is_drawn_once_per_selected_window():
    dataset = make_dataset(ink_spacing_enabled=True, ink_spacing_touch_probability=0.25)
    spans = [("A", float(x), float(x + 8)) for x in range(0, 160, 8)]

    class TrialRandom(random.Random):
        calls = 0

        def random(self):
            self.calls += 1
            assert self.calls <= 5, "touch acceptance was retried for alternative positions"
            return 0.9 if self.calls % 2 else 0.1

    rng = TrialRandom()
    samples = dataset._slice_line_image(
        Image.new("L", (160, 16)), spans, spans, 3, 12, None, rng
    )

    assert rng.calls == 5
    assert [sample.crop_left for sample in samples] == [32, 96]
    assert dataset.crop_statistics["touch_rejected"] == 3


@pytest.mark.parametrize("probability,expected_count", [(0.0, 0), (1.0, 5)])
def test_touch_probability_extremes(probability, expected_count):
    dataset = make_dataset(ink_spacing_enabled=True, ink_spacing_touch_probability=probability)
    spans = [("A", float(x), float(x + 8)) for x in range(0, 160, 8)]
    samples = dataset._slice_line_image(
        Image.new("L", (160, 16)), spans, spans, 3, 12, None, random.Random(8)
    )
    assert len(samples) == expected_count


def test_impossible_windows_are_reported_without_weakening_constraints():
    dataset = make_dataset(edge_char_min_visible_ratio=1.0, edge_fragment_max_visible_ratio=0.0)
    spans = [("A", 0.0, 160.0)]
    samples = dataset._slice_line_image(
        Image.new("L", (160, 16)), spans, spans, 3, 12, None, random.Random(8)
    )

    assert samples == []
    assert dataset.crop_statistics["unavailable_slots"] == 5


def test_explicit_and_offline_generation_share_crops_and_seed(monkeypatch):
    config = SingleLineDatasetConfig(
        task="fcn_ocr", alphabet=" ABCDEFG", channels=1,
        image_height=48, image_width=64, crop_stride=48, seed=7,
        font_check=False,
        font_paths=[str(Path("fcn_synth_generator/fonts/DejaVuSerif.ttf").resolve())],
    )
    explicit = SingleLineDataset(config)
    offline = SingleLineDataset(config)
    text = "ABCDEFG ABCDEFG ABCDEFG"
    monkeypatch.setattr(offline, "_make_line_text", lambda rng: text)
    crops = explicit.generate_crops(random.Random(config.seed), text=text)
    offline.config.samples = len(crops)
    generated = list(offline.iter_generated_samples())

    assert len(crops) > 1
    assert len(generated) == len(crops)
    for a, b in zip(crops, generated):
        assert (a.text, a.crop_left, a.source_width) == (b.text, b.crop_left, b.source_width)
        assert torch.equal(a.image, b.image)
        assert torch.equal(a.target, b.target)


@pytest.mark.parametrize("per_chunk_worker", [False, True])
@pytest.mark.parametrize("task", ["fcn_ocr", "vertical_segmentation", "baseline_detection"])
def test_crop_statistics_reach_writer_without_changing_chunk_payload(tmp_path, per_chunk_worker, task):
    config = SingleLineDatasetConfig(
        task=task, alphabet=" ABCDEFG", channels=1,
        image_height=48, image_width=64, crop_stride=48,
        samples=6, chunk_size=3, seed=7,
        font_check=False,
        font_paths=[str(Path("fcn_synth_generator/fonts/DejaVuSerif.ttf").resolve())],
    )
    dataset = SingleLineDataset(config)
    if per_chunk_worker:
        chunks = [
            generate_chunk_worker({
                "config": worker_config_data(config, dataset.font_paths, [], 3, 7 + i * 3),
                "sample_count": 3,
                "output_dir": str(tmp_path),
                "chunk_idx": i,
            })
            for i in range(2)
        ]
    else:
        chunks = generate_chunks_sequential(dataset, tmp_path, config.samples, config.chunk_size)

    statistics = Counter()
    for chunk in chunks:
        statistics.update(chunk["crop_statistics"])
        payload = torch.load(tmp_path / chunk["file"], weights_only=True)
        assert set(payload) == {"images", "texts", "targets"}
        assert payload["images"].shape == (3, 1, 48, 64)
        assert payload["images"].dtype == torch.uint8
        expected_shape = (3, 2, 48, 64) if task == "baseline_detection" else (3, 64)
        assert payload["targets"].shape == expected_shape
    assert statistics["accepted_crops"] >= config.samples
    assert statistics["planned_crops"] == (
        statistics["accepted_crops"] + statistics["unavailable_slots"] + statistics["touch_rejected"]
    )
    assert build_metadata(config, chunks).samples == 6
