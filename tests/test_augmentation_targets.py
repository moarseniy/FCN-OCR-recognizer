from __future__ import annotations

import random

import pytest
import torch

from fcn_synth_generator.dataset import SingleLineDatasetConfig
from fcn_synth_generator.gpu_augmentations import GpuTextAugmenter


def _augmenter() -> GpuTextAugmenter:
    config = SingleLineDatasetConfig(
        alphabet=" AB",
        augmentation_probabilities={"x_pad": 1.0},
        augmentations={
            "x_pad": {
                "left_px": 2,
                "right_px": 1,
                "fill_mode": "constant",
                "resize_mode": "nearest",
            }
        },
    )
    return GpuTextAugmenter(config)


def test_x_pad_resizes_ocr_content_and_labels_new_edges_as_space() -> None:
    augmenter = _augmenter()
    image = torch.linspace(0.0, 1.0, 6).reshape(1, 1, 1, 6).repeat(1, 1, 4, 1)
    target = torch.tensor([[1, 1, 1, 2, 2, 2]])

    augmented_image, augmented_target, metadata = augmenter.augment_with_metadata(
        image,
        targets=target,
        task="fcn_ocr",
    )

    assert tuple(augmented_image.shape) == tuple(image.shape)
    assert augmented_target.tolist() == [[0, 0, 1, 1, 2, 0]]
    assert metadata == [
        [
            {
                "name": "x_pad",
                "params": {
                    "pad_left": 2,
                    "pad_right": 1,
                    "content_width": 3,
                    "fill_mode": "constant",
                    "pad_reference": "output",
                    "fillcolor": 255,
                },
            }
        ]
    ]


def test_x_pad_metadata_replays_the_same_geometry_on_another_target() -> None:
    augmenter = _augmenter()
    target = torch.tensor([[1, 1, 1, 2, 2, 2]])
    metadata = [
        [
            {
                "name": "x_pad",
                "params": {"pad_left": 2, "pad_right": 1},
            }
        ]
    ]

    replayed = augmenter.apply_metadata_to_targets(
        target,
        task="fcn_ocr",
        metadata=metadata,
    )

    assert replayed.tolist() == [[0, 0, 1, 1, 2, 0]]


def test_crop_y_replays_on_both_baseline_detection_channels() -> None:
    augmenter = _augmenter()
    target = torch.zeros((1, 2, 6, 4))
    target[:, 0, 1, :] = 1.0
    target[:, 1, 4, :] = 1.0
    metadata = [
        [
            {
                "name": "crop_y",
                "params": {"crop_top": 1, "crop_bottom": 1},
            }
        ]
    ]

    replayed = augmenter.apply_metadata_to_targets(
        target,
        task="baseline_detection",
        metadata=metadata,
    )

    top_peak_y = int(replayed[0, 0].argmax()) // replayed.size(-1)
    bottom_peak_y = int(replayed[0, 1].argmax()) // replayed.size(-1)
    assert tuple(replayed.shape) == tuple(target.shape)
    assert float(replayed[:, 0].sum()) > 0.0
    assert float(replayed[:, 1].sum()) > 0.0
    assert top_peak_y < bottom_peak_y


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("cycle_shift", {"max_x": 2, "max_y": 1}),
        ("preprocess_geometry", {"scale_x": 0.2, "y_pad": 0.1}),
        ("scale", {"factor_x": 1.2, "factor_y": 0.9}),
        ("rotate", {"max_degrees": 5.0}),
        ("projective", {"max_dx": 2.0, "max_dy": 1.0}),
        ("crop_x", {"left": 2, "right": 1}),
    ],
)
def test_sampled_geometric_metadata_replays_exactly_on_ocr_targets(
    name: str,
    params: dict[str, float | int],
) -> None:
    random.seed(7)
    torch.manual_seed(7)
    config = SingleLineDatasetConfig(
        alphabet=" AB",
        augmentation_probabilities={name: 1.0},
        augmentations={name: params},
    )
    augmenter = GpuTextAugmenter(config)
    image = torch.linspace(0.0, 1.0, 12).reshape(1, 1, 1, 12).repeat(1, 1, 8, 1)
    target = torch.tensor([[0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0]])

    _, augmented_target, metadata = augmenter.augment_with_metadata(
        image,
        targets=target,
        task="fcn_ocr",
    )
    replayed_target = augmenter.apply_metadata_to_targets(
        target,
        task="fcn_ocr",
        metadata=metadata,
    )

    assert metadata and metadata[0] and metadata[0][0]["name"] == name
    assert torch.equal(augmented_target, replayed_target)
    assert augmented_target.dtype == target.dtype
    assert tuple(augmented_target.shape) == tuple(target.shape)
