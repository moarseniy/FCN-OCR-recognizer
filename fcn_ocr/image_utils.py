from __future__ import annotations

import numpy as np
from PIL import Image


def pil_fill_value(mode: str, fill: int) -> int | tuple[int, int, int]:
    value = max(0, min(255, int(fill)))
    if mode == "RGB":
        return (value, value, value)
    return value


def background_fill_value(
    image: Image.Image,
    fallback: int,
) -> int | tuple[int, int, int]:
    array = np.asarray(image)
    if array.size == 0:
        return pil_fill_value(image.mode, fallback)

    if array.ndim == 2:
        border = np.concatenate(
            (array[0, :], array[-1, :], array[:, 0], array[:, -1])
        )
        return int(np.median(border))

    if array.ndim == 3 and array.shape[2] >= 3:
        border = np.concatenate(
            (
                array[0, :, :],
                array[-1, :, :],
                array[:, 0, :],
                array[:, -1, :],
            ),
            axis=0,
        )
        values = np.median(border[:, :3], axis=0).round().astype(np.uint8).tolist()
        return tuple(int(value) for value in values[:3])

    return pil_fill_value(image.mode, fallback)
