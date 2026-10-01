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


def crop_with_fill(
    image: Image.Image,
    box: tuple[int, int, int, int],
    fill: int | tuple[int, int, int],
) -> Image.Image:
    left, top, right, bottom = box
    width = max(1, right - left)
    height = max(1, bottom - top)
    output = Image.new(image.mode, (width, height), fill)
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
        return output

    paste_xy = (source_box[0] - left, source_box[1] - top)
    output.paste(image.crop(source_box), paste_xy)
    return output
