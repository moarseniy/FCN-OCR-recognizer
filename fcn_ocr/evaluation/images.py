from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PIL import Image


class RGBImageCache:
    """Bounded in-memory cache for source images reused across Optuna trials."""

    def __init__(self, max_megabytes: float = 512.0) -> None:
        if max_megabytes < 0.0:
            raise ValueError("image cache size must be >= 0 MB")
        self.max_bytes = int(max_megabytes * 1024 * 1024)
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._images: OrderedDict[Path, tuple[Image.Image, int]] = OrderedDict()

    def load(self, path: str | Path) -> Image.Image:
        resolved = Path(path).expanduser().resolve()
        cached = self._images.get(resolved)
        if cached is not None:
            self.hits += 1
            self._images.move_to_end(resolved)
            return cached[0]

        self.misses += 1
        with Image.open(resolved) as image_file:
            image = image_file.convert("RGB")
            image.load()

        size_bytes = image.width * image.height * 3
        if self.max_bytes <= 0 or size_bytes > self.max_bytes:
            return image

        while self._images and self.current_bytes + size_bytes > self.max_bytes:
            _, (_, removed_bytes) = self._images.popitem(last=False)
            self.current_bytes -= removed_bytes
            self.evictions += 1

        self._images[resolved] = (image, size_bytes)
        self.current_bytes += size_bytes
        return image

    def stats(self) -> dict[str, int | float]:
        return {
            "items": len(self._images),
            "megabytes": self.current_bytes / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


__all__ = ["RGBImageCache"]
