from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import SingleLineDatasetConfig


class ChunkedLineDataset(Dataset):
    """Reads pre-rendered OCR line chunks saved by materialize.py."""

    def __init__(self, root_dir: str | Path, cache_size: int = 2):
        self.root_dir = Path(root_dir)
        self.cache_size = max(1, cache_size)
        manifest_path = self.root_dir / "manifest.pt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Chunk manifest not found: {manifest_path}")

        self.manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        if self.manifest.get("format") != "fcn_ocr_line_chunks_v1":
            raise ValueError(f"Unsupported chunk dataset format in {manifest_path}")
        self.chunks = self.manifest["chunks"]
        self.total_samples = int(self.manifest["total_samples"])
        self.config = SingleLineDatasetConfig.model_validate(self.manifest["config"])
        self.chunk_ends = []
        total = 0
        for chunk in self.chunks:
            total += int(chunk["samples"])
            self.chunk_ends.append(total)

        self._chunk_cache = OrderedDict()

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.total_samples
        if index < 0 or index >= self.total_samples:
            raise IndexError(index)

        chunk_idx = bisect_right(self.chunk_ends, index)
        chunk_start = 0 if chunk_idx == 0 else self.chunk_ends[chunk_idx - 1]
        local_idx = index - chunk_start
        chunk = self._load_chunk(chunk_idx)

        image = chunk["images"][local_idx].float() / 255.0
        target = chunk["targets"][local_idx].long()
        length = chunk["lengths"][local_idx].long()
        return image, target, length

    def _load_chunk(self, chunk_idx: int) -> dict:
        if chunk_idx in self._chunk_cache:
            self._chunk_cache.move_to_end(chunk_idx)
            return self._chunk_cache[chunk_idx]

        path = self.root_dir / self.chunks[chunk_idx]["file"]
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        self._chunk_cache[chunk_idx] = chunk
        self._chunk_cache.move_to_end(chunk_idx)
        while len(self._chunk_cache) > self.cache_size:
            self._chunk_cache.popitem(last=False)
        return chunk
