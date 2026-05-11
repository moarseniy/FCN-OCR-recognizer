from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ChunkedLineDataset(Dataset):
    """Reads pre-rendered OCR line chunks saved by materialize.py."""

    def __init__(self, root_dir: str | Path, cache_size: int = 2):
        self.root_dir = Path(root_dir)
        self.cache_size = max(1, cache_size)
        chunk_paths = sorted(self.root_dir.glob("chunk_*.pt"))
        if not chunk_paths:
            raise FileNotFoundError(f"No chunk_*.pt files found in {self.root_dir}")

        self.chunks = []
        self.chunk_ends = []
        total = 0
        for path in chunk_paths:
            sample_count = self._read_chunk_sample_count(path)
            self.chunks.append({"file": path.name, "samples": sample_count})
            total += sample_count
            self.chunk_ends.append(total)
        self.total_samples = total

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

        image = chunk["images"][local_idx]
        target = chunk["targets"][local_idx]
        length = chunk["lengths"][local_idx]
        return image, target, length

    def chunk_index_for_sample(self, index: int) -> int:
        if index < 0:
            index += self.total_samples
        if index < 0 or index >= self.total_samples:
            raise IndexError(index)
        return bisect_right(self.chunk_ends, index)

    def _load_chunk(self, chunk_idx: int) -> dict:
        if chunk_idx in self._chunk_cache:
            self._chunk_cache.move_to_end(chunk_idx)
            return self._chunk_cache[chunk_idx]

        path = self.root_dir / self.chunks[chunk_idx]["file"]
        chunk = self._load_torch_chunk(path)
        self._chunk_cache[chunk_idx] = chunk
        self._chunk_cache.move_to_end(chunk_idx)
        while len(self._chunk_cache) > self.cache_size:
            self._chunk_cache.popitem(last=False)
        return chunk

    def _read_chunk_sample_count(self, path: Path) -> int:
        chunk = self._load_torch_chunk(path)
        self._validate_chunk(chunk, path)
        return int(chunk["images"].shape[0])

    @staticmethod
    def _load_torch_chunk(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, TypeError):
            return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _validate_chunk(chunk: dict, path: Path) -> None:
        required_keys = {"images", "targets", "lengths"}
        missing = sorted(required_keys - set(chunk))
        if missing:
            raise KeyError(f"Chunk {path} is missing keys: {missing}")

        sample_count = chunk["images"].shape[0]
        if chunk["targets"].shape[0] != sample_count or chunk["lengths"].shape[0] != sample_count:
            raise ValueError(f"Chunk {path} has inconsistent first dimensions")
