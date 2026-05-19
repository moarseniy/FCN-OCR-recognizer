from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset
import yaml

try:
    from .dataset import SingleLineDatasetConfig
except ImportError:
    from dataset import SingleLineDatasetConfig


CHUNK_METADATA_FILENAME = "metadata.yaml"


def load_chunk_metadata(root_dir: str | Path) -> dict:
    metadata_path = Path(root_dir) / CHUNK_METADATA_FILENAME
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file) or {}

    if not isinstance(metadata, dict):
        raise ValueError(f"Chunk metadata must be a mapping: {metadata_path}")
    return metadata


class ChunkedLineDataset(Dataset):
    """Reads pre-rendered OCR line chunks saved by generate_dataset.py."""

    def __init__(
        self,
        root_dir: str | Path,
        cache_size: int = 2,
        config: SingleLineDatasetConfig | None = None,
        target_format: str = "text",
    ):
        self.root_dir = Path(root_dir)
        self.cache_size = max(1, cache_size)
        self.config = config
        self.target_format = target_format
        if self.target_format not in {"text", "dense_symbols", "binary_gaps"}:
            raise ValueError("target_format must be 'text', 'dense_symbols', or 'binary_gaps'")
        self.metadata = load_chunk_metadata(self.root_dir)
        self.char_to_index = {char: idx for idx, char in enumerate(config.alphabet)} if config else {}
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

        if self.config is None:
            raise RuntimeError("config is required to encode chunk texts into CTC targets")
        if self.target_format == "dense_symbols":
            return self._make_dense_symbol_target(image, chunk, local_idx)
        if self.target_format == "binary_gaps":
            return self._make_binary_gap_target(image, chunk, local_idx)
        return self._make_target_from_text(image, chunk["texts"][local_idx])

    def iter_texts(self):
        for chunk_idx in range(len(self.chunks)):
            chunk = self._load_chunk(chunk_idx)
            if "texts" not in chunk:
                raise KeyError(f"Chunk {self.chunks[chunk_idx]['file']} does not contain texts")
            for text in chunk["texts"]:
                yield self._normalize_text(text)

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
        required_keys = {"images", "texts"}
        missing = sorted(required_keys - set(chunk))
        if missing:
            raise KeyError(f"Chunk {path} is missing keys: {missing}")

        sample_count = chunk["images"].shape[0]
        if len(chunk["texts"]) != sample_count:
            raise ValueError(f"Chunk {path} has inconsistent first dimensions")
        if "dense_targets" in chunk and chunk["dense_targets"].shape[0] != sample_count:
            raise ValueError(f"Chunk {path} has inconsistent dense_targets first dimension")
        if "binary_gap_targets" in chunk and chunk["binary_gap_targets"].shape[0] != sample_count:
            raise ValueError(f"Chunk {path} has inconsistent binary_gap_targets first dimension")

    def _make_target_from_text(self, image: torch.Tensor, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config is None:
            raise RuntimeError("config is required to encode targets from text")
        text = self._normalize_text(text)

        missing = sorted(set(text) - set(self.config.alphabet))
        if missing:
            raise ValueError(f"text contains chars outside training alphabet: {missing}")
        if not text:
            raise ValueError("text must not be empty after space normalization")

        if len(text) > self.config.max_text_length:
            raise ValueError(
                f"text length {len(text)} exceeds max_text_length={self.config.max_text_length}: {text!r}"
            )
        target = torch.zeros(self.config.max_text_length, dtype=torch.long)
        if text:
            target[: len(text)] = torch.tensor([self.char_to_index[char] for char in text], dtype=torch.long)
        return image, target, torch.tensor(len(text), dtype=torch.long)

    def _make_dense_symbol_target(
        self,
        image: torch.Tensor,
        chunk: dict,
        local_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "dense_targets" not in chunk:
            raise KeyError(
                "Chunk does not contain dense_targets. Regenerate the dataset with "
                "save_dense_targets: true in the generation config."
            )
        target = chunk["dense_targets"][local_idx].long()
        if target.dim() != 1:
            raise ValueError(f"dense target must have shape (W,), got {tuple(target.shape)}")
        if target.size(0) != image.shape[-1]:
            raise ValueError(
                f"dense target width {target.size(0)} does not match image width {image.shape[-1]}"
            )
        return image, target, torch.tensor(-1, dtype=torch.long)

    def _make_binary_gap_target(
        self,
        image: torch.Tensor,
        chunk: dict,
        local_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "binary_gap_targets" not in chunk:
            raise KeyError(
                "Chunk does not contain binary_gap_targets. Regenerate the dataset with "
                "save_binary_gap_targets: true in the generation config."
            )
        target = chunk["binary_gap_targets"][local_idx].long()
        if target.dim() != 1:
            raise ValueError(f"binary gap target must have shape (W,), got {tuple(target.shape)}")
        if target.size(0) != image.shape[-1]:
            raise ValueError(
                f"binary gap target width {target.size(0)} does not match image width {image.shape[-1]}"
            )
        return image, target, torch.tensor(-1, dtype=torch.long)

    def _normalize_text(self, text: str) -> str:
        if self.config is None:
            return text
        space_char = self.config.space_char
        return space_char.join(part for part in text.split(space_char) if part)
