from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .chunk_metadata import (
    ChunkManifestEntry,
    ChunkMetadata,
    load_chunk_metadata,
)
from .dataset import SingleLineDatasetConfig


def load_torch_chunk(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def validate_chunk_payload(
    chunk: dict[str, Any],
    path: Path,
    metadata: ChunkMetadata,
    manifest_entry: ChunkManifestEntry,
) -> None:
    if not isinstance(chunk, dict):
        raise TypeError(
            f"Chunk {path} must contain a mapping, got {type(chunk).__name__}"
        )

    expected_keys = {"images", "texts"}
    target_keys = {
        "dense_targets": metadata.dense_targets,
        "cut_projection_targets": metadata.cut_projection_targets,
        "baseline_targets": metadata.baseline_targets,
    }
    expected_keys.update(key for key, present in target_keys.items() if present)
    missing = sorted(expected_keys - set(chunk))
    unexpected = sorted(set(chunk) - expected_keys)
    if missing:
        raise KeyError(f"Chunk {path} is missing contract keys: {missing}")
    if unexpected:
        raise KeyError(
            f"Chunk {path} contains keys absent from metadata contract: {unexpected}"
        )

    images = chunk["images"]
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Chunk {path} images must be a torch.Tensor")
    expected_image_shape = (
        manifest_entry.samples,
        metadata.channels,
        metadata.image_height,
        metadata.image_width,
    )
    if tuple(images.shape) != expected_image_shape:
        raise ValueError(
            f"Chunk {path} image shape {tuple(images.shape)} does not match "
            f"metadata {expected_image_shape}"
        )
    if images.dtype != torch.uint8:
        raise ValueError(f"Chunk {path} images dtype must be uint8, got {images.dtype}")

    texts = chunk["texts"]
    if not isinstance(texts, list) or len(texts) != manifest_entry.samples:
        raise ValueError(
            f"Chunk {path} texts must contain exactly {manifest_entry.samples} strings"
        )
    if any(not isinstance(text, str) for text in texts):
        raise TypeError(f"Chunk {path} texts must contain only strings")

    if metadata.dense_targets:
        dense = chunk["dense_targets"]
        expected_shape = (manifest_entry.samples, metadata.image_width)
        if not isinstance(dense, torch.Tensor) or tuple(dense.shape) != expected_shape:
            raise ValueError(
                f"Chunk {path} dense_targets must have shape {expected_shape}"
            )
        if dense.dtype != torch.int16:
            raise ValueError(
                f"Chunk {path} dense_targets dtype must be int16, got {dense.dtype}"
            )
        if int(dense.min()) < 0 or int(dense.max()) >= len(metadata.alphabet):
            raise ValueError(
                f"Chunk {path} dense_targets contain class indices outside metadata alphabet"
            )

    if metadata.cut_projection_targets:
        cuts = chunk["cut_projection_targets"]
        expected_shape = (manifest_entry.samples, metadata.image_width)
        if not isinstance(cuts, torch.Tensor) or tuple(cuts.shape) != expected_shape:
            raise ValueError(
                f"Chunk {path} cut_projection_targets must have shape {expected_shape}"
            )
        if cuts.dtype != torch.uint8:
            raise ValueError(
                f"Chunk {path} cut_projection_targets dtype must be uint8, got {cuts.dtype}"
            )

    if metadata.baseline_targets:
        baselines = chunk["baseline_targets"]
        expected_shape = (
            manifest_entry.samples,
            2,
            metadata.image_height,
            metadata.image_width,
        )
        if (
            not isinstance(baselines, torch.Tensor)
            or tuple(baselines.shape) != expected_shape
        ):
            raise ValueError(
                f"Chunk {path} baseline_targets must have shape {expected_shape}"
            )
        if baselines.dtype != torch.uint8:
            raise ValueError(
                f"Chunk {path} baseline_targets dtype must be uint8, got {baselines.dtype}"
            )


class ChunkedLineDataset(Dataset):
    """Reads pre-rendered OCR line chunks saved by generate_dataset.py."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        config: SingleLineDatasetConfig,
        cache_size: int = 2,
        target_format: str = "dense_symbols",
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.cache_size = max(1, cache_size)
        self.config = config
        self.target_format = target_format
        if self.target_format not in {
            "dense_symbols",
            "cut_projection",
            "baseline_heatmap",
        }:
            raise ValueError(
                "target_format must be 'dense_symbols', 'cut_projection', or 'baseline_heatmap'"
            )
        self.metadata = load_chunk_metadata(self.root_dir)
        self.metadata.require_target(self.target_format)
        manifest_files = {entry.file for entry in self.metadata.chunks}
        disk_files = {path.name for path in self.root_dir.glob("chunk_*.pt")}
        missing_files = sorted(manifest_files - disk_files)
        unexpected_files = sorted(disk_files - manifest_files)
        if missing_files or unexpected_files:
            raise ValueError(
                "Chunk files do not match metadata manifest: "
                f"missing={missing_files}, unexpected={unexpected_files}"
            )

        self.chunks = [entry.model_dump() for entry in self.metadata.chunks]
        self.chunk_ends = []
        total = 0
        for entry in self.metadata.chunks:
            total += entry.samples
            self.chunk_ends.append(total)
        self.total_samples = self.metadata.samples

        self._chunk_cache = OrderedDict()

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.total_samples
        if index < 0 or index >= self.total_samples:
            raise IndexError(index)

        chunk_idx = bisect_right(self.chunk_ends, index)
        chunk_start = 0 if chunk_idx == 0 else self.chunk_ends[chunk_idx - 1]
        local_idx = index - chunk_start
        chunk = self._load_chunk(chunk_idx)

        image = chunk["images"][local_idx]

        if self.target_format == "dense_symbols":
            return self._make_dense_symbol_target(image, chunk, local_idx)
        if self.target_format == "cut_projection":
            return self._make_cut_projection_target(image, chunk, local_idx)
        if self.target_format == "baseline_heatmap":
            return self._make_baseline_heatmap_target(image, chunk, local_idx)
        raise RuntimeError(f"unsupported target format: {self.target_format}")

    def iter_texts(self):
        for chunk_idx in range(len(self.chunks)):
            chunk = self._load_chunk(chunk_idx)
            if "texts" not in chunk:
                raise KeyError(
                    f"Chunk {self.chunks[chunk_idx]['file']} does not contain texts"
                )
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
        chunk = load_torch_chunk(path)
        validate_chunk_payload(
            chunk,
            path,
            self.metadata,
            self.metadata.chunks[chunk_idx],
        )
        self._chunk_cache[chunk_idx] = chunk
        self._chunk_cache.move_to_end(chunk_idx)
        while len(self._chunk_cache) > self.cache_size:
            self._chunk_cache.popitem(last=False)
        return chunk

    def _make_dense_symbol_target(
        self,
        image: torch.Tensor,
        chunk: dict,
        local_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "dense_targets" not in chunk:
            raise KeyError(
                "Chunk does not contain dense_targets. Regenerate the dataset with "
                "save_dense_targets: true in the generation config."
            )
        target = chunk["dense_targets"][local_idx].long()
        if target.dim() != 1:
            raise ValueError(
                f"dense target must have shape (W,), got {tuple(target.shape)}"
            )
        if target.size(0) != image.shape[-1]:
            raise ValueError(
                f"dense target width {target.size(0)} does not match image width {image.shape[-1]}"
            )
        return image, target

    def _make_cut_projection_target(
        self,
        image: torch.Tensor,
        chunk: dict,
        local_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "cut_projection_targets" not in chunk:
            raise KeyError(
                "Chunk does not contain cut_projection_targets. Regenerate the dataset with "
                "save_cut_projection_targets: true in the generation config."
            )
        raw_target = chunk["cut_projection_targets"][local_idx]
        target = raw_target.float()
        if raw_target.dtype == torch.uint8:
            target = target / 255.0
        if target.dim() != 1:
            raise ValueError(
                f"cut projection target must have shape (W,), got {tuple(target.shape)}"
            )
        if target.size(0) != image.shape[-1]:
            raise ValueError(
                f"cut projection target width {target.size(0)} does not match image width {image.shape[-1]}"
            )
        return image, target.contiguous()

    def _make_baseline_heatmap_target(
        self,
        image: torch.Tensor,
        chunk: dict,
        local_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "baseline_targets" not in chunk:
            raise KeyError(
                "Chunk does not contain baseline_targets. Regenerate the dataset with "
                "save_baseline_targets: true in the generation config."
            )
        raw_target = chunk["baseline_targets"][local_idx]
        target = raw_target.float()
        if raw_target.dtype == torch.uint8:
            target = target / 255.0
        if target.dim() != 3 or target.size(0) != 2:
            raise ValueError(
                f"baseline target must have shape (2, H, W), got {tuple(target.shape)}"
            )
        if target.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                f"baseline target shape {tuple(target.shape[-2:])} does not match image shape {tuple(image.shape[-2:])}"
            )
        return image, target.contiguous()

    def _normalize_text(self, text: str) -> str:
        space_char = self.config.space_char
        return space_char.join(part for part in text.split(space_char) if part)
