from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fcn_tasks import (
    BASELINE_DETECTION_TASK,
    FCN_OCR_TASK,
    VERTICAL_SEGMENTATION_TASK,
    normalize_task_name,
)

from .chunk_metadata import (
    ChunkManifestEntry,
    ChunkMetadata,
    load_chunk_metadata,
)
from .dataset import SingleLineDatasetConfig


def load_torch_chunk(path: Path) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)


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
        "fcn_ocr_targets": metadata.fcn_ocr_targets,
        "vertical_segmentation_targets": metadata.vertical_segmentation_targets,
        "baseline_detection_targets": metadata.baseline_detection_targets,
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

    if metadata.fcn_ocr_targets:
        fcn_ocr_targets = chunk["fcn_ocr_targets"]
        expected_shape = (manifest_entry.samples, metadata.image_width)
        if (
            not isinstance(fcn_ocr_targets, torch.Tensor)
            or tuple(fcn_ocr_targets.shape) != expected_shape
        ):
            raise ValueError(
                f"Chunk {path} fcn_ocr_targets must have shape {expected_shape}"
            )
        if fcn_ocr_targets.dtype != torch.int16:
            raise ValueError(
                f"Chunk {path} fcn_ocr_targets dtype must be int16, got {fcn_ocr_targets.dtype}"
            )
        if int(fcn_ocr_targets.min()) < 0 or int(fcn_ocr_targets.max()) >= len(
            metadata.alphabet
        ):
            raise ValueError(
                f"Chunk {path} fcn_ocr_targets contain class indices outside metadata alphabet"
            )

    if metadata.vertical_segmentation_targets:
        cuts = chunk["vertical_segmentation_targets"]
        expected_shape = (manifest_entry.samples, metadata.image_width)
        if not isinstance(cuts, torch.Tensor) or tuple(cuts.shape) != expected_shape:
            raise ValueError(
                f"Chunk {path} vertical_segmentation_targets must have shape {expected_shape}"
            )
        if cuts.dtype != torch.uint8:
            raise ValueError(
                f"Chunk {path} vertical_segmentation_targets dtype must be uint8, got {cuts.dtype}"
            )

    if metadata.baseline_detection_targets:
        baselines = chunk["baseline_detection_targets"]
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
                f"Chunk {path} baseline_detection_targets must have shape {expected_shape}"
            )
        if baselines.dtype != torch.uint8:
            raise ValueError(
                f"Chunk {path} baseline_detection_targets dtype must be uint8, got {baselines.dtype}"
            )


class ChunkedLineDataset(Dataset):
    """Reads pre-rendered OCR line chunks saved by generate_dataset.py."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        config: SingleLineDatasetConfig,
        task: str,
        cache_size: int = 2,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.cache_size = max(1, cache_size)
        self.config = config
        self.task = normalize_task_name(task)
        self.metadata = load_chunk_metadata(self.root_dir)
        self.metadata.require_task(self.task)
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

        if self.task == FCN_OCR_TASK:
            return self._make_fcn_ocr_target(image, chunk, local_idx)
        if self.task == VERTICAL_SEGMENTATION_TASK:
            return self._make_vertical_segmentation_target(image, chunk, local_idx)
        if self.task == BASELINE_DETECTION_TASK:
            return self._make_baseline_detection_target(image, chunk, local_idx)
        raise AssertionError(f"unreachable task: {self.task}")

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

    def _make_fcn_ocr_target(
        self,
        image: torch.Tensor,
        chunk: dict,
        local_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "fcn_ocr_targets" not in chunk:
            raise KeyError(
                "Chunk does not contain fcn_ocr_targets. Regenerate the dataset with "
                "save_fcn_ocr_targets: true in the generation config."
            )
        target = chunk["fcn_ocr_targets"][local_idx].long()
        if target.dim() != 1:
            raise ValueError(
                f"OCR target must have shape (W,), got {tuple(target.shape)}"
            )
        if target.size(0) != image.shape[-1]:
            raise ValueError(
                f"OCR target width {target.size(0)} does not match image width {image.shape[-1]}"
            )
        return image, target

    def _make_vertical_segmentation_target(
        self,
        image: torch.Tensor,
        chunk: dict,
        local_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "vertical_segmentation_targets" not in chunk:
            raise KeyError(
                "Chunk does not contain vertical_segmentation_targets. Regenerate the dataset with "
                "save_vertical_segmentation_targets: true in the generation config."
            )
        raw_target = chunk["vertical_segmentation_targets"][local_idx]
        target = raw_target.float()
        if raw_target.dtype == torch.uint8:
            target = target / 255.0
        if target.dim() != 1:
            raise ValueError(
                "vertical segmentation target must have shape (W,), "
                f"got {tuple(target.shape)}"
            )
        if target.size(0) != image.shape[-1]:
            raise ValueError(
                f"vertical segmentation target width {target.size(0)} does not "
                f"match image width {image.shape[-1]}"
            )
        return image, target.contiguous()

    def _make_baseline_detection_target(
        self,
        image: torch.Tensor,
        chunk: dict,
        local_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "baseline_detection_targets" not in chunk:
            raise KeyError(
                "Chunk does not contain baseline_detection_targets. Regenerate the dataset with "
                "save_baseline_detection_targets: true in the generation config."
            )
        raw_target = chunk["baseline_detection_targets"][local_idx]
        target = raw_target.float()
        if raw_target.dtype == torch.uint8:
            target = target / 255.0
        if target.dim() != 3 or target.size(0) != 2:
            raise ValueError(
                "baseline detection target must have shape (2, H, W), "
                f"got {tuple(target.shape)}"
            )
        if target.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                "baseline detection target shape "
                f"{tuple(target.shape[-2:])} does not match image shape "
                f"{tuple(image.shape[-2:])}"
            )
        return image, target.contiguous()

    def _normalize_text(self, text: str) -> str:
        space_char = self.config.space_char
        return space_char.join(part for part in text.split(space_char) if part)
