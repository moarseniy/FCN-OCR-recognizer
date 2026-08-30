from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Sampler, Subset

from fcn_synth_generator.chunk_dataset import ChunkedLineDataset

from .config import TrainingConfig


class RandomFixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, batch_count, seed=0):
        if len(dataset) <= 0:
            raise ValueError("dataset must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if batch_count < 1:
            raise ValueError("batch_count must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        dataset_size = len(self.dataset)
        for _ in range(self.batch_count):
            yield torch.randint(
                dataset_size,
                (self.batch_size,),
                generator=generator,
                dtype=torch.long,
            ).tolist()

    def __len__(self):
        return self.batch_count


class ChunkBatchSampler(Sampler):
    def __init__(
        self,
        subset,
        base_dataset,
        batch_size,
        drop_last,
        shuffle,
        seed=0,
        batch_count=None,
    ):
        self.subset = subset
        self.base_dataset = base_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.batch_count = batch_count
        self.epoch = 0
        self.groups = self._group_subset_positions_by_chunk()
        self.chunk_ids = list(self.groups)
        self.chunk_weights = torch.tensor(
            [len(self.groups[chunk_id]) for chunk_id in self.chunk_ids],
            dtype=torch.double,
        )
        if self.batch_count is not None and self.batch_count < 1:
            raise ValueError("batch_count must be >= 1")
        if not self.chunk_ids:
            raise ValueError("chunk batch sampler got an empty subset")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        if self.batch_count is not None:
            yield from self._iter_sampled_batches(generator)
            return

        chunk_ids = list(self.groups)
        if self.shuffle:
            permutation = torch.randperm(len(chunk_ids), generator=generator).tolist()
            chunk_ids = [chunk_ids[index] for index in permutation]

        for chunk_id in chunk_ids:
            positions = list(self.groups[chunk_id])
            if self.shuffle:
                permutation = torch.randperm(
                    len(positions), generator=generator
                ).tolist()
                positions = [positions[index] for index in permutation]

            for start in range(0, len(positions), self.batch_size):
                batch = positions[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self):
        if self.batch_count is not None:
            return self.batch_count

        total = 0
        for positions in self.groups.values():
            if self.drop_last:
                total += len(positions) // self.batch_size
            else:
                total += math.ceil(len(positions) / self.batch_size)
        return total

    def _group_subset_positions_by_chunk(self):
        groups = {}
        for subset_position in range(len(self.subset)):
            sample_index = self._sample_index(subset_position)
            chunk_id = self.base_dataset.chunk_index_for_sample(sample_index)
            groups.setdefault(chunk_id, []).append(subset_position)
        return groups

    def _iter_sampled_batches(self, generator):
        sampled_group_indices = torch.multinomial(
            self.chunk_weights,
            num_samples=self.batch_count,
            replacement=True,
            generator=generator,
        ).tolist()

        for group_index in sampled_group_indices:
            chunk_id = self.chunk_ids[group_index]
            positions = self.groups[chunk_id]
            if len(positions) >= self.batch_size:
                sampled_position_indices = torch.randperm(
                    len(positions),
                    generator=generator,
                )[: self.batch_size].tolist()
            else:
                sampled_position_indices = torch.randint(
                    len(positions),
                    (self.batch_size,),
                    generator=generator,
                    dtype=torch.long,
                ).tolist()
            yield [positions[index] for index in sampled_position_indices]

    def _sample_index(self, subset_position):
        if isinstance(self.subset, Subset):
            return int(self.subset.indices[subset_position])
        return subset_position


def make_data_loader(
    dataset,
    split_dataset,
    config: TrainingConfig,
    shuffle: bool,
    seed: int,
    batch_count: int | None = None,
):
    common_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.num_workers > 0:
        common_kwargs["prefetch_factor"] = config.prefetch_factor
        common_kwargs["persistent_workers"] = config.persistent_workers

    if isinstance(dataset, ChunkedLineDataset) and config.chunk_aware_batches:
        return DataLoader(
            split_dataset,
            batch_sampler=ChunkBatchSampler(
                split_dataset,
                dataset,
                config.batch_size,
                config.drop_last,
                shuffle,
                seed,
                batch_count=batch_count,
            ),
            **common_kwargs,
        )

    if batch_count is not None:
        return DataLoader(
            split_dataset,
            batch_sampler=RandomFixedBatchSampler(
                split_dataset,
                config.batch_size,
                batch_count,
                seed,
            ),
            **common_kwargs,
        )

    return DataLoader(
        split_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=config.drop_last,
        **common_kwargs,
    )


__all__ = ["ChunkBatchSampler", "RandomFixedBatchSampler", "make_data_loader"]
