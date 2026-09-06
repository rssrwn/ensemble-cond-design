import math
import torch
import numpy as np
from typing import Optional

from torch.utils.data import (
    BatchSampler,
    SubsetRandomSampler,
    SequentialSampler,
    DistributedSampler
)


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        bucket_limits: list[int],
        lengths: list[int],
        batch_cost: float,
        bucket_costs: Optional[list[float]] = None,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = False,
        round_batch_to_8: Optional[bool] = False,
        seed: Optional[int] = None
    ):

        # Modern GPUs can be more efficient when data is provided as a multiple of 8 (for 16-bit training)
        self.round_batch_to_8 = round_batch_to_8
        self.drop_last = drop_last
        self.seed = torch.initial_seed() if seed is None else seed

        if bucket_costs is not None and len(bucket_costs) != len(bucket_limits):
            raise ValueError("The number of costs and buckets must be the same.")

        if max(lengths) > max(bucket_limits):
            raise ValueError("Largest length cannot be larger than largest bucket limit.")

        bucket_limits = sorted(bucket_limits)

        # Use a constant bucket cost by default
        bucket_costs = [1] * len(bucket_limits) if bucket_costs is None else bucket_costs
        bucket_batch_sizes = [self._round_batch_size(batch_cost / cost) for cost in bucket_costs]

        # Add indices to correct bucket based on seq length
        buckets = [[] for _ in range(len(bucket_limits))]
        for seq_idx, length in enumerate(lengths):
            for b_idx, limit in enumerate(bucket_limits):
                if limit >= length:
                    buckets[b_idx].append(seq_idx)
                    break

        # Create a batch sampler for each bucket
        samplers = []
        for b_idx, (idxs, batch_size) in enumerate(zip(buckets, bucket_batch_sizes)):
            if len(idxs) == 0:
                samplers.append(None)
                continue

            if shuffle:
                generator = torch.Generator("cpu").manual_seed(self.seed + b_idx)
                sampler = SubsetRandomSampler(idxs, generator=generator)
            else:
                sampler = SequentialSampler(idxs)

            batch_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
            samplers.append(batch_sampler)

        batches_per_bucket = [len(sampler) if sampler is not None else 0 for sampler in samplers]

        print("\n*** Statistics for Data Buckets ***")
        print("Items per bucket:", [len(idxs) for idxs in buckets])
        print("Bucket batch sizes:", bucket_batch_sizes)
        print("Batches per bucket:", batches_per_bucket)

        self.buckets = buckets
        self.samplers = samplers
        self.bucket_batch_sizes = bucket_batch_sizes
        self.batches_per_bucket = batches_per_bucket
        self.batch_idx_generator = np.random.default_rng(self.seed)

    def __len__(self):
        return sum(self.batches_per_bucket)

    def __iter__(self):
        iters = [iter(sampler) if sampler is not None else None for sampler in self.samplers]
        remaining_batches = self.batches_per_bucket[:]

        while sum(remaining_batches) > 0:
            weights = np.array(remaining_batches) / sum(remaining_batches)
            b_idx = self.batch_idx_generator.choice(len(remaining_batches), p=weights)
            batch = next(iters[b_idx])
            remaining_batches[b_idx] -= 1
            yield batch

    def _round_batch_size(self, batch_size):
        if not self.round_batch_to_8:
            bs = math.floor(batch_size)
        else:
            bs = 8 * round(batch_size / 8)

        bs = 1 if bs == 0 else bs
        return bs


class MultiDatasetSampler(BatchSampler):
    """Simple sampler for multiple datasets with a single flat batch size.

    Draws homogeneous batches (all items from one dataset). Interleaves datasets
    by randomly picking which dataset to draw from, weighted by remaining batches.
    """

    def __init__(self, datasets, batch_size, drop_last=False, seed=None):
        self.drop_last = drop_last
        self.seed = torch.initial_seed() if seed is None else seed

        all_idxs = []
        offset = 0
        for dataset in datasets:
            all_idxs.append([i + offset for i in range(len(dataset))])
            offset += len(dataset)

        samplers = []
        for ds_idx, idxs in enumerate(all_idxs):
            generator = torch.Generator("cpu").manual_seed(self.seed + ds_idx)
            sampler = SubsetRandomSampler(idxs, generator=generator)
            samplers.append(BatchSampler(sampler, batch_size, drop_last=drop_last))

        batches_per_dataset = [len(s) for s in samplers]

        print("\n*** Statistics for Multi-Dataset Sampler ***")
        print("Items per dataset:", [len(idxs) for idxs in all_idxs])
        print(f"Batch size: {batch_size}")
        print("Batches per dataset:", batches_per_dataset)

        self.samplers = samplers
        self.batches_per_dataset = batches_per_dataset
        self.batch_idx_generator = np.random.default_rng(self.seed)

    def __len__(self):
        return sum(self.batches_per_dataset)

    def __iter__(self):
        iters = [iter(s) for s in self.samplers]
        remaining = self.batches_per_dataset[:]

        while sum(remaining) > 0:
            weights = np.array(remaining, dtype=np.float64) / sum(remaining)
            ds_idx = self.batch_idx_generator.choice(len(remaining), p=weights)
            batch = next(iters[ds_idx])
            remaining[ds_idx] -= 1
            yield batch


class MultiDatasetBucketSampler(BatchSampler):
    """Sampler for multiple datasets with per-dataset bucketing.

    Each dataset gets its own bucket config. Batches are always homogeneous (all items from one
    dataset). Iteration randomly picks a bucket across all datasets, weighted by remaining batches.

    For datasets without bucketing (bucket_limits=None), all items go into a single flat bucket
    with batch size = batch_cost.
    """

    def __init__(
        self,
        datasets,
        batch_costs: list[int],
        bucket_limits: Optional[list[Optional[list[int]]]] = None,
        bucket_costs: Optional[list[Optional[list[float]]]] = None,
        drop_last: bool = False,
        seed: Optional[int] = None
    ):
        n = len(datasets)
        self.drop_last = drop_last
        self.seed = torch.initial_seed() if seed is None else seed

        if len(batch_costs) != n:
            raise ValueError(f"batch_costs length ({len(batch_costs)}) must match datasets length ({n})")

        bucket_limits = [None] * n if bucket_limits is None else bucket_limits
        bucket_costs = [None] * n if bucket_costs is None else bucket_costs

        # Build buckets for each dataset, with global index offsets
        all_buckets = []
        all_batch_sizes = []
        offset = 0

        for ds_idx, dataset in enumerate(datasets):
            lengths = dataset.lengths
            ds_bucket_limits = bucket_limits[ds_idx]
            ds_bucket_costs = bucket_costs[ds_idx]
            ds_batch_cost = batch_costs[ds_idx]

            if ds_bucket_limits is None:
                # Single flat bucket — all items, batch_size = batch_cost
                global_idxs = [i + offset for i in range(len(lengths))]
                all_buckets.append(global_idxs)
                all_batch_sizes.append(max(1, int(ds_batch_cost)))
            else:
                ds_bucket_limits = sorted(ds_bucket_limits)
                ds_bucket_costs = [1] * len(ds_bucket_limits) if ds_bucket_costs is None else ds_bucket_costs
                per_bucket_batch_sizes = [max(1, math.floor(ds_batch_cost / c)) for c in ds_bucket_costs]

                buckets = [[] for _ in range(len(ds_bucket_limits))]
                for seq_idx, length in enumerate(lengths):
                    for b_idx, limit in enumerate(ds_bucket_limits):
                        if limit >= length:
                            buckets[b_idx].append(seq_idx + offset)
                            break

                for b_idx in range(len(ds_bucket_limits)):
                    if len(buckets[b_idx]) > 0:
                        all_buckets.append(buckets[b_idx])
                        all_batch_sizes.append(per_bucket_batch_sizes[b_idx])

            offset += len(lengths)

        # Create a BatchSampler per bucket
        samplers = []
        for b_idx, (idxs, batch_size) in enumerate(zip(all_buckets, all_batch_sizes)):
            generator = torch.Generator("cpu").manual_seed(self.seed + b_idx)
            sampler = SubsetRandomSampler(idxs, generator=generator)
            samplers.append(BatchSampler(sampler, batch_size, drop_last=drop_last))

        batches_per_bucket = [len(s) for s in samplers]

        print("\n*** Statistics for Multi-Dataset Buckets ***")
        print("Items per bucket:", [len(b) for b in all_buckets])
        print("Bucket batch sizes:", all_batch_sizes)
        print("Batches per bucket:", batches_per_bucket)

        self.all_buckets = all_buckets
        self.samplers = samplers
        self.batches_per_bucket = batches_per_bucket
        self.batch_idx_generator = np.random.default_rng(self.seed)

    def __len__(self):
        return sum(self.batches_per_bucket)

    def __iter__(self):
        iters = [iter(s) for s in self.samplers]
        remaining = self.batches_per_bucket[:]

        while sum(remaining) > 0:
            weights = np.array(remaining, dtype=np.float64) / sum(remaining)
            b_idx = self.batch_idx_generator.choice(len(remaining), p=weights)
            batch = next(iters[b_idx])
            remaining[b_idx] -= 1
            yield batch


class DistributedBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        bucket_limits: list[int],
        lengths: list[int],
        batch_cost: float,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        bucket_costs: Optional[list[float]] = None,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = False,
        round_batch_to_8: Optional[bool] = False,
        seed: Optional[int] = None
    ):

        # Modern GPUs can be more efficient when data is provided as a multiple of 8 (for 16-bit training)
        self.round_batch_to_8 = round_batch_to_8
        self.drop_last = drop_last
        self.seed = torch.initial_seed() if seed is None else seed

        if bucket_costs is not None and len(bucket_costs) != len(bucket_limits):
            raise ValueError("The number of costs and buckets must be the same.")

        if max(lengths) > max(bucket_limits):
            raise ValueError("Largest length cannot be larger than largest bucket limit.")

        bucket_limits = sorted(bucket_limits)

        # Use a constant bucket cost by default
        bucket_costs = [1] * len(bucket_limits) if bucket_costs is None else bucket_costs
        bucket_batch_sizes = [self._round_batch_size(batch_cost / cost) for cost in bucket_costs]

        # Add indices to correct bucket based on seq length
        buckets = [[] for _ in range(len(bucket_limits))]
        for seq_idx, length in enumerate(lengths):
            for b_idx, limit in enumerate(bucket_limits):
                if limit >= length:
                    buckets[b_idx].append(seq_idx)
                    break

        # Create a batch sampler for each bucket
        samplers = []
        for b_idx, (idxs, batch_size) in enumerate(zip(buckets, bucket_batch_sizes)):
            if len(idxs) == 0:
                samplers.append(None)
                continue

            sampler = DistributedSampler(
                idxs,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=(self.seed + b_idx),
                drop_last=drop_last
            )
            batch_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
            samplers.append(batch_sampler)

        batches_per_bucket = [len(sampler) if sampler is not None else 0 for sampler in samplers]

        print("\n*** Statistics for Data Buckets ***")
        print("Items per bucket:", [len(idxs) for idxs in buckets])
        print("Bucket batch sizes:", bucket_batch_sizes)
        print("Batches per bucket:", batches_per_bucket)

        self.buckets = buckets
        self.samplers = samplers
        self.bucket_batch_sizes = bucket_batch_sizes
        self.batches_per_bucket = batches_per_bucket
        self.batch_idx_generator = np.random.default_rng(self.seed)

    def __len__(self):
        return sum(self.batches_per_bucket)

    def __iter__(self):
        iters = [iter(sampler) if sampler is not None else None for sampler in self.samplers]
        remaining_batches = self.batches_per_bucket[:]

        while sum(remaining_batches) > 0:
            weights = np.array(remaining_batches) / sum(remaining_batches)
            b_idx = self.batch_idx_generator.choice(len(remaining_batches), p=weights)
            indices_in_bucket = next(iters[b_idx])
            batch = [self.buckets[b_idx][idx] for idx in indices_in_bucket]
            remaining_batches[b_idx] -= 1
            yield batch

    def _round_batch_size(self, batch_size):
        if not self.round_batch_to_8:
            bs = math.floor(batch_size)
        else:
            bs = 8 * round(batch_size / 8)

        bs = 1 if bs == 0 else bs
        return bs

    def set_epoch(self, epoch: int) -> None:
        """*** This needs to be called to ensure the distributed samplers have different behavious across epochs ***"""

        for batch_sampler in self.samplers:
            batch_sampler.sampler.set_epoch(epoch)
