import os
import sys
from abc import abstractmethod
from functools import partial

import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader

from enscondflow.repr import GraphMol, GraphBatch, Protein, ProteinBatch
from enscondflow.data.datasets import SmolDataset, CombinedDataset
from enscondflow.data.util import BucketBatchSampler, MultiDatasetSampler
from enscondflow.data.profile import ConfProfile


DEFAULT_MAX_WORKERS = 8


def n_available_procs():
    # sched_getaffinity is only available on linux although it generally gives the best answers
    # otherwise fallback to process_cpu_count but this can be overidden by env vars
    try:
        n_procs = len(os.sched_getaffinity(0))
    except:
        n_procs = os.process_cpu_count()

    return n_procs or 1


def graph_batch_to_dict(batch: GraphBatch) -> dict[str, torch.Tensor]:
    coords = torch.from_numpy(batch.coords).float().squeeze(1)  # remove ensemble dim
    atomics = torch.from_numpy(batch.atomics).long()
    bonds = torch.from_numpy(batch.adjacency).long()
    mask = torch.from_numpy(batch.mask).long()

    data = {
        "coords": coords,
        "atomics": atomics,
        "bonds": bonds,
        "mask": mask
    }
    return data


def protein_batch_to_dict(batch: ProteinBatch) -> dict[str, torch.Tensor]:
    coords = torch.from_numpy(batch.coords).float()
    atomics = torch.from_numpy(batch.atomics).long()
    mask = torch.from_numpy(batch.mask).long()

    data = {
        "coords": coords,
        "atomics": atomics,
        "mask": mask
    }
    return data


def profile_batch_to_dict(batch: list[ConfProfile]) -> dict:
    arrays = ConfProfile.batch_profiles(batch)
    assert len(arrays) == 7

    data = {
        "coords": torch.from_numpy(arrays[0]).float(),
        "types": torch.from_numpy(arrays[1]).long(),
        "directions": torch.from_numpy(arrays[2]).float(),
        "mask": torch.from_numpy(arrays[3]).long(),
        "rotated": torch.from_numpy(arrays[4]).long(),
        "profile_mode": torch.from_numpy(arrays[5]).long(),
        "pos_noise_std": torch.from_numpy(arrays[6]).float(),
        "raw": batch,
    }
    return data


class SmolDM(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: SmolDataset,
        val_dataset: SmolDataset,
        test_dataset: SmolDataset,
        batch_cost: int,
        bucket_limits: list[int] = None,
        bucket_cost_scale: str = "constant",
        max_workers: int = DEFAULT_MAX_WORKERS
    ):
        super().__init__()

        if bucket_cost_scale not in [None, "constant", "linear", "quadratic"]:
            raise ValueError(f"Bucket cost scale '{bucket_cost_scale}' is not supported.")

        if bucket_limits is not None:
            bucket_limits = sorted(bucket_limits)
            largest_padding = bucket_limits[-1]

            if train_dataset is not None and max(train_dataset.lengths) > largest_padding:
                raise ValueError("At least one item in train dataset is larger than largest padded size.")

            if val_dataset is not None and max(val_dataset.lengths) > largest_padding:
                raise ValueError("At least one item in val dataset is larger than largest padded size.")

            if test_dataset is not None and max(test_dataset.lengths) > largest_padding:
                raise ValueError("At least one item in test dataset is larger than largest padded size.")

        max_workers = 0 if max_workers is None else max_workers

        # Assume we will not need more than max_workers to keep memory cost down
        self._n_workers = max(0, min(max_workers, n_available_procs() - 2))
        # HDF5-backed datasets cannot be pickled by macOS/Windows spawn workers.
        # Keep loading in-process there; Linux can use fork with the research data.
        if sys.platform != "linux":
            self._n_workers = 0
        self.persist_workers = self._n_workers > 0
        self._mult_proc_context = "fork" if self._n_workers > 0 else None

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_cost = batch_cost
        self.bucket_limits = bucket_limits
        self.bucket_cost_scale = bucket_cost_scale

    @property
    def hparams(self):
        train_data = self.train_dataset
        val_data = self.val_dataset
        test_data = self.test_dataset

        train_hps = {f"train-{k}": v for k, v in train_data.hparams.items()} if train_data is not None else {}
        val_hps = {f"val-{k}": v for k, v in val_data.hparams.items()} if val_data is not None else {}
        test_hps = {f"test-{k}": v for k, v in test_data.hparams.items()} if test_data is not None else {}

        buckets = None if self.bucket_limits is None else len(self.bucket_limits)

        hparams = {
            "batch-cost": self.batch_cost,
            "buckets": buckets,
            "bucket-cost-scale": self.bucket_cost_scale,
            **train_hps,
            **val_hps,
            **test_hps
        }
        return hparams

    def train_dataloader(self):
        sampler = self._sampler(self.train_dataset, drop_last=True)
        batch_size = self.batch_cost if sampler is None else 1
        shuffle = sampler is None

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            batch_sampler=sampler,
            num_workers=self._n_workers,
            prefetch_factor=4 if self._n_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.persist_workers,
            multiprocessing_context=self._mult_proc_context if self._n_workers > 0 else None,
            collate_fn=partial(self._collate, split="train")
        )
        return dataloader

    def val_dataloader(self):
        sampler = self._sampler(self.val_dataset, drop_last=False)
        batch_size = self.batch_cost if sampler is None else 1
        n_workers = min(2, self._n_workers)
        mp_context = self._mult_proc_context if n_workers > 0 else None

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            batch_sampler=sampler,
            num_workers=n_workers,
            multiprocessing_context=mp_context,
            collate_fn=partial(self._collate, split="val")
        )
        return dataloader

    def test_dataloader(self):
        sampler = self._sampler(self.test_dataset, drop_last=False)
        batch_size = self.batch_cost if sampler is None else 1

        dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            batch_sampler=sampler,
            num_workers=min(2, self._n_workers),
            multiprocessing_context=self._mult_proc_context if self._n_workers > 0 else None,
            collate_fn=partial(self._collate, split="test")
        )
        return dataloader

    def close_datasets(self):
        if self.train_dataset is not None:
            self.train_dataset.close()
        if self.val_dataset is not None:
            self.val_dataset.close()
        if self.test_dataset is not None:
            self.test_dataset.close()

    def _sampler(self, dataset, drop_last=False):
        sampler = None
        if self.bucket_limits is not None:
            costs = self._get_bucket_costs()
            sampler = BucketBatchSampler(
                self.bucket_limits,
                dataset.lengths,
                self.batch_cost,
                bucket_costs=costs,
                drop_last=drop_last,
                round_batch_to_8=False
            )

        return sampler

    def _get_bucket_costs(self):
        if self.bucket_cost_scale is None:
            return None
        elif self.bucket_cost_scale == "constant":
            return [1] * len(self.bucket_limits)
        elif self.bucket_cost_scale == "linear":
            return self.bucket_limits
        elif self.bucket_cost_scale == "quadratic":
            # Divide by 256 and add one to approximate the linear and constant overheads
            # A molecule with 16 atoms will therefore have a cost of 1 + 1
            return [((limit ** 2) / 256) + 1 for limit in self.bucket_limits]
        else:
            raise ValueError(f"Unknown value for bucket_cost_scale '{self.bucket_cost_scale}'")

    def _collate_objs(self, batch):
        if isinstance(batch, tuple):
            return batch

        elif isinstance(batch, dict):
            return {name: self._collate_objs(obj) for name, obj in batch.items()}

        elif isinstance(batch, GraphBatch):
            return graph_batch_to_dict(batch)

        elif isinstance(batch[0], GraphMol):
            return graph_batch_to_dict(GraphBatch(list(batch)))

        elif isinstance(batch[0], Protein):
            return protein_batch_to_dict(ProteinBatch(list(batch)))

        elif isinstance(batch[0], ConfProfile):
            return profile_batch_to_dict(batch)

        # Assume that arrays all have the same shape
        elif isinstance(batch[0], np.ndarray):
            return torch.tensor(np.stack(batch))

        # Assume that tensors all have the same shape
        elif isinstance(batch[0], torch.Tensor):
            return torch.stack(batch)

        elif isinstance(batch[0], float):
            return torch.tensor(batch, dtype=torch.float)

        elif isinstance(batch[0], int):
            return torch.tensor(batch, dtype=torch.long)

        elif isinstance(batch, list):
            return [self._collate_objs(objs) for objs in batch]

        else:
            RuntimeError("Unrecognised structure of batch.")

    @abstractmethod
    def _collate(self, batch, split):
        raise NotImplementedError()


class InterpolantDM(SmolDM):
    """DataModule for flow matching with interpolation.

    Accepts lists of train/val datasets (e.g. molecules + complexes). When multiple datasets
    are provided, they are combined with a MultiDatasetSampler that draws homogeneous batches
    from each dataset using a single batch size.
    """

    def __init__(
        self,
        train_datasets,
        val_datasets,
        test_dataset,
        batch_size,
        train_interpolant=None,
        val_interpolant=None,
        test_interpolant=None,
        max_workers=DEFAULT_MAX_WORKERS
    ):

        self.train_interpolant = train_interpolant
        self.val_interpolant = val_interpolant
        self.test_interpolant = test_interpolant

        if not isinstance(train_datasets, list):
            train_datasets = [train_datasets] if train_datasets is not None else []
        if not isinstance(val_datasets, list):
            val_datasets = [val_datasets] if val_datasets is not None else []

        self._train_datasets = train_datasets
        self._val_datasets = val_datasets

        first_train = train_datasets[0] if train_datasets else None
        first_val = val_datasets[0] if val_datasets else None
        super().__init__(
            first_train,
            first_val,
            test_dataset,
            batch_size,
            max_workers=max_workers
        )

    @property
    def hparams(self):
        interps = [self.train_interpolant, self.val_interpolant, self.test_interpolant]
        datasets = ["train", "val", "test"]

        hparams = []
        for dataset, interp in zip(datasets, interps):
            if interp is not None:
                interp_hparams = {f"{dataset}-{k}": v for k, v in interp.hparams.items()}
                hparams.append(interp_hparams)

        hparams = {k: v for interp_hparams in hparams for k, v in interp_hparams.items()}
        return {**hparams, **super().hparams}

    def train_dataloader(self):
        if len(self._train_datasets) <= 1:
            return super().train_dataloader()

        return self._multi_dataset_loader(self._train_datasets, split="train", drop_last=True)

    def val_dataloader(self):
        if len(self._val_datasets) <= 1:
            return super().val_dataloader()

        return self._multi_dataset_loader(self._val_datasets, split="val", drop_last=False)

    def _multi_dataset_loader(self, datasets, split, drop_last):
        combined = CombinedDataset(datasets)
        sampler = MultiDatasetSampler(datasets, self.batch_cost, drop_last=drop_last)

        is_train = split == "train"
        n_workers = self._n_workers if is_train else min(2, self._n_workers)
        mp_context = self._mult_proc_context if n_workers > 0 else None

        kwargs = {}
        if is_train and n_workers > 0:
            kwargs = {"prefetch_factor": 4, "pin_memory": torch.cuda.is_available(), "persistent_workers": self.persist_workers}

        return DataLoader(
            combined,
            batch_sampler=sampler,
            num_workers=n_workers,
            multiprocessing_context=mp_context,
            collate_fn=partial(self._collate, split=split),
            **kwargs
        )

    def close_datasets(self):
        for ds in self._train_datasets:
            ds.close()
        for ds in self._val_datasets:
            ds.close()

        if self.test_dataset is not None:
            self.test_dataset.close()

    def _collate(self, batch, split):
        if split == "train" and self.train_interpolant is not None:
            objs = self.train_interpolant.interpolate(batch)

        elif split == "val" and self.val_interpolant is not None:
            objs = self.val_interpolant.interpolate(batch)

        elif split == "test" and self.test_interpolant is not None:
            objs = self.test_interpolant.interpolate(batch)

        return super()._collate_objs(objs)
