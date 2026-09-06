from __future__ import annotations

import bisect
import tqdm
import torch
import numpy as np
from typing import Union
from more_itertools import grouper
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

from enscondflow.repr import GraphBatch, ComplexBatch


def _preload_complex_batch(batch_complexes):
    """Compute binding profiles and ligand properties for a batch of complexes."""

    from enscondflow.data.features import BindingProfile, Qed, LogP
    from enscondflow.data.profile import ConfProfile

    binding_profile = BindingProfile(raise_on_err=False)
    qed_feat = Qed()
    logp_feat = LogP()

    results = []
    for system in batch_complexes:
        profile = binding_profile(system)
        if isinstance(profile, ConfProfile):
            system.ligand.meta[BindingProfile.CACHE_KEY] = profile

        system.ligand.meta["qed"] = qed_feat(system.ligand)
        system.ligand.meta["logp"] = logp_feat(system.ligand)
        system.interactions = None
        results.append(system)

    return results


def _preload_batch(batch_mols, features, n_confs):
    """Compute features and subsample conformers for a batch of pre-read molecules."""

    results = []
    for mol in batch_mols:
        for feat in features:
            mol.meta[feat.name] = feat(mol)

        if n_confs is not None and mol.confs is not None and len(mol.confs) > n_confs:
            mol = mol.copy_with(confs=mol.confs.uniform_sample(n_confs))

        results.append(mol)

    return results


# **********************************************
# *** Abstract class for all Smol data types ***
# **********************************************


class SmolDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, smol_data, transform=None):
        super().__init__()

        self._data = smol_data
        self.transform = transform

    @property
    def hparams(self):
        return {}

    @property
    def lengths(self):
        return self._data.lengths

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        molecule = self._data[item]
        if self.transform is not None:
            molecule = self.transform(molecule)

        return molecule

    @abstractmethod
    def sample(self, n_items, replacement=False):
        pass

    @abstractmethod
    def close(self):
        pass

    @staticmethod
    @abstractmethod
    def load(data_path, transform=None, n_shards=None):
        pass


# ***********************************
# *** SmolDataset implementations ***
# ***********************************


class GraphDataset(SmolDataset):
    def sample(self, n_items, replacement=False):
        mol_samples = np.random.choice(list(self._data), n_items, replace=replacement)
        hdf5_files = self._data._open_fps
        data = GraphBatch(mol_samples, hdf5_file=hdf5_files)
        return GraphDataset(data, transform=self.transform)

    def subset(self, idxs):
        subset_mols = [self._data[idx] for idx in idxs]
        hdf5_files = self._data._open_fps
        data = GraphBatch(subset_mols, hdf5_file=hdf5_files)
        return GraphDataset(data, transform=self.transform)

    def select(self, select_fn):
        """select_fn should return True to keep the mol, False to remove"""

        selected = [mol for mol in self._data if select_fn(mol)]
        hdf5_files = self._data._open_fps
        data = GraphBatch(selected, hdf5_file=hdf5_files)
        return GraphDataset(data, transform=self.transform)

    def split(self, split: Union[int, list[int]]) -> tuple[GraphDataset, GraphDataset]:
        """split is either a number of mols to split off or a list of mol indices"""

        if isinstance(split, int):
            split = np.random.choice(list(range(len(self._data))), split, replace=False)

        split_idxs_set = set(split)
        non_split_idxs = [idx for idx in range(len(self._data)) if idx not in split_idxs_set]

        split_subset = self.subset(split)
        other_subset = self.subset(non_split_idxs)

        return split_subset, other_subset

    def preload(self, features=None, n_confs=None, batch_size=100, max_workers=None):
        """Precompute features and read all molecules into memory.

        Args:
            features: List of MolFeature instances to precompute and store in mol.meta.
            n_confs: If set, subsample each molecule's conformers to this many via uniform_sample.
            batch_size: Number of molecules per worker batch.
            max_workers: Max processes for feature computation. Defaults to ProcessPoolExecutor default.
        """

        features = features or []
        mols = self._data._mols
        n_mols = len(mols)

        # Materialise LazyData in the main process (h5py objects are not picklable)
        for i in tqdm.tqdm(range(n_mols), desc="Reading HDF5"):
            mols[i] = mols[i].read()
        self._data.close_hdf5()

        # Compute features and subsample conformers in parallel
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for batch in grouper(range(n_mols), batch_size, incomplete="fill", fillvalue=None):
                batch_idxs = tuple(i for i in batch if i is not None)
                batch_mols = [mols[i] for i in batch_idxs]
                future = pool.submit(_preload_batch, batch_mols, features, n_confs)
                futures[future] = batch_idxs

            pbar = tqdm.tqdm(total=n_mols, desc="Computing features")
            for future in as_completed(futures):
                batch_idxs = futures[future]
                results = future.result()
                for idx, mol in zip(batch_idxs, results):
                    mols[idx] = mol
                pbar.update(len(batch_idxs))
            pbar.close()

    def close(self):
        self._data.close_hdf5()

    @staticmethod
    def load(data_path, transform=None, n_shards=None):
        batch = GraphBatch.load(data_path, n_shards=n_shards)
        return GraphDataset(batch, transform=transform)


class ComplexDataset(SmolDataset):
    def close(self):
        self._data.close_hdf5()

    def preload(self, batch_size=50, max_workers=None):
        """Precompute binding profiles and ligand properties, read all complexes into memory."""

        complexes = self._data._complexes
        n = len(complexes)

        for i in tqdm.tqdm(range(n), desc="Reading HDF5"):
            complexes[i] = complexes[i].read()
        self._data.close_hdf5()

        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for batch in grouper(range(n), batch_size, incomplete="fill", fillvalue=None):
                batch_idxs = tuple(i for i in batch if i is not None)
                batch_complexes = [complexes[i] for i in batch_idxs]
                future = pool.submit(_preload_complex_batch, batch_complexes)
                futures[future] = batch_idxs

            pbar = tqdm.tqdm(total=n, desc="Computing features")
            for future in as_completed(futures):
                batch_idxs = futures[future]
                results = future.result()
                for idx, cx in zip(batch_idxs, results):
                    complexes[idx] = cx
                pbar.update(len(batch_idxs))
            pbar.close()

    def replicate(self, n_times: int):
        replicated = [system for system in self._data for _ in range(n_times)]
        hdf5_files = self._data._open_fps
        data = ComplexBatch(replicated, hdf5_file=hdf5_files)
        return ComplexDataset(data, transform=self.transform)

    def sample(self, n_items, replacement=False):
        samples = np.random.choice(list(self._data), n_items, replace=replacement)
        hdf5_files = self._data._open_fps
        data = ComplexBatch(samples, hdf5_file=hdf5_files)
        return ComplexDataset(data, transform=self.transform)

    def select(self, select_fn):
        """select_fn should return True to keep the mol, False to remove"""

        selected = [system for system in self._data if select_fn(system)]
        hdf5_files = self._data._open_fps
        data = ComplexBatch(selected, hdf5_file=hdf5_files)
        return ComplexDataset(data, transform=self.transform)

    @staticmethod
    def load(data_path, transform=None, n_shards=None):
        batch = ComplexBatch.load(data_path, n_shards=n_shards)
        return ComplexDataset(batch, transform=transform)


# *********************************
# *** Combine datasets together ***
# *********************************


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[Union[CombinedDataset, SmolDataset]]):
        total = 0
        cumulative_sizes = []
        for dataset in datasets:
            total += len(dataset)
            cumulative_sizes.append(total)

        self.datasets = datasets
        self._cumulative_sizes = cumulative_sizes

    @property
    def hparams(self):
        return {}

    @property
    def lengths(self):
        dataset_lengths = []
        for dataset in self.datasets:
            dataset_lengths.extend(dataset.lengths)

        return dataset_lengths

    def __len__(self):
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, item):
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, item)
        local_idx = item if dataset_idx == 0 else item - self._cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][local_idx]

    def close(self):
        for dataset in self.datasets:
            dataset.close()
