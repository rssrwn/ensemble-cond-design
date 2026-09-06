from __future__ import annotations

import h5py
import copy
import pickle
from pathlib import Path
from more_itertools import grouper
from collections.abc import Sequence
from typing import Union, Optional

import numpy as np
from rdkit import Chem
from scipy.spatial.transform import Rotation

import enscondflow.util.rdkit as smolRD
import enscondflow.util.functional as smolF
from enscondflow.repr.atoms import AtomSet
from enscondflow.repr.bonds import BondSet
from enscondflow.repr.confs import ConfSet
from enscondflow.repr.util import PICKLE_PROTOCOL, check_dict_key, check_type


# Type aliases
TArr = np.ndarray


class GraphMol:
    def __init__(
        self,
        atoms: AtomSet,
        bonds: BondSet,
        confs: Optional[ConfSet] = None,
        meta: Optional[dict[str, str]] = None
    ):
        self._check_mol(atoms, bonds, confs=confs)

        meta = {} if meta is None else meta

        self.atoms = atoms
        self.bonds = bonds
        self.confs = confs
        self.meta = meta


    # *** Publicly exposed properties, others can still be accessible through mol.atoms.X etc ***

    @property
    def n_atoms(self) -> int:
        return len(self)

    @property
    def n_heavy_atoms(self) -> int:
        # NOTE this requires reading the atomics into memory (they are not kept though)
        atom_mask = self.atomics != 1
        return atom_mask.sum().item()

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def n_conformers(self) -> int:
        return 0 if self.confs is None else len(self.confs)

    @property
    def atomics(self) -> TArr:
        return self.atoms.atomics.astype(np.long)

    @property
    def charges(self) -> TArr:
        return self.atoms.charges.astype(np.long)

    @property
    def charged_symbols(self) -> list[str]:
        return self.atoms.charged_symbols

    @property
    def coords(self) -> Union[TArr, None]:
        return self.confs.coords if self.confs is not None else None

    @property
    def conf_weights(self) -> Union[TArr, None]:
        return self.confs.weights if self.confs is not None else None

    @property
    def bond_indices(self) -> TArr:
        return self.bonds.indices.astype(np.long)

    @property
    def bond_types(self) -> TArr:
        return self.bonds.types.astype(np.long)

    @property
    def adjacency(self) -> TArr:
        return self.bonds.adj_matrix(len(self))

    @property
    def seq_length(self) -> int:
        return len(self)


    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        return len(self.atoms)

    def __str__(self) -> str:
        if self.meta is not None and "str_id" in self.meta:
            return self.meta["str_id"]

        return super().__str__()

    def read(self) -> GraphMol:
        """Force the data to be read into memory if it isn't already"""

        atoms = self.atoms.read()
        bonds = self.bonds.read()
        confs = self.confs.read()

        mol = GraphMol(atoms, bonds, confs=confs, meta=self.meta)
        return mol

    def get_conformer(self, idx: int) -> TArr:
        return self.confs.get_conformer(idx)

    def copy_with(
        self,
        atoms: Optional[AtomSet] = None,
        bonds: Optional[BondSet] = None,
        confs: Optional[ConfSet] = None
    ) -> GraphMol:

        atoms = self.atoms.copy() if atoms is None else atoms.copy()
        bonds = self.bonds.copy() if bonds is None else bonds.copy()

        if confs is not None:
            confs = confs.copy()
        elif self.confs is not None:
            confs = self.confs.copy()

        meta = copy.deepcopy(self.meta) if self.meta is not None else None
        return GraphMol(atoms, bonds, confs=confs, meta=meta)

    def copy(self) -> GraphMol:
        return self.copy_with()

    def mol_with_conformer(self, conf_idx: int) -> GraphMol:
        """Return a copy of the mol with only the conformer at index conf_idx"""

        conf = ConfSet(self.get_conformer(conf_idx))
        return self.copy_with(confs=conf)

    def permute(self, indices: Union[list[int], TArr], in_place: bool = False) -> GraphMol:
        if in_place:
            self.atoms.permute_atoms(indices, in_place=True)
            self.bonds.permute_atoms(indices, in_place=True)
            if self.confs is not None:
                self.confs.permute_atoms(indices, in_place=True)

            return self

        atoms = self.atoms.permute_atoms(indices)
        bonds = self.bonds.permute_atoms(indices)
        confs = self.confs.permute_atoms(indices) if self.confs is not None else None
        return self.copy_with(atoms=atoms, bonds=bonds, confs=confs)

    def remove_hs(self) -> GraphMol:
        indices = np.arange(len(self.atomics))
        non_h_idxs = indices[self.atomics != 1]
        return self.permute(non_h_idxs)

    def drop_3d(self) -> GraphMol:
        """Returns a copy of this molecule without confs"""

        new_mol = self.copy()
        new_mol.confs = None
        return new_mol

    def order_by_bonds(self, canonical: bool = True) -> GraphMol:
        """Returns a permuted version of the molecule where the ordering of atoms is defined by RDKit

        The RDkit molecule must be sanitisable, otherwise this function will throw an error.

        Args:
            canonical (bool): Whether to use RDKit's canonical atom ordering, default True.

        Returns:
            GraphMol: Permuted version of the molecule.
        """

        rdkit_mol = self.to_rdkit(sanitise=True)
        _ = Chem.MolToSmiles(rdkit_mol, canonical=canonical, doRandom=not canonical)
        atom_order = rdkit_mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
        return self.permute(list(atom_order))

    def pad(
        self,
        n_atoms: int,
        pad_atomic: int = 0,
        pad_charge: int = 0,
        pad_coord_mode: str = "com",
        in_place: bool = False
    ) -> GraphMol:
        """Pad the mol to length n_atoms using pad_atomics and pad_charge as pad tokens for atomics and charges"""

        # Note we don't add any bonds to the mol
        # The 'pad' atoms will therefore be disconnected from the rest of the graph
        if in_place:
            self.atoms.pad(n_atoms, pad_atomic=pad_atomic, pad_charge=pad_charge, in_place=True)
            if self.confs is not None:
                self.confs.pad(n_atoms, pad_mode=pad_coord_mode, in_place=True)

            return self

        atoms = self.atoms.pad(n_atoms, pad_atomic=pad_atomic, pad_charge=pad_charge)
        confs = self.confs.pad(n_atoms, pad_mode=pad_coord_mode) if self.confs is not None else None
        return self.copy_with(atoms=atoms, confs=confs)

    def _check_mol(self, atoms, bonds, confs = None):
        check_type(atoms, AtomSet, "atoms")
        check_type(bonds, BondSet, "bonds")

        if confs is not None:
            check_type(confs, ConfSet, "confs")

        if len(bonds) > 0 and bonds.min_index < 0:
            raise RuntimeError("Bond indices within a molecule cannot be negative.")
        if len(bonds) > 0 and bonds.max_index >= len(atoms):
            raise RuntimeError("The maximum atom index in bonds cannot be larger the largest atom index.")
        if confs is not None and confs.n_atoms != len(atoms):
            raise RuntimeError("The number of atoms in the atom set and conf set must be the same.")


    # *** Geometric specific functions ***

    def zero_com(self) -> GraphMol:
        confs = self.confs.zero_com()
        return self.copy_with(confs=confs)

    def rotate(self, rotation: Union[Rotation, list[Rotation]]) -> GraphMol:
        confs = self.confs.rotate(rotation)
        return self.copy_with(confs=confs)

    def shift(self, shift: Union[TArr, list[TArr]]) -> GraphMol:
        confs = self.confs.shift(shift)
        return self.copy_with(confs=confs)

    def scale(self, scale: float) -> GraphMol:
        confs = self.confs.scale(scale)
        return self.copy_with(confs=confs)


    # *** IO and conversion utility functions ***

    @staticmethod
    def from_rdkit(rdkit_mol: Chem.rdchem.Mol) -> GraphMol:
        atoms = AtomSet.from_rdkit(rdkit_mol)
        bonds = BondSet.from_rdkit(rdkit_mol)

        confs = None
        if rdkit_mol.GetNumConformers() > 0:
            confs = ConfSet.from_rdkit(rdkit_mol)

        mol = GraphMol(atoms, bonds, confs=confs)
        return mol

    @staticmethod
    def _from_core_repr(dict_repr: dict[str, dict[str, TArr]]) -> GraphMol:
        check_type(dict_repr, dict, "unpickled object")
        check_dict_key(dict_repr, "atoms")
        check_dict_key(dict_repr, "bonds")

        atoms = AtomSet.from_dict(dict_repr["atoms"])
        bonds = BondSet.from_dict(dict_repr["bonds"])

        confs = dict_repr.get("confs")
        confs = ConfSet.from_dict(confs) if confs is not None else None

        meta = dict_repr.get("meta")

        mol = GraphMol(atoms, bonds, confs=confs, meta=meta)
        return mol

    @staticmethod
    def from_bytes(data: bytes) -> GraphMol:
        obj = pickle.loads(data)
        return GraphMol._from_core_repr(obj)

    def to_rdkit(self, sanitise: bool = False) -> Chem.rdchem.Mol:
        rdkit_mol = smolRD.mol_from_atoms(
            self.atomics,
            self.bonds.bonds,
            coords=self.coords,
            charges=self.charges,
            sanitise=sanitise
        )

        # Set conformer weights within the mol if they exist
        if self.confs is not None and self.confs.weights is not None:
            assert rdkit_mol.GetNumConformers() == self.n_conformers
            for idx, conf in enumerate(rdkit_mol.GetConformers()):
                conf.SetProp("weight", str(self.confs.weights[idx].item()))

        return rdkit_mol

    def _to_core_repr(self) -> dict[str, dict[str, TArr]]:
        """A representation of the molecule using only built-in types and numpy arrays."""

        dict_repr = {
            "atoms": self.atoms.to_dict(),
            "bonds": self.bonds.to_dict()
        }

        if self.confs is not None:
            dict_repr["confs"] = self.confs.to_dict()

        if self.meta is not None:
            dict_repr["meta"] = self.meta

        return dict_repr

    def to_bytes(self) -> bytes:
        dict_repr = self._to_core_repr()
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj


# ***************************************************************
# ***************** Batched Representations *********************
# ***************************************************************


class GraphBatch(Sequence):
    """Utility class predominantly for loading, saving and batching Molecule objects."""

    def __init__(self, mols: list[GraphMol], hdf5_file: Union[h5py.File, list[h5py.File], None] = None):
        for mol in mols:
            check_type(mol, GraphMol, "molecule object")

        open_fps = []
        if hdf5_file is not None:
            if isinstance(hdf5_file, h5py.File):
                open_fps = [hdf5_file]
            elif isinstance(hdf5_file, list):
                check_type(hdf5_file[0], h5py.File, "hdf5 file list item")
                open_fps = hdf5_file
            else:
                raise TypeError("hdf5_file must be either an h5py.File or a list of h5py.File objects.")

        self._mols = mols
        self._open_fps = open_fps


    # *** Publicly exposed properties, others can still be accessible through mol.atoms.X etc ***

    @property
    def lengths(self) -> list[int]:
        return [len(mol) for mol in self._mols]

    @property
    def mask(self) -> TArr:
        return smolF.pad_arrays([np.ones(mol.seq_length) for mol in self._mols])

    @property
    def atomics(self) -> TArr:
        return smolF.pad_arrays([mol.atomics for mol in self._mols])

    @property
    def charges(self) -> TArr:
        return smolF.pad_arrays([mol.charges for mol in self._mols])

    @property
    def coords(self) -> Union[TArr, None]:
        n_confs = [mol.n_conformers for mol in self._mols]
        if any([n in [None, 0] for n in n_confs]) or any([n != n_confs[0] for n in n_confs]):
            raise RuntimeError("All mols in the batch must have the same number of conformers.")

        # Transpose conf and atom dims for padding, then transpose back
        coords = [mol.coords.transpose((1, 0, 2)) for mol in self._mols]
        padded = smolF.pad_arrays(coords).transpose((0, 2, 1, 3))
        return padded

    @property
    def bonds(self) -> TArr:
        return smolF.pad_arrays([mol.bonds for mol in self._mols])

    @property
    def bond_indices(self) -> TArr:
        return smolF.pad_arrays([mol.bond_indices for mol in self._mols])

    @property
    def bond_types(self) -> TArr:
        return smolF.pad_arrays([mol.bond_types for mol in self._mols])

    @property
    def adjacency(self) -> TArr:
        max_length = max(self.lengths)
        adjs = [mol.bonds.adj_matrix(max_length) for mol in self._mols]
        return np.stack(adjs, axis=0)


    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        return len(self._mols)

    def __getitem__(self, index: int) -> GraphMol:
        return self._mols[index]

    def subset(self, idxs: list[int]) -> GraphBatch:
        # Take all fps since we currently don't have a way of knowing which correspond to subset
        subset_mols = [self._mols[idx] for idx in idxs]
        batch = GraphBatch(subset_mols, self._open_fps)
        return batch

    # *** IO and conversion utility functions ***

    @staticmethod
    def _from_core_repr(obj: list[dict[str, dict[str, TArr]]]) -> GraphBatch:
        mols = [GraphMol._from_core_repr(mol_data) for mol_data in obj]
        return GraphBatch(mols)

    @staticmethod
    def from_bytes(data: bytes) -> GraphBatch:
        obj = pickle.loads(data)
        return GraphMol._from_core_repr(obj)

    @staticmethod
    def from_batches(batches: list[GraphBatch]) -> GraphBatch:
        """Accumulate a list of MolBatch objects into one batch"""

        mols = [mol for batch in batches for mol in batch]
        open_fps = [fp for batch in batches for fp in batch._open_fps]
        batch = GraphBatch(mols, hdf5_file=open_fps)
        return batch

    @staticmethod
    def load(save_path: Union[str, Path], n_shards: int = None) -> GraphBatch:
        """Load data from a folder that was saved using the save function.
        
        If shards is provided, only the first <n_shards> shards will be loaded. This is useful for debugging or
        loading only a subset of the dataset.
        """

        save_path = Path(save_path)

        if not (save_path.exists() and save_path.is_dir()):
            raise RuntimeError(f"The folder was not found at path {str(save_path)}")

        shard_paths = [path for path in save_path.iterdir() if path.suffix == ".hdf5"]
        sorted_paths = list(sorted(shard_paths, key=lambda p: int(p.stem)))

        if n_shards is not None:
            n_shards = min(len(sorted_paths) - 1, n_shards)
            sorted_paths = sorted_paths[:n_shards]

        shards = [GraphBatch.load_hdf5_shard(shard_path) for shard_path in sorted_paths]
        batch = GraphBatch.from_batches(shards)
        return batch

    @staticmethod
    def load_hdf5_shard(save_file: Union[str, Path]) -> GraphBatch:
        save_file = Path(save_file)

        if save_file.suffix != ".hdf5":
            raise RuntimeError("Save file must have an hdf5 suffix.")

        hdf5_file = h5py.File(save_file, "r")
        mols = GraphBatch._load_from_group(hdf5_file)
        batch = GraphBatch(mols, hdf5_file=hdf5_file)
        return batch

    @staticmethod
    def _load_from_group(group: h5py.Group) -> list[GraphMol]:
        """Load molecules from an HDF5 group."""

        atom_arrays = {name: arr for name, arr in group["atoms"].items()}
        atom_sets = AtomSet.atoms_from_arrays(atom_arrays)

        bond_arrays = {name: arr for name, arr in group["bonds"].items()}
        bond_sets = BondSet.bonds_from_arrays(bond_arrays)

        n_atoms = len(atom_sets)
        n_bonds = len(bond_sets)

        if n_atoms != n_bonds:
            raise RuntimeError(f"Number of AtomSet and BondSet must be equal, got {n_atoms} and {n_bonds}")

        conf_sets = [None] * len(atom_sets)
        if "confs" in group:
            conf_arrays = {name: arr for name, arr in group["confs"].items()}
            conf_sets = ConfSet.confs_from_arrays(conf_arrays)

        if len(conf_sets) != n_atoms:
            raise RuntimeError(f"Number of ConfSet molecules must match number of AtomSet molecules.")

        meta_bytes = group["meta"].attrs["metas"].tobytes()
        metas = [None if len(meta) == 0 else meta for meta in pickle.loads(meta_bytes)]

        zipped = zip(atom_sets, bond_sets, conf_sets, metas)
        mols = [GraphMol(atoms, bonds, confs, meta=meta) for atoms, bonds, confs, meta in zipped]
        return mols

    def _to_core_repr(self) -> list[dict[str, dict[str, TArr]]]:
        """A representation of the molecule using only built-in types and numpy arrays."""

        dict_list = [mol._to_core_repr() for mol in self._mols]
        return dict_list

    def to_bytes(self) -> bytes:
        dict_repr = self._to_core_repr()
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    def save(self, save_path: Union[str, Path], shard_size: Optional[int] = None) -> None:
        """Save the batch of data under the directory given by save_path."""

        save_path = Path(save_path)

        # Allow save_path to exist only if it is an empty directory
        # Otherwise there is always a risk of accidentally losing data
        if save_path.exists():
            if not (save_path.is_dir() and len(list(save_path.iterdir())) == 0):
                raise RuntimeError("Save path must point to an empty or non-existing directory.")

        save_path.mkdir(exist_ok=True, parents=True)

        shard_size = len(self) if shard_size is None else shard_size
        mol_shards = [[mol for mol in mols if mol is not None] for mols in grouper(self, shard_size)]

        for idx, shard in enumerate(mol_shards):
            shard_batch = GraphBatch(shard)
            save_file = save_path / f"{idx}.hdf5"
            shard_batch.save_hdf5_shard(save_file)

    def save_hdf5_shard(self, save_file: Union[str, Path]) -> None:
        hdf5_path = Path(save_file)

        if save_file.exists():
            raise RuntimeError(f"File {str(save_file)} already exists.")

        if hdf5_path.suffix != ".hdf5":
            raise ValueError(f"save_file must end in .hdf5, got {save_file}")

        with h5py.File(hdf5_path, "x") as f:
            self._save_to_group(f)

    def _save_to_group(self, group: h5py.Group) -> None:
        """Save molecule data to an HDF5 group."""

        atoms = [mol.atoms for mol in self._mols]
        bonds = [mol.bonds for mol in self._mols]
        confs = [mol.confs for mol in self._mols if mol.confs is not None]

        atom_arrays = AtomSet.arrays_from_atoms(atoms)
        bond_arrays = BondSet.arrays_from_bonds(bonds)
        conf_arrays = ConfSet.arrays_from_confs(confs) if len(confs) > 0 else None

        metas = [mol.meta if mol.meta is not None else {} for mol in self._mols]
        metas = np.void(pickle.dumps(metas, protocol=PICKLE_PROTOCOL))

        atom_group = group.create_group("atoms")
        for name, arr in atom_arrays.items():
            atom_group.create_dataset(name, data=arr)

        bond_group = group.create_group("bonds")
        for name, arr in bond_arrays.items():
            bond_group.create_dataset(name, data=arr)

        if conf_arrays is not None:
            conf_group = group.create_group("confs")
            for name, arr in conf_arrays.items():
                conf_group.create_dataset(name, data=arr)

        meta_group = group.create_group("meta", track_order=True)
        meta_group.attrs["metas"] = metas

    def close_hdf5(self) -> None:
        """Closes any HDF5 file associated with this batch.

        If the batch was read from an HDF5 file this will close the underlying file, stopping any further reads. If the
        batch did not originate from HDF5 data this function does not do anything.

        NOTE even if the molecules in the batch have been transferred to a different MolBatch object this will stop
        reads from any data within the HDF5 file, so only close the file if you are sure the data will not be read.
        """

        # Close the files but don't set them to None, then we can open them again if needed
        if self._open_fps is not None:
            for fp in self._open_fps:
                fp.close() if fp is not None else None
