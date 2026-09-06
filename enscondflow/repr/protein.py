from __future__ import annotations

import h5py
import copy
import pickle
import tempfile
from pathlib import Path
from more_itertools import grouper
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import prolif as plf
import MDAnalysis as mda
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from biotite.structure import AtomArray

import enscondflow.util.rdkit as smolRD
import enscondflow.util.functional as smolF
from enscondflow.repr.atoms import AtomSet
from enscondflow.repr.bonds import BondSet
from enscondflow.repr.confs import ConfSet
from enscondflow.repr.util import PICKLE_PROTOCOL, check_type, check_dict_key


TArr = np.ndarray


# *** Some util functions for proteins ***

def _pad_string_lists(string_lists: list[list[str]], pad_value: str = "") -> TArr:
    """Pad a list of string lists to the same length and return as numpy array."""

    if len(string_lists) == 0:
        return np.array([])

    max_len = max(len(lst) for lst in string_lists)
    padded = []

    for lst in string_lists:
        if len(lst) < max_len:
            padded_lst = list(lst) + [pad_value] * (max_len - len(lst))
            padded.append(padded_lst)
        else:
            padded.append(list(lst))

    return np.array(padded)


# *****************************************************************************
# *************************** Protein Representation **************************
# *****************************************************************************


class Protein:
    """Representation of a single protein chain."""

    def __init__(
        self,
        atoms: AtomSet,
        bonds: BondSet,
        confs: ConfSet,
        meta: Optional[dict[str, str]] = None
    ):
        self._check_pocket(atoms, bonds, confs=confs)

        meta = {} if meta is None else meta

        self.atoms = atoms
        self.bonds = bonds
        self.confs = confs
        self.meta = meta


    # *** Publicly exposed properties ***

    @property
    def n_atoms(self) -> int:
        return len(self)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def n_residues(self) -> int:
        return len(set(self.atoms.res_ids.tolist()))

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
    def res_names(self) -> list[str]:
        return self.atoms.res_names.tolist()

    @property
    def atom_names(self) -> list[str]:
        return self.atoms.atom_names.tolist()

    @property
    def res_ids(self) -> TArr:
        return self.atoms.res_ids.astype(np.long)

    @property
    def chain_id(self) -> str:
        """Returns chain ID from metadata, or empty string if not set"""
        return self.meta.get("chain_id", "")

    @property
    def coords(self) -> TArr:
        # NOTE we don't support multiple confs for protein for now, this return arr shape [N, 3]
        return self.confs.coords.astype(np.float32)[0,]

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

    def read(self) -> Protein:
        """Force the data to be read into memory if it isn't already"""

        atoms = self.atoms.read()
        bonds = self.bonds.read()
        confs = self.confs.read()

        pocket = Protein(atoms, bonds, confs, meta=self.meta)
        return pocket

    def copy_with(
        self,
        atoms: Optional[AtomSet] = None,
        bonds: Optional[BondSet] = None,
        confs: Optional[ConfSet] = None
    ) -> Protein:

        atoms = self.atoms.copy() if atoms is None else atoms.copy()
        bonds = self.bonds.copy() if bonds is None else bonds.copy()
        confs = self.confs.copy() if confs is None else confs.copy()

        meta = copy.deepcopy(self.meta) if self.meta is not None else None
        return Protein(atoms, bonds, confs, meta=meta)

    def copy(self) -> Protein:
        return self.copy_with()

    def permute(self, indices: Union[list[int], TArr]) -> Protein:
        """Permute or subset atoms by indices"""

        atoms = self.atoms.permute_atoms(indices)
        bonds = self.bonds.permute_atoms(indices)
        confs = self.confs.permute_atoms(indices)
        return self.copy_with(atoms=atoms, bonds=bonds, confs=confs)

    def remove_hs(self) -> Protein:
        """Returns a copy with hydrogen atoms removed"""

        indices = np.arange(len(self.atomics))
        non_h_idxs = indices[self.atomics != 1]
        return self.permute(non_h_idxs)


    # *** Geometric functions ***

    def zero_com(self) -> Protein:
        """Returns a copy with centre of mass at origin"""

        confs = self.confs.zero_com()
        return self.copy_with(confs=confs)

    def shift(self, shift: TArr) -> Protein:
        """Returns a copy with coordinates shifted by the given vector"""

        confs = self.confs.shift(shift)
        return self.copy_with(confs=confs)

    def rotate(self, rotation) -> Protein:
        """Returns a copy with coordinates rotated"""

        confs = self.confs.rotate(rotation)
        return self.copy_with(confs=confs)


    # *** IO and conversion functions ***

    @staticmethod
    def from_biotite(atom_array: AtomArray, infer_bonds: bool = True) -> Protein:
        """Create a Protein from a biotite AtomArray.

        Args:
            atom_array: Biotite AtomArray with protein structure
            infer_bonds: If True, infer bonds using biotite's connect_via_residue_names, otherwise it will look for
                    a BondSet object attached to atom_array, otherwise bonds will

        Returns:
            Protein instance
        """

        # Handle bonds first to make sure they exist if infer_bonds is False
        bond_list = atom_array.bonds

        if infer_bonds:
            bond_list = struc.connect_via_residue_names(atom_array, inter_residue=True)

        if bond_list is None:
            raise ValueError("Bonds must be attached to atom_array if infer_bonds is False.")

        atoms = AtomSet.from_biotite(atom_array)
        bonds = BondSet.from_biotite(bond_list)
        confs = ConfSet(atom_array.coord)

        # Extract chain_id and store in meta
        unique_chains = list(set(atom_array.chain_id.tolist()))
        if len(unique_chains) > 1:
            print(f"Warning: Multiple chain IDs found ({unique_chains}), combining into single chain_id")
            chain_id = "_".join(sorted(unique_chains))
        else:
            chain_id = unique_chains[0] if unique_chains else ""

        meta = {"chain_id": chain_id}
        return Protein(atoms, bonds, confs, meta=meta)

    @staticmethod
    def _from_core_repr(dict_repr: dict[str, dict[str, TArr]]) -> Protein:
        check_type(dict_repr, dict, "unpickled object")

        check_dict_key(dict_repr, "atoms")
        check_dict_key(dict_repr, "bonds")
        check_dict_key(dict_repr, "confs")

        atoms = AtomSet.from_dict(dict_repr["atoms"])
        bonds = BondSet.from_dict(dict_repr["bonds"])
        confs = ConfSet.from_dict(dict_repr["confs"])

        meta = dict_repr.get("meta")

        return Protein(atoms, bonds, confs, meta=meta)

    @staticmethod
    def from_bytes(data: bytes) -> Protein:
        obj = pickle.loads(data)
        return Protein._from_core_repr(obj)

    def _to_core_repr(self) -> dict[str, dict[str, TArr]]:
        """A representation using only built-in types and numpy arrays."""

        dict_repr = {
            "atoms": self.atoms.to_dict(),
            "bonds": self.bonds.to_dict(),
            "confs": self.confs.to_dict()
        }

        if self.meta is not None:
            dict_repr["meta"] = self.meta

        return dict_repr

    def to_bytes(self) -> bytes:
        dict_repr = self._to_core_repr()
        return pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)

    def to_prolif(self) -> plf.Molecule:
        """Convert the protein to a Prolif Molecule for interaction detection.

        Creates a temporary PDB file from the protein structure and reads it back
        using MDAnalysis with bond guessing, then converts to a Prolif Molecule.
        This roundabout approach is necessary because Prolif requires specific
        molecular representations that aren't directly available from our data structures.

        Returns:
            Prolif Molecule object for the protein.
        """

        n_atoms = len(self)
        atoms = AtomArray(len(self))
        elements = [smolRD.PT.symbol_from_atomic(a) for a in self.atomics.tolist()]

        atoms.coord = self.coords
        atoms.element = np.array(elements)
        atoms.res_name = self.res_names
        atoms.atom_name = self.atom_names
        atoms.res_id = self.res_ids
        atoms.chain_id = np.array(["A"] * n_atoms)  # Prolif expects single char chain IDs

        with tempfile.TemporaryDirectory() as tmp_dir:
            write_path = Path(tmp_dir) / "pocket.pdb"

            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, atoms)
            pdb_file.write(write_path)

            protein_mda = mda.Universe(str(write_path.resolve()), guess_bonds=True)
            protein_mol = plf.Molecule.from_mda(protein_mda)

        return protein_mol


    # *** Validation ***

    def _check_pocket(self, atoms: AtomSet, bonds: BondSet, confs: ConfSet):
        check_type(atoms, AtomSet, "atoms")
        check_type(bonds, BondSet, "bonds")
        check_type(confs, ConfSet, "confs")

        if not atoms.has_residue_annotations:
            raise ValueError("ProteinPocket requires AtomSet with residue annotations (res_names, atom_names, etc.)")
        if len(bonds) > 0 and bonds.min_index < 0:
            raise RuntimeError("Bond indices within a pocket cannot be negative.")
        if len(bonds) > 0 and bonds.max_index >= len(atoms):
            raise RuntimeError("The maximum atom index in bonds cannot be larger than the largest atom index.")
        if confs.n_atoms != len(atoms):
            raise RuntimeError("The number of atoms in the atom set and conf set must be the same.")


# *****************************************************************************
# ********************** Batched Protein Representation ***********************
# *****************************************************************************


class ProteinBatch(Sequence):
    """Utility class for loading, saving and batching Protein objects."""

    def __init__(self, proteins: list[Protein], hdf5_file: Union[h5py.File, list[h5py.File], None] = None):
        for protein in proteins:
            check_type(protein, Protein, "protein object")

        open_fps = []
        if hdf5_file is not None:
            if isinstance(hdf5_file, h5py.File):
                open_fps = [hdf5_file]
            elif isinstance(hdf5_file, list):
                if len(hdf5_file) > 0:
                    check_type(hdf5_file[0], h5py.File, "hdf5 file list item")
                open_fps = hdf5_file
            else:
                raise TypeError("hdf5_file must be either an h5py.File or a list of h5py.File objects.")

        self._proteins = proteins
        self._open_fps = open_fps


    # *** Publicly exposed properties ***

    @property
    def lengths(self) -> list[int]:
        return [len(protein) for protein in self._proteins]

    @property
    def mask(self) -> TArr:
        return smolF.pad_arrays([np.ones(protein.seq_length) for protein in self._proteins])

    @property
    def atomics(self) -> TArr:
        return smolF.pad_arrays([protein.atomics for protein in self._proteins])

    @property
    def charges(self) -> TArr:
        return smolF.pad_arrays([protein.charges for protein in self._proteins])

    @property
    def res_names(self) -> TArr:
        return _pad_string_lists([protein.res_names for protein in self._proteins])

    @property
    def atom_names(self) -> TArr:
        return _pad_string_lists([protein.atom_names for protein in self._proteins])

    @property
    def res_ids(self) -> TArr:
        return smolF.pad_arrays([protein.res_ids for protein in self._proteins])

    @property
    def coords(self) -> TArr:
        # Protein.coords is 2D [n_atoms, 3], so just pad directly
        coords = [protein.coords for protein in self._proteins]
        padded = smolF.pad_arrays(coords)
        return padded

    @property
    def bond_indices(self) -> TArr:
        return smolF.pad_arrays([protein.bond_indices for protein in self._proteins])

    @property
    def bond_types(self) -> TArr:
        return smolF.pad_arrays([protein.bond_types for protein in self._proteins])

    @property
    def adjacency(self) -> TArr:
        max_length = max(self.lengths)
        adjs = [protein.bonds.adj_matrix(max_length) for protein in self._proteins]
        return np.stack(adjs, axis=0)


    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        return len(self._proteins)

    def __getitem__(self, index: int) -> Protein:
        return self._proteins[index]

    def subset(self, idxs: list[int]) -> ProteinBatch:
        # Take all fps since we currently don't have a way of knowing which correspond to subset
        subset_proteins = [self._proteins[idx] for idx in idxs]
        batch = ProteinBatch(subset_proteins, self._open_fps)
        return batch


    # *** IO and conversion utility functions ***

    @staticmethod
    def _from_core_repr(obj: list[dict[str, dict[str, TArr]]]) -> ProteinBatch:
        proteins = [Protein._from_core_repr(protein_data) for protein_data in obj]
        return ProteinBatch(proteins)

    @staticmethod
    def from_bytes(data: bytes) -> ProteinBatch:
        obj = pickle.loads(data)
        return ProteinBatch._from_core_repr(obj)

    @staticmethod
    def from_batches(batches: list[ProteinBatch]) -> ProteinBatch:
        """Accumulate a list of ProteinBatch objects into one batch"""

        proteins = [protein for batch in batches for protein in batch]
        open_fps = [fp for batch in batches for fp in batch._open_fps]
        batch = ProteinBatch(proteins, hdf5_file=open_fps)
        return batch

    @staticmethod
    def load(save_path: Union[str, Path], n_shards: int = None) -> ProteinBatch:
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

        shards = [ProteinBatch.load_hdf5_shard(shard_path) for shard_path in sorted_paths]
        batch = ProteinBatch.from_batches(shards)
        return batch

    @staticmethod
    def load_hdf5_shard(save_file: Union[str, Path]) -> ProteinBatch:
        save_file = Path(save_file)

        if save_file.suffix != ".hdf5":
            raise RuntimeError("Save file must have an hdf5 suffix.")

        hdf5_file = h5py.File(save_file, "r")
        proteins = ProteinBatch._load_from_group(hdf5_file)
        batch = ProteinBatch(proteins, hdf5_file=hdf5_file)
        return batch

    @staticmethod
    def _load_from_group(group: h5py.Group) -> list[Protein]:
        """Load proteins from an HDF5 group."""

        atom_arrays = {name: arr for name, arr in group["atoms"].items()}
        atom_sets = AtomSet.atoms_from_arrays(atom_arrays)

        bond_arrays = {name: arr for name, arr in group["bonds"].items()}
        bond_sets = BondSet.bonds_from_arrays(bond_arrays)

        n_atoms = len(atom_sets)
        n_bonds = len(bond_sets)

        if n_atoms != n_bonds:
            raise RuntimeError(f"Number of AtomSet and BondSet must be equal, got {n_atoms} and {n_bonds}")

        if "confs" not in group:
            raise RuntimeError("Protein data must contain conformers.")

        conf_arrays = {name: arr for name, arr in group["confs"].items()}
        conf_sets = ConfSet.confs_from_arrays(conf_arrays)

        if len(conf_sets) != n_atoms:
            raise RuntimeError(f"Number of ConfSet proteins must match number of AtomSet proteins.")

        meta_bytes = group["meta"].attrs["metas"].tobytes()
        metas = [None if len(meta) == 0 else meta for meta in pickle.loads(meta_bytes)]

        zipped = zip(atom_sets, bond_sets, conf_sets, metas)
        proteins = [Protein(atoms, bonds, confs, meta=meta) for atoms, bonds, confs, meta in zipped]
        return proteins

    def _to_core_repr(self) -> list[dict[str, dict[str, TArr]]]:
        """A representation of the proteins using only built-in types and numpy arrays."""

        dict_list = [protein._to_core_repr() for protein in self._proteins]
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
        protein_shards = [[protein for protein in ps if protein is not None] for ps in grouper(self, shard_size)]

        for idx, shard in enumerate(protein_shards):
            shard_batch = ProteinBatch(shard)
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
        """Save protein data to an HDF5 group."""

        atoms = [protein.atoms for protein in self._proteins]
        bonds = [protein.bonds for protein in self._proteins]
        confs = [protein.confs for protein in self._proteins]

        atom_arrays = AtomSet.arrays_from_atoms(atoms)
        bond_arrays = BondSet.arrays_from_bonds(bonds)
        conf_arrays = ConfSet.arrays_from_confs(confs)

        metas = [protein.meta if protein.meta is not None else {} for protein in self._proteins]
        metas = np.void(pickle.dumps(metas, protocol=PICKLE_PROTOCOL))

        atom_group = group.create_group("atoms")
        for name, arr in atom_arrays.items():
            atom_group.create_dataset(name, data=arr)

        bond_group = group.create_group("bonds")
        for name, arr in bond_arrays.items():
            bond_group.create_dataset(name, data=arr)

        conf_group = group.create_group("confs")
        for name, arr in conf_arrays.items():
            conf_group.create_dataset(name, data=arr)

        meta_group = group.create_group("meta", track_order=True)
        meta_group.attrs["metas"] = metas

    def close_hdf5(self) -> None:
        """Closes any HDF5 file associated with this batch.

        If the batch was read from an HDF5 file this will close the underlying file, stopping any further reads. If the
        batch did not originate from HDF5 data this function does not do anything.

        NOTE even if the proteins in the batch have been transferred to a different ProteinBatch object this will stop
        reads from any data within the HDF5 file, so only close the file if you are sure the data will not be read.
        """

        # Close the files but don't set them to None, then we can open them again if needed
        if self._open_fps is not None:
            for fp in self._open_fps:
                fp.close() if fp is not None else None
