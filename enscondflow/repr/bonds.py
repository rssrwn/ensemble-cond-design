from __future__ import annotations

import h5py
import torch
import numpy as np
from rdkit import Chem
from typing import Optional, Union
from collections.abc import Sequence
from biotite.structure import BondList

import enscondflow.util.functional as smolF
from enscondflow.repr.lazydata import LazyData
from enscondflow.repr.vocab import BondVocab
from enscondflow.repr.util import check_shape_len, check_dim_shape, check_type, check_dict_key


SUPPORTED_STEREO = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ
]


TArr = np.ndarray


class BondSet(Sequence):
    def __init__(self, bonds: Union[TArr, LazyData]):
        check_type(bonds, [np.ndarray, LazyData], "bonds")
        check_shape_len(bonds, 2, "bonds")
        check_dim_shape(bonds, 1, 3, "bonds")

        if isinstance(bonds, np.ndarray):
            bonds = bonds.astype(np.int32)

        self._bonds = bonds


    # *** Publicly exposed properties ***

    @property
    def bonds(self) -> TArr:
        """Returns array of shape [n_bonds, 3]"""

        if isinstance(self._bonds, LazyData):
            return self._bonds.read().astype(np.int32)

        return self._bonds

    @property
    def seq_length(self) -> int:
        return len(self)

    @property
    def indices(self) -> TArr:
        return self.bonds[:, :2]

    @property
    def types(self) -> TArr:
        return self.bonds[:, 2]

    @property
    def min_index(self) -> Union[int, None]:
        if len(self) == 0:
            return None

        index = self.indices.min().item()
        return index

    @property
    def max_index(self) -> Union[int, None]:
        if len(self) == 0:
            return None
        
        index = self.indices.max().item()
        return index


    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        # Both np arr and H5Data provide length functions, so we don't need to force a read
        return len(self._bonds)

    def __getitem__(self, index: Union[int, TArr]) -> Union[BondSet, tuple[int, int, int]]:
        """Get an item from the set of bonds.

        The index can be provided as either an int or a numpy array. If index is an array this provides numpy style
        indexing and returns another BondSet of indexed bonds. If index is an int this function just provides simple
        indexing and returns a tuple of (start index, end index, bond type).
        """

        if isinstance(index, int):
            bond = self.bonds[index]
            return tuple(bond.tolist())

        if isinstance(index, np.ndarray):
            bonds = self.bonds[index]
            bonds = self.copy_with(bonds=bonds)
            return bonds

        raise TypeError("index must be either an int or an np array.")

    def adj_matrix(self, n_atoms: int) -> TArr:
        if len(self) > 0 and self.indices.max().item() >= n_atoms:
            raise ValueError("The largest atom index is larger than the number of atoms requested.")

        # TODO create numpy version of this function
        bonds = torch.tensor(self.bonds, dtype=torch.int32)
        adj = smolF.adj_from_edges(bonds[:, :2], bonds[:, 2], n_atoms, symmetric=True)
        adj = adj.numpy()

        return adj

    def read(self) -> BondSet:
        return BondSet(self.bonds)

    def copy_with(
        self,
        bonds: Optional[TArr] = None,
        bond_indices: Optional[TArr] = None,
        bond_types: Optional[TArr] = None
    ) -> BondSet:

        if bonds is not None and (bond_indices is not None or bond_types is not None):
            raise ValueError("BondSet copy_with cannot be provided with both bonds and bond indices or types")

        if bond_indices is not None and bond_types is not None and bond_indices.shape[0] != bond_types.shape[0]:
            raise ValueError("The length of bond_indices and bond_types must be the same.")

        if bond_indices is not None and bond_types is None and bond_indices.shape[0] != len(self):
            raise ValueError("The length of bond_indices must match the current bonds, if types are not provided.")

        if bond_indices is None and bond_types is not None and len(self) != bond_types.shape[0]:
            raise ValueError("The length of bond_type must match the current bonds, if indices are not provided.")

        if bonds is not None:
            check_shape_len(bonds, 2, "bonds")
            check_dim_shape(bonds, 1, 3, "bonds")

            bond_indices = bonds[:, :2]
            bond_types = bonds[:, 2]

        bond_indices = self.indices.copy() if bond_indices is None else bond_indices.copy()
        bond_types = self.types.copy() if bond_types is None else bond_types.copy()
        bond_types = np.expand_dims(bond_types, axis=1)

        bonds = np.concat((bond_indices, bond_types), axis=1)
        bonds = BondSet(bonds)
        return bonds

    def copy(self) -> BondSet:
        return self.copy_with()

    def permute_atoms(self, indices: Union[list[int], TArr], in_place: bool = False) -> BondSet:
        """Used for permuting atom order. In this case the indices are the same as given to the AtomSet permute function."""

        if len(set(indices)) != len(indices):
            raise ValueError(f"Indices list cannot contain duplicates.")

        indices = np.array(indices)

        # Remove any bonds which have an atom which does not exist in the index list
        mask = np.isin(self.indices, indices)
        mask = mask.all(axis=1)

        rem_bond_indices = self.indices[mask, :]
        bond_types = self.types[mask]

        # Create a tensor which will map curr indices to new indices
        index_map = np.array([-1] * (indices.max().item() + 1))
        index_map[indices] = np.arange(indices.shape[0])

        # Then use the index map to convert the atom indices within the bond list
        bond_indices = index_map[rem_bond_indices]

        if not in_place:
            return self.copy_with(bond_indices=bond_indices, bond_types=bond_types)

        bonds = np.concat((bond_indices, bond_types[:, None]), axis=1).astype(np.int32)
        self._bonds = bonds

        return self


    # *** IO and conversion utility functions ***

    @staticmethod
    def from_rdkit(mol: Chem.rdchem.Mol) -> BondSet:
        # We need to kekulize the mol first so that aromatic bonds are converted to single and double
        # The aromatic property is still stored however

        kekul_mol = Chem.Mol(mol)
        Chem.Kekulize(kekul_mol)

        bond_list = []

        for bond in kekul_mol.GetBonds():
            bond_start = bond.GetBeginAtomIdx()
            bond_end = bond.GetEndAtomIdx()

            # Bonds in RDKit are not duplicated so we can switch the indices to keep upper tri only
            if bond_start > bond_end:
                bond_start = bond.GetEndAtomIdx()
                bond_end = bond.GetBeginAtomIdx()

            bond_type = bond.GetBondType()
            is_arom = bond.GetIsAromatic()
            stereo = bond.GetStereo()

            # Only keep stereo on non-arom double bonds — single/triple bonds can occassionally carry E/Z
            is_stereo_bond = bond_type == Chem.BondType.DOUBLE and not is_arom
            stereo = stereo if stereo in SUPPORTED_STEREO and is_stereo_bond else Chem.BondStereo.STEREONONE

            # Will throw an error if the bond cannot be found in the vocab
            bond_index = BondVocab.get_index(bond_type, is_arom, stereo)
            bond_list.append([bond_start, bond_end, bond_index])

        bonds = BondSet(np.array(bond_list))
        return bonds

    @staticmethod
    def from_biotite(bond_list: BondList) -> BondSet:
        """Create a BondSet from a biotite BondList.

        Biotite BondType values:
            0: ANY, 1: SINGLE, 2: DOUBLE, 3: TRIPLE, 4: QUADRUPLE,
            5: AROMATIC_SINGLE, 6: AROMATIC_DOUBLE, 7: AROMATIC_TRIPLE,
            8: COORDINATION, 9: AROMATIC (generic)

        Args:
            bond_list: Biotite BondList object

        Returns:
            BondSet with bonds encoded using BondVocab
        """

        if bond_list is None or len(bond_list.as_array()) == 0:
            return BondSet(np.zeros((0, 3), dtype=np.int32))

        bond_arr = bond_list.as_array()

        # Map biotite bond types to (RDKit BondType, is_aromatic)
        # Unsupported types (ANY=0, QUADRUPLE=4, COORDINATION=8) will raise an error
        biotite_to_rdkit = {
            1: (Chem.BondType.SINGLE, False),   # SINGLE
            2: (Chem.BondType.DOUBLE, False),   # DOUBLE
            3: (Chem.BondType.TRIPLE, False),   # TRIPLE
            5: (Chem.BondType.SINGLE, True),    # AROMATIC_SINGLE
            6: (Chem.BondType.DOUBLE, True),    # AROMATIC_DOUBLE
            7: (Chem.BondType.TRIPLE, True),    # AROMATIC_TRIPLE
            9: (Chem.BondType.SINGLE, True),    # AROMATIC (generic) -> treat as aromatic single
        }

        converted_bonds = []
        for bond in bond_arr:
            start, end, bond_type = bond[0], bond[1], bond[2]

            # Ensure upper triangular (start < end)
            if start > end:
                start, end = end, start

            if bond_type not in biotite_to_rdkit:
                raise ValueError(f"Unsupported biotite bond type {bond_type}.")

            rdkit_type, is_aromatic = biotite_to_rdkit[bond_type]
            bond_index = BondVocab.get_index(rdkit_type, is_aromatic)
            converted_bonds.append([start, end, bond_index])

        return BondSet(np.array(converted_bonds, dtype=np.int32))

    @staticmethod
    def from_dict(dict_repr) -> BondSet:
        bonds = dict_repr["bonds"]
        return BondSet(bonds)

    def to_dict(self) -> dict[str, np.ndarray]:
        dict_repr = {"bonds": self.bonds}
        return dict_repr

    @staticmethod
    def bonds_from_arrays(array_map: dict[str, Union[np.ndarray, h5py.Dataset]]) -> list[BondSet]:
        """Convert the merged arrays from arrays_from_bonds back into BondSets."""

        check_dict_key(array_map, "sizes", "bond array map")
        check_dict_key(array_map, "bonds", "bond array map")

        bond_arr = array_map["bonds"]
        is_hdf5 = isinstance(bond_arr, h5py.Dataset)

        if (not is_hdf5) and (not isinstance(bond_arr, np.ndarray)):
            raise TypeError(f"Bond array must be either np.ndarray or h5py.Dataset, got {type(bond_arr)}")

        # We need to read the sizes of bond lists into memory to recreate them
        sizes = np.array(array_map["sizes"][()]).tolist()
        bond_arr = bond_arr.copy() if not is_hdf5 else bond_arr

        curr_idx = 0
        bond_sets = []

        for n_bonds in sizes:
            if is_hdf5:
                bonds = LazyData(bond_arr, curr_idx, n_bonds)
            else:
                bonds = bond_arr[curr_idx : curr_idx + n_bonds].copy()

            bond_set = BondSet(bonds)
            bond_sets.append(bond_set)
            curr_idx += n_bonds

        return bond_sets

    @staticmethod
    def arrays_from_bonds(bond_sets: list[BondSet]) -> dict[str, np.ndarray]:
        """Merge multiple bond sets into a dictionary of numpy arrays."""

        bonds = np.concat([bond_set.bonds for bond_set in bond_sets], axis=0)
        sizes = np.array([len(bond_set) for bond_set in bond_sets])

        arrays = {
            "bonds": bonds,
            "sizes": sizes
        }
        return arrays
