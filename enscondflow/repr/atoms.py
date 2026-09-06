
from __future__ import annotations

import h5py
import numpy as np
from rdkit import Chem
from typing import Optional, Union
from collections.abc import Sequence
from biotite.structure import AtomArray

import enscondflow.util.rdkit as smolRD
from enscondflow.repr.lazydata import LazyData
from enscondflow.repr.util import check_shape_len, check_shapes_equal, check_type, check_dict_key


TArr = np.ndarray

# Maximum string lengths for fixed-length string storage
MAX_RES_NAME_LEN = 3
MAX_ATOM_NAME_LEN = 4


# *** Some util functions specific to AtomSet ***

def _check_annotation_consistency(atom_sets: list[AtomSet], attr_name: str) -> bool:
    """Check consistency of optional annotations - all atom sets must either have or not have each annotation"""

    has_attr = [getattr(atoms, attr_name) is not None for atoms in atom_sets]
    if any(has_attr) and not all(has_attr):
        raise ValueError(f"Inconsistent {attr_name}: some atom sets have {attr_name} but others do not")
    return all(has_attr)


def _check_string_lengths(arr: Union[np.ndarray, LazyData], max_len: int, name: str) -> None:
    """Validate that all strings in array are within max length. Skip validation for LazyData."""

    if isinstance(arr, LazyData):
        return

    for i, s in enumerate(arr):
        if len(s) > max_len:
            raise ValueError(f"{name}[{i}] = '{s}' exceeds max length {max_len}")


# *** AtomSet Class ***

class AtomSet(Sequence):
    def __init__(
        self,
        atomics: Union[TArr, LazyData],
        charges: Union[TArr, LazyData, None] = None,
        res_names: Union[TArr, LazyData, None] = None,
        atom_names: Union[TArr, LazyData, None] = None,
        res_ids: Union[TArr, LazyData, None] = None
    ):
        check_type(atomics, [np.ndarray, LazyData], "atomics")
        check_shape_len(atomics, 1, "atomics")

        # Assume all formal charges to 0 if they are not provided
        # This is not true for the other optional features
        charges = np.zeros_like(atomics) if charges is None else charges

        check_type(charges, [np.ndarray, LazyData], "charges")
        check_shape_len(charges, 1, "charges")
        check_shapes_equal(atomics, charges, 0)

        # Validate shapes of optional annotations if provided
        if res_names is not None:
            check_shape_len(res_names, 1, "res names")
            check_shapes_equal(atomics, res_names, 0)
            _check_string_lengths(res_names, MAX_RES_NAME_LEN, "res_names")

        if atom_names is not None:
            check_shape_len(atom_names, 1, "atom names")
            check_shapes_equal(atomics, atom_names, 0)
            _check_string_lengths(atom_names, MAX_ATOM_NAME_LEN, "atom_names")

        if res_ids is not None:
            check_shape_len(res_ids, 1, "res ids")
            check_shapes_equal(atomics, res_ids, 0)

        # Store data in 16 bit arrays if the data has been read from disk
        if isinstance(atomics, np.ndarray):
            atomics = atomics.astype(np.int16)
        if isinstance(charges, np.ndarray):
            charges = charges.astype(np.int16)

        # Optional residue-level annotations (used for proteins)
        if res_ids is not None and isinstance(res_ids, np.ndarray):
            res_ids = res_ids.astype(np.int32)

        self._atomics = atomics
        self._charges = charges
        self._res_names = res_names
        self._atom_names = atom_names
        self._res_ids = res_ids


    # *** Publicly exposed properties ***

    @property
    def atomics(self) -> TArr:
        """Returns array of shape [n_atoms,]"""

        if isinstance(self._atomics, LazyData):
            return self._atomics.read().astype(np.int16)

        return self._atomics

    @property
    def charges(self) -> TArr:
        """Returns array of shape [n_atoms,]"""

        if isinstance(self._charges, LazyData):
            return self._charges.read().astype(np.int16)

        return self._charges

    @property
    def charged_symbols(self) -> list[str]:
        atoms = [smolRD.PT.symbol_from_atomic(atomic) for atomic in self.atomics.tolist()]
        charged_atoms = [f"{atom}_{str(charge)}" for atom, charge in zip(atoms, self.charges.tolist())]
        return charged_atoms

    @property
    def res_names(self) -> Union[TArr, None]:
        """Returns array of residue names, shape [n_atoms,], or None if not set"""

        if self._res_names is None:
            return None

        if isinstance(self._res_names, LazyData):
            arr = self._res_names.read()
            return AtomSet._decode_bytes(arr)

        return self._res_names

    @property
    def atom_names(self) -> Union[TArr, None]:
        """Returns array of atom names, shape [n_atoms,], or None if not set"""

        if self._atom_names is None:
            return None

        if isinstance(self._atom_names, LazyData):
            arr = self._atom_names.read()
            return AtomSet._decode_bytes(arr)

        return self._atom_names

    @property
    def res_ids(self) -> Union[TArr, None]:
        """Returns array of residue IDs, shape [n_atoms,], or None if not set"""

        if self._res_ids is None:
            return None

        if isinstance(self._res_ids, LazyData):
            return self._res_ids.read().astype(np.int32)

        return self._res_ids

    @property
    def has_residue_annotations(self) -> bool:
        """Returns True if all residue-level annotations are present"""

        has_annotations = [
            self._res_names is not None,
            self._atom_names is not None,
            self._res_ids is not None
        ]
        return all(has_annotations)

    @property
    def seq_length(self) -> int:
        return len(self)


    # *** Basic indexing and utility functions ***

    def __len__(self) -> int:
        # Both np arr and H5Data provide length functions, so we don't need to force a read
        return len(self._atomics)

    def __getitem__(self, index: Union[int, TArr]) -> Union[AtomSet, tuple[int, int]]:
        """Get an item from the set of atoms.

        The index can be provided as either an int or a numpy array. If index is an array this provides numpy style
        indexing and returns another AtomSet of indexed atoms. If index is an int this function just provides simple
        indexing and returns a tuple of (atomic, charge)
        """

        if isinstance(index, int):
            atomic = self.atomics[index].item()
            charge = self.charges[index].item()
            return atomic, charge

        if isinstance(index, np.ndarray):
            atomics = self.atomics[index]
            charges = self.charges[index]

            res_names = self.res_names[index] if self.res_names is not None else None
            atom_names = self.atom_names[index] if self.atom_names is not None else None
            res_ids = self.res_ids[index] if self.res_ids is not None else None

            atoms = AtomSet(
                atomics,
                charges=charges,
                res_names=res_names,
                atom_names=atom_names,
                res_ids=res_ids
            )
            return atoms

        raise TypeError("index must be either an int or an np array.")

    def read(self) -> AtomSet:
        """Force the data to read into memory if it's not already"""

        atoms = AtomSet(
            self.atomics,
            charges=self.charges,
            res_names=self.res_names,
            atom_names=self.atom_names,
            res_ids=self.res_ids
        )
        return atoms

    def copy_with(
        self,
        atomics: Optional[TArr] = None,
        charges: Optional[TArr] = None,
        res_names: Optional[TArr] = None,
        atom_names: Optional[TArr] = None,
        res_ids: Optional[TArr] = None
    ) -> AtomSet:

        atomics = self.atomics.copy() if atomics is None else atomics.copy()
        charges = self.charges.copy() if charges is None else charges.copy()

        if res_names is None and self.res_names is not None:
            res_names = self.res_names.copy()
        if atom_names is None and self.atom_names is not None:
            atom_names = self.atom_names.copy()
        if res_ids is None and self.res_ids is not None:
            res_ids = self.res_ids.copy()

        atoms = AtomSet(
            atomics,
            charges=charges,
            res_names=res_names,
            atom_names=atom_names,
            res_ids=res_ids
        )
        return atoms

    def copy(self) -> AtomSet:
        return self.copy_with()

    def permute_atoms(self, indices: Union[list[int], TArr], in_place: bool = False) -> AtomSet:
        """Used for permuting atom order.

        Can be used for reordering or taking a subset, but not for duplicating.

        Args:
            indices (list[int] or np.ndarray): Ordering of atoms requested for the new set.
            in_place (bool): If True, mutate this AtomSet in place and return self.

        Returns:
            AtomSet: Atom set with atoms reodered according to indices.
        """

        indices = np.array(indices)

        if len(set(indices.tolist())) != len(indices.tolist()):
            raise ValueError(f"Indices list cannot contain duplicates.")

        if indices.min().item() < 0:
            raise ValueError("Indices cannot be negative.")

        if indices.max().item() >= self.seq_length:
            raise ValueError(f"Index {max(indices)} is out of bounds for atom set with {self.seq_length} atoms.")

        if not in_place:
            return self[indices]

        self._atomics = self.atomics[indices]
        self._charges = self.charges[indices]

        if self._res_names is not None:
            self._res_names = self.res_names[indices]
        if self._atom_names is not None:
            self._atom_names = self.atom_names[indices]
        if self._res_ids is not None:
            self._res_ids = self.res_ids[indices]

        return self

    def pad(
        self,
        n_atoms: int,
        pad_atomic: int = 0,
        pad_charge: int = 0,
        pad_res_name: str = "PAD",
        pad_atom_name: str = "PAD",
        pad_res_id: int = -1,
        in_place: bool = False
    ) -> AtomSet:
        """Pad the atoms to length n_atoms using provided pad tokens for each field"""

        if n_atoms < len(self):
            raise ValueError(f"Cannot pad to fewer atoms than exist in the molecule.")

        if n_atoms == len(self):
            return self if in_place else self.copy()

        n_pad_atoms = n_atoms - len(self)
        pad_atomics = np.array([pad_atomic] * n_pad_atoms)
        pad_charges = np.array([pad_charge] * n_pad_atoms)

        atomics = np.concat((self.atomics, pad_atomics), axis=0)
        charges = np.concat((self.charges, pad_charges), axis=0)

        res_names = None
        atom_names = None
        res_ids = None

        if self.res_names is not None:
            res_names = np.concat((self.res_names, np.array([pad_res_name] * n_pad_atoms)), axis=0)
        if self.atom_names is not None:
            atom_names = np.concat((self.atom_names, np.array([pad_atom_name] * n_pad_atoms)), axis=0)
        if self.res_ids is not None:
            res_ids = np.concat((self.res_ids, np.array([pad_res_id] * n_pad_atoms)), axis=0)

        if not in_place:
            atoms_copy = self.copy_with(
                atomics=atomics,
                charges=charges,
                res_names=res_names,
                atom_names=atom_names,
                res_ids=res_ids,
            )
            return atoms_copy

        self._atomics = atomics.astype(np.int16)
        self._charges = charges.astype(np.int16)

        if res_names is not None:
            self._res_names = res_names
        if atom_names is not None:
            self._atom_names = atom_names
        if res_ids is not None:
            self._res_ids = res_ids.astype(np.int32)

        return self


    # *** IO and conversion utility functions ***

    @staticmethod
    def from_rdkit(mol: Chem.rdchem.Mol) -> AtomSet:
        atomics = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

        atomics = np.array(atomics)
        charges = np.array(charges)

        atoms = AtomSet(atomics, charges=charges)
        return atoms

    @staticmethod
    def from_biotite(atom_array: AtomArray) -> AtomSet:
        elements = [el if el != "SE" else "Se" for el in atom_array.element.tolist()]
        atomics = np.array([smolRD.PT.atomic_from_symbol(el) for el in elements])

        charges = np.zeros(len(atomics), dtype=np.int16)
        if "charge" in atom_array.get_annotation_categories():
            charges = atom_array.charge.astype(np.int16)

        res_names = atom_array.res_name
        atom_names = atom_array.atom_name
        res_ids = atom_array.res_id

        atoms = AtomSet(
            atomics,
            charges=charges,
            res_names=res_names,
            atom_names=atom_names,
            res_ids=res_ids
        )
        return atoms

    @staticmethod
    def from_dict(dict_repr) -> AtomSet:
        check_dict_key(dict_repr, "atomics")
        check_dict_key(dict_repr, "charges")

        atomics = dict_repr["atomics"]
        charges = dict_repr["charges"]

        # Check for optional features
        res_names = dict_repr.get("res_names")
        atom_names = dict_repr.get("atom_names")
        res_ids = dict_repr.get("res_ids")

        atoms = AtomSet(
            atomics,
            charges=charges,
            res_names=res_names,
            atom_names=atom_names,
            res_ids=res_ids
        )
        return atoms

    def to_dict(self) -> dict[str, np.ndarray]:
        dict_repr = {
            "atomics": self.atomics,
            "charges": self.charges
        }

        if self.res_names is not None:
            dict_repr["res_names"] = self.res_names
        if self.atom_names is not None:
            dict_repr["atom_names"] = self.atom_names
        if self.res_ids is not None:
            dict_repr["res_ids"] = self.res_ids

        return dict_repr

    @staticmethod
    def atoms_from_arrays(array_map: dict[str, Union[np.ndarray, h5py.Dataset]]) -> list[AtomSet]:
        """Convert merged arrays back into AtomSets"""

        check_dict_key(array_map, "sizes", "atom array map")
        check_dict_key(array_map, "atomics", "atom array map")
        check_dict_key(array_map, "charges", "atom array map")

        # Read sizes of atom sets into memory
        sizes = np.array(array_map["sizes"][()]).tolist()

        # Check all provided arrays have same total number of atoms (except sizes which measures n_mols)
        array_lens = [arr.shape[0] for name, arr in array_map.items() if name != "sizes"]
        if not all([arr_len == array_lens[0] for arr_len in array_lens]):
            raise RuntimeError("All provided atom arrays must be the same length.")

        split_arr_map = {name: AtomSet._split_array(arr, sizes) for name, arr in array_map.items()}

        atomics_arrs = split_arr_map["atomics"]
        charges_arrs = split_arr_map["charges"]

        # Optional annotation arrays - decode bytes to strings for string-type annotations
        res_names_arrs = split_arr_map.get("res_names", [None] * len(sizes))
        atom_names_arrs = split_arr_map.get("atom_names", [None] * len(sizes))
        res_ids_arrs = split_arr_map.get("res_ids", [None] * len(sizes))

        # Decode byte strings to unicode (only for in-memory arrays, LazyData will decode on read)
        res_names_arrs = [AtomSet._decode_bytes(arr) for arr in res_names_arrs]
        atom_names_arrs = [AtomSet._decode_bytes(arr) for arr in atom_names_arrs]

        zipped = zip(atomics_arrs, charges_arrs, res_names_arrs, atom_names_arrs, res_ids_arrs)

        atom_sets = []
        for atomics, charges, res_names, atom_names, res_ids in zipped:
            atom_set = AtomSet(
                atomics,
                charges=charges,
                res_names=res_names,
                atom_names=atom_names,
                res_ids=res_ids
            )
            atom_sets.append(atom_set)

        return atom_sets

    @staticmethod
    def arrays_from_atoms(atom_sets: list[AtomSet]) -> dict[str, np.ndarray]:
        """Merge multiple atom sets into a dictionary of arrays.

        Many atom sets are stored together to allow efficient initialisation.

        Each array will correspond to one 'column' of the data (eg. atomics, charges) but only if these are present
        in all provided atom sets. The indices into the dictionary correspond to internal names given to the features.

        All provided atom sets must have the same set of columns.
        """

        # All atom sets will have atomics and charges (charges are set to 0 if not provided)
        atomics = np.concat([atoms.atomics for atoms in atom_sets]).astype(np.int16)
        charges = np.concat([atoms.charges for atoms in atom_sets]).astype(np.int16)
        sizes = np.array([len(atoms) for atoms in atom_sets])

        arrays = {
            "atomics": atomics,
            "charges": charges,
            "sizes": sizes
        }

        all_have_res_names = _check_annotation_consistency(atom_sets, "res_names")
        all_have_atom_names = _check_annotation_consistency(atom_sets, "atom_names")
        all_have_res_ids = _check_annotation_consistency(atom_sets, "res_ids")

        # Use fixed-length byte strings for HDF5 compatibility
        if all_have_res_names:
            arrays["res_names"] = np.concat([atoms.res_names for atoms in atom_sets]).astype(f'S{MAX_RES_NAME_LEN}')
        if all_have_atom_names:
            arrays["atom_names"] = np.concat([atoms.atom_names for atoms in atom_sets]).astype(f'S{MAX_ATOM_NAME_LEN}')
        if all_have_res_ids:
            arrays["res_ids"] = np.concat([atoms.res_ids for atoms in atom_sets]).astype(np.int32)

        return arrays

    @staticmethod
    def _split_array(arr: Union[np.ndarray, h5py.Dataset], sizes: list[int]) -> list[Union[LazyData, np.ndarray]]:
        is_hdf5 = isinstance(arr, h5py.Dataset)

        if (not is_hdf5) and (not isinstance(arr, np.ndarray)):
            raise TypeError(f"Each array must be either np.ndarray or h5py.Dataset, got {type(arr)}")

        arr = arr.copy() if not is_hdf5 else arr

        curr_idx = 0
        splits = []

        for n_atoms in sizes:
            if is_hdf5:
                split_arr = LazyData(arr, curr_idx, n_atoms)
            else:
                split_arr = arr[curr_idx : curr_idx + n_atoms]

            splits.append(split_arr)
            curr_idx += n_atoms

        return splits

    @staticmethod
    def _decode_bytes(arr: Union[np.ndarray, LazyData, None]) -> Union[np.ndarray, LazyData, None]:
        """Decode fixed-length byte string arrays to unicode strings. Returns None or LazyData unchanged."""

        if arr is None or isinstance(arr, LazyData):
            return arr

        # Check if this is a fixed-length byte string array that needs decoding
        if arr.dtype.kind == 'S':
            return arr.astype('U')

        return arr
