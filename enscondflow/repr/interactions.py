from __future__ import annotations

import h5py
import pickle
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import prolif as plf

from enscondflow.repr.complex import BindingComplex
from enscondflow.repr.util import PICKLE_PROTOCOL, check_dict_key


TArr = np.ndarray


# All possible prolif interaction types.
# PiStacking includes both edge-to-face and face-to-face configurations.
# Users must use a subset of this list when specifying interactions.
PROLIF_INTERACTIONS = [
    "Hydrophobic",
    "VdWContact",
    "MetalAcceptor",
    "MetalDonor",
    "Cationic",
    "Anionic",
    "XBAcceptor",
    "XBDonor",
    "CationPi",
    "PiCation",
    "PiStacking",
    "HBAcceptor",
    "HBDonor"
]

# Maps for grouping related interaction types into a single category.
# These are used to reduce the dimensionality of the interaction representation
# by combining symmetric or related interactions (e.g. donor/acceptor pairs).

GROUP_METAL_MAP = {
    "MetalAcceptor": "metal-interaction",
    "MetalDonor": "metal-interaction"
}

GROUP_IONIC_MAP = {
    "Cationic": "ionic-interaction",
    "Anionic": "ionic-interaction"
}

GROUP_HALOGEN_MAP = {
    "XBAcceptor": "halogen-bond",
    "XBDonor": "halogen-bond"
}

GROUP_PI_MAP = {
    "CationPi": "pi-cation",
    "PiCation": "pi-cation"
}

GROUP_HBOND_MAP = {
    "HBAcceptor": "hydrogen-bond",
    "HBDonor": "hydrogen-bond"
}


# *** Validation functions ***

def _check_interaction_types_exist(interaction_types: list[str]) -> None:
    """Validate that all interaction types are recognised prolif interaction names."""

    for int_type in interaction_types:
        if int_type not in PROLIF_INTERACTIONS:
            raise ValueError(f"Interaction type {int_type} is not recognised.")


def _check_interaction_profile(check_interactions: list[str], interaction_profile: list[str]) -> None:
    """Check that all interactions in check_interactions exist in interaction_profile."""

    for int_type in check_interactions:
        if int_type not in interaction_profile:
            raise ValueError(f"Interaction type {int_type} not found in current interaction profile.")


# *********************
# *** Interaction   ***
# *********************


@dataclass(frozen=True)
class Interaction:
    """Immutable representation of a single protein-ligand interaction.

    Stores the atom indices involved in the interaction for both protein and ligand,
    along with the interaction type. Uses tuples to support multi-atom interactions
    (e.g. pi-stacking between ring systems).
    """

    protein_atoms: tuple[int, ...]
    ligand_atoms: tuple[int, ...]
    interaction_type: str

    def with_type(self, new_type: str) -> Interaction:
        """Return a new Interaction with the type changed."""

        return Interaction(self.protein_atoms, self.ligand_atoms, new_type)

    def remap_atoms(self, protein_map: dict[int, int], ligand_map: dict[int, int]) -> Optional[Interaction]:
        """Return a new Interaction with atom indices remapped.

        Args:
            protein_map: Mapping from old protein atom indices to new indices.
            ligand_map: Mapping from old ligand atom indices to new indices.

        Returns:
            New Interaction with remapped indices, or None if any atom is not in the maps.
        """

        new_protein_atoms = []
        for idx in self.protein_atoms:
            if idx not in protein_map:
                return None
            new_protein_atoms.append(protein_map[idx])

        new_ligand_atoms = []
        for idx in self.ligand_atoms:
            if idx not in ligand_map:
                return None
            new_ligand_atoms.append(ligand_map[idx])

        return Interaction(tuple(new_protein_atoms), tuple(new_ligand_atoms), self.interaction_type)


# ***********************
# *** InteractionSet  ***
# ***********************


class InteractionSet:
    """Sparse representation of protein-ligand interactions.

    Stores interactions as a collection of Interaction objects rather than a dense array.
    This is more memory-efficient when interactions are sparse (which is typical) and
    preserves multi-atom groupings from prolif.

    Dense arrays are generated on-demand via the array property.
    """

    def __init__(self, interactions: Sequence[Interaction], n_protein_atoms: int, n_ligand_atoms: int):
        self._interactions = tuple(interactions)
        self._n_protein_atoms = n_protein_atoms
        self._n_ligand_atoms = n_ligand_atoms
        self._interaction_types: Optional[list[str]] = None


    # *** Properties ***

    @property
    def n_protein_atoms(self) -> int:
        return self._n_protein_atoms

    @property
    def n_ligand_atoms(self) -> int:
        return self._n_ligand_atoms

    @property
    def interaction_types(self) -> list[str]:
        """Return unique interaction types present in this set, in sorted order."""

        if self._interaction_types is None:
            types_set = set()
            for interaction in self._interactions:
                types_set.add(interaction.interaction_type)
            self._interaction_types = sorted(types_set)

        return self._interaction_types

    @property
    def array(self) -> np.ndarray:
        """Generate dense array on-demand.

        Returns:
            Array of shape [n_protein_atoms, n_ligand_atoms, n_interaction_types].
            Each entry is 1 if the interaction exists, 0 otherwise.
        """

        int_types = self.interaction_types
        int_type_to_idx = {int_type: idx for idx, int_type in enumerate(int_types)}

        arr = np.zeros((self._n_protein_atoms, self._n_ligand_atoms, len(int_types)), dtype=np.int8)

        for interaction in self._interactions:
            int_idx = int_type_to_idx[interaction.interaction_type]
            for p_idx in interaction.protein_atoms:
                for l_idx in interaction.ligand_atoms:
                    arr[p_idx, l_idx, int_idx] = 1

        return arr

    def __len__(self) -> int:
        return len(self._interactions)


    # *** Creation ***

    @staticmethod
    def from_array(interaction_types: list[str], arr: np.ndarray) -> InteractionSet:
        """Create an InteractionSet from a dense array.

        Each non-zero entry arr[p, l, t] = 1 becomes one Interaction with single-atom tuples.
        Multi-atom groupings cannot be reconstructed from dense arrays.

        Args:
            interaction_types: List of interaction type names corresponding to the last dimension.
            arr: Dense array of shape [n_protein_atoms, n_ligand_atoms, n_interaction_types].

        Returns:
            InteractionSet with one Interaction per non-zero entry.
        """

        if len(arr.shape) != 3:
            raise ValueError(f"arr must have 3 dimensions, got shape {arr.shape}")

        n_protein, n_ligand, n_types = arr.shape
        if len(interaction_types) != n_types:
            raise ValueError(f"Number of interaction types must match last dim, got {len(interaction_types)} and {n_types}")

        interactions = []
        for p_idx in range(n_protein):
            for l_idx in range(n_ligand):
                for t_idx in range(n_types):
                    if arr[p_idx, l_idx, t_idx] != 0:
                        interaction = Interaction(
                            protein_atoms=(p_idx,),
                            ligand_atoms=(l_idx,),
                            interaction_type=interaction_types[t_idx]
                        )
                        interactions.append(interaction)

        return InteractionSet(interactions, n_protein, n_ligand)

    @staticmethod
    def from_prolif_ifp(ifp: dict, n_protein: int, n_ligand: int) -> InteractionSet:
        """Create an InteractionSet from a Prolif interaction fingerprint.

        Preserves multi-atom groupings (e.g. all atoms in a ring for pi-stacking).

        Args:
            ifp: Prolif interaction fingerprint dict for a single frame.
            n_protein: Number of protein atoms.
            n_ligand: Number of ligand atoms.

        Returns:
            InteractionSet with multi-atom interactions preserved.
        """

        interactions = []

        # ifp is keyed by (ligand_reskey, protein_reskey) tuples
        for _, res_interactions in ifp.items():

            # res_interactions is keyed by interaction type
            for int_type, interaction_list in res_interactions.items():

                # Each interaction contains atom indices for both molecules
                for interaction_data in interaction_list:
                    l_atom_idxs = tuple(interaction_data["parent_indices"]["ligand"])
                    p_atom_idxs = tuple(interaction_data["parent_indices"]["protein"])

                    interaction = Interaction(
                        protein_atoms=p_atom_idxs,
                        ligand_atoms=l_atom_idxs,
                        interaction_type=int_type
                    )
                    interactions.append(interaction)

        return InteractionSet(interactions, n_protein, n_ligand)

    @staticmethod
    def from_system(system: BindingComplex, interaction_types: list[str] = None) -> InteractionSet:
        """Extract interactions from a BindingComplex using Prolif.

        Args:
            system: The protein-ligand complex to analyze.
            interaction_types: List of interaction types to detect. If None, all types are detected.

        Returns:
            InteractionSet with detected interactions.
        """

        if interaction_types is not None:
            _check_interaction_types_exist(interaction_types)

        interaction_types = PROLIF_INTERACTIONS if interaction_types is None else interaction_types

        # Run prolif to get interaction fingerprint
        plf_fp = InteractionSet._interaction_fp(system, interaction_types)

        # ifp[0] because we only have a single frame (no MD trajectory)
        interactions = InteractionSet.from_prolif_ifp(plf_fp.ifp[0], len(system.protein), len(system.ligand))
        return interactions


    # *** Operations ***

    def subset(self, interaction_types: list[str]) -> InteractionSet:
        """Return a new InteractionSet with only the specified interaction types."""

        _check_interaction_profile(interaction_types, self.interaction_types)

        type_set = set(interaction_types)
        filtered = [i for i in self._interactions if i.interaction_type in type_set]
        return InteractionSet(filtered, self._n_protein_atoms, self._n_ligand_atoms)

    def subset_atoms(self, protein_mask: np.ndarray, ligand_mask: np.ndarray) -> InteractionSet:
        """Return a new InteractionSet with only atoms matching the masks.

        Args:
            protein_mask: Boolean array of length n_protein_atoms.
            ligand_mask: Boolean array of length n_ligand_atoms.

        Returns:
            New InteractionSet with remapped atom indices.
        """

        # Build mappings from old indices to new indices
        protein_map = {}
        new_idx = 0
        for old_idx, keep in enumerate(protein_mask):
            if keep:
                protein_map[old_idx] = new_idx
                new_idx += 1
        n_new_protein = new_idx

        ligand_map = {}
        new_idx = 0
        for old_idx, keep in enumerate(ligand_mask):
            if keep:
                ligand_map[old_idx] = new_idx
                new_idx += 1
        n_new_ligand = new_idx

        # Remap interactions
        remapped = []
        for interaction in self._interactions:
            new_interaction = interaction.remap_atoms(protein_map, ligand_map)
            if new_interaction is not None:
                remapped.append(new_interaction)

        return InteractionSet(remapped, n_new_protein, n_new_ligand)

    def group(self, group_map: dict[str, str]) -> InteractionSet:
        """Return a new InteractionSet with interaction types remapped.

        Args:
            group_map: Dict mapping old interaction type names to new grouped names.
                       Types not in the map are kept unchanged.

        Returns:
            New InteractionSet with remapped interaction types.
        """

        grouped = []
        for interaction in self._interactions:
            new_type = group_map.get(interaction.interaction_type, interaction.interaction_type)
            grouped.append(interaction.with_type(new_type))

        return InteractionSet(grouped, self._n_protein_atoms, self._n_ligand_atoms)


    # *** Serialization ***

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""

        # Store interactions as a list of dicts for easy unpickling
        interaction_data = []
        for i in self._interactions:
            interaction_data.append({
                "protein_atoms": i.protein_atoms,
                "ligand_atoms": i.ligand_atoms,
                "interaction_type": i.interaction_type
            })

        data_dict = {
            "interactions": interaction_data,
            "n_protein_atoms": self._n_protein_atoms,
            "n_ligand_atoms": self._n_ligand_atoms
        }
        return pickle.dumps(data_dict, protocol=PICKLE_PROTOCOL)

    @staticmethod
    def from_bytes(data: bytes) -> InteractionSet:
        """Deserialize from bytes."""

        obj = pickle.loads(data)

        check_dict_key(obj, "interactions")
        check_dict_key(obj, "n_protein_atoms")
        check_dict_key(obj, "n_ligand_atoms")

        interactions = []
        for i_data in obj["interactions"]:
            interaction = Interaction(
                protein_atoms=i_data["protein_atoms"],
                ligand_atoms=i_data["ligand_atoms"],
                interaction_type=i_data["interaction_type"]
            )
            interactions.append(interaction)

        return InteractionSet(interactions, obj["n_protein_atoms"], obj["n_ligand_atoms"])

    def copy(self) -> InteractionSet:
        """Return a copy of this InteractionSet."""

        # Interactions are immutable, so we just need a new container
        return InteractionSet(self._interactions, self._n_protein_atoms, self._n_ligand_atoms)


    # *** HDF5 batching ***

    @staticmethod
    def save_to_group(int_sets: list[Optional[InteractionSet]], group: h5py.Group) -> None:
        """Save interaction sets to an HDF5 group as serialized bytes."""

        # Serialize each InteractionSet to bytes
        serialized = []
        sizes = []
        for int_set in int_sets:
            if int_set is None:
                sizes.append(0)
            else:
                data = int_set.to_bytes()
                serialized.append(data)
                sizes.append(len(data))

        # Concatenate all bytes and store with sizes
        all_bytes = b''.join(serialized)
        group.create_dataset("data", data=np.frombuffer(all_bytes, dtype=np.uint8))
        group.create_dataset("sizes", data=np.array(sizes, dtype=np.int64))

    @staticmethod
    def load_from_group(group: h5py.Group) -> list[Optional[InteractionSet]]:
        """Load interaction sets from an HDF5 group."""

        data = np.array(group["data"][()])
        sizes = np.array(group["sizes"][()])

        interaction_sets = []
        offset = 0
        for size in sizes:
            if size == 0:
                interaction_sets.append(None)
            else:
                byte_data = data[offset:offset + size].tobytes()
                interaction_sets.append(InteractionSet.from_bytes(byte_data))
                offset += size

        return interaction_sets


    # *** Prolif helpers ***

    @staticmethod
    def _interaction_fp(system: BindingComplex, interaction_types: list[str] = None) -> plf.Fingerprint:
        """Run the Prolif interaction detection algorithm.

        Args:
            system: The protein-ligand complex to analyze.
            interaction_types: List of interaction types to detect. If None, all types are detected.

        Returns:
            Prolif Fingerprint object containing detected interactions.
        """

        if interaction_types is not None:
            _check_interaction_types_exist(interaction_types)

        interaction_types = PROLIF_INTERACTIONS if interaction_types is None else interaction_types
        protein_mol, ligand_mol = InteractionSet._prepare_prolif_mols(system)

        # count=True enables counting multiple instances of same interaction type per residue pair
        plf_fp = plf.Fingerprint(interaction_types, count=True)

        # residues="all" ensures we check all protein residues, not just those near the ligand
        plf_fp.run_from_iterable([ligand_mol], protein_mol, residues="all", progress=False)
        return plf_fp

    @staticmethod
    def _prepare_prolif_mols(
        system: BindingComplex,
        lig_resname: str = "LIG",
        lig_resnumber: int = 1
    ) -> tuple[plf.Molecule, plf.Molecule]:
        """Create Prolif Molecule objects for protein and ligand.

        Args:
            system: The protein-ligand complex.
            lig_resname: Residue name to assign to ligand atoms.
            lig_resnumber: Residue number to assign to ligand atoms.

        Returns:
            Tuple of (protein_mol, ligand_mol) as Prolif Molecule objects.
        """

        protein_mol = system.protein.to_prolif()
        ligand_rdkit = system.ligand.to_rdkit()
        ligand_mol = plf.Molecule.from_rdkit(ligand_rdkit, resname=lig_resname, resnumber=lig_resnumber)
        return protein_mol, ligand_mol
