import numpy as np
from typing import Optional, Union
from scipy.optimize import linear_sum_assignment

from enscondflow.repr import AtomSet, BondSet, ConfSet, GraphMol, Protein, BindingComplex, AtomVocab, BondVocab


TArr = np.ndarray


# *****************************************************************************
# ******************************* Prior Sampler *******************************
# *****************************************************************************


class GraphNoise:
    def __init__(
        self,
        coord_noise: str = "gaussian",
        cat_noise: str = "uniform",
        zero_com: bool = True
    ):
        if coord_noise != "gaussian":
            raise NotImplementedError(f"Coord noise {coord_noise} is not supported.")

        if cat_noise not in ["uniform", "mask"]:
            raise ValueError(f"cat_noise must be either 'uniform' or 'mask', got {cat_noise}")

        self.coord_noise = coord_noise
        self.cat_noise = cat_noise
        self.n_atom_types = len(AtomVocab)
        self.n_bond_types = len(BondVocab)
        self.atom_mask_idx = AtomVocab.get_index("MASK")
        self.bond_mask_idx = BondVocab.get_index("MASK")
        self.zero_com = zero_com

    @property
    def hparams(self):
        return {
            "coord-noise": self.coord_noise,
            "cat-noise": self.cat_noise,
            "zero-com": self.zero_com
        }

    def sample_molecule(self, n_atoms: int, n_conformers: int = 1) -> GraphMol:
        confs = None

        if n_conformers not in [None, 0]:
            coords = np.random.standard_normal((n_conformers, n_atoms, 3))
            confs = ConfSet(coords)

        atoms, bonds = self._sample_discrete(n_atoms)

        # This will create 0s (unknown charge) for all charges, although the atom types could be charged
        mol = GraphMol(atoms, bonds, confs=confs)
        mol = mol.zero_com() if self.zero_com and confs is not None else mol

        return mol

    def _sample_discrete(self, n_atoms):
        bond_indices = np.stack(np.triu_indices(n_atoms), axis=1)
        n_bonds = bond_indices.shape[0]

        # Assumes that the last atom and bond type are mask tokens and are not sampled
        if self.cat_noise == "uniform":
            atomics = np.random.randint(0, self.n_atom_types - 1, size=(n_atoms,))
            bond_types = np.random.randint(0, self.n_bond_types - 1, size=(n_bonds,))

        elif self.cat_noise == "mask":
            atomics = np.array([self.atom_mask_idx] * n_atoms)
            bond_types = np.array([self.bond_mask_idx] * n_bonds)

        else:
            raise ValueError(f"cat_noise must be either 'uniform' or 'mask', got {self.cat_noise}")

        bond_arr = np.concat((bond_indices, bond_types[:, None]), axis=1)
        atoms = AtomSet(atomics)
        bonds = BondSet(bond_arr)

        return atoms, bonds


# *****************************************************************************
# ***************************** Graph Interpolant *****************************
# *****************************************************************************


class GraphInterpolant:
    def __init__(
        self,
        prior_sampler: GraphNoise,
        coord_noise_std: float = 0.0,
        perm_ot: bool = False,
        time_alpha: float = 1.0,
        time_beta: float = 1.0,
        pad_to: Optional[int] = None,
        pad_coord_mode: str = "com"
    ):
        self.prior_sampler = prior_sampler
        self.coord_noise_std = coord_noise_std
        self.perm_ot = perm_ot

        self.time_alpha = time_alpha
        self.time_beta = time_beta

        self.pad_to = pad_to
        self.pad_coord_mode = pad_coord_mode

    @property
    def hparams(self):
        prior_hparams = {f"prior-{k}": v for k, v in self.prior_sampler.hparams.items()}
        hparams = {
            "coord-noise-std": self.coord_noise_std,
            "perm-ot": self.perm_ot,
            "time-alpha": self.time_alpha,
            "time-beta": self.time_beta,
            "pad-to": self.pad_to,
            "pad-coord-mode": self.pad_coord_mode,
            **prior_hparams
        }
        return hparams

    def interpolate(self, to_mols: list[Union[GraphMol, BindingComplex]]) -> dict:
        # NOTE this requires that everything in the batch has the same type, either pocket or no pocket for whole batch
        include_pocket = len(to_mols) > 0 and isinstance(to_mols[0], BindingComplex)

        # Copy everything up front since some changes are done in place to save mem transfers
        to_mol_copies = [mol.copy() for mol in to_mols]
        feature_mols = to_mol_copies

        pockets = None
        if include_pocket:
            pockets = [system.protein for system in to_mol_copies]
            to_mol_copies = [system.ligand for system in to_mol_copies]
            feature_mols = to_mol_copies

        # Convert mol atomics to hold indices into atom vocab, then pad in place
        converted_to_mols = [self.convert_mol_atoms(mol) for mol in to_mol_copies]
        padded = self._possibly_pad_mols(converted_to_mols)

        # Sample noise at padded size and permutation-match over entire padded molecule
        from_mols = [self.prior_sampler.sample_molecule(len(mol), mol.n_conformers) for mol in padded]
        from_mols = [self._match_mols(fm, tm) for fm, tm in zip(from_mols, padded)]

        # Sample times and interpolate mols, possibly with different time schedules for different modalities
        times = np.random.beta(self.time_alpha, self.time_beta, size=(len(padded),)).tolist()
        interp_mols = [self._interpolate_mol(*t) for t in zip(from_mols, padded, times)]

        batch = {
            "prior_mols": from_mols,
            "interp_mols": interp_mols,
            "data_mols": padded,
            "times": times
        }

        # Read pre-computed features from meta if available
        if feature_mols and "features" in feature_mols[0].meta:
            all_feats = [mol.meta["features"] for mol in feature_mols]
            batch["features"] = {k: [fs[k] for fs in all_feats] for k in all_feats[0].keys()}

        if pockets is not None:
            pocket_rotations = [int(p.meta.get("rotated", 0)) for p in pockets]
            raw_pockets = tuple(pockets)
            pockets = [self.convert_pocket_atoms(p) for p in pockets]
            batch["pocket"] = {
                "proteins": pockets,
                "rotated": pocket_rotations,
                "raw_proteins": raw_pockets
            }

        return batch

    def _match_mols(self, from_mol: GraphMol, to_mol: GraphMol) -> GraphMol:
        """Permutation-align the from_mol to best match the to_mol using Hungarian algorithm.

        Computes pairwise squared distances between from_mol and to_mol coordinates
        and finds the optimal assignment that minimises total transport cost.
        """

        if len(to_mol) > len(from_mol):
            raise RuntimeError(f"from_mol must have at least as many atoms as to_mol.")

        # Keep the same number of atoms as the data mol in the noise mol
        from_mol.permute(list(range(len(to_mol))), in_place=True)

        if not self.perm_ot:
            return from_mol

        # Hungarian permutation alignment over entire padded molecule
        # Centre both point clouds before computing cost so that any global translation
        # (from coord shift augmentation) doesn't dominate the assignment
        to_coords = to_mol.get_conformer(0)
        from_coords = from_mol.get_conformer(0)

        to_centered = to_coords - to_coords.mean(axis=0)
        from_centered = from_coords - from_coords.mean(axis=0)

        diffs = to_centered[:, None, :] - from_centered[None, :, :]
        cost_matrix = np.sum(diffs * diffs, axis=2)
        _, from_mol_indices = linear_sum_assignment(cost_matrix)
        from_mol.permute(from_mol_indices.tolist(), in_place=True)

        return from_mol

    def _interpolate_mol(self, from_mol: GraphMol, to_mol: GraphMol, t: float) -> GraphMol:
        """Interpolates mols which have already been sampled according to OT map, if required"""

        if from_mol.seq_length != to_mol.seq_length:
            raise RuntimeError(f"Both molecules to be interpolated must have the same number of atoms.")

        # Interpolate coords and add gaussian noise
        coords_mean = (from_mol.coords * (1 - t)) + (to_mol.coords * t)
        coord_noise = np.random.randn(*coords_mean.shape) * self.coord_noise_std
        coords = coords_mean + coord_noise
        confs = ConfSet(coords)

        # Interpolate discrete features
        # Note both mask and sample interpolation are done the same way
        atomics = self._interp_discrete(t, from_mol.atomics, to_mol.atomics)
        adj = self._interp_discrete(t, from_mol.adjacency, to_mol.adjacency)

        bond_indices = np.triu_indices(len(from_mol))
        bond_types = adj[bond_indices[0], bond_indices[1]]
        bond_arr = np.stack((*bond_indices, bond_types), axis=1)

        atoms = AtomSet(atomics)
        bonds = BondSet(bond_arr)

        # For now don't pass a charge, then the interpolated charge will be index 0 -> unknown charge
        interp_mol = GraphMol(atoms, bonds, confs=confs)
        return interp_mol

    def _interp_discrete(self, t, from_idxs, to_idxs, mask=None):
        if mask is None:
            mask = np.random.rand(*from_idxs.shape) > t

        interp_idxs = to_idxs.copy()
        interp_idxs[mask] = from_idxs[mask]
        return interp_idxs

    def _possibly_pad_mols(self, mols: list[GraphMol]):
        if self.pad_to is None:
            return mols

        padded = [mol.pad(self.pad_to, pad_coord_mode=self.pad_coord_mode, in_place=True) for mol in mols]
        return padded

    @staticmethod
    def convert_mol_atoms(mol: GraphMol) -> GraphMol:
        """Convert atoms to their indices

        This will reuse the same molecular format as the actual data, but now the molecule's atomic numbers are indices
        into the atom vocab. Bond types are stored as before with an index into the bond vocab.
        """

        atomic_indices = np.array(AtomVocab.indices_from_tokens(mol.charged_symbols))
        atoms = mol.atoms.copy_with(atomics=atomic_indices)
        transformed = mol.copy_with(atoms=atoms)
        return transformed

    @staticmethod
    def convert_pocket_atoms(pocket: Protein) -> Protein:
        """Convert atoms to their indices

        This will reuse the same molecular format as the actual data, but now the molecule's atomic numbers are indices
        into the atom vocab. Bond types are stored as before with an index into the bond vocab.
        """

        atomic_indices = np.array(AtomVocab.indices_from_tokens(pocket.charged_symbols))
        atoms = pocket.atoms.copy_with(atomics=atomic_indices)
        transformed = pocket.copy_with(atoms=atoms)
        return transformed
