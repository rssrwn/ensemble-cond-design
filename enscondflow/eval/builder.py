import torch
import numpy as np
from rdkit import RDLogger
from functools import partial
from more_itertools import grouper
from concurrent.futures import ProcessPoolExecutor

import enscondflow.util.rdkit as smolRD
import enscondflow.util.functional as smolF
from enscondflow.repr import AtomVocab
from enscondflow.repr.util import PAD_TOKEN, MASK_TOKEN


# *****************************************************************************
# *************************** Global Builder Class ****************************
# *****************************************************************************


class MolBuilder:
    """Global helper class for building and converting molecules"""

    @staticmethod
    def mols_from_smiles(smiles, preserve_hs=False, n_workers=16, batch_size=1000):
        mols = MolBuilder._batched_proc(
            smolRD.mol_from_smiles,
            smiles,
            n_workers=n_workers,
            batch_size=batch_size,
            preserve_hs=preserve_hs
        )
        return mols

    @staticmethod
    def mols_from_tensors(
        atom_dists,
        bond_dists=None,
        coords=None,
        mask=None,
        sanitise=True,
        n_workers=None,
        batch_size=100
    ):
        extracted = MolBuilder._separate_batch(
            atom_dists,
            bond_dists=bond_dists,
            coords=coords,
            mask=mask
        )

        outs = MolBuilder._batched_proc(
            MolBuilder._parse_mol,
            extracted,
            n_workers=n_workers,
            batch_size=batch_size,
            sanitise=sanitise
        )
        return outs

    @staticmethod
    def _separate_batch(atom_dists, bond_dists=None, coords=None, mask=None):
        """Separate each molecule from the batch"""

        atom_dists = atom_dists.cpu()
        bond_dists = bond_dists.cpu() if bond_dists is not None else None
        coords = coords.cpu() if coords is not None else None
        mask = mask.cpu() if mask is not None else None

        atom_dists_list = []
        bond_dists_list = []
        coords_list = []
        mask_list = []

        for idx in range(atom_dists.size(0)):
            mol_token_dists = atom_dists[idx,]
            mol_bs = bond_dists[idx,] if bond_dists is not None else None
            mol_cs = coords[idx,] if coords is not None else None
            mol_mask = mask[idx,] if mask is not None else None

            atom_dists_list.append(mol_token_dists)
            bond_dists_list.append(mol_bs)
            coords_list.append(mol_cs)
            mask_list.append(mol_mask)

        zipped = list(zip(atom_dists_list, bond_dists_list, coords_list, mask_list))
        return zipped

    @staticmethod
    def _parse_mol(mol_info, sanitise=True):
        atom_dists, bond_dists, coords, mask = mol_info

        tokens, bool_mask = MolBuilder._extract_tokens(atom_dists, mask=mask)
        splits = [token.split("_") if "_" in token else (token, 0) for token in tokens]

        # Transpose mol coords if we have an ensemble of confs, shape after transpose [n_confs, n_atoms, 3]
        if coords is not None:
            coords = coords[bool_mask,].float().numpy()
            coords = coords.transpose(1,0,2) if len(coords.shape) == 3 else coords

        try:
            atoms, charges = tuple(zip(*splits))
            atomics = np.array([smolRD.PT.atomic_from_symbol(atom) for atom in atoms])
            charges = np.array([int(charge) for charge in charges])
        except:
            print("\n*** Error in parsing sampled tokens ***")
            print("Sampled tokens:", tokens)
            return None

        bonds = None
        if bond_dists is not None:
            bond_types = torch.argmax(bond_dists, dim=-1) if len(bond_dists.shape) == 3 else bond_dists
            bond_types = bond_types[bool_mask, :][:, bool_mask]
            bonds = smolF.bonds_from_adj(bond_types).long().numpy()

        mol = smolRD.mol_from_atoms(atomics, bonds, coords=coords, charges=charges, sanitise=sanitise)
        return mol

    @staticmethod
    def _extract_tokens(atoms, mask=None):
        vocab_indices = torch.argmax(atoms, dim=-1) if len(atoms.shape) == 2 else atoms

        pad_index = AtomVocab.get_index(PAD_TOKEN)
        mask_index = AtomVocab.get_index(MASK_TOKEN)

        bool_mask = (vocab_indices != pad_index) & (vocab_indices != mask_index)
        bool_mask = (mask == 1) & bool_mask if mask is not None else bool_mask

        vocab_indices = vocab_indices[bool_mask].tolist()
        tokens = AtomVocab.tokens_from_indices(vocab_indices)
        return tokens, bool_mask

    @staticmethod
    def _batched_proc(proc_fn, items, n_workers, batch_size, *args, **kwargs):
        """Runs a batched version of the given function across the list of items

        Args:
            proc_fn (fn): The function to be applied to each item in the list
            items (list): The list of items to be processed
            n_workers (int): Max number of worker processes
            batch_size (int): Size of each batch of items

        Returns:
            list: List of processed items, flattened to original size
        """

        if n_workers in [None, 0]:
            results = [proc_fn(i, *args, **kwargs) for i in items if i is not None]
            return results

        batch_fn = partial(MolBuilder._batched_proc_fn, proc_fn, *args, **kwargs)
        grouped_items = list(grouper(items, batch_size, incomplete="fill", fillvalue=None))

        executor = ProcessPoolExecutor(max_workers=n_workers)
        futures = [executor.submit(batch_fn, group) for group in grouped_items]
        outs = [item for future in futures for item in future.result()]
        executor.shutdown()

        return outs

    @staticmethod
    def _batched_proc_fn(proc_fn, group, *args, **kwargs):
        # Disable RDKit logging in each worker process
        RDLogger.DisableLog('rdApp.*')

        return [proc_fn(i, *args, **kwargs) for i in group if i is not None]
