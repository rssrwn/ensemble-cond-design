import numpy as np
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from enscondflow.util.geometry.energy import calc_energy_mmff
from enscondflow.util.geometry.optimise import optimise_mol_mmff


# *****************************************************************************
# ***************************** Helper functions ******************************
# *****************************************************************************


def _dedup_conformers(mol, rmsd_threshold=0.5):
    mol_copy = Chem.Mol(mol)

    n_confs = mol_copy.GetNumConformers()
    if n_confs <= 1:
        return list(range(n_confs))

    curr_indices = [0]

    for i in range(1, n_confs):
        is_unique = True

        for j in curr_indices:
            rmsd = rdMolAlign.AlignMol(mol_copy, mol_copy, i, j)
            if rmsd < rmsd_threshold:
                is_unique = False
                break

        if is_unique:
            curr_indices.append(i)

    return curr_indices


def _calc_weights(energies, temp=300):
    """Energies in kcal/mol"""

    kT = 0.001987 * temp

    # Shift energies relative to minimum
    energies = np.array(energies)
    relative_energies = energies - np.min(energies)

    # Calculate Boltzmann factors and normalise
    boltzmann_factors = np.exp(-relative_energies / kT)
    weights = boltzmann_factors / np.sum(boltzmann_factors)
    return weights


# *****************************************************************************
# ************************** Conf sampling functions **************************
# *****************************************************************************


def sample_conformers(
    mol: Chem.rdchem.Mol,
    n_confs: int = 1,
    max_attempts: int = 10,
    fast_conf: bool = False,
    opt_iters: Optional[int] = None,
    n_threads: int = 1
) -> Chem.rdchem.Mol:
    """Create a (set of) conformer(s) for a molecule using the RDKit ETKDGv3 method

    The molecule is copied and the input is not modified.

    NOTE if the input molecule contains at least one H atom, no further Hs will be added before the conf gen.
    If there are no Hs, Hs will be added with RDkit.

    If any conf fails, None is returned.

    Args:
        mol (Chem.Mol): RDKit molecule (existing conformers will be ignored)
        n_confs (int): The number of conformers to generate, default 1
        max_attempts (int): Max num of attempts per conformer, default 1
        fast_conf (bool): Whether to use a faster version of conf gen with lower quality results, default False
        opt_iters (int, optional): Optional apply some number of MMFF optimisation iters to each conf
        n_threads (int): Number of threads to ask RDKit to use (0 means use all available processors)

    Returns:
        Chem.Mol: Copied molecule with conformers added (or None on failure)
    """

    mol_copy = Chem.Mol(mol)

    # Return None if mol is not valid
    try:
        Chem.SanitizeMol(mol_copy)
    except:
        return None

    # Hs are added and will be returned in the sampled conformers
    contains_hs = mol_copy.GetNumAtoms() != mol_copy.GetNumHeavyAtoms()
    mol_copy = Chem.AddHs(mol_copy) if not contains_hs else mol_copy

    params = AllChem.ETKDGv3()
    params.maxIterations = max_attempts
    params.numThreads = n_threads

    # Turn off some optimisations and use slightly higher tolerence if speed is important
    if fast_conf:
        params.optimizerForceTol = 0.002
        params.useBasicKnowledge = False
        params.useExpTorsionAnglePrefs = False

    try:
        outs = AllChem.EmbedMultipleConfs(mol_copy, n_confs, params)
    except:
        return None

    if len(outs) != n_confs:
        return None

    if opt_iters not in [None, 0]:
        mol_copy = optimise_mol_mmff(mol_copy, max_iters=opt_iters, n_threads=n_threads, allow_unconverged=True)
        if mol_copy is None:
            return None

    return mol_copy


def sample_ensemble(
    mol,
    max_confs=128,
    max_conf_attempts=100,
    max_opt_iters=1000,
    dedup_rmsd_threshold=0.5,
    strain_filter=6.0,
    temp=300.0,
    n_threads=1
):
    """Generate a full set of conformers and weights for a given molecule.

    Samples conformers for a molecular graph using RDKit ETKDG, applies MMFF optimisation to each and applies
    RMSD-based deduplication before calculating boltzmann weights. This will return a copy of the mol with new confs.

    Args:
        mol (Chem.Mol): RDKit molecule for an ensemble to be sampled
        max_confs (int): Max number of conformers to attempt to sample
        max_opt_iters (int): Maximum number of MMFF optimisation steps to apply
        strain_filter (float): Filter out conformers with global strain higher than this
        dedup_rmsd_threshold (float): RMSD threshold for removing deduplicate conformers
        temp (float): Temperature for boltzmann weight calculation
        n_threads (int): Number of threads to ask RDKit to use (0 means use all available processors)

    Returns:
        (Chem.Mol, np.ndarray): Molecule with embedded confs and conformer weights array
    """

    # Sample initial set of conformers (possibly minimised)
    embedded = sample_conformers(
        mol,
        n_confs=max_confs,
        max_attempts=max_conf_attempts,
        opt_iters=max_opt_iters,
        n_threads=n_threads
    )

    if embedded is None:
        return None

    # Deduplicate the conformers and add remaining to new mol object
    if dedup_rmsd_threshold not in [None, 0.0]:
        conf_indices = _dedup_conformers(embedded, rmsd_threshold=dedup_rmsd_threshold)
    else:
        conf_indices = list(range(embedded.GetNumConformers()))

    confs = [embedded.GetConformer(idx) for idx in conf_indices]

    if len(confs) == 0:
        return None

    emb_dedup = Chem.Mol(embedded)
    emb_dedup.RemoveAllConformers()

    for conf in confs:
        emb_dedup.AddConformer(conf, assignId=True)

    # Calculate energies for each conf
    energies = calc_energy_mmff(emb_dedup)
    energies = [energies] if not isinstance(energies, list) else energies

    assert len(energies) == len(confs)

    # Filter out confs which failed energy calculation
    is_valid = [energy is not None for energy in energies]
    confs = [conf for valid, conf in zip(is_valid, confs) if valid]
    energies = [energy for valid, energy in zip(is_valid, energies) if valid]

    if len(energies) == 0:
        return None

    global_strains = np.array(energies) - min(energies)

    # Filter out confs too far from minimum energy
    is_valid = (global_strains <= strain_filter).tolist()
    confs = [conf for valid, conf in zip(is_valid, confs) if valid]
    energies = [energy for valid, energy in zip(is_valid, energies) if valid]

    assert len(energies) == len(confs)

    final_mol = Chem.Mol(emb_dedup)
    final_mol.RemoveAllConformers()

    for conf in confs:
        final_mol.AddConformer(conf, assignId=True)

    weights = _calc_weights(energies, temp=temp)
    e_min = min(energies)
    return final_mol, weights, e_min
