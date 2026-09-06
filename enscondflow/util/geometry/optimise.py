import os

# Resolves segfault issues when running xtb
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional
from scipy.optimize import minimize
from xtb.interface import Param, Calculator
from xtb.libxtb import VERBOSITY_MUTED

from enscondflow.util.geometry.energy import possibly_add_hs


BOHR_PER_ANGSTROM = 1.8897259886

_XTB_METHOD_MAP = {
    "GFN1-xTB": Param.GFN1xTB,
    "GFN2-xTB": Param.GFN2xTB,
    "GFN-FF": Param.GFNFF
}


# *****************************************************************************
# ***************************** Helper functions ******************************
# *****************************************************************************


def _make_xtb_calculator(atomic_nums, positions_bohr, method, accuracy, electronic_temperature, solvent):
    """Create a muted xTB calculator with the given settings."""

    calc = Calculator(_XTB_METHOD_MAP[method], atomic_nums, positions_bohr)
    calc.set_verbosity(VERBOSITY_MUTED)
    calc.set_accuracy(accuracy)
    calc.set_electronic_temperature(electronic_temperature)

    if solvent is not None:
        calc.set_solvent(solvent)

    return calc


def _check_positions(positions):
    """Return True if positions are valid for xTB (no NaN/Inf, no overlapping atoms)."""

    if not np.all(np.isfinite(positions)):
        return False

    n_atoms = len(positions)
    if n_atoms < 2:
        return True

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 0.1:
                return False

    return True


def _xtb_energy_and_gradient(calc, coords_flat, n_atoms):
    """Run an xTB singlepoint and return (energy, flat gradient)."""

    coords = coords_flat.reshape(n_atoms, 3)

    if not _check_positions(coords):
        raise ValueError("Invalid positions: NaN/Inf or overlapping atoms")

    calc.update(coords)
    res = calc.singlepoint()
    return res.get_energy(), res.get_gradient().flatten()


def _set_conf_positions(mol, positions):
    """Return a copy of mol with a single conformer set to the given positions."""

    mol_out = Chem.Mol(mol)
    mol_out.RemoveAllConformers()
    conf = Chem.Conformer(mol_out.GetNumAtoms())

    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos.tolist())

    mol_out.AddConformer(conf, assignId=True)
    return mol_out


# *****************************************************************************
# ************************ MMFF geometry optimisation *************************
# *****************************************************************************


def optimise_mol_mmff(
    mol: Chem.rdchem.Mol,
    max_iters: int = 1000,
    n_threads: int = 1,
    allow_unconverged: bool = True,
    return_energy: bool = False
):
    """Optimise the conformation of an RDKit molecule using the MMFF forcefield.

    The molecule is copied so the original is not modified.

    If the input molecule contains multiple conformers, this function will attempt to optimise all conformers and
    return a new molecule object with the same number of conformers.

    When return_energy is True, energies are computed on the optimised with-Hs structure (same forcefield state used
    for optimisation) before copying coords back, giving accurate post-optimisation energies.

    Args:
        mol (Chem.Mol): RDKit molecule
        max_iters (int): Max iterations for the conformer optimisation algorithm
        n_threads (int): Number of threads to ask RDKit to use (0 means use all available processors)
        allow_unconverged (bool): Whether to allow returning partially converged molecules
        return_energy (bool): If True, also compute and return MMFF energies for each conformer

    Returns:
        Without return_energy: Chem.Mol or None
        With return_energy: (Chem.Mol, float|list[float]) or None — energy is a single float for one conformer,
            a list for multiple. Individual energies may be None if calculation failed for that conformer.
    """

    mol_copy = Chem.Mol(mol)

    # Return None if mol is not valid
    try:
        Chem.SanitizeMol(mol_copy)
    except:
        return None

    contains_hs = mol_copy.GetNumAtoms() != mol_copy.GetNumHeavyAtoms()
    mol_copy = Chem.AddHs(mol_copy, addCoords=True) if not contains_hs else mol_copy

    # Copy mol again for optimisation
    opt_mol = Chem.Mol(mol_copy)

    try:
        out = AllChem.MMFFOptimizeMoleculeConfs(opt_mol, maxIters=max_iters, numThreads=n_threads)
    except:
        return None

    if len(out) == 0:
        return None

    exitcodes, _ = tuple(zip(*out))
    converged = [code == 0 for code in exitcodes]

    # Return None if allow_unconverged is not set and not all confs converged
    if not allow_unconverged and not all(converged):
        return None

    # Compute energies on the optimised with-Hs structure before copying coords back
    energies = None
    if return_energy:
        energies = []
        for c_idx in range(opt_mol.GetNumConformers()):
            try:
                mmff_props = AllChem.MMFFGetMoleculeProperties(opt_mol, mmffVariant="MMFF94")
                ff = AllChem.MMFFGetMoleculeForceField(opt_mol, mmff_props, confId=c_idx)
                energies.append(ff.CalcEnergy())
            except:
                energies.append(None)

        energies = energies[0] if len(energies) == 1 else energies

    # Copy the mol and pass the opt conf info since MMFF will change aromatic atom props
    mol_copy.RemoveAllConformers()

    for conf in opt_mol.GetConformers():
        mol_copy.AddConformer(conf, assignId=True)

    mol_out = mol_copy if contains_hs else Chem.RemoveAllHs(mol_copy)

    if return_energy:
        return mol_out, energies

    return mol_out


# *****************************************************************************
# ************************* xTB geometry optimisation *************************
# *****************************************************************************


def optimise_mol_xtb(
    mol: Chem.rdchem.Mol,
    conf_idx: int = 0,
    max_iters: int = 200,
    method: str = "GFN2-xTB",
    accuracy: float = 1.0,
    electronic_temperature: float = 300.0,
    solvent: Optional[str] = None
):
    """Optimise a conformer using the xTB semi-empirical method and return the minimised mol and energy.

    Uses xtb-python for energy/gradient evaluation and scipy L-BFGS-B for geometry optimisation.
    The molecule is copied so the original is not modified. Hs are added if not already present since xTB
    requires all atoms. The returned molecule will have the same H-atom status as the input.

    Args:
        mol (Chem.Mol): RDKit molecule with at least one conformer
        conf_idx (int): Index of the conformer to optimise, default 0
        max_iters (int): Maximum number of geometry optimisation steps
        method (str): xTB method to use, one of "GFN1-xTB", "GFN2-xTB", "GFN-FF"
        accuracy (float): Numerical accuracy for the calculation (lower = tighter, 1.0 is default)
        electronic_temperature (float): Electronic temperature in Kelvin for Fermi smearing
        solvent (str, optional): ALPB solvent name (e.g. "water", "methanol"), None for gas phase

    Returns:
        (Chem.Mol, float, float): Tuple of (optimised molecule, final energy in Hartree, initial energy in Hartree),
            or None if optimisation fails
    """

    if method not in _XTB_METHOD_MAP:
        raise ValueError(f"Unknown xTB method '{method}', must be one of {list(_XTB_METHOD_MAP.keys())}")

    mol_copy = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(mol_copy)
    except:
        return None

    contains_hs = mol_copy.GetNumAtoms() != mol_copy.GetNumHeavyAtoms()
    mol_copy = possibly_add_hs(mol_copy, max_iters=50) if not contains_hs else mol_copy

    if mol_copy is None:
        return None

    positions = np.array(mol_copy.GetConformer(conf_idx).GetPositions())
    atomics = np.array([atom.GetAtomicNum() for atom in mol_copy.GetAtoms()])
    n_atoms = len(atomics)

    positions_bohr = positions * BOHR_PER_ANGSTROM
    if not _check_positions(positions_bohr):
        return None

    try:
        calc = _make_xtb_calculator(atomics, positions_bohr, method, accuracy, electronic_temperature, solvent)

        x0 = positions_bohr.flatten()
        initial_energy, _ = _xtb_energy_and_gradient(calc, x0, n_atoms)

        _cache = {}

        def cached_eval(coords_flat):
            key = coords_flat.tobytes()
            if key not in _cache:
                _cache.clear()
                _cache[key] = _xtb_energy_and_gradient(calc, coords_flat, n_atoms)
            return _cache[key]

        result = minimize(
            fun=lambda x: cached_eval(x)[0],
            x0=x0,
            jac=lambda x: cached_eval(x)[1],
            method="L-BFGS-B",
            options={"maxiter": max_iters, "gtol": 1e-5},
        )

    except Exception as e:
        print(f"[optimise_mol_xtb] failed: {e}")
        return None

    opt_positions = result.x.reshape(n_atoms, 3) / BOHR_PER_ANGSTROM
    opt_mol = _set_conf_positions(mol_copy, opt_positions)
    opt_mol = opt_mol if contains_hs else Chem.RemoveAllHs(opt_mol)
    return opt_mol, result.fun, initial_energy
