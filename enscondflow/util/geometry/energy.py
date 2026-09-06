from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Union


# *************************************************************************************************
# ********************************* Conformer ensemble functions **********************************
# *************************************************************************************************


def possibly_add_hs(mol: Chem.Mol, max_iters: int = 100) -> Union[Chem.Mol, None]:
    """Adds Hs (and applies a small minimisation to them) to the molecule if they are needed.

    The function will only add Hs if there are none in the molecule, otherwise none will be added.

    If the molecule has at least one conformer, coords are added to the Hs and a small minimisation is applied with
    the MMFF forcefield, keeping all heavy atoms fixed.

    Args:
        mol (Chem.Mol): Molecule to have Hs added. The molecule is not modified.
        max_iters (int): Maximum number of optimisation steps applied to H atoms.

    Returns:
        Chem.Mol: A copy of mol with Hs added
    """

    mol_hs = Chem.Mol(mol)

    add_hs = mol_hs.GetNumAtoms() == mol_hs.GetNumHeavyAtoms()

    try:
        mol_hs = Chem.AddHs(mol_hs, addCoords=True) if add_hs else mol_hs
    except:
        return None

    if not add_hs:
        return mol_hs

    heavy_idxs = [idx for idx, atom in enumerate(mol_hs.GetAtoms()) if atom.GetAtomicNum() != 1]

    try:
        ff_props = AllChem.MMFFGetMoleculeProperties(mol_hs)
        ff = AllChem.MMFFGetMoleculeForceField(mol_hs, ff_props)

        for idx in heavy_idxs:
            ff.MMFFAddPositionConstraint(idx, 0, 10000.0)

        AllChem.OptimizeMoleculeConfs(mol_hs, ff, maxIters=max_iters)
    except:
        return None

    return mol_hs


def calc_energy_mmff(mol: Chem.Mol, per_atom: bool = False) -> Union[float, list[float]]:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield

    The molecule is copied so the original is not modified. If multiple conformers exist in the molecule the energies
    are calculated independently and returned as a list. The conformer ids must be continuous and start from 0.

    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by number of atoms in mol, default False

    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """

    # Add Hs for energy calculation if there are none
    # This will also create a copy of the mol
    mol_copy = possibly_add_hs(mol)

    if mol_copy is None:
        return None

    n_atoms = mol_copy.GetNumAtoms()

    energies = []
    for c_idx in range(mol_copy.GetNumConformers()):
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94")
            ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=c_idx)
            energy = ff.CalcEnergy()
            energy = energy / n_atoms if per_atom else energy
        except:
            energy = None

        energies.append(energy)

    energy = energies[0] if len(energies) == 1 else energies
    return energy
