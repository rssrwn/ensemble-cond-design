import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Union, Optional

from enscondflow.repr.vocab import BondVocab
from enscondflow.repr.util import check_dim_shape, check_shape_len, check_shapes_equal


TArr = np.ndarray


def mol_is_valid(mol: Chem.rdchem.Mol, with_hs: bool = True, connected: bool = True) -> bool:
    """Whether the mol can be sanitised and, optionally, whether it's fully connected
    
    Args:
        mol (Chem.Mol): RDKit molecule to check
        with_hs (bool): Whether to check validity including hydrogens (if they are in the input mol), default True
        connected (bool): Whether to also assert that the mol must not have disconnected atoms, default True

    Returns:
        bool: Whether the mol is valid
    """

    if mol is None:
        return False

    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)

    try:
        AllChem.SanitizeMol(mol_copy)
    except:
        return False

    n_frags = len(AllChem.GetMolFrags(mol_copy))
    if connected and n_frags != 1:
        return False

    return True


# TODO could allow more args
def smiles_from_mol(mol: Chem.rdchem.Mol, canonical: bool = True, explicit_hs: bool = False) -> Union[str, None]:
    """Create a SMILES string from a molecule

    Args:
        mol (Chem.Mol): RDKit molecule object
        canonical (bool): Whether to create a canonical SMILES, default True
        explicit_hs (bool): Whether to embed hydrogens in the mol before creating a SMILES, default False. If True 
                this will create a new mol with all hydrogens embedded. Note that the SMILES created by doing this
                is not necessarily the same as creating a SMILES showing implicit hydrogens.

    Returns:
        str: SMILES string which could be None if the SMILES generation failed
    """

    if mol is None:
        return None
    
    if explicit_hs:
        mol = Chem.AddHs(mol)

    try:
        smiles = Chem.MolToSmiles(mol, canonical=canonical)
    except:
        smiles = None

    return smiles


def mol_from_smiles(smiles: str, preserve_hs: bool = True, embed_hs: bool = False) -> Union[Chem.rdchem.Mol, None]:
    """Create a RDKit molecule from a SMILES string

    Args:
        smiles (str): SMILES string
        preserve_hs (bool): Whether to preserve hydrogen atoms that are present in the given SMILES.
        embed_hs (bool): Whether to embed explicit hydrogens into the mol. This could change the number of atoms in
                the molecule, even if hydrogens are provided in the input SMILES.

    Returns:
        Chem.Mol: RDKit molecule object or None if one cannot be created from the SMILES
    """

    if smiles is None:
        return None

    smi_params = Chem.SmilesParserParams()
    if preserve_hs:
        smi_params.removeHs = False

    try:
        mol = Chem.MolFromSmiles(smiles, smi_params)
        mol = Chem.AddHs(mol) if embed_hs else mol
    except:
        mol = None

    return mol


def mol_from_atoms(
    atomics: TArr,
    bonds: TArr,
    coords: Optional[TArr] = None,
    charges: Optional[TArr] = None,
    sanitise: bool = True
):
    """Create RDKit mol from atomic numbers, bonds and, optionally, coords and charges

    If any of the atomics are not valid (do not exist on the periodic table), None will be returned. It's the caller's
    responsibility to ensure the provided bonds are valid. If charges are not provided they are assumed to be 0 for 
    all atoms.

    Args:
        atomics (np.ndarray): Atomic numbers, length must be n_atoms
        bonds (np.ndarray): Bond indices and types, shape [n_bonds, 3]
        coords (np.ndarray, optional): Coordinate tensor, shape [n_atoms, 3] or [n_confs, n_atoms, 3]
        charges (np.ndarray, optional): Charge for each atom, shape [n_atoms]
        sanitise (bool): Whether to apply RDKit sanitization to the molecule, default True

    Returns:
        Chem.rdchem.Mol: RDKit molecule or None if one cannot be created
    """

    check_shape_len(atomics, 1, "atomics")
    check_shape_len(bonds, 2, "bonds")
    check_dim_shape(bonds, 1, 3, "bonds")

    if coords is not None:
        if len(coords.shape) == 2:
            check_dim_shape(coords, 1, 3, "coords")
            check_shapes_equal(atomics, coords, 0)

            # If there is only one conf create a conf dim with only one item
            # This ensures the shape will be same as the multi conf case
            coords = np.expand_dims(coords, (0))

        elif len(coords.shape) == 3:
            check_dim_shape(coords, 2, 3, "coords")
            if coords.shape[1] != atomics.shape[0]:
                raise ValueError("Coords and atomics must have the same number of atoms.")

        else:
            raise ValueError("Coords must have shape either [n_confs, n_atoms, 3] or [n_atoms, 3].")

    if charges is not None:
        check_shape_len(charges, 1, "charges")
        check_shapes_equal(atomics, charges, 0)

    charges = charges.tolist() if charges is not None else [0] * atomics.shape[0]

    mol = Chem.RWMol()

    # Add atom types and charges
    for idx, atomic in enumerate(atomics.tolist()):
        atom = Chem.Atom(atomic)
        atom.SetFormalCharge(charges[idx])
        mol.AddAtom(atom)

    # Add bonds, along with aromatic and stereo info
    for bond in bonds.astype(np.int32).tolist():
        start, end, bond_index = bond

        # Don't add self connections
        if start == end:
            continue

        bond_type = BondVocab.get_bond_type(bond_index)
        is_arom = BondVocab.get_is_aromatic(bond_index)
        stereo = BondVocab.get_stereo(bond_index)

        # Ignore non-RDKit bonds (eg. NONE or MASK)
        if not isinstance(bond_type, Chem.BondType):
            continue

        b_idx = mol.AddBond(start, end, bond_type)

        # Add extra bond info to newly created bond
        bond = mol.GetBondWithIdx(b_idx - 1)
        bond.SetIsAromatic(is_arom)
        bond.SetStereo(stereo)

    try:
        mol = mol.GetMol()
    except:
        return None

    # Add 3D coords for all conformers, if they were provided
    if coords is not None:
        for conf_coords in list(coords):
            conf = Chem.Conformer(conf_coords.shape[0])
            for idx, coord in enumerate(conf_coords.tolist()):
                conf.SetAtomPosition(idx, coord)

            mol.AddConformer(conf, assignId=True)

        # Try to add stereo info from coords, but ignore on failure
        # This will use the default (first) conformer to assign stereo info
        try:
            Chem.AssignStereochemistryFrom3D(mol)
        except:
            pass

    if sanitise:
        try:
            Chem.SanitizeMol(mol)
        except:
            return None

    return mol
