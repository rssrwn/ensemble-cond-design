from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from enscondflow.data.profile import ConfProfile
from enscondflow.util.rdkit.pharmacophore import PharmacophoreFinder


# Map from pharmacophore feature family name to RDKit colour name for PUBCHEM_PHARMACOPHORE_FEATURES
_PHARM_NAME_TO_COLOUR = {
    "Aromatic": "rings",
    "Donor": "donor",
    "Acceptor": "acceptor",
    "Cation": "cation",
    "Anion": "anion",
    "Hydrophobe": "hydrophobe",
}


# *****************************************************************************
# **************************** Alignment functions ****************************
# *****************************************************************************


def set_pharm_features_from_profile(mol: Chem.rdchem.Mol, profile: ConfProfile) -> Chem.rdchem.Mol:
    """Return a copy of mol with PUBCHEM_PHARMACOPHORE_FEATURES set from a ConfProfile.

    Iterates pharmacophore points (type >= 2 in the profile), maps each to an RDKit colour name,
    and sets the PUBCHEM_PHARMACOPHORE_FEATURES property on the copy so that rdShapeAlign uses
    these specific features for colour scoring.

    Args:
        mol (Chem.Mol): RDKit molecule (not modified)
        profile (ConfProfile): Profile containing types and atom_ids

    Returns:
        Chem.Mol: Copy of mol with pharmacophore features set (this can be empty)
    """

    mol_copy = Chem.Mol(mol)

    lines = []
    for point_type, atom_id_tuple in zip(profile.types, profile.atom_ids):
        point_type = int(point_type)

        # Only pharmacophore points (type >= 2); type 0 = padding, type 1 = shape
        if point_type < 2:
            continue

        pharm_name = PharmacophoreFinder.get_feature_name(point_type - 2)
        colour_name = _PHARM_NAME_TO_COLOUR.get(pharm_name)
        if colour_name is None:
            continue

        n_atoms = len(atom_id_tuple)

        # NOTE the pubchem features expect atom indices starting from 1
        atom_ids_str = " ".join(str(a + 1) for a in atom_id_tuple)
        lines.append(f"{n_atoms} {atom_ids_str} {colour_name}")

    header = str(len(lines))
    mol_copy.SetProp("PUBCHEM_PHARMACOPHORE_FEATURES", header + "\n" + "\n".join(lines))

    return mol_copy


def detect_and_set_pharm_features(mol: Chem.rdchem.Mol, conf_idx: int = -1) -> Chem.rdchem.Mol:
    """Return a copy of mol with PUBCHEM_PHARMACOPHORE_FEATURES set from auto-detected features.

    Runs PharmacophoreFinder on the molecule and converts all detected features to the PUBCHEM
    format so that rdShapeAlign uses these features for colour scoring.

    Args:
        mol (Chem.Mol): RDKit molecule (not modified)
        conf_idx (int): Conformer index to use for feature detection, default -1

    Returns:
        Chem.Mol: Copy of mol with pharmacophore features set (this can be empty)
    """

    mol_copy = Chem.Mol(mol)
    pharm_feats = PharmacophoreFinder.run_mol(mol_copy, conf_idx=conf_idx)

    lines = []
    for feat in pharm_feats:
        pharm_name = PharmacophoreFinder.get_feature_name(feat.type)
        colour_name = _PHARM_NAME_TO_COLOUR.get(pharm_name)
        if colour_name is None:
            continue

        n_atoms = len(feat.atom_ids)

        # NOTE the pubchem features expect atom indices starting from 1
        atom_ids_str = " ".join(str(a + 1) for a in feat.atom_ids)
        lines.append(f"{n_atoms} {atom_ids_str} {colour_name}")

    header = str(len(lines))
    mol_copy.SetProp("PUBCHEM_PHARMACOPHORE_FEATURES", header + "\n" + "\n".join(lines))

    return mol_copy


def align_conf(mol, ref_mol, align_weight=0.5, ref_profile=None):
    """Align a single-conformer mol to ref_mol and return the aligned mol with shape/colour scores.

    Args:
        mol (Chem.Mol): Molecule with a single conformer (not modified)
        ref_mol (Chem.Mol): Reference molecule to align to
        align_weight (float): Weight for shape vs colour alignment (1.0 = shape only, 0.0 = colour only)
        ref_profile (ConfProfile, optional): If provided, sets custom pharmacophore features on ref_mol
            (from profile) and mol (from auto-detection) for colour scoring

    Returns:
        (Chem.Mol, float, float): Tuple of (aligned mol, shape tanimoto, colour tanimoto)
    """

    if align_weight < 0.0 or align_weight > 1.0:
        raise ValueError("align_weight must be between 0.0 and 1.0")

    mol_copy = Chem.Mol(mol)
    ref_copy = Chem.Mol(ref_mol)

    if ref_profile is not None:
        ref_copy = set_pharm_features_from_profile(ref_copy, ref_profile)
        mol_copy = detect_and_set_pharm_features(mol_copy)

    # mol_copy will be modified in-place with the aligned conf
    shape_tani, colour_tani = rdShapeAlign.AlignMol(
        ref_copy,
        mol_copy,
        opt_param=align_weight,
        useColors=True
    )

    return mol_copy, shape_tani, colour_tani


def align_best_conf(mol, ref_mol, align_weight=0.5, ref_profile=None):
    """Find and return the best-aligned conformer from an ensemble.

    Searches all conformers in mol for the one with the best combined shape + colour tanimoto
    against ref_mol, then returns a new mol with just that aligned conformer.

    Args:
        mol (Chem.Mol): Molecule with a set of query conformers
        ref_mol (Chem.Mol): Reference molecule to compare to
        align_weight (float): Weight for alignment (1.0 for only shape, 0.0 for only colour)
        ref_profile (ConfProfile, optional): If provided, sets custom pharmacophore features on both
            ref_mol (from profile) and mol (from auto-detection) for colour scoring

    Returns:
        (Chem.Mol, float, float): Tuple of:
            1. Molecule with single conformer aligned to ref
            2. Shape tanimoto similarity
            3. Colour tanimoto similarity
    """

    if align_weight < 0.0 or align_weight > 1.0:
        raise ValueError("align_weight must be between 0.0 and 1.0")

    mol_copy = Chem.Mol(mol)
    ref_copy = Chem.Mol(ref_mol)
    n_confs = mol_copy.GetNumConformers()
    assert n_confs >= 1

    if ref_profile is not None:
        ref_copy = set_pharm_features_from_profile(ref_copy, ref_profile)
        mol_copy = detect_and_set_pharm_features(mol_copy)

    best_idx = None
    best_score = None
    all_scores = []

    for c_idx in range(n_confs):
        # useColors=True so we get a colour score regardless of align_weight
        scores = rdShapeAlign.AlignMol(
            ref_copy,
            mol_copy,
            probeConfId=c_idx,
            opt_param=align_weight,
            useColors=True
        )
        all_scores.append(scores)

        score = (scores[0] * align_weight) + ((1 - align_weight) * scores[1])
        if best_idx is None or score > best_score:
            best_idx = c_idx
            best_score = score

    shape_tani, colour_tani = all_scores[best_idx]

    # Extract the best conformer (already aligned in-place by AlignMol)
    best_conf = Chem.Conformer(mol_copy.GetConformer(best_idx))
    mol_copy.RemoveAllConformers()
    mol_copy.AddConformer(best_conf, assignId=True)

    return mol_copy, shape_tani, colour_tani


def score_conf(mol, ref_mol, ref_profile=None):
    """Score shape/colour tanimotos without alignment

    Args:
        mol (Chem.Mol): Molecule with a single conformer (not modified)
        ref_mol (Chem.Mol): Reference molecule (not modified)
        ref_profile (ConfProfile, optional): If provided, sets custom pharmacophore features on ref_mol
            (from profile) and mol (from auto-detection) for colour scoring

    Returns:
        (float, float): Tuple of (shape tanimoto, colour tanimoto)
    """

    mol_copy = Chem.Mol(mol)
    ref_copy = Chem.Mol(ref_mol)

    if ref_profile is not None:
        ref_copy = set_pharm_features_from_profile(ref_copy, ref_profile)
        mol_copy = detect_and_set_pharm_features(mol_copy)

    # Use colours to get a non-zero colour tani score
    opts = rdShapeAlign.ShapeInputOptions()
    opts.useColors = True

    shape_tani, colour_tani = rdShapeAlign.ScoreMol(ref_copy, mol_copy, opts, opts)
    return shape_tani, colour_tani
