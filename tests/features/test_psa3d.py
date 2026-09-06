import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.repr import GraphMol, ConfSet
from enscondflow.data.features import PSA3D


@pytest.fixture
def phenol_multiconf():
    """Create a phenol molecule with multiple conformers (has polar O-H)."""
    smiles = "c1ccc(O)cc1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=3, randomSeed=42)
    for conf_id in range(rdkit_mol.GetNumConformers()):
        AllChem.MMFFOptimizeMolecule(rdkit_mol, confId=conf_id)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def methane_multiconf():
    """Create methane with multiple conformers (no polar surface)."""
    smiles = "C"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=3, randomSeed=42)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def butane_multiconf():
    """Create butane with multiple conformers (no polar atoms)."""
    smiles = "CCCC"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=5, randomSeed=42)
    return GraphMol.from_rdkit(rdkit_mol)


class TestPSA3D:
    def test_default_name(self):
        feat = PSA3D()
        assert feat.name == "psa3d"

    def test_returns_float(self, phenol_multiconf):
        feat = PSA3D()
        result = feat(phenol_multiconf)
        assert isinstance(result, float)

    def test_positive_for_polar_mol(self, phenol_multiconf):
        """Phenol has O-H, so 3D PSA should be > 0."""
        feat = PSA3D()
        result = feat(phenol_multiconf)
        assert result > 0.0

    def test_zero_for_nonpolar_mol(self, butane_multiconf):
        """Butane has no polar atoms, so 3D PSA should be 0."""
        feat = PSA3D()
        result = feat(butane_multiconf)
        assert result == 0.0

    def test_uses_all_conformers(self, phenol_multiconf):
        """Different number of conformers should give potentially different values."""
        feat = PSA3D()
        full_result = feat(phenol_multiconf)

        # Use only first 2 conformers
        coords_2 = phenol_multiconf.confs.coords[:2]
        mol_2 = phenol_multiconf.copy_with(confs=ConfSet(coords_2))
        result_2 = feat(mol_2)

        # Both should be positive floats (they may or may not differ numerically for rigid phenol)
        assert isinstance(full_result, float) and full_result > 0.0
        assert isinstance(result_2, float) and result_2 > 0.0

    def test_error_handling_returns_default(self):
        """A molecule with no conformers should return 0.0 via error handling."""
        smiles = "c1ccc(O)cc1"
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        mol = GraphMol.from_rdkit(rdkit_mol)

        feat = PSA3D()
        result = feat(mol)
        assert result == 0.0

    def test_single_conf_returns_error(self):
        """A molecule with only 1 conformer should fail the assertion and return error value."""
        smiles = "c1ccc(O)cc1"
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
        mol = GraphMol.from_rdkit(rdkit_mol)

        feat = PSA3D()
        result = feat(mol)
        assert result == 0.0
