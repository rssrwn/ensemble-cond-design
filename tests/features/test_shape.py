import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.repr import GraphMol
from enscondflow.data.features import Shape


# *** Fixtures ***

@pytest.fixture
def methane_mol():
    """Create a simple methane molecule for testing."""
    smiles = "C"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def benzene_mol():
    """Create a benzene molecule with 6 heavy atoms."""
    smiles = "c1ccccc1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def multi_conf_mol():
    """Create a molecule with multiple conformers."""
    smiles = "CCCC"  # Butane - can have multiple conformers
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=3, randomSeed=42)
    return GraphMol.from_rdkit(rdkit_mol)


# *** Tests ***

class TestShapeInit:
    def test_default_init(self):
        """Test default initialization of Shape feature."""
        shape = Shape()
        assert shape.name == "shape"
        assert shape.conf_idx == 0

    def test_custom_init(self):
        """Test custom initialization parameters."""
        shape = Shape(name="custom-shape", conf_idx=1)
        assert shape.name == "custom-shape"
        assert shape.conf_idx == 1


class TestShapeRun:
    def test_returns_ndarray(self, benzene_mol):
        """Test that Shape returns a numpy array."""
        shape = Shape()
        result = shape(benzene_mol)
        assert isinstance(result, np.ndarray)

    def test_shape_removes_hydrogens(self, benzene_mol):
        """Test that shape positions are for heavy atoms only."""
        shape = Shape()
        result = shape(benzene_mol)

        # Benzene has 6 heavy atoms
        assert len(result) == 6

    def test_shape_3d_coordinates(self, benzene_mol):
        """Test that result has 3D coordinates."""
        shape = Shape()
        result = shape(benzene_mol)

        assert result.shape == (6, 3)

    def test_positions_match_conformer(self, benzene_mol):
        """Test that positions match the molecule conformer."""
        shape = Shape()
        result = shape(benzene_mol)

        benzene_no_hs = benzene_mol.remove_hs()
        expected = benzene_no_hs.get_conformer(0)

        np.testing.assert_array_almost_equal(result, expected, decimal=5)


class TestShapeConformer:
    def test_conf_idx_selection(self, multi_conf_mol):
        """Test that conf_idx selects the correct conformer."""
        shape_0 = Shape(conf_idx=0)
        result_0 = shape_0(multi_conf_mol)

        shape_1 = Shape(conf_idx=1)
        result_1 = shape_1(multi_conf_mol)

        # Different conformers should have different coordinates
        assert not np.allclose(result_0, result_1)


class TestShapeErrorHandling:
    def test_error_returns_fallback(self, benzene_mol):
        """Test that errors return fallback value when raise_on_err=False."""
        # Use invalid conf_idx to trigger error
        shape = Shape(raise_on_err=False, return_on_err=0.0, conf_idx=999)
        result = shape(benzene_mol)

        # Should return array of return_on_err values
        assert isinstance(result, np.ndarray)

    def test_error_raises_when_requested(self, benzene_mol):
        """Test that errors raise when raise_on_err=True."""
        # Use invalid conf_idx to trigger error
        shape = Shape(raise_on_err=True, conf_idx=999)

        with pytest.raises(Exception):
            shape(benzene_mol)
