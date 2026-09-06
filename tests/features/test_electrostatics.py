import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.repr import GraphMol
from enscondflow.data.features import Electrostatics


# *** Fixtures ***

@pytest.fixture
def methane_mol():
    """Create a simple methane molecule for testing."""
    smiles = "C"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def phenol_mol():
    """Create a phenol molecule with polar atoms."""
    smiles = "c1ccc(O)cc1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def charged_mol():
    """Create a molecule with formal charges (carboxylic acid)."""
    smiles = "CC(=O)O"  # Acetic acid
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def multi_conf_mol():
    """Create a molecule with multiple conformers."""
    smiles = "CCCC"  # Butane
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=3, randomSeed=42)
    return GraphMol.from_rdkit(rdkit_mol)


# *** Tests ***

class TestElectrostaticsInit:
    def test_default_init(self):
        """Test default initialization of Electrostatics feature."""
        electro = Electrostatics()
        assert electro.name == "electrostatics"
        assert electro.conf_idx == 0
        assert electro.raise_on_err is False
        assert electro.return_on_err == 0.0

    def test_custom_init(self):
        """Test custom initialization parameters."""
        electro = Electrostatics(
            name="custom-electro",
            conf_idx=1,
            raise_on_err=True,
            return_on_err=-1.0
        )
        assert electro.name == "custom-electro"
        assert electro.conf_idx == 1
        assert electro.raise_on_err is True
        assert electro.return_on_err == -1.0


class TestElectrostaticsRun:
    def test_returns_ndarray(self, phenol_mol):
        """Test that Electrostatics returns a numpy array."""
        electro = Electrostatics()
        result = electro(phenol_mol)
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_heavy_atoms(self, phenol_mol):
        """Test that output has one value per heavy atom."""
        electro = Electrostatics()
        result = electro(phenol_mol)

        n_heavy = len(phenol_mol.remove_hs())
        assert len(result) == n_heavy

    def test_output_is_1d(self, phenol_mol):
        """Test that output is a 1D array."""
        electro = Electrostatics()
        result = electro(phenol_mol)

        assert result.ndim == 1


class TestElectrostaticsValues:
    def test_nonzero_partial_charges(self, phenol_mol):
        """Test that polar molecules have non-zero partial charges."""
        electro = Electrostatics()
        result = electro(phenol_mol)

        # Phenol should have non-zero partial charges
        assert not np.allclose(result, 0.0)

    def test_charge_neutrality(self, phenol_mol):
        """Test that total charge is approximately neutral for neutral molecules."""
        electro = Electrostatics()
        result = electro(phenol_mol)

        # Sum of partial charges should be close to zero for neutral molecule
        assert np.abs(result.sum()) < 0.1

    def test_oxygen_has_negative_charge(self, phenol_mol):
        """Test that oxygen atom has negative partial charge."""
        electro = Electrostatics()
        result = electro(phenol_mol)

        # Find oxygen atom index in heavy-atom-only molecule
        rdkit_mol = phenol_mol.remove_hs().to_rdkit()
        o_idx = None
        for i, atom in enumerate(rdkit_mol.GetAtoms()):
            if atom.GetAtomicNum() == 8:
                o_idx = i
                break

        assert o_idx is not None
        assert result[o_idx] < 0  # Oxygen should have negative partial charge

    def test_carbon_varies_by_environment(self, charged_mol):
        """Test that carbon charges vary based on chemical environment."""
        electro = Electrostatics()
        result = electro(charged_mol)

        # Find carbon atoms in heavy-atom-only molecule
        rdkit_mol = charged_mol.remove_hs().to_rdkit()
        c_indices = []
        for i, atom in enumerate(rdkit_mol.GetAtoms()):
            if atom.GetAtomicNum() == 6:
                c_indices.append(i)

        # At least two carbons in acetic acid
        assert len(c_indices) >= 2

        # Carbon charges should differ based on environment
        c_charges = [result[i] for i in c_indices]
        assert not np.allclose(c_charges[0], c_charges[1])


class TestElectrostaticsConformer:
    def test_conf_idx_selection(self, multi_conf_mol):
        """Test that conf_idx affects conformer used."""
        electro_0 = Electrostatics(conf_idx=0)
        electro_1 = Electrostatics(conf_idx=1)

        # For the same molecule structure, charges should be similar
        # but computed from different conformers
        result_0 = electro_0(multi_conf_mol)
        result_1 = electro_1(multi_conf_mol)

        # Charges depend on geometry, so may differ slightly
        assert result_0.shape == result_1.shape


class TestElectrostaticsMMFF:
    def test_uses_mmff_forcefield(self, phenol_mol):
        """Test that MMFF forcefield is used for partial charges (heavy atoms only)."""
        electro = Electrostatics()
        result = electro(phenol_mol)

        # Verify by computing MMFF charges on the heavy-atom-only molecule
        phenol_no_hs = phenol_mol.remove_hs()
        rdkit_mol = phenol_no_hs.to_rdkit(sanitise=True)
        mmff_props = AllChem.MMFFGetMoleculeProperties(rdkit_mol)
        expected_charges = [mmff_props.GetMMFFPartialCharge(i) for i in range(rdkit_mol.GetNumAtoms())]

        np.testing.assert_array_almost_equal(result, expected_charges, decimal=6)


class TestElectrostaticsErrorHandling:
    def test_error_returns_fallback(self, phenol_mol):
        """Test that errors return fallback value when raise_on_err=False."""
        # Use invalid conf_idx to trigger error
        electro = Electrostatics(raise_on_err=False, return_on_err=0.0, conf_idx=999)
        result = electro(phenol_mol)

        # Should return array of return_on_err values
        assert isinstance(result, np.ndarray)

    def test_error_raises_when_requested(self, phenol_mol):
        """Test that errors raise when raise_on_err=True."""
        # Use invalid conf_idx to trigger error
        electro = Electrostatics(raise_on_err=True, conf_idx=999)

        with pytest.raises(Exception):
            electro(phenol_mol)


class TestElectrostaticsCallable:
    def test_callable_interface(self, phenol_mol):
        """Test that Electrostatics can be called directly."""
        electro = Electrostatics()

        # Both call methods should work
        result_call = electro(phenol_mol)
        result_run = electro.run(phenol_mol)

        np.testing.assert_array_equal(result_call, result_run)
