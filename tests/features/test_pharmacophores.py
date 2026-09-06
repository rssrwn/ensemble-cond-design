import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.repr import GraphMol
from enscondflow.data.features import Pharmacophores
from enscondflow.util.rdkit import PharmacophoreFinder


# *** Fixtures ***

@pytest.fixture
def benzene_mol():
    """Create a benzene molecule with aromatic ring."""
    smiles = "c1ccccc1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def phenol_mol():
    """Create a phenol molecule with H-bond donor/acceptor."""
    smiles = "c1ccc(O)cc1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def aniline_mol():
    """Create an aniline molecule with amine donor."""
    smiles = "c1ccc(N)cc1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def aspirin_mol():
    """Create aspirin with multiple pharmacophore types."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def multi_conf_mol():
    """Create a molecule with multiple conformers."""
    smiles = "c1ccc(O)cc1"  # Phenol
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=3, randomSeed=42)
    return GraphMol.from_rdkit(rdkit_mol)


# *** Tests ***

class TestPharmacophoresInit:
    def test_default_init(self):
        """Test default initialization of Pharmacophores feature."""
        pharm = Pharmacophores()
        assert pharm.name == "pharmacophores"
        assert pharm.conf_idx == 0

    def test_custom_init(self):
        """Test custom initialization parameters."""
        pharm = Pharmacophores(name="custom-pharm", conf_idx=1)
        assert pharm.name == "custom-pharm"
        assert pharm.conf_idx == 1


class TestPharmacophoresRun:
    def test_returns_tuple(self, phenol_mol):
        """Test that Pharmacophores returns a tuple of (types, coords, atom_ids, directions)."""
        pharm = Pharmacophores()
        result = pharm(phenol_mol)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_types_and_coords_shapes(self, phenol_mol):
        """Test that types and coords have matching lengths."""
        pharm = Pharmacophores()
        types, coords, _, _ = pharm(phenol_mol)

        assert len(types) == len(coords)

    def test_coords_are_3d(self, phenol_mol):
        """Test that coordinates are 3D."""
        pharm = Pharmacophores()
        types, coords, _, _ = pharm(phenol_mol)

        if len(coords) > 0:
            assert coords.shape[1] == 3


class TestPharmacophoreDetection:
    def test_aromatic_detected(self, benzene_mol):
        """Test that aromatic ring is detected."""
        pharm = Pharmacophores()
        types, coords, _, _ = pharm(benzene_mol)

        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic")
        assert aromatic_idx in types

    def test_donor_detected(self, aniline_mol):
        """Test that H-bond donor is detected."""
        pharm = Pharmacophores()
        types, coords, _, _ = pharm(aniline_mol)

        donor_idx = PharmacophoreFinder.get_feature_index("Donor")
        assert donor_idx in types

    def test_acceptor_detected(self, phenol_mol):
        """Test that H-bond acceptor is detected."""
        pharm = Pharmacophores()
        types, coords, _, _ = pharm(phenol_mol)

        acceptor_idx = PharmacophoreFinder.get_feature_index("Acceptor")
        assert acceptor_idx in types

    def test_multiple_types_detected(self, aspirin_mol):
        """Test that multiple pharmacophore types are detected."""
        pharm = Pharmacophores()
        types, coords, _, _ = pharm(aspirin_mol)

        # Aspirin should have multiple pharmacophore types
        unique_types = set(types)
        assert len(unique_types) >= 2


class TestPharmacophorePositions:
    def test_positions_within_molecular_bounds(self, phenol_mol):
        """Test that pharmacophore positions are within molecular bounds."""
        pharm = Pharmacophores()
        types, coords, _, _ = pharm(phenol_mol)

        mol_coords = phenol_mol.get_conformer(0)
        mol_center = mol_coords.mean(axis=0)
        mol_radius = np.linalg.norm(mol_coords - mol_center, axis=1).max()

        for coord in coords:
            dist_from_center = np.linalg.norm(coord - mol_center)
            # Pharmacophore should be within reasonable distance of molecule
            assert dist_from_center < mol_radius + 5.0

    def test_aromatic_position_is_ring_center(self, benzene_mol):
        """Test that aromatic pharmacophore is at ring center."""
        pharm = Pharmacophores()
        types, coords, _, _ = pharm(benzene_mol)

        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic")
        aromatic_mask = np.array(types) == aromatic_idx
        aromatic_pos = coords[aromatic_mask]

        # Get ring center from molecular coords
        benzene_no_hs = benzene_mol.remove_hs()
        heavy_coords = benzene_no_hs.get_conformer(0)
        ring_center = heavy_coords.mean(axis=0)

        # Aromatic pharmacophore should be near ring center
        assert len(aromatic_pos) > 0
        np.testing.assert_array_almost_equal(aromatic_pos[0], ring_center, decimal=1)


class TestPharmacophoreConformer:
    def test_conf_idx_selection(self, multi_conf_mol):
        """Test that conf_idx selects the correct conformer."""
        pharm_0 = Pharmacophores(conf_idx=0)
        _, coords_0, _, _ = pharm_0(multi_conf_mol)

        pharm_1 = Pharmacophores(conf_idx=1)
        _, coords_1, _, _ = pharm_1(multi_conf_mol)

        # Different conformers should have different pharmacophore positions
        if len(coords_0) > 0 and len(coords_1) > 0:
            assert not np.allclose(coords_0, coords_1)


class TestPharmacophoreErrorHandling:
    def test_error_returns_empty_arrays(self, benzene_mol):
        """Test that errors return empty arrays when raise_on_err=False."""
        pharm = Pharmacophores(raise_on_err=False, conf_idx=999)
        types, coords, _, directions = pharm(benzene_mol)

        assert len(types) == 0
        assert coords.shape == (0, 3)
        assert directions.shape == (0, 3)

    def test_error_raises_when_requested(self, benzene_mol):
        """Test that errors raise when raise_on_err=True."""
        pharm = Pharmacophores(raise_on_err=True, conf_idx=999)

        with pytest.raises(Exception):
            pharm(benzene_mol)


class TestPharmacophoreTypeIndices:
    def test_type_indices_are_valid(self, aspirin_mol):
        """Test that type indices are within valid range."""
        pharm = Pharmacophores()
        types, _, _, _ = pharm(aspirin_mol)

        vocab_size = PharmacophoreFinder.get_vocab_size()

        for t in types:
            assert 0 <= t < vocab_size

    def test_types_match_feature_vocab(self, phenol_mol):
        """Test that types correspond to known pharmacophore features."""
        pharm = Pharmacophores()
        types, _, _, _ = pharm(phenol_mol)

        vocab = PharmacophoreFinder.get_feature_vocab()

        for t in types:
            # Should be able to get feature name
            name = PharmacophoreFinder.get_feature_name(t)
            assert name in vocab


class TestPharmacophoreAtomIds:
    def test_atom_ids_returned(self, phenol_mol):
        """Test that atom_ids are returned as a list of tuples."""
        pharm = Pharmacophores()
        types, coords, atom_ids, _ = pharm(phenol_mol)

        assert isinstance(atom_ids, list)
        assert len(atom_ids) == len(types)
        for aid in atom_ids:
            assert isinstance(aid, tuple)

    def test_atom_ids_empty_on_error(self, benzene_mol):
        """Test that atom_ids are empty list on error."""
        pharm = Pharmacophores(raise_on_err=False, conf_idx=999)
        _, _, atom_ids, _ = pharm(benzene_mol)

        assert atom_ids == []


class TestPharmacophoreDirections:
    def test_aniline_donor_expansion(self, aniline_mol):
        """Test that aniline NH2 produces 2 donor features (one per N-H bond)."""
        pharm = Pharmacophores()
        types, coords, atom_ids, directions = pharm(aniline_mol)

        donor_idx = PharmacophoreFinder.get_feature_index("Donor")
        donor_mask = types == donor_idx
        n_donors = donor_mask.sum()

        # Aniline has NH2, so 2 N-H bonds → 2 expanded donor features
        assert n_donors == 2

    def test_donor_directions_unit_length(self, aniline_mol):
        """Test that donor direction vectors are approximately unit length."""
        pharm = Pharmacophores()
        types, coords, atom_ids, directions = pharm(aniline_mol)

        donor_idx = PharmacophoreFinder.get_feature_index("Donor")
        donor_mask = types == donor_idx
        donor_dirs = directions[donor_mask]

        for d in donor_dirs:
            norm = np.linalg.norm(d)
            np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_non_donor_non_aromatic_directions_are_zero(self, aspirin_mol):
        """Test that non-donor, non-aromatic features have zero direction vectors."""
        pharm = Pharmacophores()
        types, coords, atom_ids, directions = pharm(aspirin_mol)

        donor_idx = PharmacophoreFinder.get_feature_index("Donor")
        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic")

        for t, d in zip(types, directions):
            if t != donor_idx and t != aromatic_idx:
                np.testing.assert_array_equal(d, np.zeros(3))

    def test_directions_shape_matches_types(self, phenol_mol):
        """Test that directions array has same length as types."""
        pharm = Pharmacophores()
        types, coords, atom_ids, directions = pharm(phenol_mol)

        assert len(directions) == len(types)
        assert directions.shape == (len(types), 3)

    def test_phenol_donor_has_one_direction(self, phenol_mol):
        """Test that phenol O-H produces 1 donor feature with direction."""
        pharm = Pharmacophores()
        types, coords, atom_ids, directions = pharm(phenol_mol)

        donor_idx = PharmacophoreFinder.get_feature_index("Donor")
        donor_mask = types == donor_idx
        n_donors = donor_mask.sum()

        # Phenol has one O-H bond → 1 expanded donor feature
        assert n_donors == 1

        donor_dir = directions[donor_mask][0]
        assert np.linalg.norm(donor_dir) > 0.9

    def test_aromatic_direction_is_ring_normal(self, benzene_mol):
        """Aromatic directions should be unit-length ring normals."""
        pharm = Pharmacophores()
        types, coords, atom_ids, directions = pharm(benzene_mol)

        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic")
        aromatic_mask = types == aromatic_idx
        aromatic_dirs = directions[aromatic_mask]

        assert len(aromatic_dirs) > 0
        for d in aromatic_dirs:
            np.testing.assert_almost_equal(np.linalg.norm(d), 1.0, decimal=5)
