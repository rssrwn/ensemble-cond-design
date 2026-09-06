import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.repr import GraphMol
from enscondflow.data.features import InteractionProfile
from enscondflow.data.profile import ConfProfile


# *** Fixtures ***

def _embed_mol(smiles, n_confs=1):
    """Helper to create a molecule with conformer(s) and explicit Hs."""
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    if n_confs == 1:
        AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(rdkit_mol)
    else:
        AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=n_confs, randomSeed=42)
    return rdkit_mol


@pytest.fixture
def benzene_mol():
    """Create a benzene molecule with aromatic ring (no explicit Hs)."""
    return GraphMol.from_rdkit(_embed_mol("c1ccccc1"))


@pytest.fixture
def phenol_mol():
    """Create a phenol molecule with H-bond donor/acceptor and aromatic ring (no explicit Hs)."""
    return GraphMol.from_rdkit(_embed_mol("c1ccc(O)cc1"))


@pytest.fixture
def aspirin_mol():
    """Create aspirin with multiple pharmacophore types (no explicit Hs)."""
    return GraphMol.from_rdkit(_embed_mol("CC(=O)Oc1ccccc1C(=O)O"))


@pytest.fixture
def multi_conf_mol():
    """Create a molecule with multiple conformers (no explicit Hs)."""
    return GraphMol.from_rdkit(_embed_mol("c1ccc(O)cc1", n_confs=3))


# *** Tests ***

class TestInteractionProfileInit:
    def test_default_init(self):
        """Test default initialization of InteractionProfile."""
        profile = InteractionProfile()
        assert profile.name == "profile"
        assert profile.include_pharms is True

    def test_custom_init(self):
        """Test custom initialization parameters."""
        profile = InteractionProfile(name="custom-profile", include_pharms=False, conf_idx=1)
        assert profile.name == "custom-profile"
        assert profile.include_pharms is False

    def test_no_pharms_init(self):
        """Test initialization with pharmacophores disabled."""
        profile = InteractionProfile(include_pharms=False)
        assert profile.include_pharms is False


class TestInteractionProfileRun:
    def test_returns_conf_profile(self, phenol_mol):
        """Test that InteractionProfile returns a ConfProfile."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert isinstance(result, ConfProfile)

    def test_conf_profile_has_positions(self, phenol_mol):
        """Test that result has positions."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert result.positions is not None
        assert len(result.positions) > 0

    def test_conf_profile_has_types(self, phenol_mol):
        """Test that result has types."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert result.types is not None
        assert len(result.types) > 0


class TestInteractionProfileShape:
    def test_shape_points_from_heavy_atoms(self, benzene_mol):
        """Test that shape points come from heavy atoms."""
        profile = InteractionProfile()
        result = profile(benzene_mol)

        benzene_no_hs = benzene_mol.remove_hs()
        n_heavy_atoms = len(benzene_no_hs)

        # Count shape points (type == 1)
        shape_mask = result.types == 1
        n_shape_points = shape_mask.sum()

        # Should have at least n_heavy_atoms shape points
        assert n_shape_points >= n_heavy_atoms

    def test_shape_positions_3d(self, phenol_mol):
        """Test that shape positions are 3D coordinates."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert result.positions.shape[1] == 3


class TestInteractionProfileTypeIndexing:
    def test_type_0_not_present(self, phenol_mol):
        """Test that type 0 (padding) is not present."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert 0 not in result.types

    def test_type_1_is_shape(self, phenol_mol):
        """Test that type 1 represents shape points."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert 1 in result.types

    def test_pharmacophore_types_offset_by_2(self, phenol_mol):
        """Test that pharmacophore types are offset by 2."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        # Pharmacophore types should be >= 2
        pharm_mask = result.types > 1
        if pharm_mask.sum() > 0:
            min_pharm_type = result.types[pharm_mask].min()
            assert min_pharm_type >= 2


class TestInteractionProfileIncludePharms:
    def test_no_pharms_only_shape(self, benzene_mol):
        """Test that include_pharms=False produces only shape points."""
        profile = InteractionProfile(include_pharms=False)
        result = profile(benzene_mol)

        benzene_no_hs = benzene_mol.remove_hs()
        n_heavy_atoms = len(benzene_no_hs)

        # Count shape points
        shape_mask = result.types == 1
        n_shape_points = shape_mask.sum()

        # Should have exactly n_heavy_atoms shape points
        assert n_shape_points == n_heavy_atoms

        # Should have no pharmacophore points
        pharm_mask = result.types > 1
        assert pharm_mask.sum() == 0

    def test_with_pharms_has_pharmacophores(self, aspirin_mol):
        """Test that include_pharms=True produces pharmacophore points."""
        profile = InteractionProfile(include_pharms=True)
        result = profile(aspirin_mol)

        pharm_mask = result.types > 1
        assert pharm_mask.sum() > 0


class TestInteractionProfileConformer:
    def test_conf_idx_selection(self, multi_conf_mol):
        """Test that conf_idx selects the correct conformer."""
        profile_0 = InteractionProfile(conf_idx=0)
        result_0 = profile_0(multi_conf_mol)

        profile_1 = InteractionProfile(conf_idx=1)
        result_1 = profile_1(multi_conf_mol)

        # Different conformers should have different positions
        assert not np.allclose(result_0.positions, result_1.positions)


class TestInteractionProfileCombination:
    def test_shape_and_pharm_combined(self, phenol_mol):
        """Test that shape and pharmacophore points are combined."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        # Should have both shape points (type 1) and pharmacophore points (type > 1)
        assert 1 in result.types
        assert any(t > 1 for t in result.types)

    def test_positions_types_aligned(self, phenol_mol):
        """Test that positions and types are aligned."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert len(result.positions) == len(result.types)


class TestInteractionProfileErrorHandling:
    def test_always_raises_on_error(self, phenol_mol):
        """Test that InteractionProfile raises on error with invalid conf_idx."""
        profile = InteractionProfile(conf_idx=999, raise_on_err=True)

        with pytest.raises(Exception):
            profile(phenol_mol)


class TestInteractionProfileCallable:
    def test_callable_interface(self, phenol_mol):
        """Test that InteractionProfile can be called directly."""
        profile = InteractionProfile()

        # Both call methods should work
        result_call = profile(phenol_mol)

        assert isinstance(result_call, ConfProfile)


class TestInteractionProfileAtomIds:
    def test_atom_ids_present(self, phenol_mol):
        """Test that atom_ids are set on the returned ConfProfile."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert result.atom_ids is not None
        assert len(result.atom_ids) == len(result.types)

    def test_shape_atom_ids_are_single_element(self, benzene_mol):
        """Test that shape point atom_ids are single-element tuples."""
        profile = InteractionProfile(include_pharms=False)
        result = profile(benzene_mol)

        # With no pharms, all points are shape points
        for aid in result.atom_ids:
            assert isinstance(aid, tuple)
            assert len(aid) == 1

    def test_shape_atom_ids_are_sequential(self, benzene_mol):
        """Test that shape atom_ids correspond to heavy atom indices."""
        profile = InteractionProfile(include_pharms=False)
        result = profile(benzene_mol)

        benzene_no_hs = benzene_mol.remove_hs()
        n_heavy = len(benzene_no_hs)

        # First n_heavy atom_ids should be (0,), (1,), ..., (n_heavy-1,)
        for i in range(n_heavy):
            assert result.atom_ids[i] == (i,)

    def test_atom_ids_include_pharmacophores(self, phenol_mol):
        """Test that atom_ids include pharmacophore entries."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        phenol_no_hs = phenol_mol.remove_hs()
        n_heavy = len(phenol_no_hs)

        # Total atom_ids should be more than just shape points
        n_pharm = (result.types > 1).sum()
        assert len(result.atom_ids) == n_heavy + n_pharm


class TestInteractionProfileDirections:
    def test_directions_exist(self, phenol_mol):
        """Test that result has directions field with correct shape."""
        profile = InteractionProfile()
        result = profile(phenol_mol)

        assert result.directions is not None
        assert result.directions.shape == result.positions.shape

    def test_shape_points_have_zero_directions(self, benzene_mol):
        """Test that shape points have zero direction vectors."""
        profile = InteractionProfile(include_pharms=False)
        result = profile(benzene_mol)

        np.testing.assert_array_equal(result.directions, 0.0)

    def test_donor_features_have_nonzero_directions(self, phenol_mol):
        """Test that donor features have non-zero directions."""
        from enscondflow.util.rdkit import PharmacophoreFinder

        profile = InteractionProfile()
        result = profile(phenol_mol)

        donor_idx = PharmacophoreFinder.get_feature_index("Donor") + 2
        donor_mask = result.types == donor_idx

        if donor_mask.sum() > 0:
            donor_dirs = result.directions[donor_mask]
            for d in donor_dirs:
                assert np.linalg.norm(d) > 0.9
