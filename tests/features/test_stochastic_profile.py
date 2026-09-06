import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.repr import GraphMol
from enscondflow.data.features import InteractionProfile, BindingProfile, StochasticProfile
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
    return GraphMol.from_rdkit(_embed_mol("c1ccccc1"))


@pytest.fixture
def phenol_mol():
    return GraphMol.from_rdkit(_embed_mol("c1ccc(O)cc1"))


@pytest.fixture
def aspirin_mol():
    return GraphMol.from_rdkit(_embed_mol("CC(=O)Oc1ccccc1C(=O)O"))


# *** Tests ***

class TestStochasticProfileInit:
    def test_default_init(self):
        """Test default initialization wrapping InteractionProfile."""
        inner = InteractionProfile()
        profile = StochasticProfile(inner)
        assert profile.name == "profile"
        assert profile.pos_std_dev == 0.0
        assert profile.shape_resample == 0.0
        assert profile.local_pharm_dropout == 0.0
        assert profile.global_pharm_dropout == 0.0
        assert profile.rotate_prob == 0.0

    def test_custom_init(self):
        """Test custom initialization parameters."""
        inner = InteractionProfile(name="custom")
        profile = StochasticProfile(
            inner,
            pos_std_dev=0.5,
            shape_resample=0.3,
            local_pharm_dropout=0.2,
            global_pharm_dropout=0.1,
            rotate_prob=0.5
        )
        assert profile.name == "custom"
        assert profile.pos_std_dev == 0.5
        assert profile.shape_resample == 0.3
        assert profile.local_pharm_dropout == 0.2
        assert profile.global_pharm_dropout == 0.1
        assert profile.rotate_prob == 0.5

    def test_rejects_non_profile_feature(self):
        """Test that non-profile features are rejected."""
        from enscondflow.data.features import Shape
        with pytest.raises(TypeError, match="must be either"):
            StochasticProfile(Shape())

    def test_rejects_invalid_shape_resample(self):
        """Test that shape_resample outside [0, 1] is rejected."""
        inner = InteractionProfile()
        with pytest.raises(ValueError, match="shape_resample"):
            StochasticProfile(inner, shape_resample=1.5)

        with pytest.raises(ValueError, match="shape_resample"):
            StochasticProfile(inner, shape_resample=-0.1)

    def test_tuple_pos_std_dev(self):
        """Test that pos_std_dev can be a tuple for random sampling."""
        inner = InteractionProfile()
        profile = StochasticProfile(inner, pos_std_dev=(0.1, 0.5))
        assert profile.pos_std_dev == (0.1, 0.5)


class TestStochasticProfileRun:
    def test_returns_conf_profile(self, phenol_mol):
        """Test that StochasticProfile returns a ConfProfile."""
        inner = InteractionProfile()
        profile = StochasticProfile(inner)
        result = profile(phenol_mol)

        assert isinstance(result, ConfProfile)

    def test_no_stochasticity_matches_base(self, phenol_mol):
        """Test that zero stochasticity produces the same types as the base profile."""
        inner = InteractionProfile()
        base_result = inner(phenol_mol)

        stoch = StochasticProfile(inner)
        stoch_result = stoch(phenol_mol)

        # Types should match (positions get noise with std=0 which is identity)
        np.testing.assert_array_equal(stoch_result.types, base_result.types)
        assert len(stoch_result.atom_ids) == len(base_result.atom_ids)


class TestStochasticProfileNoise:
    def test_position_noise_applied(self, phenol_mol):
        """Test that position noise changes positions."""
        inner = InteractionProfile()
        no_noise = StochasticProfile(inner, pos_std_dev=0.0)
        result_clean = no_noise(phenol_mol)

        noisy = StochasticProfile(inner, pos_std_dev=1.0)
        result_noisy = noisy(phenol_mol)

        assert not np.allclose(result_clean.positions, result_noisy.positions)

    def test_tuple_std_dev_samples_uniformly(self, phenol_mol):
        """Test that tuple std_dev samples from uniform distribution."""
        inner = InteractionProfile()
        profile = StochasticProfile(inner, pos_std_dev=(0.0, 1.0))

        results = []
        for i in range(10):
            np.random.seed(i)
            result = profile(phenol_mol)
            results.append(result.positions)

        all_same = all(np.allclose(results[0], r) for r in results[1:])
        assert not all_same


class TestStochasticProfileResample:
    def test_resample_adds_shape_points(self, benzene_mol):
        """Test that shape resampling adds extra shape points."""
        inner = InteractionProfile(include_pharms=False)
        profile = StochasticProfile(inner, shape_resample=1.0)

        np.random.seed(42)
        result = profile(benzene_mol)

        benzene_no_hs = benzene_mol.remove_hs()
        n_heavy = len(benzene_no_hs)

        # With resample=1.0, every point is resampled (duplicated)
        shape_mask = result.types == 1
        n_shape = shape_mask.sum()
        assert n_shape == n_heavy * 2

    def test_resample_zero_no_extra_points(self, benzene_mol):
        """Test that resample=0 adds no extra points."""
        inner = InteractionProfile(include_pharms=False)
        profile = StochasticProfile(inner, shape_resample=0.0)
        result = profile(benzene_mol)

        benzene_no_hs = benzene_mol.remove_hs()
        n_heavy = len(benzene_no_hs)

        shape_mask = result.types == 1
        assert shape_mask.sum() == n_heavy

    def test_resample_partial(self, benzene_mol):
        """Test that partial resampling adds some extra points."""
        inner = InteractionProfile(include_pharms=False)
        profile = StochasticProfile(inner, shape_resample=0.5)

        np.random.seed(42)
        result = profile(benzene_mol)

        benzene_no_hs = benzene_mol.remove_hs()
        n_heavy = len(benzene_no_hs)

        shape_mask = result.types == 1
        assert shape_mask.sum() >= n_heavy


class TestStochasticProfilePharmDropout:
    def test_global_dropout_removes_all_pharms(self, aspirin_mol):
        """Test that global_pharm_dropout=1.0 removes all pharmacophores."""
        inner = InteractionProfile()
        profile = StochasticProfile(inner, global_pharm_dropout=1.0)

        result = profile(aspirin_mol)
        pharm_mask = result.types >= 2
        assert pharm_mask.sum() == 0

    def test_global_dropout_zero_keeps_pharms(self, aspirin_mol):
        """Test that global_pharm_dropout=0.0 keeps all pharmacophores."""
        inner = InteractionProfile()

        base_result = inner(aspirin_mol)
        n_pharms_base = (base_result.types >= 2).sum()

        profile = StochasticProfile(inner, global_pharm_dropout=0.0)
        result = profile(aspirin_mol)
        n_pharms = (result.types >= 2).sum()

        assert n_pharms == n_pharms_base

    def test_local_dropout_full_removes_all_pharms(self, aspirin_mol):
        """Test that local_pharm_dropout=1.0 removes all pharmacophores."""
        inner = InteractionProfile()
        profile = StochasticProfile(inner, local_pharm_dropout=1.0)

        result = profile(aspirin_mol)
        pharm_mask = result.types >= 2
        assert pharm_mask.sum() == 0

    def test_local_dropout_zero_keeps_pharms(self, aspirin_mol):
        """Test that local_pharm_dropout=0.0 keeps all pharmacophores."""
        inner = InteractionProfile()

        base_result = inner(aspirin_mol)
        n_pharms_base = (base_result.types >= 2).sum()

        profile = StochasticProfile(inner, local_pharm_dropout=0.0)
        result = profile(aspirin_mol)
        n_pharms = (result.types >= 2).sum()

        assert n_pharms == n_pharms_base

    def test_local_dropout_partial(self, aspirin_mol):
        """Test that partial local dropout reduces pharmacophore count on average."""
        inner = InteractionProfile()

        base_result = inner(aspirin_mol)
        n_pharms_base = (base_result.types >= 2).sum()

        if n_pharms_base == 0:
            pytest.skip("No pharmacophores to test dropout on")

        total_pharms = 0
        n_runs = 20
        profile = StochasticProfile(inner, local_pharm_dropout=0.5)
        for i in range(n_runs):
            np.random.seed(i)
            result = profile(aspirin_mol)
            total_pharms += (result.types >= 2).sum()

        avg_pharms = total_pharms / n_runs
        assert avg_pharms < n_pharms_base
        assert avg_pharms > 0

    def test_global_dropout_keeps_shape_points(self, aspirin_mol):
        """Test that global pharmacophore dropout preserves shape points."""
        inner = InteractionProfile()
        base_result = inner(aspirin_mol)
        n_shape_base = (base_result.types == 1).sum()

        profile = StochasticProfile(inner, global_pharm_dropout=1.0)
        result = profile(aspirin_mol)
        n_shape = (result.types == 1).sum()

        assert n_shape == n_shape_base


class TestStochasticProfileRotation:
    def test_rotation_changes_positions(self, phenol_mol):
        """Test that random rotation changes positions when the coin flip lands on rotate."""
        inner = InteractionProfile()

        no_rot = StochasticProfile(inner, rotate_prob=0.0)
        result_no_rot = no_rot(phenol_mol)

        # Rotation happens 50% of the time — retry until we get a rotated result
        rot = StochasticProfile(inner, rotate_prob=1.0)
        for seed in range(100):
            np.random.seed(seed)
            result_rot = rot(phenol_mol)
            if result_rot.rotated:
                break

        assert result_rot.rotated
        assert not np.allclose(result_no_rot.positions, result_rot.positions)

    def test_rotation_preserves_distances(self, phenol_mol):
        """Test that rotation preserves interatomic distances."""
        inner = InteractionProfile()

        no_rot = StochasticProfile(inner, rotate_prob=0.0)
        result_no_rot = no_rot(phenol_mol)

        # Retry until rotation actually happens
        rot = StochasticProfile(inner, rotate_prob=1.0)
        for seed in range(100):
            np.random.seed(seed)
            result_rot = rot(phenol_mol)
            if result_rot.rotated:
                break

        assert result_rot.rotated

        def pairwise_distances(coords):
            diff = coords[:, None, :] - coords[None, :, :]
            return np.sqrt((diff ** 2).sum(axis=-1))

        dists = pairwise_distances(result_rot.positions)
        dists_no_rot = pairwise_distances(result_no_rot.positions)

        np.testing.assert_array_almost_equal(dists, dists_no_rot, decimal=5)


class TestStochasticProfileAtomIds:
    def test_atom_ids_preserved(self, phenol_mol):
        """Test that atom_ids are maintained through stochastic transformations."""
        inner = InteractionProfile()
        profile = StochasticProfile(inner)
        result = profile(phenol_mol)

        assert result.atom_ids is not None
        assert len(result.atom_ids) == len(result.types)

    def test_atom_ids_grow_with_resample(self, benzene_mol):
        """Test that atom_ids grow when resampling adds points."""
        inner = InteractionProfile(include_pharms=False)

        np.random.seed(42)
        profile = StochasticProfile(inner, shape_resample=0.5)
        result = profile(benzene_mol)

        benzene_no_hs = benzene_mol.remove_hs()
        n_heavy = len(benzene_no_hs)

        # Should have more atom_ids than just heavy atoms if any were resampled
        assert len(result.atom_ids) >= n_heavy

    def test_atom_ids_shrink_with_dropout(self, aspirin_mol):
        """Test that atom_ids shrink when pharmacophores are dropped."""
        inner = InteractionProfile()
        base_result = inner(aspirin_mol)

        profile = StochasticProfile(inner, global_pharm_dropout=1.0)
        result = profile(aspirin_mol)

        assert len(result.atom_ids) <= len(base_result.atom_ids)


class TestStochasticProfileConsistency:
    def test_types_positions_electros_aligned(self, phenol_mol):
        """Test that all arrays remain aligned after stochastic transforms."""
        inner = InteractionProfile()
        profile = StochasticProfile(
            inner,
            pos_std_dev=0.3,
            shape_resample=0.5,
            local_pharm_dropout=0.3,
            rotate_prob=1.0
        )

        np.random.seed(42)
        result = profile(phenol_mol)

        assert len(result.positions) == len(result.types)
        assert len(result.atom_ids) == len(result.types)

    def test_no_padding_type_in_output(self, phenol_mol):
        """Test that type 0 (padding) does not appear in the output."""
        inner = InteractionProfile()
        profile = StochasticProfile(inner, shape_resample=0.5, local_pharm_dropout=0.3)

        np.random.seed(42)
        result = profile(phenol_mol)

        assert 0 not in result.types


class TestStochasticProfileDirections:
    def test_rotation_transforms_directions(self, phenol_mol):
        """Test that rotation is applied to directions."""
        inner = InteractionProfile()

        no_rot = StochasticProfile(inner, rotate_prob=0.0)
        result_no_rot = no_rot(phenol_mol)

        # Retry until rotation actually happens
        rot = StochasticProfile(inner, rotate_prob=1.0)
        for seed in range(100):
            np.random.seed(seed)
            result_rot = rot(phenol_mol)
            if result_rot.rotated:
                break

        assert result_rot.rotated

        # Check if any directions are non-zero (donors should be)
        has_nonzero = np.any(np.linalg.norm(result_no_rot.directions, axis=1) > 0.5)
        if has_nonzero:
            assert not np.allclose(result_no_rot.directions, result_rot.directions)

    def test_noise_does_not_affect_directions(self, phenol_mol):
        """Test that position noise does not change directions."""
        inner = InteractionProfile()

        no_noise = StochasticProfile(inner, pos_std_dev=0.0)
        result_clean = no_noise(phenol_mol)

        np.random.seed(42)
        noisy = StochasticProfile(inner, pos_std_dev=1.0)
        result_noisy = noisy(phenol_mol)

        np.testing.assert_array_equal(result_clean.directions, result_noisy.directions)

    def test_directions_aligned_after_transforms(self, phenol_mol):
        """Test that directions remain aligned with other arrays after stochastic transforms."""
        inner = InteractionProfile()
        profile = StochasticProfile(
            inner,
            pos_std_dev=0.3,
            shape_resample=0.5,
            local_pharm_dropout=0.3,
            rotate_prob=1.0,
        )

        np.random.seed(42)
        result = profile(phenol_mol)

        assert len(result.directions) == len(result.types)
        assert result.directions.shape == result.positions.shape

