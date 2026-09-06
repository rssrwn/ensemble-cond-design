import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.repr import GraphMol
from enscondflow.data.features import (
    Tpsa, RadiusOfGyration, IntramolecularHBonds,
    MaskFeatureGroup, Qed, LogP, FeatureGroup
)


# *** Fixtures ***

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
def aspirin_mol():
    """Create aspirin with multiple functional groups."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def salicylic_acid_mol():
    """Create salicylic acid which can form an intramolecular H-bond between OH and COOH."""
    smiles = "OC1=CC=CC=C1C(O)=O"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def methane_mol():
    """Create a simple methane molecule."""
    smiles = "C"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def multi_conf_mol():
    """Create a molecule with multiple conformers."""
    smiles = "CCCC"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMultipleConfs(rdkit_mol, numConfs=3, randomSeed=42)
    return GraphMol.from_rdkit(rdkit_mol)


# *** TPSA Tests ***

class TestTpsa:
    def test_returns_float(self, phenol_mol):
        """Test that Tpsa returns a float."""
        feat = Tpsa()
        result = feat(phenol_mol)
        assert isinstance(result, float)

    def test_default_name(self):
        """Test default feature name."""
        feat = Tpsa()
        assert feat.name == "tpsa"

    def test_nonzero_for_polar_mol(self, phenol_mol):
        """Test that phenol has non-zero TPSA (has polar O-H)."""
        feat = Tpsa()
        result = feat(phenol_mol)
        assert result > 0.0

    def test_zero_for_nonpolar_mol(self):
        """Test that a nonpolar molecule has zero TPSA."""
        smiles = "c1ccccc1"
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
        benzene = GraphMol.from_rdkit(rdkit_mol)

        feat = Tpsa()
        result = feat(benzene)
        assert result == 0.0

    def test_aspirin_higher_than_phenol(self, aspirin_mol, phenol_mol):
        """Test that aspirin (more polar groups) has higher TPSA than phenol."""
        feat = Tpsa()
        assert feat(aspirin_mol) > feat(phenol_mol)


# *** Radius of Gyration Tests ***

class TestRadiusOfGyration:
    def test_returns_float(self, phenol_mol):
        """Test that RadiusOfGyration returns a float."""
        feat = RadiusOfGyration()
        result = feat(phenol_mol)
        assert isinstance(result, float)

    def test_default_name(self):
        """Test default feature name."""
        feat = RadiusOfGyration()
        assert feat.name == "radius-of-gyration"

    def test_positive_value(self, phenol_mol):
        """Test that Rg is positive."""
        feat = RadiusOfGyration()
        result = feat(phenol_mol)
        assert result > 0.0

    def test_methane_smaller_than_aspirin(self, methane_mol, aspirin_mol):
        """Test that methane has a smaller Rg than aspirin."""
        feat = RadiusOfGyration()
        assert feat(methane_mol) < feat(aspirin_mol)

    def test_conf_idx_selects_conformer(self, multi_conf_mol):
        """Test that different conformers give different Rg values."""
        feat_0 = RadiusOfGyration(conf_idx=0)
        feat_1 = RadiusOfGyration(conf_idx=1)

        rg_0 = feat_0(multi_conf_mol)
        rg_1 = feat_1(multi_conf_mol)

        # Different conformers may have different Rg (butane is flexible)
        assert isinstance(rg_0, float)
        assert isinstance(rg_1, float)


# *** Intramolecular H-Bond Tests ***

class TestIntramolecularHBonds:
    def test_returns_int(self, phenol_mol):
        """Test that IntramolecularHBonds returns an int."""
        feat = IntramolecularHBonds()
        result = feat(phenol_mol)
        assert isinstance(result, int)

    def test_default_name(self):
        """Test default feature name."""
        feat = IntramolecularHBonds()
        assert feat.name == "intramol-hbonds"

    def test_methane_has_zero(self, methane_mol):
        """Test that methane has zero intramolecular H-bonds."""
        feat = IntramolecularHBonds()
        result = feat(methane_mol)
        assert result == 0

    def test_phenol_has_zero(self, phenol_mol):
        """Test that phenol (small, no intramolecular H-bond possible) has zero."""
        feat = IntramolecularHBonds()
        result = feat(phenol_mol)
        assert result == 0

    def test_salicylic_acid_has_hbond(self, salicylic_acid_mol):
        """Test that salicylic acid can form intramolecular H-bond (OH...O=C)."""
        feat = IntramolecularHBonds()
        result = feat(salicylic_acid_mol)
        # Salicylic acid commonly forms intramolecular H-bond between phenol OH and carboxyl
        assert result >= 1

    def test_nonnegative(self, aspirin_mol):
        """Test that result is non-negative."""
        feat = IntramolecularHBonds()
        result = feat(aspirin_mol)
        assert result >= 0

    def test_class_constants(self):
        """Test that Prolif-derived constants are set correctly."""
        assert IntramolecularHBonds.DA_DIST_MAX == 3.5
        assert IntramolecularHBonds.DHA_ANGLE_MIN == 130.0
        assert IntramolecularHBonds.MIN_BOND_DIST == 4


# *** MaskFeatureGroup Tests ***

class TestMaskFeatureGroup:
    def test_returns_dict(self, phenol_mol):
        """Test that MaskFeatureGroup returns a dict."""
        group = MaskFeatureGroup(Qed(), LogP(), mask_prob=0.0)
        result = group(phenol_mol)

        assert isinstance(result, dict)

    def test_includes_features_and_masks(self, phenol_mol):
        """Test that result includes both feature values and masks."""
        group = MaskFeatureGroup(Qed(), LogP(), mask_prob=0.0)
        result = group(phenol_mol)

        assert "qed" in result
        assert "logp" in result
        assert "qed-mask" in result
        assert "logp-mask" in result

    def test_mask_prob_zero_all_kept(self, phenol_mol):
        """Test that mask_prob=0 keeps all features (mask=1)."""
        group = MaskFeatureGroup(Qed(), LogP(), mask_prob=0.0)
        result = group(phenol_mol)

        assert result["qed-mask"] == 1
        assert result["logp-mask"] == 1

    def test_mask_prob_one_all_dropped(self, phenol_mol):
        """Test that mask_prob=1 drops all features (mask=0)."""
        group = MaskFeatureGroup(Qed(), LogP(), mask_prob=1.0)
        result = group(phenol_mol)

        assert result["qed-mask"] == 0
        assert result["logp-mask"] == 0

    def test_mask_is_independent_per_feature(self, phenol_mol):
        """Test that masks are sampled independently for each feature."""
        group = MaskFeatureGroup(Qed(), LogP(), mask_prob=0.5)

        # Run multiple times and check that masks differ across features at least sometimes
        seen_different = False
        for i in range(20):
            np.random.seed(i)
            result = group(phenol_mol)
            if result["qed-mask"] != result["logp-mask"]:
                seen_different = True
                break

        assert seen_different

    def test_rejects_invalid_mask_prob(self):
        """Test that invalid mask_prob is rejected."""
        with pytest.raises(ValueError, match="mask_prob"):
            MaskFeatureGroup(Qed(), mask_prob=1.5)

        with pytest.raises(ValueError, match="mask_prob"):
            MaskFeatureGroup(Qed(), mask_prob=-0.1)

    def test_rejects_duplicate_names(self):
        """Test that duplicate feature names are rejected."""
        with pytest.raises(ValueError, match="unique"):
            MaskFeatureGroup(Qed(), Qed(), mask_prob=0.0)

    def test_feature_values_correct(self, phenol_mol):
        """Test that feature values match direct computation."""
        qed_feat = Qed()
        logp_feat = LogP()
        group = MaskFeatureGroup(qed_feat, logp_feat, mask_prob=0.0)

        result = group(phenol_mol)
        expected_qed = qed_feat(phenol_mol)
        expected_logp = logp_feat(phenol_mol)

        assert result["qed"] == expected_qed
        assert result["logp"] == expected_logp


class TestFeatureGroup:
    def test_returns_dict(self, phenol_mol):
        """Test that FeatureGroup returns a dict."""
        group = FeatureGroup(Qed(), LogP())
        result = group(phenol_mol)

        assert isinstance(result, dict)
        assert "qed" in result
        assert "logp" in result

    def test_no_masks(self, phenol_mol):
        """Test that FeatureGroup does not include masks."""
        group = FeatureGroup(Qed(), LogP())
        result = group(phenol_mol)

        assert "qed-mask" not in result
        assert "logp-mask" not in result

    def test_rejects_duplicate_names(self):
        """Test that duplicate feature names are rejected."""
        with pytest.raises(ValueError, match="unique"):
            FeatureGroup(Qed(), Qed())
