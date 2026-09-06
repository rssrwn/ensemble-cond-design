import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.repr import AtomSet, BondSet, ConfSet, GraphMol, Protein, BindingComplex, Interaction, InteractionSet
from enscondflow.data.features import BindingProfile
from enscondflow.data.profile import ConfProfile
from enscondflow.util.rdkit import PharmacophoreFinder


# *** Fixtures ***

@pytest.fixture
def benzene_ligand():
    """Create a benzene molecule as a simple test ligand with an aromatic ring."""
    smiles = "c1ccccc1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def phenol_ligand():
    """Create a phenol molecule with H-bond donor/acceptor capability."""
    smiles = "c1ccc(O)cc1"
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(rdkit_mol)
    return GraphMol.from_rdkit(rdkit_mol)


@pytest.fixture
def simple_protein():
    """Create a minimal protein-like structure for testing."""
    # Simple protein with 10 atoms (C, N, O pattern)
    atomics = np.array([6, 7, 8, 6, 7, 8, 6, 7, 8, 6])  # C, N, O repeating
    charges = np.zeros(10, dtype=np.int32)
    res_names = np.array(["ALA"] * 10)
    atom_names = np.array(["CA", "N", "O", "CA", "N", "O", "CA", "N", "O", "CA"])
    res_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])

    atoms = AtomSet(atomics, charges=charges, res_names=res_names, atom_names=atom_names, res_ids=res_ids)

    # Simple linear bonds
    bonds_arr = np.array([
        [0, 1, 1],
        [1, 2, 1],
        [2, 3, 1],
        [3, 4, 1],
        [4, 5, 1],
        [5, 6, 1],
        [6, 7, 1],
        [7, 8, 1],
        [8, 9, 1],
    ], dtype=np.int32)
    bonds = BondSet(bonds_arr)

    # Random coordinates
    coords = np.random.randn(10, 3).astype(np.float32) * 5
    confs = ConfSet(coords)

    return Protein(atoms, bonds, confs)


@pytest.fixture
def hbond_interactions(phenol_ligand, simple_protein):
    """Create interaction set with H-bond interactions."""
    hb_donor = Interaction(
        protein_atoms=(2,),
        ligand_atoms=(4,),
        interaction_type="HBDonor"
    )

    hb_acceptor = Interaction(
        protein_atoms=(1,),
        ligand_atoms=(4,),
        interaction_type="HBAcceptor"
    )

    return InteractionSet(
        [hb_donor, hb_acceptor],
        n_protein_atoms=len(simple_protein),
        n_ligand_atoms=len(phenol_ligand)
    )


@pytest.fixture
def pi_interactions(benzene_ligand, simple_protein):
    """Create interaction set with pi-stacking interactions."""
    pi_stacking = Interaction(
        protein_atoms=(0, 3, 6),
        ligand_atoms=(0, 1, 2, 3, 4, 5),
        interaction_type="PiStacking"
    )

    return InteractionSet(
        [pi_stacking],
        n_protein_atoms=len(simple_protein),
        n_ligand_atoms=len(benzene_ligand)
    )


@pytest.fixture
def cation_pi_interactions(benzene_ligand, simple_protein):
    """Create interaction set with cation-pi interaction (cation on ligand)."""
    cation_pi = Interaction(
        protein_atoms=(0, 3, 6),
        ligand_atoms=(0,),
        interaction_type="CationPi"
    )

    return InteractionSet(
        [cation_pi],
        n_protein_atoms=len(simple_protein),
        n_ligand_atoms=len(benzene_ligand)
    )


@pytest.fixture
def cationic_interactions(benzene_ligand, simple_protein):
    """Create interaction set with a Cationic interaction."""
    cationic = Interaction(
        protein_atoms=(2,),
        ligand_atoms=(0,),
        interaction_type="Cationic"
    )

    return InteractionSet(
        [cationic],
        n_protein_atoms=len(simple_protein),
        n_ligand_atoms=len(benzene_ligand)
    )


@pytest.fixture
def anionic_interactions(phenol_ligand, simple_protein):
    """Create interaction set with an Anionic interaction."""
    anionic = Interaction(
        protein_atoms=(1,),
        ligand_atoms=(4,),
        interaction_type="Anionic"
    )

    return InteractionSet(
        [anionic],
        n_protein_atoms=len(simple_protein),
        n_ligand_atoms=len(phenol_ligand)
    )


@pytest.fixture
def complex_with_cationic(benzene_ligand, simple_protein, cationic_interactions):
    """Create a BindingComplex with Cationic interactions."""
    return BindingComplex(simple_protein, benzene_ligand, interactions=cationic_interactions)


@pytest.fixture
def complex_with_anionic(phenol_ligand, simple_protein, anionic_interactions):
    """Create a BindingComplex with Anionic interactions."""
    return BindingComplex(simple_protein, phenol_ligand, interactions=anionic_interactions)


@pytest.fixture
def complex_with_hbonds(phenol_ligand, simple_protein, hbond_interactions):
    """Create a BindingComplex with H-bond interactions."""
    return BindingComplex(simple_protein, phenol_ligand, interactions=hbond_interactions)


@pytest.fixture
def complex_with_pi(benzene_ligand, simple_protein, pi_interactions):
    """Create a BindingComplex with pi-stacking interactions."""
    return BindingComplex(simple_protein, benzene_ligand, interactions=pi_interactions)


@pytest.fixture
def complex_with_cation_pi(benzene_ligand, simple_protein, cation_pi_interactions):
    """Create a BindingComplex with cation-pi interactions."""
    return BindingComplex(simple_protein, benzene_ligand, interactions=cation_pi_interactions)


@pytest.fixture
def complex_no_interactions(benzene_ligand, simple_protein):
    """Create a BindingComplex with empty interactions."""
    empty_interactions = InteractionSet(
        [],
        n_protein_atoms=len(simple_protein),
        n_ligand_atoms=len(benzene_ligand)
    )
    return BindingComplex(simple_protein, benzene_ligand, interactions=empty_interactions)


# *** Tests ***

class TestBindingProfileInit:
    def test_default_init(self):
        """Test default initialization of BindingProfile."""
        profile = BindingProfile()
        assert profile.name == "binding-profile"

    def test_custom_init(self):
        """Test custom initialization parameters."""
        profile = BindingProfile(name="custom-profile", raise_on_err=True)
        assert profile.name == "custom-profile"
        assert profile.raise_on_err is True


class TestBindingProfileRun:
    def test_requires_interactions(self, benzene_ligand, simple_protein):
        """Test that BindingProfile raises error when interactions are missing."""
        complex_no_int = BindingComplex(simple_protein, benzene_ligand, interactions=None)
        profile = BindingProfile(raise_on_err=True)

        with pytest.raises(ValueError, match="must have interactions"):
            profile(complex_no_int)

    def test_returns_conf_profile(self, complex_with_hbonds):
        """Test that BindingProfile returns a ConfProfile."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        assert isinstance(result, ConfProfile)
        assert result.positions is not None
        assert result.types is not None

    def test_shape_positions_from_heavy_atoms(self, complex_with_hbonds):
        """Test that shape positions come from ligand heavy atoms."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        ligand_no_hs = complex_with_hbonds.ligand.remove_hs()
        n_heavy_atoms = len(ligand_no_hs)

        # Shape types are 1, so count them
        shape_mask = result.types == 1
        n_shape_points = shape_mask.sum()

        assert n_shape_points == n_heavy_atoms


class TestHBondExtraction:
    def test_hbond_donor_extraction(self, complex_with_hbonds):
        """Test that HBDonor interactions are extracted correctly."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        donor_idx = PharmacophoreFinder.get_feature_index("Donor") + 2
        donor_mask = result.types == donor_idx

        assert donor_mask.sum() > 0

    def test_hbond_acceptor_extraction(self, complex_with_hbonds):
        """Test that HBAcceptor interactions are extracted correctly."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        acceptor_idx = PharmacophoreFinder.get_feature_index("Acceptor") + 2
        acceptor_mask = result.types == acceptor_idx

        assert acceptor_mask.sum() > 0

    def test_hbond_positions_from_ligand(self, complex_with_hbonds):
        """Test that H-bond positions come from ligand atoms."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        ligand_coords = complex_with_hbonds.ligand.get_conformer(0)

        donor_idx = PharmacophoreFinder.get_feature_index("Donor") + 2
        donor_mask = result.types == donor_idx
        donor_positions = result.positions[donor_mask]

        # The O atom in phenol is at index 4 (with Hs)
        expected_pos = ligand_coords[4]

        assert len(donor_positions) > 0
        np.testing.assert_array_almost_equal(donor_positions[0], expected_pos, decimal=5)


class TestPiInteractionExtraction:
    def test_pi_stacking_extraction(self, complex_with_pi):
        """Test that PiStacking interactions are extracted correctly."""
        profile = BindingProfile()
        result = profile(complex_with_pi)

        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic") + 2
        aromatic_mask = result.types == aromatic_idx

        assert aromatic_mask.sum() > 0

    def test_pi_stacking_position_is_ring_center(self, complex_with_pi):
        """Test that PiStacking position is the center of the ring."""
        profile = BindingProfile()
        result = profile(complex_with_pi)

        ligand_coords = complex_with_pi.ligand.get_conformer(0)

        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic") + 2
        aromatic_mask = result.types == aromatic_idx
        aromatic_positions = result.positions[aromatic_mask]

        # Ring atoms are 0-5, so center should be mean of those coords
        ring_center = ligand_coords[:6].mean(axis=0)

        assert len(aromatic_positions) == 1
        np.testing.assert_array_almost_equal(aromatic_positions[0], ring_center, decimal=5)

    def test_cation_pi_skipped_without_matching_feature(self, complex_with_cation_pi):
        """Test that CationPi is skipped when no Cation feature is detected on the ligand."""
        profile = BindingProfile()
        result = profile(complex_with_cation_pi)

        cation_idx = PharmacophoreFinder.get_feature_index("Cation") + 2
        cation_mask = result.types == cation_idx

        # Benzene has no Cation pharmacophore feature, so CationPi interaction is skipped
        assert cation_mask.sum() == 0


class TestIonicInteractionExtraction:
    def test_cationic_skipped_without_matching_feature(self, complex_with_cationic):
        """Test that Cationic interactions are skipped when no Cation feature is detected."""
        profile = BindingProfile()
        result = profile(complex_with_cationic)

        cation_idx = PharmacophoreFinder.get_feature_index("Cation") + 2
        cation_mask = result.types == cation_idx

        # Benzene has no Cation pharmacophore feature
        assert cation_mask.sum() == 0

    def test_anionic_skipped_without_matching_feature(self, complex_with_anionic):
        """Test that Anionic interactions are skipped when no Anion feature is detected."""
        profile = BindingProfile()
        result = profile(complex_with_anionic)

        anion_idx = PharmacophoreFinder.get_feature_index("Anion") + 2
        anion_mask = result.types == anion_idx

        # Phenol has no Anion pharmacophore feature
        assert anion_mask.sum() == 0


class TestUnsupportedInteractions:
    def test_hydrophobic_skipped_when_not_in_vocab(self, benzene_ligand, simple_protein):
        """Test that Hydrophobic interactions are silently skipped when Hydrophobe is not in vocab."""
        hydrophobic = Interaction(
            protein_atoms=(0,),
            ligand_atoms=(0,),
            interaction_type="Hydrophobic"
        )
        interactions = InteractionSet(
            [hydrophobic],
            n_protein_atoms=len(simple_protein),
            n_ligand_atoms=len(benzene_ligand)
        )
        system = BindingComplex(simple_protein, benzene_ligand, interactions=interactions)

        profile = BindingProfile()
        result = profile(system)

        # Should only have shape points since Hydrophobe is not in the pharmacophore vocab
        pharm_mask = result.types > 1
        assert pharm_mask.sum() == 0


class TestEmptyInteractions:
    def test_no_interactions(self, complex_no_interactions):
        """Test handling of complex with no interactions."""
        profile = BindingProfile()
        result = profile(complex_no_interactions)

        # Should still have shape points
        shape_mask = result.types == 1
        assert shape_mask.sum() > 0

        # Should not have any pharmacophore points (types > 1)
        pharm_mask = result.types > 1
        assert pharm_mask.sum() == 0


class TestTypeIndexing:
    def test_type_indices_correct(self, complex_with_hbonds):
        """Test that type indices follow the expected scheme."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        # Type 0 should not appear (reserved for padding)
        assert 0 not in result.types

        # Type 1 should appear (shape points)
        assert 1 in result.types

        # Pharmacophore types should be >= 2
        pharm_mask = result.types > 1
        if pharm_mask.sum() > 0:
            min_pharm_type = result.types[pharm_mask].min()
            assert min_pharm_type >= 2


class TestBindingProfileAtomIds:
    def test_atom_ids_present(self, complex_with_hbonds):
        """Test that atom_ids are set on the returned ConfProfile."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        assert result.atom_ids is not None
        assert len(result.atom_ids) == len(result.types)

    def test_shape_atom_ids_are_single_element(self, complex_with_hbonds):
        """Test that shape point atom_ids are single-element tuples."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        ligand_no_hs = complex_with_hbonds.ligand.remove_hs()
        n_heavy = len(ligand_no_hs)

        # First n_heavy atom_ids should be single-element tuples
        for i in range(n_heavy):
            assert isinstance(result.atom_ids[i], tuple)
            assert len(result.atom_ids[i]) == 1

    def test_hbond_atom_ids_single_element(self, complex_with_hbonds):
        """Test that H-bond pharmacophore atom_ids are single-element tuples."""
        profile = BindingProfile()
        result = profile(complex_with_hbonds)

        ligand_no_hs = complex_with_hbonds.ligand.remove_hs()
        n_heavy = len(ligand_no_hs)

        # Pharmacophore atom_ids (after shape points)
        pharm_atom_ids = result.atom_ids[n_heavy:]
        assert len(pharm_atom_ids) > 0

        for aid in pharm_atom_ids:
            assert isinstance(aid, tuple)
            assert len(aid) == 1  # H-bonds are single atoms

    def test_pi_stacking_atom_ids_multi_element(self, complex_with_pi):
        """Test that pi-stacking pharmacophore atom_ids are multi-element tuples (ring atoms)."""
        profile = BindingProfile()
        result = profile(complex_with_pi)

        ligand_no_hs = complex_with_pi.ligand.remove_hs()
        n_heavy = len(ligand_no_hs)

        # Pharmacophore atom_ids (after shape points)
        pharm_atom_ids = result.atom_ids[n_heavy:]
        assert len(pharm_atom_ids) == 1  # One pi-stacking interaction

        # Ring atom_ids should contain multiple atoms
        assert len(pharm_atom_ids[0]) > 1
        assert pharm_atom_ids[0] == (0, 1, 2, 3, 4, 5)

    def test_empty_interactions_atom_ids(self, complex_no_interactions):
        """Test that atom_ids are present even with no interactions."""
        profile = BindingProfile()
        result = profile(complex_no_interactions)

        assert result.atom_ids is not None
        # Should only have shape atom_ids
        ligand_no_hs = complex_no_interactions.ligand.remove_hs()
        n_heavy = len(ligand_no_hs)
        assert len(result.atom_ids) == n_heavy
