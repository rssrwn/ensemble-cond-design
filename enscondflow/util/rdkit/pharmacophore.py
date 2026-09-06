import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures


RDKIT_SMARTS_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
SMARTS_PATH = os.path.join(os.path.dirname(__file__), "../../../defs/pharmacophore.fdef")


TArr = np.ndarray


@dataclass
class PharmacophoreFeature:
    type: int
    position: TArr
    atom_ids: tuple[int, ...]
    direction: TArr = None

    def __post_init__(self):
        if self.direction is None:
            self.direction = np.zeros(3)


class _PharmacophoreFinder:
    _feature_factory = ChemicalFeatures.BuildFeatureFactory(SMARTS_PATH)

    @staticmethod
    def set_default_features():
        factory = ChemicalFeatures.BuildFeatureFactory(SMARTS_PATH)
        _PharmacophoreFinder._feature_factory = factory

    @staticmethod
    def set_rdkit_features():
        factory = ChemicalFeatures.BuildFeatureFactory(RDKIT_SMARTS_PATH)
        _PharmacophoreFinder._feature_factory = factory

    @staticmethod
    def set_custom_features(path):
        factory = ChemicalFeatures.BuildFeatureFactory(path)
        _PharmacophoreFinder._feature_factory = factory

    @staticmethod
    def get_vocab_size() -> int:
        return len(_PharmacophoreFinder.get_feature_vocab())

    @staticmethod
    def get_feature_vocab() -> list[str]:
        vocab = _PharmacophoreFinder._feature_factory.GetFeatureFamilies()
        return list(vocab)

    @staticmethod
    def get_feature_index(feature: str) -> int:
        feat_idx_map = {feat: idx for idx, feat in enumerate(_PharmacophoreFinder.get_feature_vocab())}
        return feat_idx_map[feature]

    @staticmethod
    def get_feature_name(feat_idx: int) -> str:
        n_feats = _PharmacophoreFinder.get_vocab_size()
        if feat_idx >= n_feats:
            raise RuntimeError(f"Tried to access feature index {feat_idx} with only {n_feats} features.")

        features = _PharmacophoreFinder.get_feature_vocab()
        return features[feat_idx]

    @staticmethod
    def run_mol(mol: Chem.rdchem.Mol, conf_idx: Optional[int] = None) -> list[PharmacophoreFeature]:
        # Default conformer if None provided
        conf_idx = -1 if conf_idx is None else conf_idx

        feats = _PharmacophoreFinder._feature_factory.GetFeaturesForMol(mol, confId=conf_idx)

        results = []
        for feature in feats:
            pos = feature.GetPos()
            results.append(PharmacophoreFeature(
                type=_PharmacophoreFinder.get_feature_index(feature.GetFamily()),
                position=np.array([pos.x, pos.y, pos.z]),
                atom_ids=tuple(feature.GetAtomIds()),
            ))

        return results

    @staticmethod
    def build_heavy_to_full_map(mol_with_hs: Chem.rdchem.Mol) -> dict[int, int]:
        """Map heavy-atom indices (after removeHs) to full-mol indices (with Hs)."""
        mapping = {}
        heavy_idx = 0
        for i in range(mol_with_hs.GetNumAtoms()):
            if mol_with_hs.GetAtomWithIdx(i).GetAtomicNum() != 1:
                mapping[heavy_idx] = i
                heavy_idx += 1
        return mapping

    @staticmethod
    def expand_donor_directions(
        mol_with_hs: Chem.rdchem.Mol,
        heavy_to_full: dict[int, int],
        features: list[PharmacophoreFeature],
        conf_idx: int = -1,
    ) -> list[PharmacophoreFeature]:
        """Expand each donor feature into one feature per bonded H, with direction vectors.

        For donors: direction = normalised(H_pos - heavy_atom_pos) for each bonded H.
        For non-donors: pass through with zero direction.
        If a donor has no bonded Hs, keep original with zero direction.

        The input mol must have explicit Hs with 3D coordinates. Use the original embedded mol
        rather than Chem.AddHs(addCoords=True) which can produce poor H geometries.
        """

        if mol_with_hs.GetNumAtoms() == mol_with_hs.GetNumHeavyAtoms():
            raise ValueError(
                "expand_donor_directions requires a mol with explicit Hs. "
                "Pass the original embedded mol before removing Hs."
            )

        donor_type_idx = _PharmacophoreFinder.get_feature_index("Donor")
        conf = mol_with_hs.GetConformer(conf_idx)
        expanded = []

        for feat in features:
            if feat.type != donor_type_idx:
                expanded.append(feat)
                continue

            # Collect all bonded H atoms across all atoms in this feature
            h_atoms = []
            for heavy_idx in feat.atom_ids:
                full_idx = heavy_to_full.get(heavy_idx)
                if full_idx is None:
                    continue

                atom = mol_with_hs.GetAtomWithIdx(full_idx)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        h_atoms.append((full_idx, neighbor.GetIdx()))

            if not h_atoms:
                expanded.append(feat)
                continue

            for heavy_full_idx, h_full_idx in h_atoms:
                heavy_pos = np.array(conf.GetAtomPosition(heavy_full_idx))
                h_pos = np.array(conf.GetAtomPosition(h_full_idx))
                direction = h_pos - heavy_pos
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm

                expanded.append(PharmacophoreFeature(
                    type=feat.type,
                    position=feat.position.copy(),
                    atom_ids=feat.atom_ids,
                    direction=direction,
                ))

        return expanded

    @staticmethod
    def expand_aromatic_directions(
        mol: Chem.rdchem.Mol,
        features: list[PharmacophoreFeature],
        conf_idx: int = -1,
    ) -> list[PharmacophoreFeature]:
        """Set direction vectors for aromatic features to the ring normal.

        The ring normal is computed from the cross product of two in-plane vectors
        formed by the ring atom positions. Sign follows the right-hand rule with
        respect to the atom index ordering. Non-aromatic features pass through unchanged.
        """

        aromatic_type_idx = _PharmacophoreFinder.get_feature_index("Aromatic")
        conf = mol.GetConformer(conf_idx)
        expanded = []

        for feat in features:
            if feat.type != aromatic_type_idx or len(feat.atom_ids) < 3:
                expanded.append(feat)
                continue

            positions = np.array([list(conf.GetAtomPosition(idx)) for idx in feat.atom_ids])
            v1 = positions[1] - positions[0]
            v2 = positions[2] - positions[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                normal = normal / norm

            expanded.append(PharmacophoreFeature(
                type=feat.type,
                position=feat.position.copy(),
                atom_ids=feat.atom_ids,
                direction=normal,
            ))

        return expanded


# Make a single global object available

PharmacophoreFinder = _PharmacophoreFinder()
