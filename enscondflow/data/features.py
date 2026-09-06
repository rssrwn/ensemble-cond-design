import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors, AllChem, rdFreeSASA, rdqueries

from enscondflow.repr import GraphMol, ConfSet
from enscondflow.data.profile import ConfProfile
from enscondflow.repr.complex import BindingComplex
from enscondflow.util.rdkit import PharmacophoreFinder


TArr = np.ndarray


# *************************************************************************************************
# ******************************** Feature interface and wrappers *********************************
# *************************************************************************************************


class MolFeature(ABC):
    def __init__(self, name, raise_on_err=False, return_on_err=0.0):
        self.name = name
        self.raise_on_err = raise_on_err
        self.return_on_err = return_on_err

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def run(self, mol: GraphMol) -> Any:
        try:
            out = self._run(mol)
        except Exception as err:
            return self._handle_error(err, len(mol))

        return out

    def _handle_error(self, err, mol_size):
        print(f"Failed processing for feature '{self.name}': {type(err).__name__}: {err}")

        if self.raise_on_err:
            raise err

        return self.return_on_err

    @abstractmethod
    def _run(self, mol: GraphMol) -> Any:
        pass


class FeatureGroup:
    def __init__(self, *features: MolFeature):
        feat_names = [feat.name for feat in features]
        if len(set(feat_names)) != len(feat_names):
            raise ValueError("All MolFeatures must have unique names.")

        self.features = features

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def run(self, mol: GraphMol) -> dict[str, Any]:
        return {feat.name: feat(mol) for feat in self.features}


class MaskFeatureGroup:
    """Same as FeatureGroup but adds a mask for each feature to the dictionary.

    Mask value of 0 means the feature is dropped out, 1 for kept. The mask array will be directly added to the
    returned dictionary with feature stored under key '<feature.name>' and mask '<feature.name>-mask'.

    The masking is controlled with probability mask_prob.
    """

    def __init__(self, *features: MolFeature, mask_prob=0.0):
        feat_names = [feat.name for feat in features]
        if len(set(feat_names)) != len(feat_names):
            raise ValueError("All MolFeatures must have unique names.")

        if mask_prob < 0.0 or mask_prob > 1.0:
            raise ValueError("mask_prob must be between 0 and 1.")

        self.features = features
        self.mask_prob = mask_prob

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def run(self, mol: GraphMol) -> dict[str, Any]:
        features = {feat.name: feat(mol) for feat in self.features}
        masks = {f"{feat}-mask": self._mask_feat() for feat in features.keys()}
        out = {**features, **masks}
        return out

    def _mask_feat(self):
        return np.array(np.random.rand() > self.mask_prob, dtype=np.int64)


class SavedFeature(MolFeature):
    """A feature stored in each molecule's metadata.

    This will look for key <name> in the molecule's metadata dict and throw an error if it doesn't exist
    """

    def __init__(self, name):
        # Raise an error if the feature does not exist in the metadata
        super().__init__(name, raise_on_err=True)

    def _run(self, mol: GraphMol) -> float:
        return mol.meta[self.name]


# *************************************************************************************************
# ************************************** Sampled features *****************************************
# *************************************************************************************************


class GaussianFeature(MolFeature):
    """Sample a feature value from a gaussian distribution (useful for sampling)"""

    def __init__(self, name, mean, std_dev, max_val=None):
        # Raise on error since only problems should come from user inputs
        super().__init__(name, raise_on_err=True)

        if std_dev < 0.0:
            raise ValueError("std_dev of Gaussian must be non-negative.")

        self.mean = mean
        self.std_dev = std_dev
        self.max_val = max_val

    def _run(self, mol: GraphMol) -> TArr:
        sample = np.random.normal(self.mean, self.std_dev)
        if self.max_val is not None:
            sample = min(sample, self.max_val)

        return np.array(sample)


class UniformFeature(MolFeature):
    """Sample a feature value from a uniform distribution (useful for sampling)"""

    def __init__(self, name, low, high):
        # Raise on error since only problems should come from user inputs
        super().__init__(name, raise_on_err=True)

        self.low = low
        self.high = high

    def _run(self, mol: GraphMol) -> TArr:
        return np.array(np.random.uniform(self.low, self.high))


# *************************************************************************************************
# *********************************** Basic molecular properties **********************************
# *************************************************************************************************


class ScaledFeature(MolFeature):
    """Scale an existing MolFeature to be between 0 and 1 with precomputed scale factors

    For a raw, unscaled value, x, this will compute: (x - min_val) / (max_val - min_val)
    """

    def __init__(self, feature, min_val, max_val, return_on_err=-1.0):
        super().__init__(f"scaled-{feature.name}", raise_on_err=feature.raise_on_err, return_on_err=return_on_err)

        self.feature = feature
        self.min_val = min_val
        self.max_val = max_val

    def _run(self, mol: GraphMol) -> float:
        feat_val = self.feature._run(mol)
        return (feat_val - self.min_val) / (self.max_val - self.min_val)


class StandardisedFeature(MolFeature):
    """Scale an existing MolFeature by subtracting a mean and dividing by a standard deviation"""

    def __init__(self, feature, mean, std, return_on_err=-10.0):
        super().__init__(
            f"standardised-{feature.name}",
            raise_on_err=feature.raise_on_err,
            return_on_err=return_on_err
        )

        self.feature = feature
        self.mean = mean
        self.std = std

    def _run(self, mol: GraphMol) -> float:
        # Call inner _run directly so any exception propagates up and hits this wrapper's
        # run()/_handle_error — otherwise the inner's return_on_err fallback is silently
        # standardised (e.g. raw 0 → −2.1 for PSA3D) and the wrapper's OOD sentinel never fires.
        feat_val = self.feature._run(mol)
        return (feat_val - self.mean) / self.std


# *************************************************************************************************
# *********************************** Basic molecular properties **********************************
# *************************************************************************************************


class SavedPharmacophores(MolFeature):
    """Precompute pharmacophore topology (types, atom_ids, direction metadata) once per molecule.

    Stores only the information needed to compute coords and directions from numpy arrays
    at runtime, avoiding the expensive per-conformer loop over RDKit conformers during preload.

    direction_meta is a list with one entry per expanded feature:
        ("none",)                              — zero direction vector
        ("donor", heavy_full_idx, h_full_idx)  — direction = normalize(h_pos - heavy_pos), with-Hs indices
        ("aromatic", idx0, idx1, idx2)         — direction = normalize(cross(v1, v2)), ring normal from 3 atom positions
    """

    def __init__(self):
        super().__init__(Pharmacophores.CACHE_KEY, raise_on_err=False)

    def _run(self, mol: GraphMol):
        mol_with_hs = mol.to_rdkit(sanitise=True)
        heavy_to_full = PharmacophoreFinder.build_heavy_to_full_map(mol_with_hs)

        mol_no_hs = mol.remove_hs()
        rdkit_mol = mol_no_hs.to_rdkit(sanitise=True)
        pharm_feats = PharmacophoreFinder.run_mol(rdkit_mol, conf_idx=0)

        if len(pharm_feats) == 0:
            return {"types": np.array([]), "atom_ids": [], "direction_meta": []}

        donor_idx = PharmacophoreFinder.get_feature_index("Donor")
        aromatic_idx = PharmacophoreFinder.get_feature_index("Aromatic")

        expanded_types = []
        expanded_atom_ids = []
        direction_meta = []

        for feat in pharm_feats:
            if feat.type == donor_idx:
                h_atoms = []
                for heavy_idx in feat.atom_ids:
                    full_idx = heavy_to_full.get(heavy_idx)
                    if full_idx is None:
                        continue
                    atom = mol_with_hs.GetAtomWithIdx(full_idx)
                    for nbr in atom.GetNeighbors():
                        if nbr.GetAtomicNum() == 1:
                            h_atoms.append((full_idx, nbr.GetIdx()))

                if not h_atoms:
                    expanded_types.append(feat.type)
                    expanded_atom_ids.append(feat.atom_ids)
                    direction_meta.append(("none",))
                else:
                    for heavy_full, h_full in h_atoms:
                        expanded_types.append(feat.type)
                        expanded_atom_ids.append(feat.atom_ids)
                        direction_meta.append(("donor", heavy_full, h_full))

            elif feat.type == aromatic_idx and len(feat.atom_ids) >= 3:
                ring_full_ids = tuple(heavy_to_full[i] for i in feat.atom_ids[:3] if i in heavy_to_full)
                if len(ring_full_ids) >= 3:
                    expanded_types.append(feat.type)
                    expanded_atom_ids.append(feat.atom_ids)
                    direction_meta.append(("aromatic", *ring_full_ids[:3]))
                else:
                    expanded_types.append(feat.type)
                    expanded_atom_ids.append(feat.atom_ids)
                    direction_meta.append(("none",))

            else:
                expanded_types.append(feat.type)
                expanded_atom_ids.append(feat.atom_ids)
                direction_meta.append(("none",))

        return {
            "types": np.array(expanded_types),
            "atom_ids": expanded_atom_ids,
            "direction_meta": direction_meta,
        }

    def _handle_error(self, err, mol_size):
        return {"types": np.array([]), "atom_ids": [], "direction_meta": []}


class Qed(MolFeature):
    """QED from RDKit"""

    def __init__(self, name="qed", raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)

    def _run(self, mol: GraphMol) -> float:
        rdkit_mol = mol.to_rdkit(sanitise=True)
        return QED.default(rdkit_mol)


class LogP(MolFeature):
    """LogP from RDKit Crippen implementation"""

    def __init__(self, name="logp", raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)

    def _run(self, mol: GraphMol) -> float:
        rdkit_mol = mol.to_rdkit(sanitise=True)
        return Crippen.MolLogP(rdkit_mol)


class AromaticRings(MolFeature):
    """Number of aromatic rings in mol"""

    def __init__(self, name="n-aromatic-rings", raise_on_err=False, return_on_err=-1):
        super().__init__(name, raise_on_err, return_on_err)

    def _run(self, mol: GraphMol) -> int:
        rdkit_mol = mol.to_rdkit(sanitise=True)
        n_rings = rdMolDescriptors.CalcNumAromaticRings(rdkit_mol)
        return n_rings


class HeavyAtoms(MolFeature):
    """Number of heavy atoms in mol"""

    def __init__(self, name="n-heavy-atoms", raise_on_err=False, return_on_err=-1):
        super().__init__(name, raise_on_err, return_on_err)

    def _run(self, mol: GraphMol) -> int:
        rdkit_mol = mol.to_rdkit(sanitise=True)
        n_heavy = rdMolDescriptors.CalcNumHeavyAtoms(rdkit_mol)
        return n_heavy


class Weight(MolFeature):
    """Mol weight from RDKit CalcExactMolWt function (as a float)"""

    def __init__(self, name="mol-weight", raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)

    def _run(self, mol: GraphMol) -> float:
        rdkit_mol = mol.to_rdkit(sanitise=True)
        mol_weight = rdMolDescriptors.CalcExactMolWt(rdkit_mol)
        return mol_weight


class Tpsa(MolFeature):
    """Topological Polar Surface Area from RDKit"""

    def __init__(self, name="tpsa", raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)

    def _run(self, mol: GraphMol) -> float:
        rdkit_mol = mol.to_rdkit(sanitise=True)
        tpsa = rdMolDescriptors.CalcTPSA(rdkit_mol)
        return tpsa


# *************************************************************************************************
# ************************************* Ensemble properties ***************************************
# *************************************************************************************************


class RMSF(MolFeature):
    """Root mean square fluctuation across an ensemble of conformers.

    For each reference conformer choice, aligns all conformers to the reference (after centering),
    computes per-atom RMSF, and averages across atoms. The final scalar is the mean over n_refs
    different reference choices — the single-pass Kabsch-to-ref scheme is not invariant to the
    reference choice, so averaging reduces that variation. n_refs=1 picks the first conformer
    (matching the original behaviour); n_refs>1 picks the first n_refs conformers as references.
    """

    def __init__(self, name="rmsf", n_refs=1, raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)

        if n_refs < 1:
            raise ValueError(f"n_refs must be >= 1, got {n_refs}")

        self.n_refs = n_refs

    def _run(self, mol: GraphMol) -> float:
        n_confs = 0 if mol.confs is None else len(mol.confs)
        assert n_confs >= 2, f"RMSF requires >= 2 conformers, got {n_confs}"

        # Centre each conformer on its own CoM
        coords = mol.confs.coords.copy()  # [n_confs, n_atoms, 3]
        coords = coords - coords.mean(axis=1, keepdims=True)

        n_refs = min(self.n_refs, n_confs)
        rmsfs = [self._rmsf_for_ref(coords, ref_idx) for ref_idx in range(n_refs)]
        return float(np.mean(rmsfs))

    @staticmethod
    def _rmsf_for_ref(coords: TArr, ref_idx: int) -> float:
        ref = coords[ref_idx]

        aligned = []
        for i in range(len(coords)):
            if i == ref_idx:
                aligned.append(ref)
            else:
                rot, _ = Rotation.align_vectors(ref, coords[i])
                aligned.append(rot.apply(coords[i]))

        # Aligned shape [n_confs, n_atoms, 3]
        aligned = np.stack(aligned)

        # Per-atom RMSF: sqrt(mean_over_confs(||pos - mean_pos||^2))
        mean_struct = aligned.mean(axis=0)
        sq_dists = ((aligned - mean_struct[None]) ** 2).sum(axis=2)
        per_atom_rmsf = np.sqrt(sq_dists.mean(axis=0))

        return per_atom_rmsf.mean().item()


class MeanPairwiseRMSD(MolFeature):
    """Mean pairwise RMSD across an ensemble of conformers after Kabsch alignment.

    Captures conformational diversity without depending on a single reference conformer
    (unlike RMSF). For ensembles with more than max_pairs conformer pairs, a random subset
    of upper-triangle pairs is sampled to bound runtime.
    """

    def __init__(self, name="mean-pairwise-rmsd", max_pairs=200, raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)
        self.max_pairs = max_pairs

    def _run(self, mol: GraphMol) -> float:
        n_confs = 0 if mol.confs is None else len(mol.confs)
        assert n_confs >= 2, f"MeanPairwiseRMSD requires >= 2 conformers, got {n_confs}"

        coords = mol.confs.coords - mol.confs.coords.mean(axis=1, keepdims=True)

        idx_i, idx_j = np.triu_indices(n_confs, k=1)
        n_pairs = len(idx_i)
        if n_pairs > self.max_pairs:
            sel = np.random.choice(n_pairs, size=self.max_pairs, replace=False)
            idx_i, idx_j = idx_i[sel], idx_j[sel]

        rmsds = []
        for i, j in zip(idx_i, idx_j):
            rot, _ = Rotation.align_vectors(coords[i], coords[j])
            aligned = rot.apply(coords[j])
            rmsd = np.sqrt(((aligned - coords[i]) ** 2).sum(axis=1).mean())
            rmsds.append(rmsd)

        return float(np.mean(rmsds))


class PSA3D(MolFeature):
    """Mean 3D Polar Surface Area across an ensemble of conformers.

    For each conformer, computes SASA via rdFreeSASA restricted to polar atoms
    (N, O, and H atoms bonded to N or O), then averages across all conformers.
    """

    def __init__(self, name="psa3d", raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)

    def _run(self, mol: GraphMol) -> float:
        n_confs = 0 if mol.confs is None else len(mol.confs)
        assert n_confs >= 2, f"PSA3D requires >= 2 conformers, got {n_confs}"

        rdkit_mol = mol.to_rdkit(sanitise=True)

        # Tag polar atoms: N (7), O (8), and H atoms bonded to N or O
        has_polar = False
        for atom in rdkit_mol.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum in (7, 8):
                atom.SetBoolProp("_polar", True)
                has_polar = True
            elif anum == 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() in (7, 8):
                        atom.SetBoolProp("_polar", True)
                        has_polar = True
                        break

        if not has_polar:
            return 0.0

        radii = [Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in rdkit_mol.GetAtoms()]
        polar_query = rdqueries.HasPropQueryAtom("_polar")
        psa_values = []

        for conf_idx in range(mol.n_conformers):
            polar_sasa = rdFreeSASA.CalcSASA(rdkit_mol, radii, confIdx=conf_idx, query=polar_query)
            psa_values.append(polar_sasa)

        return float(np.mean(psa_values))


class ConfEntropy(MolFeature):
    """Configurational entropy of an ensemble of molecule conformers"""

    def __init__(self, name="conf-entropy", raise_on_err=False, return_on_err=-1.0, eps=1e-10):
        super().__init__(name, raise_on_err, return_on_err)
        self.eps = eps

    def _run(self, mol: GraphMol) -> float:
        assert mol.confs.has_weights

        weights = mol.confs.weights.copy()
        weight_mask = weights >= self.eps
        weights = weights[weight_mask]

        if len(weights) == 0:
            raise RuntimeError(f"All molecular conformers had weight less than epsilon.")

        neg_entropy = (weights * np.log(weights)).sum()
        entropy = (- neg_entropy).item()
        return entropy


# *************************************************************************************************
# ************************************* Conformer properties **************************************
# *************************************************************************************************


class Shape(MolFeature):
    def __init__(self, name="shape", conf_idx=0, raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)

        self.conf_idx = conf_idx

    def _run(self, mol: GraphMol) -> TArr:
        mol = mol.remove_hs()
        coords = mol.get_conformer(self.conf_idx)
        return coords

    def _handle_error(self, err, mol_size):
        print(f"Failed processing for feature '{self.name}'")

        if self.raise_on_err:
            raise err

        if not isinstance(self.return_on_err, TArr):
            return np.array([self.return_on_err] * mol_size)

        return self.return_on_err


class Electrostatics(MolFeature):
    def __init__(self, name="electrostatics", conf_idx=0, raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)

        self.conf_idx = conf_idx

    def _run(self, mol: GraphMol) -> TArr:
        mol = mol.remove_hs()

        if self.conf_idx != 0 or mol.n_conformers != 1:
            conf_arr = mol.confs.get_conformer(self.conf_idx)
            mol = mol.copy_with(confs=ConfSet(conf_arr))

        rdkit_mol = mol.to_rdkit(sanitise=True)
        mmff_props = AllChem.MMFFGetMoleculeProperties(rdkit_mol)
        p_charges = [mmff_props.GetMMFFPartialCharge(i) for i in range(rdkit_mol.GetNumAtoms())]
        return np.array(p_charges)

    def _handle_error(self, err, mol_size):
        # This one does fail occassionally, but let's not spam the terminal with messages
        # print(f"Failed processing for feature '{self.name}'")

        if self.raise_on_err:
            raise err

        if not isinstance(self.return_on_err, TArr):
            return np.array([self.return_on_err] * mol_size)

        return self.return_on_err


class Pharmacophores(MolFeature):
    CACHE_KEY = "_pharmacophores"

    def __init__(self, name="pharmacophores", conf_idx=0, raise_on_err=False):
        super().__init__(name, raise_on_err)

        self.conf_idx = conf_idx

    def _run(self, mol: GraphMol) -> tuple[TArr, TArr, list[tuple[int, ...]], TArr]:
        # Full cache (coords + directions already computed in data_transform)
        if self.CACHE_KEY in mol.meta and "coords" in mol.meta[self.CACHE_KEY]:
            cached = mol.meta[self.CACHE_KEY]
            return cached["types"], cached["coords"], cached["atom_ids"], cached["directions"]

        # No cache — compute from scratch (mol must have Hs)
        mol_with_hs = mol.to_rdkit(sanitise=True)
        heavy_to_full = PharmacophoreFinder.build_heavy_to_full_map(mol_with_hs)

        mol = mol.remove_hs()
        rdkit_mol = mol.to_rdkit(sanitise=True)

        pharm_feats = PharmacophoreFinder.run_mol(rdkit_mol, conf_idx=self.conf_idx)
        pharm_feats = PharmacophoreFinder.expand_donor_directions(
            mol_with_hs,
            heavy_to_full,
            pharm_feats,
            conf_idx=self.conf_idx
        )
        pharm_feats = PharmacophoreFinder.expand_aromatic_directions(rdkit_mol, pharm_feats, conf_idx=self.conf_idx)

        types = np.array([f.type for f in pharm_feats])
        coords = np.array([f.position for f in pharm_feats]) if pharm_feats else np.zeros((0, 3))
        atom_ids = [f.atom_ids for f in pharm_feats]
        directions = np.array([f.direction for f in pharm_feats]) if pharm_feats else np.zeros((0, 3))

        return types, coords, atom_ids, directions

    def _handle_error(self, err, mol_size):
        print(f"Failed processing for feature '{self.name}'")

        if self.raise_on_err:
            raise err

        # If there is an error return empty arrays with correct shape
        types = np.array([])
        coords = np.zeros((0, 3))

        return types, coords, [], np.zeros((0, 3))


class RadiusOfGyration(MolFeature):
    """Radius of gyration for a single conformer, computed from all atom positions"""

    def __init__(self, name="radius-of-gyration", conf_idx=0, raise_on_err=False, return_on_err=0.0):
        super().__init__(name, raise_on_err, return_on_err)
        self.conf_idx = conf_idx

    def _run(self, mol: GraphMol) -> float:
        coords = mol.get_conformer(self.conf_idx)
        center = coords.mean(axis=0)
        rg = np.sqrt(((coords - center) ** 2).sum(axis=1).mean())
        return rg.item()


class IntramolecularHBonds(MolFeature):
    """Count intramolecular hydrogen bonds for a single conformer.

    Uses the same donor/acceptor SMARTS and geometric criteria as Prolif (Bouysset & Fiorucci, J. Cheminform. 2021):
        - Donor-Acceptor distance <= 3.5 A
        - D-H...A angle >= 130 degrees
        - Donor/acceptor must be >= 4 bonds apart (excludes covalent neighbors)

    Donor and acceptor SMARTS are taken from Prolif, which adapted them from Pharmit (Koes & Camacho)
    and RDKit's BaseFeatures.fdef.
    """

    _DONOR_SMARTS = Chem.MolFromSmarts("[$([O,S,#7;+0]),$([Nv4+1]),$([n+]c[nH])]-[H]")
    _ACCEPTOR_SMARTS = Chem.MolFromSmarts(
        "[$([N&!$([NX3]-*=[O,N,P,S])&!$([NX3]-[a])&!$([Nv4+1])&!$(N=C(-[C,N])-N)]),"
        "$([n+0&!X3&!$([n&r5]:[n+&r5])]),"
        "$([O&!$([OX2](C)C=O)&!$(O(~a)~a)&!$(O=N-*)&!$([O-]-N=O)]),"
        "$([o+0]),"
        "$([F&$(F-[#6])&!$(F-[#6][F,Cl,Br,I])])]"
    )

    DA_DIST_MAX = 3.5
    DHA_ANGLE_MIN = 130.0
    MIN_BOND_DIST = 4

    def __init__(self, name="intramol-hbonds", conf_idx=0, raise_on_err=False, return_on_err=0):
        super().__init__(name, raise_on_err, return_on_err)
        self.conf_idx = conf_idx

    def _run(self, mol: GraphMol) -> int:
        rdkit_mol = mol.to_rdkit(sanitise=True)
        conf = rdkit_mol.GetConformer(self.conf_idx)

        donors = rdkit_mol.GetSubstructMatches(self._DONOR_SMARTS)
        acceptors = rdkit_mol.GetSubstructMatches(self._ACCEPTOR_SMARTS)

        if not donors or not acceptors:
            return 0

        topo_dist = Chem.GetDistanceMatrix(rdkit_mol)
        acceptor_idxs = [match[0] for match in acceptors]

        count = 0
        for donor_heavy_idx, h_idx in donors:
            donor_pos = np.array(conf.GetAtomPosition(donor_heavy_idx))
            h_pos = np.array(conf.GetAtomPosition(h_idx))

            for acc_idx in acceptor_idxs:
                if acc_idx == donor_heavy_idx:
                    continue

                if topo_dist[donor_heavy_idx, acc_idx] < self.MIN_BOND_DIST:
                    continue

                acc_pos = np.array(conf.GetAtomPosition(acc_idx))

                da_dist = np.linalg.norm(donor_pos - acc_pos)
                if da_dist > self.DA_DIST_MAX:
                    continue

                # D-H...A angle: vectors from H to donor and H to acceptor
                hd = donor_pos - h_pos
                ha = acc_pos - h_pos
                cos_angle = np.dot(hd, ha) / (np.linalg.norm(hd) * np.linalg.norm(ha) + 1e-10)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                if angle >= self.DHA_ANGLE_MIN:
                    count += 1

        return count


# *************************************************************************************************
# **************************** Interaction/binding profile features *******************************
# *************************************************************************************************


class InteractionProfile(MolFeature):
    def __init__(self, name="profile", include_pharms=True, conf_idx=0, raise_on_err=False):
        super().__init__(name, raise_on_err=raise_on_err)

        self.include_pharms = include_pharms

        self.shape = Shape(conf_idx=conf_idx, raise_on_err=raise_on_err)
        self.pharm = Pharmacophores(conf_idx=conf_idx, raise_on_err=raise_on_err)

    def _run(self, mol: GraphMol) -> ConfProfile:
        shape_pos = self.shape(mol)
        shape_atom_ids = [(i,) for i in range(len(shape_pos))]

        if self.include_pharms:
            types, pharm_pos, pharm_atom_ids, pharm_directions = self.pharm(mol)
        else:
            types = np.array([])
            pharm_pos = np.zeros((0, 3))
            pharm_atom_ids = []
            pharm_directions = np.zeros((0, 3))

        positions = np.concat((shape_pos, pharm_pos), axis=0)
        atom_ids = shape_atom_ids + pharm_atom_ids

        shape_directions = np.zeros_like(shape_pos)
        directions = np.concat((shape_directions, pharm_directions), axis=0) if len(pharm_directions) > 0 else shape_directions

        # Type indices will refer to the following:
        # 0 - Left aside for padding
        # 1 - Shape point
        # 2..N - Pharmacophore types ordered according to the result from PharmacophoreFinder
        shape_types = np.ones(len(shape_pos))
        types = np.concat((shape_types, types + 2), axis=0)

        assert len(types) == len(positions)
        assert len(atom_ids) == len(types)

        profile = ConfProfile(positions, types, atom_ids, directions=directions)
        return profile


class BindingProfile(MolFeature):
    """A ConfProfile from a protein-ligand binding complex.

    The complex must have the interactions already extracted. The types correspond to the index into
    PharmacophoreFinder. Interactions whose pharmacophore name is not in the current PharmacophoreFinder
    vocab are silently skipped.

    Each binding interaction is mapped to the closest PharmacophoreFinder feature of the same type,
    ensuring positions use centroid representation consistent with training data. Interactions that
    don't match any detected pharmacophore feature are skipped.

    If the profile has been precomputed and stored in system.ligand.meta[CACHE_KEY], returns
    the cached profile directly without recomputing.
    """

    CACHE_KEY = "_binding_profile"

    # Map from prolif interaction types to pharmacophore feature names
    # Single-atom: one pharmacophore point per ligand atom
    SINGLE_ATOM_TYPE_MAP = {
        "HBDonor": "Donor",
        "HBAcceptor": "Acceptor",
        "Cationic": "Cation",
        "Anionic": "Anion",
        "Hydrophobic": "Hydrophobe",
    }

    # Pi interactions: ring center or single atom depending on geometry
    PI_TYPE_MAP = {
        "PiStacking": "Aromatic",
        "PiCation": "Aromatic",
        "CationPi": "Cation",
    }

    def __init__(self, name="binding-profile", raise_on_err=False):
        super().__init__(name, raise_on_err)

        self.profile = InteractionProfile(include_pharms=False, raise_on_err=raise_on_err)

    def _run(self, system: BindingComplex) -> ConfProfile:
        cached = system.ligand.meta.get(self.CACHE_KEY)
        if cached is not None:
            return cached

        if system.interactions is None:
            raise ValueError("BindingComplex must have interactions extracted.")

        # Get shape points from interaction profile
        profile = self.profile(system.ligand)

        # Extract pharmacophore positions and types from xtal interactions
        pharm_types, pharm_pos, pharm_atom_ids, pharm_directions = self._extract_interaction_pharmacophores(system)

        # Combine shape and pharmacophore data
        positions = np.concat((profile.positions, pharm_pos), axis=0)
        atom_ids = profile.atom_ids + pharm_atom_ids

        shape_directions = np.zeros_like(profile.positions)
        directions = np.concat((shape_directions, pharm_directions), axis=0) if len(pharm_directions) > 0 else shape_directions

        # Type indices:
        # 0 - Padding
        # 1 - Shape point
        # 2..N - Pharmacophore types from PharmacophoreFinder
        shape_types = np.ones(len(profile.positions))
        types = np.concat((shape_types, pharm_types + 2), axis=0) if len(pharm_types) > 0 else shape_types

        profile = ConfProfile(positions, types, atom_ids=atom_ids, directions=directions)
        return profile

    def _extract_interaction_pharmacophores(
        self, system: BindingComplex
    ) -> tuple[TArr, TArr, list[tuple[int, ...]], TArr]:
        """Extract pharmacophore types, positions, atom_ids and directions from binding interactions.

        Each interaction is mapped to a PharmacophoreFinder feature on the reference ligand by matching
        pharmacophore type and overlapping atom_ids. This ensures positions are centroids consistent with
        training data. Interactions that don't match any detected feature are skipped.

        Returns:
            Tuple of (types, positions, atom_ids, directions) for pharmacophores derived from interactions.
        """

        # Build mapping from full-ligand atom indices to heavy-atom-only indices
        full_rdkit = system.ligand.to_rdkit(sanitise=True)

        heavy_atom_map = {}
        heavy_idx = 0
        for i in range(full_rdkit.GetNumAtoms()):
            if full_rdkit.GetAtomWithIdx(i).GetAtomicNum() != 1:
                heavy_atom_map[i] = heavy_idx
                heavy_idx += 1

        # Work in heavy-atom space from here on
        ligand_no_hs = system.ligand.remove_hs()
        rdkit_mol = ligand_no_hs.to_rdkit(sanitise=True)

        # Run PharmacophoreFinder to get centroid-based features
        detected_feats = PharmacophoreFinder.run_mol(rdkit_mol)

        # Expand donors with direction vectors using mol with Hs
        heavy_to_full = PharmacophoreFinder.build_heavy_to_full_map(full_rdkit)
        detected_feats = PharmacophoreFinder.expand_donor_directions(full_rdkit, heavy_to_full, detected_feats)

        features_by_type = {}
        for feat in detected_feats:
            features_by_type.setdefault(feat.type, []).append(feat)

        seen = set()
        points = []
        for interaction in system.interactions._interactions:
            int_type = interaction.interaction_type

            # Remap to heavy-atom indices, dropping any H-only atoms
            ligand_atoms = tuple(heavy_atom_map[a] for a in interaction.ligand_atoms if a in heavy_atom_map)
            if not ligand_atoms:
                continue

            # Determine pharmacophore type from interaction type
            if int_type in self.SINGLE_ATOM_TYPE_MAP:
                pharm_name = self.SINGLE_ATOM_TYPE_MAP[int_type]
            elif int_type in self.PI_TYPE_MAP:
                pharm_name = self.PI_TYPE_MAP[int_type]
            else:
                continue

            if pharm_name not in PharmacophoreFinder.get_feature_vocab():
                continue

            pharm_idx = PharmacophoreFinder.get_feature_index(pharm_name)

            # Find the PharmacophoreFinder feature matching this interaction
            matched = self._find_matching_feature(pharm_idx, ligand_atoms, features_by_type)
            if matched is None:
                continue

            # Use PharmacophoreFinder's centroid position and atom_ids
            dir_tuple = tuple(matched.direction.tolist())
            key = (pharm_idx, matched.atom_ids, dir_tuple)
            if key not in seen:
                seen.add(key)
                points.append((pharm_idx, matched.position, matched.atom_ids, matched.direction))

        if not points:
            return np.array([]), np.zeros((0, 3)), [], np.zeros((0, 3))

        types, positions, atom_ids, directions = zip(*points)
        return np.array(types), np.array(positions), list(atom_ids), np.array(directions)

    @staticmethod
    def _find_matching_feature(pharm_idx, interaction_atoms, features_by_type):
        """Find the PharmacophoreFinder feature that best matches a binding interaction.

        Match criteria: same pharmacophore type, at least one overlapping atom_id.
        If multiple features match, pick the one with the most atom overlap.
        """

        candidates = features_by_type.get(pharm_idx, [])
        best_feat = None
        best_overlap = 0

        interaction_set = set(interaction_atoms)
        for feat in candidates:
            overlap = len(interaction_set & set(feat.atom_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                best_feat = feat

        return best_feat


class StochasticProfile(MolFeature):
    """Add stochasicity to interaction or binding profiles.

    Adding noise is often useful for training or for simulating a binding profile given all possible pharmacophores
    that a molecule could form.

    This supports various forms of stochasicity:
        - Adding noise to positions (gaussian noise controlled with pos_std_dev)
        - Resampling shape points (shape_resample refers to prob of each point being resampled)
        - Random rotations of 3D points (rotate_prob)
        - Profile mode dropout via independent coin flips for shape and pharmacophore components:
            - global_pharm_dropout: probability of dropping all pharmacophores (shape-only, mode 1)
            - global_shape_dropout: probability of dropping all shape points (pharma-only, mode 2)
            - If both fire simultaneously, neither is dropped (both, mode 0). This keeps the model
              from ever seeing an empty profile and biases towards the full-info mode when both
              dropout probs are high.
            - local_pharm_dropout: probability of dropping each pharmacophore point independently
    """

    def __init__(
        self,
        profile_feat,
        pos_std_dev=0.0,
        rotate_prob=0.0,
        shape_resample=0.0,
        local_pharm_dropout=0.0,
        global_pharm_dropout=0.0,
        global_shape_dropout=0.0,
        raise_on_err=False
    ):
        if not (isinstance(profile_feat, InteractionProfile) or isinstance(profile_feat, BindingProfile)):
            raise TypeError("profile_feat must be either an InteractionProfile or a BindingProfile feature.")

        super().__init__(profile_feat.name, raise_on_err=raise_on_err)

        if shape_resample < 0.0 or shape_resample > 1.0:
            raise ValueError("shape_resample must be between 0 and 1.")
        if rotate_prob < 0.0 or rotate_prob > 1.0:
            raise ValueError("rotate_prob must be between 0 and 1.")

        self.profile_feat = profile_feat
        self.pos_std_dev = pos_std_dev
        self.rotate_prob = rotate_prob
        self.shape_resample = shape_resample
        self.local_pharm_dropout = local_pharm_dropout
        self.global_pharm_dropout = global_pharm_dropout
        self.global_shape_dropout = global_shape_dropout

    def _run(self, mol) -> ConfProfile:
        profile = self.profile_feat(mol)
        if not isinstance(profile, ConfProfile):
            raise RuntimeError("Inner profile feature failed, see preceding error.")

        # Separate shape points from pharmacophore points
        shape_mask = profile.types == 1
        pharm_mask = profile.types >= 2

        shape_pos = profile.positions[shape_mask]
        shape_directions = profile.directions[shape_mask]
        shape_atom_ids = [aid for aid, m in zip(profile.atom_ids, shape_mask) if m]

        pharm_pos = profile.positions[pharm_mask]
        pharm_types = profile.types[pharm_mask]
        pharm_directions = profile.directions[pharm_mask]
        pharm_atom_ids = [aid for aid, m in zip(profile.atom_ids, pharm_mask) if m]

        # Shape resampling — duplicate each shape point with probability shape_resample
        if self.shape_resample > 0.0:
            resample_mask = np.random.rand(len(shape_pos)) < self.shape_resample
            shape_pos = np.concat((shape_pos, shape_pos[resample_mask]), axis=0)
            shape_directions = np.concat((shape_directions, shape_directions[resample_mask]), axis=0)
            shape_atom_ids = shape_atom_ids + [shape_atom_ids[i] for i, m in enumerate(resample_mask) if m]

        # If both are dropped, revert to mode 0 (both) so the model never sees an empty profile
        drop_pharm = self.global_pharm_dropout > 0 and np.random.rand() < self.global_pharm_dropout
        drop_shape = self.global_shape_dropout > 0 and np.random.rand() < self.global_shape_dropout

        if drop_pharm and not drop_shape:
            profile_mode = 1
        elif drop_shape and not drop_pharm:
            profile_mode = 2
        else:
            profile_mode = 0

        # Keep shape points only
        if profile_mode == 1:
            pharm_pos = np.zeros((0, 3))
            pharm_types = np.array([])
            pharm_directions = np.zeros((0, 3))
            pharm_atom_ids = []

        # Keep pharmacophores only
        elif profile_mode == 2:
            shape_pos = np.zeros((0, 3))
            shape_directions = np.zeros((0, 3))
            shape_atom_ids = []

        # Local pharmacophore dropout — drop each pharmacophore independently (when mode is 0 or 2)
        if profile_mode != 1 and self.local_pharm_dropout > 0.0 and len(pharm_pos) > 0:
            keep_mask = np.random.rand(len(pharm_pos)) >= self.local_pharm_dropout
            pharm_pos = pharm_pos[keep_mask]
            pharm_types = pharm_types[keep_mask]
            pharm_directions = pharm_directions[keep_mask]
            pharm_atom_ids = [aid for aid, m in zip(pharm_atom_ids, keep_mask) if m]

        # Add Gaussian noise to shape positions only
        if isinstance(self.pos_std_dev, tuple):
            pos_noise_std = np.random.uniform(self.pos_std_dev[0], self.pos_std_dev[1])
        else:
            pos_noise_std = float(self.pos_std_dev)

        shape_pos = np.random.normal(shape_pos, pos_noise_std)

        # Recombine shape and pharmacophore points
        shape_types = np.ones(len(shape_pos))
        types = np.concat((shape_types, pharm_types), axis=0) if len(pharm_types) > 0 else shape_types
        positions = np.concat((shape_pos, pharm_pos), axis=0)
        directions = np.concat((shape_directions, pharm_directions), axis=0) if len(pharm_directions) > 0 else shape_directions
        atom_ids = shape_atom_ids + pharm_atom_ids

        # Apply random 3D rotation to both positions and directions
        if self.rotate_prob > 0.0:
            rotated = np.random.rand() < self.rotate_prob
            if rotated:
                rotation = Rotation.random()
                positions = rotation.apply(positions)
                if len(directions) > 0:
                    directions = rotation.apply(directions)
        else:
            rotated = False

        profile = ConfProfile(
            positions,
            types,
            atom_ids,
            directions=directions,
            rotated=rotated,
            profile_mode=profile_mode,
            pos_noise_std=pos_noise_std,
        )
        return profile
