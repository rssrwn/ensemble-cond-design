import os
import math
import torch
import numpy as np
from collections import namedtuple
from functools import partial
from torchmetrics import Metric
from posebusters import PoseBusters
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs, QED, Crippen

import enscondflow.util.rdkit as smolRD
import enscondflow.util.geometry as Geom
import enscondflow.data.features as Features
from enscondflow.repr import GraphMol
from enscondflow.util.rdkit.pharmacophore import PharmacophoreFinder


HARTREE_TO_KCAL = 627.509


AlignedConf = namedtuple("AlignedConf", ["aligned", "shape_tani", "colour_tani", "ref_profile", "energy", "strain"])


# *****************************************************************************
# ************************** Some helper functions ****************************
# *****************************************************************************


def _is_valid_valence(valence, allowed, charge):
    if isinstance(allowed, int):
        valid = allowed == valence

    elif isinstance(allowed, list):
        valid = valence in allowed

    elif isinstance(allowed, dict):
        allowed = allowed.get(charge)
        if allowed is None:
            return False

        valid = _is_valid_valence(valence, allowed, charge)

    return valid


def _is_valid_float(num):
    return num not in [None, float("inf"), float("-inf"), float("nan")]


def _try_remove_hs(mol):
    if mol is None:
        return None

    try:
        mol_no_hs = Chem.RemoveAllHs(mol)
    except:
        return None

    if mol_no_hs is None:
        return None

    # Ensure a copy is created
    return Chem.Mol(mol_no_hs)


def _entropy(weights, eps=1e-8):
    weight_mask = weights >= eps
    weights = weights[weight_mask]

    if len(weights) == 0:
        print("All molecular conformers had weight less than epsilon.")
        return None

    neg_entropy = (weights * np.log(weights)).sum()
    entropy = (- neg_entropy).item()
    return entropy


def _score_single(mol, ref, profile, align, align_weight, optimise, max_opt_iters):
    """Score a single mol-ref pair, optionally optimising and/or aligning first. Returns AlignedConf or None"""

    if mol is None or ref is None:
        return None
    if mol.GetNumAtoms() < 4 or ref.GetNumAtoms() < 4:
        return None
    if mol.GetNumConformers() == 0 or ref.GetNumConformers() == 0:
        return None

    energy = None
    strain = None
    if optimise:
        opt_result = Geom.optimise_mol_xtb(mol, max_iters=max_opt_iters)
        if opt_result is None:
            return None

        mol, energy, initial_energy = opt_result
        strain = initial_energy - energy

    mol = _try_remove_hs(mol)
    ref_clean = _try_remove_hs(ref)
    if mol is None or ref_clean is None:
        return None

    if align:
        mol, shape_tani, colour_tani = Geom.align_conf(mol, ref_clean, align_weight=align_weight, ref_profile=profile)
    else:
        shape_tani, colour_tani = Geom.score_conf(mol, ref_clean, ref_profile=profile)

    return AlignedConf(mol, shape_tani, colour_tani, profile, energy, strain)


def score_confs(
    mols,
    refs,
    ref_profiles,
    align=False,
    align_weight=0.5,
    optimise=False,
    max_opt_iters=100,
    n_workers=None
):
    """Score generated conformers against references.

    Optionally optimise with xTB and/or align to reference before scoring.
    Returns list[AlignedConf | None].
    """

    score_fn = partial(
        _score_single,
        align=align,
        align_weight=align_weight,
        optimise=optimise,
        max_opt_iters=max_opt_iters
    )

    desc = ""
    if optimise:
        desc = f"xtb opt"
    if align:
        desc = f"{desc} + align" if len(desc) != 0 else "align"

    if n_workers is not None and n_workers > 0:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(score_fn, m, r, p) for m, r, p in zip(mols, refs, ref_profiles)]
            results = [f.result() for f in tqdm(futures, total=len(mols), desc=desc)]

    else:
        results = [score_fn(m, r, p) for m, r, p in tqdm(zip(mols, refs, ref_profiles), total=len(mols), desc=desc)]

    n_success = sum(r is not None for r in results)
    print(f"Scored {n_success}/{len(results)} molecules ({n_success / len(results):.1%})")
    return results


# *****************************************************************************
# ************************ Metric Interface Classes ***************************
# *****************************************************************************


class GenerativeMetric(Metric):
    # TODO add metric attributes - see torchmetrics doc

    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class PairMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, mols: list[Chem.rdchem.Mol], refs: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class AlignmentMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, mols: list[Chem.rdchem.Mol], refs: list[Chem.rdchem.Mol], aligned_confs: list[AlignedConf]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class EnsembleMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, ensembles: list[Chem.rdchem.Mol], aligned_confs: list[AlignedConf]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


# *****************************************************************************
# **************************** Generative Metrics *****************************
# *****************************************************************************


class Validity(GenerativeMetric):
    def __init__(self, connected=False, **kwargs):
        super().__init__(**kwargs)
        self.connected = connected

        self.add_state("valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        is_valid = [smolRD.mol_is_valid(mol, connected=self.connected) for mol in mols if mol is not None]
        self.valid += sum(is_valid)
        self.total += len(mols)

    def compute(self) -> torch.Tensor:
        return self.valid.float() / self.total


# TODO I don't think this will work with DDP
class Uniqueness(GenerativeMetric):
    """Note: only tracks uniqueness of molecules which can be converted into SMILES"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_smiles = []

    def reset(self):
        self.valid_smiles = []

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [smolRD.smiles_from_mol(mol, canonical=True) for mol in mols if mol is not None]
        valid_smiles = [smi for smi in smiles if smi is not None]
        self.valid_smiles.extend(valid_smiles)

    def compute(self) -> torch.Tensor:
        num_unique = len(set(self.valid_smiles))
        uniqueness = torch.tensor(num_unique) / len(self.valid_smiles)
        return uniqueness


class Novelty(GenerativeMetric):
    def __init__(self, existing_mols: list[Chem.rdchem.Mol], **kwargs):
        super().__init__(**kwargs)

        n_workers = min(16, len(os.sched_getaffinity(0)))
        executor = ProcessPoolExecutor(max_workers=n_workers)

        futures = [executor.submit(smolRD.smiles_from_mol, mol, canonical=True) for mol in existing_mols]
        smiles = [future.result() for future in futures]
        smiles = [smi for smi in smiles if smi is not None]

        executor.shutdown()

        self.smiles = set(smiles)

        self.add_state("novel", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        smiles = [smolRD.smiles_from_mol(mol, canonical=True) for mol in mols if mol is not None]
        valid_smiles = [smi for smi in smiles if smi is not None]
        novel = [smi not in self.smiles for smi in valid_smiles]

        self.novel += sum(novel)
        self.total += len(novel)

    def compute(self) -> torch.Tensor:
        return self.novel.float() / self.total


class AverageSize(GenerativeMetric):
    """Average size of the RDKit valid molecules generated"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_length", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valid_mols = [mol for mol in mols if smolRD.mol_is_valid(mol, connected=False)]

        # This will take the number of explicit atoms in the mol, so if Hs are explicit they will be included
        lengths = [mol.GetNumAtoms() for mol in valid_mols]

        self.n_valid += len(valid_mols)
        self.total_length += sum(lengths)

    def compute(self) -> torch.Tensor:
        return self.total_length.float() / self.n_valid


class AverageQED(GenerativeMetric):
    """Average QED (Quantitative Estimate of Drug-likeness) of valid generated molecules."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("qed_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        for mol in mols:
            if mol is None or not smolRD.mol_is_valid(mol):
                continue

            try:
                self.qed_sum += QED.default(mol)
                self.n_valid += 1
            except:
                continue

    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)

        return self.qed_sum / self.n_valid


class AverageLogP(GenerativeMetric):
    """Average Crippen LogP of valid generated molecules."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("logp_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        for mol in mols:
            if mol is None or not smolRD.mol_is_valid(mol):
                continue

            try:
                self.logp_sum += Crippen.MolLogP(mol)
                self.n_valid += 1
            except:
                continue

    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)

        return self.logp_sum / self.n_valid


class PBValidity(GenerativeMetric):
    """PoseBusters validity of de novo unconditionally generated molecules.

    NOTE this only counts the proportion of PB valid mols from mols which are not None in the input lists.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.buster = PoseBusters(config="mol", max_workers=8, chunk_size=10)

        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        # Don't count None mols since these count as RDKit invalid
        mols = [mol for mol in mols if mol is not None]
        if len(mols) == 0:
            return

        results_df = self.buster.bust(mols)
        results = results_df.fillna(1.0).astype(bool).all(axis=1).tolist()

        self.n_valid += sum(results)
        self.total += len(results)

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


# *****************************************************************************
# ****************************** Energy Metrics *******************************
# *****************************************************************************


class EnergyValidity(GenerativeMetric):
    def __init__(self, optimise=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise

        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        num_mols = len(mols)

        if self.optimise:
            mols = [Geom.optimise_mol_mmff(mol) for mol in mols if mol is not None]

        energies = [Geom.calc_energy_mmff(mol) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]

        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated

    Note that the energy cannot be calculated for some molecules (specifically invalid ones) and the pose optimisation
    is not guaranteed to succeed. Molecules for which the energy cannot be calculated do not count towards the metric.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, optimise=False, per_atom=False, n_workers=None, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise
        self.per_atom = per_atom
        self.n_workers = n_workers

        self.add_state("energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid_energies", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valid_energies = self._compute_valid_energies(mols)

        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        return self.energy / self.n_valid_energies

    def _compute_valid_energies(self, mols: list[Chem.rdchem.Mol]) -> list[float]:
        energies = [self._compute_mol_energy(mol) for mol in mols]
        valid_energies = [energy for energy in energies if _is_valid_float(energy)]
        return valid_energies

    def _compute_mol_energy(self, mol: Chem.rdchem.Mol) -> float:
        if mol is None:
            return None

        mol = Geom.optimise_mol_mmff(mol) if self.optimise else mol
        if mol is None:
            return None

        energy = Geom.calc_energy_mmff(mol, per_atom=self.per_atom)

        # Handle case of mol having multiple conformers
        if isinstance(energy, list):
            valid_energies = [en for en in energy if _is_valid_float(en)]
            if len(valid_energies) == 0:
                return None

            energy = sum(valid_energies) / len(valid_energies)

        return energy


class AverageRelaxEnergy(GenerativeMetric):
    """Average relaxation energy (energy diff between pose and optimised pose).

    Only calculated when all of the following are true:
    1. The molecule is valid and an energy can be calculated
    2. The pose optimisation succeeds
    3. The energy can be calculated for the optimised pose

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results. Use the EnergyValidity metric with the optimise flag set to True to track the proportion of
    molecules for which this metric can be calculated.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    def __init__(self, per_atom=False, n_workers=None, **kwargs):
        super().__init__(**kwargs)

        self.per_atom = per_atom
        self.n_workers = n_workers

        self.add_state("total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        energy_diffs = self._compute_valid_strains(mols)

        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        return self.total_energy_diff / self.n_valid

    def _compute_valid_strains(self, mols: list[Chem.rdchem.Mol]) -> list[float]:
        strains = [self._compute_mol_strain(mol) for mol in mols]
        valid_strains = [energy for energy in strains if _is_valid_float(energy)]
        return valid_strains

    def _compute_mol_strain(self, mol: Chem.rdchem.Mol) -> float:
        if mol is None:
            return None

        opt_mol = Geom.optimise_mol_mmff(mol, max_iters=1000, allow_unconverged=True)
        if opt_mol is None:
            return None

        opt_energy = Geom.calc_energy_mmff(opt_mol, per_atom=self.per_atom)
        orig_energy = Geom.calc_energy_mmff(mol, per_atom=self.per_atom)

        # Handle case of mol having multiple conformers
        if isinstance(opt_energy, list):
            strains = []
            for opt_en, orig_en in zip(opt_energy, orig_energy):
                if _is_valid_float(opt_en) and _is_valid_float(orig_en):
                    strains.append(orig_en - opt_en)

            if len(strains) == 0:
                return None

            strain = sum(strains) / len(strains)
            return strain

        if _is_valid_float(opt_energy) and _is_valid_float(orig_energy):
            return orig_energy - opt_energy

        return None


# *****************************************************************************
# ***************************** Similarity Metrics ****************************
# *****************************************************************************


class ReconstructionAccuracy(PairMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols, refs):
        assert len(mols) == len(refs)

        mol_smis = [smolRD.smiles_from_mol(mol, canonical=True) if mol is not None else None for mol in mols]
        ref_smis = [smolRD.smiles_from_mol(ref, canonical=True) if ref is not None else None for ref in refs]
        matching = [m == r for m, r in zip(mol_smis, ref_smis) if m is not None and r is not None]

        self.n_correct += sum(matching)
        self.total += len(matching)

    def compute(self) -> torch.Tensor:
        return self.n_correct.float() / self.total


class ECFPTanimoto(PairMetric):
    """Extended Connectivity FingerPrint (Morgan fingerprint) Tanimoto similarity between generated and reference molecules.

    ECFP captures circular neighborhoods around atoms and is widely used for measuring structural similarity.
    Higher values indicate more similar molecules (range 0-1).

    Args:
        radius: The radius of the Morgan fingerprint (default 2, which corresponds to ECFP4)
        n_bits: Number of bits in the fingerprint (default 2048)
        use_features: Whether to use feature-based invariants instead of atom-based (default False)
        remove_hs: Whether to remove hydrogens before computing fingerprints (default True)
    """

    def __init__(self, radius=2, n_bits=2048, use_features=False, remove_hs=True, **kwargs):
        super().__init__(**kwargs)

        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
        self.remove_hs = remove_hs

        self.add_state("tanimoto_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_pairs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols, refs):
        assert len(mols) == len(refs)

        if self.remove_hs:
            mols = [_try_remove_hs(mol) for mol in mols]
            refs = [_try_remove_hs(ref) for ref in refs]

        # Filter out None molecules
        valid_pairs = [(m, r) for m, r in zip(mols, refs) if m is not None and r is not None]

        if len(valid_pairs) == 0:
            return

        fp_fn = partial(
            AllChem.GetMorganFingerprintAsBitVect,
            radius=self.radius,
            nBits=self.n_bits,
            useFeatures=self.use_features
        )

        # Compute fingerprints and similarities
        for mol, ref in valid_pairs:
            try:
                mol_fp = fp_fn(mol)
                ref_fp = fp_fn(ref)
                tanimoto = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)

                self.tanimoto_sum += tanimoto
                self.n_pairs += 1

            except:
                continue

    def compute(self) -> torch.Tensor:
        if self.n_pairs == 0:
            return torch.tensor(0.0)
        return self.tanimoto_sum / self.n_pairs


class MACCSTanimoto(PairMetric):
    """MACCS Keys Tanimoto similarity between generated and reference molecules.

    MACCS keys are 166 predefined structural keys that capture common chemical features.
    Useful for measuring similarity based on presence/absence of specific substructures.
    Higher values indicate more similar molecules (range 0-1).

    Args:
        remove_hs: Whether to remove hydrogens before computing fingerprints (default True)
    """

    def __init__(self, remove_hs=True, **kwargs):
        super().__init__(**kwargs)

        self.remove_hs = remove_hs

        self.add_state("tanimoto_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_pairs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols, refs):
        assert len(mols) == len(refs)

        if self.remove_hs:
            mols = [_try_remove_hs(mol) for mol in mols]
            refs = [_try_remove_hs(ref) for ref in refs]

        # Filter out None molecules
        valid_pairs = [(m, r) for m, r in zip(mols, refs) if m is not None and r is not None]

        if len(valid_pairs) == 0:
            return

        # Compute fingerprints and similarities
        for mol, ref in valid_pairs:
            try:
                mol_fp = MACCSkeys.GenMACCSKeys(mol)
                ref_fp = MACCSkeys.GenMACCSKeys(ref)
                tanimoto = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)

                self.tanimoto_sum += tanimoto
                self.n_pairs += 1

            except:
                continue

    def compute(self) -> torch.Tensor:
        if self.n_pairs == 0:
            return torch.tensor(0.0)

        return self.tanimoto_sum / self.n_pairs


# *****************************************************************************
# **************************** Alignment Metrics ******************************
# *****************************************************************************


class ShapeTanimoto(AlignmentMetric):
    """Best shape tanimoto from conformer ensemble alignment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("shape_tanimoto_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols, refs, aligned_confs):
        assert len(mols) == len(refs) == len(aligned_confs)

        tanis = [a.shape_tani for a in aligned_confs if a is not None]
        valid_tanis = [tani for tani in tanis if not math.isnan(tani)]
        tani_sum = sum(valid_tanis)

        self.shape_tanimoto_sum += tani_sum
        self.n_valid += len(valid_tanis)

    def compute(self) -> torch.Tensor:
        return self.shape_tanimoto_sum / self.n_valid


class ColourTanimoto(AlignmentMetric):
    """Best colour tanimoto from conformer ensemble alignment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("colour_tanimoto_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols, refs, aligned_confs):
        assert len(mols) == len(refs) == len(aligned_confs)

        tanis = [a.colour_tani for a in aligned_confs if a is not None]
        valid_tanis = [tani for tani in tanis if not math.isnan(tani)]
        tani_sum = sum(valid_tanis)

        self.colour_tanimoto_sum += tani_sum
        self.n_valid += len(valid_tanis)

    def compute(self) -> torch.Tensor:
        return self.colour_tanimoto_sum / self.n_valid


class InteractionRecovery(AlignmentMetric):
    """Proportion of reference pharmacophore features recovered by generated molecules.

    For each ref-gen pair, compares reference pharmacophore features against those detected in
    the aligned generated molecule. A ref feature is "recovered" if the generated molecule has
    a feature of the same type within ``dist_threshold`` angstroms.

    When a ref_profile is available in the AlignedConf, pharmacophore features are taken from
    the profile (i.e. only the interactions actually conditioned on). Otherwise, features are
    auto-detected from the reference molecule via PharmacophoreFinder.
    """

    def __init__(self, dist_threshold=2.0, pharm_filter="all", **kwargs):
        super().__init__(**kwargs)

        if pharm_filter not in ("all", "hydrophobe", "non-hydrophobe"):
            raise ValueError("pharm_filter must be 'all', 'hydrophobe', or 'non-hydrophobe'")

        self.dist_threshold = dist_threshold
        self.pharm_filter = pharm_filter
        self._hydrophobe_idx = PharmacophoreFinder.get_feature_index("Hydrophobe")

        self.add_state("recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols, refs, aligned_confs):
        assert len(mols) == len(refs) == len(aligned_confs)

        for aligned in aligned_confs:
            if aligned is None:
                continue

            # Use only the provided ref pharmacophores, possibly not all interactions available in the ref mol
            ref_types, ref_positions = self._get_profile_pharm_features(aligned.ref_profile)

            # Filter by pharmacophore type if requested
            if self.pharm_filter == "hydrophobe":
                keep = [i for i, t in enumerate(ref_types) if t == self._hydrophobe_idx]
            elif self.pharm_filter == "non-hydrophobe":
                keep = [i for i, t in enumerate(ref_types) if t != self._hydrophobe_idx]
            else:
                keep = list(range(len(ref_types)))

            ref_types = [ref_types[i] for i in keep]
            ref_positions = [ref_positions[i] for i in keep]

            if len(ref_types) == 0:
                continue

            # Detect pharmacophores in the aligned gen mol. The aligned mol may have Hs but
            # PharmacophoreFinder SMARTS patterns only match heavy atoms so results are the same.
            gen_feats = PharmacophoreFinder.run_mol(aligned.aligned)

            # Group gen features by type for efficient lookup
            gen_by_type: dict[int, list[np.ndarray]] = {}
            for feat in gen_feats:
                gen_by_type.setdefault(feat.type, []).append(feat.position)

            n_preserved = 0
            for ref_type, ref_pos in zip(ref_types, ref_positions):
                gen_positions = gen_by_type.get(ref_type)
                if gen_positions is None:
                    continue

                dists = np.linalg.norm(np.array(gen_positions) - ref_pos, axis=1)
                if dists.min() <= self.dist_threshold:
                    n_preserved += 1

            self.recall_sum += n_preserved / len(ref_types)
            self.n_valid += 1

    @staticmethod
    def _get_profile_pharm_features(profile):
        """Extract pharmacophore types and positions from a ConfProfile.

        Returns only the pharmacophore points (type >= 2), with types mapped back to
        PharmacophoreFinder indices (type - 2).

        Returns:
            Tuple of (types, positions) where types is a list of int and positions is a list of ndarray.
        """

        pharm_mask = profile.types >= 2
        if not pharm_mask.any():
            return [], []

        types = (profile.types[pharm_mask] - 2).astype(int).tolist()
        positions = list(profile.positions[pharm_mask])
        return types, positions

    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)

        return self.recall_sum / self.n_valid


class XTBLocalStrain(AlignmentMetric):
    """Mean local xTB strain energy (initial - optimised) in kcal/mol."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("strain_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols, refs, aligned_confs):
        for a in aligned_confs:
            if a is not None and a.strain is not None and _is_valid_float(a.strain):
                self.strain_sum += a.strain * HARTREE_TO_KCAL
                self.n_valid += 1

    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(float("nan"))

        return self.strain_sum / self.n_valid


# *****************************************************************************
# ***************************** Ensemble Metrics ******************************
# *****************************************************************************


class EnsembleEntropy(EnsembleMetric):
    """Entropy of Boltzmann weights from conformer ensemble sampling."""

    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(**kwargs)

        self.eps = eps

        self.add_state("entropy_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, ensembles, aligned_confs):
        # Ignore other args, just to keep the interface the same

        assert len(ensembles) == len(aligned_confs)

        entropies = []
        for ens in ensembles:
            if ens is None:
                continue

            _, weights, _ = ens
            ent = _entropy(weights, self.eps)

            if ent is not None:
                entropies.append(ent)

        self.entropy_sum += sum(entropies)
        self.n_valid += len(entropies)

    def compute(self) -> torch.Tensor:
        return self.entropy_sum / self.n_valid


class StrainEnergy(EnsembleMetric):
    """Strain energy: difference between minimized model conf energy and ensemble E_min.

    For each molecule, strain = aligned_conf.energy - ensemble_e_min. This measures how
    far the model's generated conformer is from the lowest-energy conformer found by
    ensemble sampling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("strain_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, ensembles, aligned_confs):
        assert len(ensembles) == len(aligned_confs)

        for ens, aligned in zip(ensembles, aligned_confs):
            if ens is None or aligned is None:
                continue

            if aligned.energy is None or not _is_valid_float(aligned.energy):
                continue

            _, _, e_min = ens
            if not _is_valid_float(e_min):
                continue

            strain = aligned.energy - e_min
            self.strain_sum += strain
            self.n_valid += 1

    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)

        return self.strain_sum / self.n_valid


class LowStrainRate(EnsembleMetric):
    """Proportion of molecules with strain energy below a threshold."""

    def __init__(self, threshold=6.0, **kwargs):
        super().__init__(**kwargs)

        self.threshold = threshold

        self.add_state("n_low", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, ensembles, aligned_confs):
        assert len(ensembles) == len(aligned_confs)

        for ens, aligned in zip(ensembles, aligned_confs):
            if ens is None or aligned is None:
                continue

            if aligned.energy is None or not _is_valid_float(aligned.energy):
                continue

            _, _, e_min = ens
            if not _is_valid_float(e_min):
                continue

            strain = aligned.energy - e_min
            self.n_valid += 1
            if strain <= self.threshold:
                self.n_low += 1

    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)

        return self.n_low.float() / self.n_valid


class EnsembleMeanPairwiseRMSD(EnsembleMetric):
    """Mean pairwise RMSD across conformer ensembles.

    Reuses the MeanPairwiseRMSD feature from enscondflow.data.features via GraphMol conversion.
    """

    def __init__(self, max_pairs=200, **kwargs):
        super().__init__(**kwargs)

        self._feat = Features.MeanPairwiseRMSD(max_pairs=max_pairs, raise_on_err=False, return_on_err=None)

        self.add_state("rmsd_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, ensembles, aligned_confs):
        for ens in ensembles:
            if ens is None:
                continue

            mol, _, _ = ens
            if mol.GetNumConformers() < 2:
                continue

            graph_mol = GraphMol.from_rdkit(mol)
            rmsd = self._feat(graph_mol)
            if rmsd is not None:
                self.rmsd_sum += rmsd
                self.n_valid += 1

    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)

        return self.rmsd_sum / self.n_valid


class EnsemblePSA3D(EnsembleMetric):
    """Mean 3D Polar Surface Area across conformer ensembles.

    Reuses the PSA3D feature from enscondflow.data.features via GraphMol conversion.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._feat = Features.PSA3D(raise_on_err=False, return_on_err=None)

        self.add_state("psa3d_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, ensembles, aligned_confs):
        for ens in ensembles:
            if ens is None:
                continue

            mol, _, _ = ens
            if mol.GetNumConformers() < 2:
                continue

            graph_mol = GraphMol.from_rdkit(mol)
            psa = self._feat(graph_mol)
            if psa is not None:
                self.psa3d_sum += psa
                self.n_valid += 1

    def compute(self) -> torch.Tensor:
        if self.n_valid == 0:
            return torch.tensor(0.0)

        return self.psa3d_sum / self.n_valid
