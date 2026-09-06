"""Util file for EnsCondFlow scripts"""

import math
import torch
import tempfile
import resource
import numpy as np
import biotite.structure.io.pdb as pdb_io
from tqdm import tqdm
from pathlib import Path
from functools import partial
from rdkit import RDLogger, Chem
from biotite.structure import AtomArray
from torchmetrics import MetricCollection
from scipy.spatial.transform import Rotation

import enscondflow.util.rdkit as smolRD
import enscondflow.util.geometry as Geom
import enscondflow.eval.metrics as Metrics
from enscondflow.repr import ConfSet, AtomVocab, BondVocab
from enscondflow.eval.integrator import Integrator
from enscondflow.data.features import Qed, LogP, Pharmacophores
from enscondflow.data.profile import ConfProfile
from enscondflow.models import EnsCondFlow, FeatureEncoder, HybridGenerator
from enscondflow.eval.docking import prepare_receptor_batch, dock_batch_parallel


# Declarations to be used in scripts
PROJECT_PREFIX = "enscondflow"


# Approx mean and std for features so that we pass in approx Gaussian dists
MEAN_PAIRWISE_RMSD_MEAN = 2.3
MEAN_PAIRWISE_RMSD_STD = 0.5

PSA3D_MEAN = 105.0
PSA3D_STD = 50.0


# *****************************************************************************
# *************************** Util setup functions ****************************
# *****************************************************************************


def disable_lib_stdout():
    RDLogger.DisableLog('rdApp.*')


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"


def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except Exception:
            print("Limit change unsuccessful. Using torch file_system file sharing strategy instead.")

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


# TODO support multi gpus
def calc_train_steps(dm, epochs, acc_batches):
    dm.setup("train")
    steps_per_epoch = math.ceil(len(dm.train_dataloader()) / acc_batches)
    return steps_per_epoch * epochs


# *****************************************************************************
# ************************* Util conversion functions *************************
# *****************************************************************************


def to_device(obj, device):
    """Send a (possibly hierarchical) dict to device and return the new dict

    Anything that is not a Tensor wrapped in dict/tuple/list will be left unchanged
    """

    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_device(item, device) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def to_numpy(obj):
    """Convert (possibly hierarchical) construction of tensors to numpy arrays
    
    Anything that is not an np array wrapped in dict/tuple/list will be left unchanged
    """

    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_numpy(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.numpy()
    else:
        return obj


# *****************************************************************************
# ****************************** Data transform *******************************
# *****************************************************************************


def _compute_directions(direction_meta, with_hs_conf):
    """Compute pharmacophore direction vectors from precomputed topology metadata.

    Args:
        direction_meta: List of tuples describing how to compute each direction
        with_hs_conf: Conformer coords with Hs, shape [n_atoms_full, 3]
    """

    directions = np.zeros((len(direction_meta), 3))
    for i, meta in enumerate(direction_meta):
        if meta[0] == "donor":
            heavy_pos = with_hs_conf[meta[1]]
            h_pos = with_hs_conf[meta[2]]
            d = h_pos - heavy_pos
            norm = np.linalg.norm(d)
            if norm > 1e-8:
                directions[i] = d / norm

        elif meta[0] == "aromatic":
            p0, p1, p2 = with_hs_conf[meta[1]], with_hs_conf[meta[2]], with_hs_conf[meta[3]]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                directions[i] = normal / norm

    return directions


def mol_transform(molecule, rand_rot=False, conf_sample=True, profile_feat=None, feat_dropout=0.0, shift_std=0.0):
    """Sample a conformer, apply a random rotation, centre the CoM, and compute features.

    If profile_feat is provided, computes the profile and stores it along with property features
    and per-feature masks in molecule.meta["features"]. If not provided, only the pharmacophore
    cache is updated (for compat with eval scripts that compute features separately).
    """

    assert molecule.confs is not None

    qed_feat = Qed()
    logp_feat = LogP()

    # Compute features that don't exist already (only for trial_run without preload)
    if qed_feat.name not in molecule.meta:
        molecule.meta[qed_feat.name] = qed_feat(molecule)
    if logp_feat.name not in molecule.meta:
        molecule.meta[logp_feat.name] = logp_feat(molecule)

    # NOTE for these we cannot compute the real values since the ensemble will already have been downsampled
    # So we just set dummy values (0.0 standardised = training mean). These need to be precomputed before downsampling.
    if "standardised-mean-pairwise-rmsd" not in molecule.meta:
        molecule.meta["standardised-mean-pairwise-rmsd"] = 0.0
    if "standardised-psa3d" not in molecule.meta:
        molecule.meta["standardised-psa3d"] = 0.0

    # Sample a conformer randomly or take the lowest energy one
    if conf_sample:
        conf_coords = molecule.confs.get_conformer(np.random.randint(0, len(molecule.confs)))
    else:
        conf_coords = molecule.confs.select_topk(k=1).coords[0]

    # Compute pharmacophore coords/directions from the sampled conformer
    # No need to copy full mol here with remove_hs, just mask directly on the confs
    no_hs_conf = conf_coords[molecule.atomics != 1]

    pharm_cache = molecule.meta.get(Pharmacophores.CACHE_KEY)
    if pharm_cache is not None and "direction_meta" in pharm_cache and len(pharm_cache["atom_ids"]) > 0:
        pharm_types = pharm_cache["types"]
        pharm_atom_ids = pharm_cache["atom_ids"]
        pharm_coords = np.array([no_hs_conf[list(aids)].mean(axis=0) for aids in pharm_atom_ids])
        pharm_directions = _compute_directions(pharm_cache["direction_meta"], conf_coords)

    elif pharm_cache is not None:
        pharm_types = pharm_cache["types"]
        pharm_atom_ids = pharm_cache["atom_ids"]
        pharm_coords = np.zeros((0, 3))
        pharm_directions = np.zeros((0, 3))

    # Fallback: compute from scratch (e.g. trial_run without preload)
    else:
        temp_mol = molecule.copy_with(confs=ConfSet(conf_coords[np.newaxis]))
        pharm = Pharmacophores(conf_idx=0, raise_on_err=False)
        pharm_types, pharm_coords, pharm_atom_ids, pharm_directions = pharm(temp_mol)

    molecule = molecule.copy_with(confs=ConfSet(conf_coords[np.newaxis]))
    rotation = Rotation.random() if rand_rot else None
    molecule = molecule.rotate(rotation) if rotation is not None else molecule

    if rotation is not None and len(pharm_coords) > 0:
        pharm_coords = rotation.apply(pharm_coords)
        nonzero_dirs = np.linalg.norm(pharm_directions, axis=1) > 0.5
        if nonzero_dirs.any():
            pharm_directions[nonzero_dirs] = rotation.apply(pharm_directions[nonzero_dirs])

    molecule = molecule.remove_hs()

    # Apply the same CoM shift to cached pharmacophore coords (directions are vectors, not positions)
    com = molecule.get_conformer(0).mean(axis=0)
    molecule = molecule.zero_com()

    if len(pharm_coords) > 0:
        pharm_coords = pharm_coords - com

    if shift_std > 0:
        coord_shift = np.random.randn(3) * shift_std
        molecule = molecule.shift(coord_shift)
        if len(pharm_coords) > 0:
            pharm_coords = pharm_coords + coord_shift

    molecule.meta[Pharmacophores.CACHE_KEY] = {
        "types": pharm_types,
        "coords": pharm_coords,
        "atom_ids": pharm_atom_ids,
        "directions": pharm_directions
    }

    # Compute profile and feature masks, store in meta["features"] for the interpolant to read
    if profile_feat is not None:
        mask_feat = lambda: np.array(np.random.rand() > feat_dropout, dtype=np.int64)
        molecule.meta["features"] = {
            "profile": profile_feat(molecule),
            "standardised-mean-pairwise-rmsd": molecule.meta["standardised-mean-pairwise-rmsd"],
            "standardised-psa3d": molecule.meta["standardised-psa3d"],
            "standardised-mean-pairwise-rmsd-mask": mask_feat(),
            "standardised-psa3d-mask": mask_feat()
        }

    return molecule


def complex_transform(system, rand_rot=True, profile_feat=None, feat_dropout=0.0, pocket_rotate_prob=0.5):
    """Prepare a BindingComplex for pocket-conditioned training.

    Computes binding profile and molecular properties from the original system (with Hs and
    interactions), then removes Hs, centers on pocket COM, and applies augmentations:

    1. Random rotation of the entire complex (data augmentation for learning equivariance).
       The profile positions are shifted and rotated to match.
    2. Independent random rotation of the pocket only (prob pocket_rotate_prob). This breaks the
       pocket out of the ligand's reference frame. The rotated flag (0/1) tells the pocket encoder
       whether the pocket is in the same frame as the ligand (0) or an arbitrary frame (1).
    3. StochasticProfile noise/rotation/dropout is already baked into the profile from step 0.
       Its own rotation flag indicates whether the profile is in the ligand's frame.
    """

    assert system.ligand.n_conformers == 1

    # Compute profile from original system (needs Hs and interactions, unless precomputed)
    profile = None
    if profile_feat is not None:
        result = profile_feat(system)
        if isinstance(result, ConfProfile):
            profile = result

    # remove_hs() returns a new object — set interactions to None on the copy, not the original,
    # since replicate() creates multiple references to the same BindingComplex
    system = system.remove_hs()
    system.interactions = None

    # Shift the system to pocket CoM (aligns with what we will use at eval time)
    pocket_com_shift = -system.protein.coords.mean(axis=0)
    system = system.zero_pocket_com()

    # Randomly rotate the entire complex to allow learned equivariance
    aug_rotation = Rotation.random() if rand_rot else None
    if aug_rotation is not None:
        system = system.rotate(aug_rotation)

    # Independently rotate the pocket only — breaks pocket out of ligand frame
    pocket_rotated = pocket_rotate_prob > 0 and np.random.rand() < pocket_rotate_prob
    if pocket_rotated:
        pocket_rotation = Rotation.random()
        system = system.copy_with(protein=system.protein.rotate(pocket_rotation))

    system.protein.meta["rotated"] = int(pocket_rotated)

    # Transform profile positions into the ligand's final frame
    if profile is not None:
        profile_positions = profile.positions + pocket_com_shift
        if aug_rotation is not None:
            profile_positions = aug_rotation.apply(profile_positions)

        profile_directions = profile.directions.copy()
        nonzero = np.linalg.norm(profile_directions, axis=1) > 0.5
        if aug_rotation is not None and nonzero.any():
            profile_directions[nonzero] = aug_rotation.apply(profile_directions[nonzero])

        mask_feat = lambda: np.array(np.random.rand() > feat_dropout, dtype=np.int64)
        profile = ConfProfile(
            positions=profile_positions,
            types=profile.types,
            atom_ids=profile.atom_ids,
            directions=profile_directions,
            rotated=profile.rotated,
            profile_mode=profile.profile_mode,
            pos_noise_std=profile.pos_noise_std
        )

        system.ligand.meta["features"] = {
            "profile": profile,
            "standardised-mean-pairwise-rmsd": 0.0,
            "standardised-psa3d": 0.0,
            "standardised-mean-pairwise-rmsd-mask": mask_feat(),
            "standardised-psa3d-mask": mask_feat()
        }

    return system


# *****************************************************************************
# ******************* Util loading functions for evaluation *******************
# *****************************************************************************


def load_integrator(
    hparams,
    integration_steps=None,
    step_size=None,
    cat_strategy=None,
    cat_noise_level=None,
    coord_sch=None,
    bond_sch=None,
    corr_sch_a=None,
    corr_sch_b=None
):
    integration_steps = hparams["integration-steps"] if integration_steps is None else integration_steps
    step_size = hparams["step-size"] if step_size is None else step_size
    cat_strategy = hparams["integration-cat-strategy"] if cat_strategy is None else cat_strategy
    cat_noise_level = hparams["integration-cat-noise-level"] if cat_noise_level is None else cat_noise_level
    coord_sch = hparams["integration-coord-schedule"] if coord_sch is None else coord_sch
    bond_sch = hparams["integration-bond-schedule"] if bond_sch is None else bond_sch
    corr_sch_a = hparams["corrector-schedule-a"] if corr_sch_a is None else corr_sch_a
    corr_sch_b = hparams["corrector-schedule-b"] if corr_sch_b is None else corr_sch_b

    integrator = Integrator(
        integration_steps,
        step_size=step_size,
        cat_strategy=cat_strategy,
        cat_noise_level=cat_noise_level,
        coord_sch=coord_sch,
        bond_sch=bond_sch,
        corrector_sch_a=corr_sch_a,
        corrector_sch_b=corr_sch_b
    )
    return integrator


def load_model(ckpt_path, integrator=None, arch="hybrid"):
    """Load a compatible Lightning checkpoint, including locally trained models."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    hparams = dict(checkpoint["hyper_parameters"])
    # Compilation is a runtime choice, not part of the learned architecture.
    hparams["compile_model"] = False
    integrator = load_integrator(hparams) if integrator is None else integrator

    n_atom_types = len(AtomVocab)
    n_bond_types = len(BondVocab)

    if arch != "hybrid":
        raise ValueError(f"Unsupported arch '{arch}'")

    generator = HybridGenerator(
        hparams["d_model"],
        hparams["n_heads"],
        hparams["d_edge"],
        hparams["n_layers"],
        hparams["n_blocks"],
        n_atom_types,
        n_bond_types,
        ff_factor=hparams["ff_factor"],
        dropout=hparams["dropout"],
        d_emb=hparams.get("d_emb", 64)
    )

    encoder = FeatureEncoder(
        hparams["enc-d_model"],
        hparams["enc-n_heads"],
        hparams["enc-n_layers"],
        hparams["enc-n_props"],
        d_out=hparams["enc-d_out"],
        include_pocket=hparams.get("enc-include_pocket", False)
    )

    fm_model = EnsCondFlow(
        generator=generator, encoder=encoder, integrator=integrator, **hparams
    )
    # torch.compile wrappers affect key paths but not learned tensor values.
    state = {k.replace("._orig_mod.", "."): v for k, v in checkpoint["state_dict"].items()}
    fm_model.load_state_dict(state, strict=True)
    fm_model.eval()
    return fm_model


# ***************************************************************************
# ************************ Sampling util functions **************************
# ***************************************************************************


def generate_molecules(
    model,
    val_dl,
    cfg_gamma,
    profile_cond: str = "profile",
    drop_pocket: bool = False,
    drop_props: bool = False,
    sanitise=True
):
    """Generate molecules (possibly) using CFG on discrete, continuous or both."""

    gen_mols = []
    data_mols = []

    for batch in tqdm(val_dl):
        batch = to_device(batch, model.device)
        data_mols_batch = model.generate_mols(batch["data_mols"], sanitise=True)

        gen_mols_batch = model.predict(
            batch,
            cfg_gamma=cfg_gamma,
            profile_cond=profile_cond,
            drop_pocket=drop_pocket,
            drop_props=drop_props,
            sanitise=sanitise
        )

        gen_mols.extend(gen_mols_batch)
        data_mols.extend(data_mols_batch)

    assert len(gen_mols) == len(data_mols)

    return gen_mols, data_mols


# *****************************************************************************
# ************************ Evaluation util functions **************************
# *****************************************************************************


def _per_mol_metric(make_metric, items):
    """Per-molecule metric values: run a fresh metric on each item (the aggregate is just their mean).
    Returns one float per item, or None where the molecule was invalid / the metric had no valid sample."""
    out = []
    for args in items:
        metric = make_metric()
        try:
            metric.update(*args)
            n_valid = getattr(metric, "n_valid", None)
            if n_valid is not None and int(n_valid) == 0:
                out.append(None)
                continue

            val = float(metric.compute())
            out.append(None if np.isnan(val) else val)
        except Exception:
            out.append(None)

    return out


def score_molecules(mols, data_mols=None, return_per_mol=False):
    """Score molecules. Pass data_mols to also include ref-comparison metrics (ref-match, ecfp-tanimoto)."""

    # mols are RDKit objects but must contain None where the mol was not sanitisable
    gen_metrics = {
        "validity": Metrics.Validity(),
        "fc-validity": Metrics.Validity(connected=True),
        "uniqueness": Metrics.Uniqueness(),
        "energy": Metrics.AverageEnergy(),
        "average-size": Metrics.AverageSize(),
        "qed": Metrics.AverageQED(),
        "logp": Metrics.AverageLogP()
    }

    gen_group = MetricCollection(gen_metrics, compute_groups=False)
    gen_group.update(mols)
    gen_results = gen_group.compute()

    if data_mols is None:
        result = dict(gen_results)
    else:
        assert len(mols) == len(data_mols)

        pair_metrics = {
            "ref-match": Metrics.ReconstructionAccuracy(),
            "ecfp-tanimoto": Metrics.ECFPTanimoto()
        }

        pair_group = MetricCollection(pair_metrics, compute_groups=False)
        pair_group.update(mols, data_mols)
        result = {**gen_results, **pair_group.compute()}

    if not return_per_mol:
        return result

    per_mol = {
        "validity": _per_mol_metric(lambda: Metrics.Validity(), [([m],) for m in mols]),
        "fc-validity": _per_mol_metric(lambda: Metrics.Validity(connected=True), [([m],) for m in mols]),
        "qed": _per_mol_metric(lambda: Metrics.AverageQED(), [([m],) for m in mols])
    }
    if data_mols is not None:
        per_mol["ecfp-tanimoto"] = _per_mol_metric(lambda: Metrics.ECFPTanimoto(), [([m], [d]) for m, d in zip(mols, data_mols)])

    return result, per_mol


def score_docking(
    gen_mols,
    ref_mols,
    systems,
    exhaustiveness=32,
    n_workers=8
):
    """Score generated and reference molecules by docking into their target pockets.

    Generated molecules are fully docked (mode="dock") since they are in an arbitrary
    rotational frame. Reference molecules are scored in place (mode="score") since they
    are already in the correct binding pose.

    Args:
        gen_mols: List of generated RDKit molecules (may contain None).
        ref_mols: List of reference RDKit molecules.
        systems: List of BindingComplex objects.
        exhaustiveness: Number of Monte Carlo runs for Vina.
        n_workers: Number of parallel workers.

    Returns:
        Dict with "gen-dock-score", "ref-dock-score", and "dock-success" metrics.
    """

    proteins = [s.protein.read() for s in systems]
    ref_coords_list = [s.ligand.get_conformer(0) for s in systems]

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Preparing receptors...")
        receptor_paths = prepare_receptor_batch(proteins, tmp_dir, n_workers=n_workers)
        n_prepared = sum(1 for p in receptor_paths if p is not None)
        print(f"Prepared {n_prepared}/{len(proteins)} receptors")

        print("Docking generated molecules...")
        gen_results = dock_batch_parallel(
            gen_mols,
            receptor_paths,
            ref_coords_list,
            mode="dock",
            exhaustiveness=exhaustiveness,
            n_workers=n_workers,
        )
        n_gen_docked = sum(1 for r in gen_results if r is not None)
        print(f"Docked {n_gen_docked}/{len(gen_mols)} generated molecules")

        print("Scoring reference molecules...")
        ref_results = dock_batch_parallel(
            ref_mols,
            receptor_paths,
            ref_coords_list,
            mode="score",
            n_workers=n_workers,
        )
        n_ref_scored = sum(1 for r in ref_results if r is not None)
        print(f"Scored {n_ref_scored}/{len(ref_mols)} reference molecules")

    gen_affinities = [r.best_affinity for r in gen_results if r is not None]
    ref_affinities = [r.best_affinity for r in ref_results if r is not None]

    n_gen_valid = sum(1 for m in gen_mols if m is not None)
    dock_success = n_gen_docked / n_gen_valid if n_gen_valid > 0 else 0.0

    return {
        "gen-dock-score": np.mean(gen_affinities) if gen_affinities else float("nan"),
        "ref-dock-score": np.mean(ref_affinities) if ref_affinities else float("nan"),
        "dock-success": dock_success,
    }


def score_pose_quality(
    gen_aligned_mols,
    ref_mols,
    systems,
    box_padding=5.0,
    n_workers=8,
    return_per_mol=False
):
    """Score and minimize aligned generated molecules and reference molecules in their target pockets.

    Generated aligned molecules are scored in place and then locally minimized. Reference molecules
    are also scored and minimized. This gives an indication of how well the aligned generated conformer
    fits the binding pocket without full redocking.

    Args:
        gen_aligned_mols: List of aligned generated RDKit molecules (may contain None).
        ref_mols: List of reference RDKit molecules.
        systems: List of BindingComplex objects.
        box_padding: Angstroms of padding around reference ligand for docking box.
        n_workers: Number of parallel workers.

    Returns:
        Dict with score and minimize metrics for both generated and reference molecules.
    """

    proteins = [s.protein.read() for s in systems]
    ref_coords_list = [s.ligand.get_conformer(0) for s in systems]

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Preparing receptors...")
        receptor_paths = prepare_receptor_batch(proteins, tmp_dir, box_padding=box_padding, n_workers=n_workers)
        n_prepared = sum(1 for p in receptor_paths if p is not None)
        print(f"Prepared {n_prepared}/{len(proteins)} receptors")

        print("Scoring reference molecules...")
        ref_score_results = dock_batch_parallel(
            ref_mols,
            receptor_paths,
            ref_coords_list,
            mode="score",
            box_padding=box_padding,
            n_workers=n_workers,
        )

        print("Minimizing reference molecules...")
        ref_min_results = dock_batch_parallel(
            ref_mols,
            receptor_paths,
            ref_coords_list,
            mode="minimize",
            box_padding=box_padding,
            n_workers=n_workers,
        )

        print("Scoring aligned generated molecules...")
        gen_score_results = dock_batch_parallel(
            gen_aligned_mols,
            receptor_paths,
            ref_coords_list,
            mode="score",
            box_padding=box_padding,
            n_workers=n_workers,
        )

        print("Minimizing aligned generated molecules...")
        gen_min_results = dock_batch_parallel(
            gen_aligned_mols,
            receptor_paths,
            ref_coords_list,
            mode="minimize",
            box_padding=box_padding,
            n_workers=n_workers,
        )

    def _mean_affinity(results):
        affinities = [r.best_affinity for r in results if r is not None]
        return np.mean(affinities) if affinities else float("nan")

    n_gen_valid = sum(1 for m in gen_aligned_mols if m is not None)
    n_gen_scored = sum(1 for r in gen_score_results if r is not None)

    agg = {
        "ref-score": _mean_affinity(ref_score_results),
        "ref-minimized": _mean_affinity(ref_min_results),
        "gen-score": _mean_affinity(gen_score_results),
        "gen-minimized": _mean_affinity(gen_min_results),
        "gen-score-success": n_gen_scored / n_gen_valid if n_gen_valid > 0 else 0.0,
    }
    if not return_per_mol:
        return agg

    def _affin(results):
        return [float(r.best_affinity) if r is not None else None for r in results]

    per_mol = {
        "ref-score": _affin(ref_score_results),
        "ref-minimized": _affin(ref_min_results),
        "gen-score": _affin(gen_score_results),
        "gen-minimized": _affin(gen_min_results)
    }
    return agg, per_mol


def write_output(output_dir, gen_mols, ref_mols, results, systems=None, per_mol=None):
    """Write evaluation output: generated/reference mols, results JSON, and optionally protein structures.

    Args:
        output_dir: Directory to write output to
        gen_mols: List of generated RDKit mols (may contain None)
        ref_mols: List of reference RDKit mols
        results: Dict of metric name -> value to write as JSON
        systems: List of BindingComplex objects. When provided, writes protein PDB files alongside ligands.
    """

    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write results JSON
    def _coerce(v):
        if isinstance(v, torch.Tensor):
            return v.item() if v.numel() == 1 else v.tolist()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    json_results = {k: _coerce(v) for k, v in results.items()}
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Write per-system mol files
    mols_dir = output_dir / "mols"
    mols_dir.mkdir(exist_ok=True)

    n_written = 0
    for i, (gen_mol, ref_mol) in enumerate(zip(gen_mols, ref_mols)):
        if gen_mol is None:
            continue

        system_dir = mols_dir / f"{i:04d}"
        system_dir.mkdir(exist_ok=True)

        writer = Chem.SDWriter(str(system_dir / "gen.sdf"))
        writer.write(gen_mol)
        writer.close()

        writer = Chem.SDWriter(str(system_dir / "ref.sdf"))
        writer.write(ref_mol)
        writer.close()

        # Write protein PDB if systems are provided
        if systems is not None:
            protein = systems[i].protein.read()
            n_atoms = len(protein)
            atoms = AtomArray(n_atoms)
            elements = [smolRD.PT.symbol_from_atomic(a) for a in protein.atomics.tolist()]

            atoms.coord = protein.coords
            atoms.element = np.array(elements)
            atoms.res_name = protein.res_names
            atoms.atom_name = protein.atom_names
            atoms.res_id = protein.res_ids
            atoms.chain_id = np.array(["A"] * n_atoms)

            pdb_file = pdb_io.PDBFile()
            pdb_io.set_structure(pdb_file, atoms)
            pdb_file.write(system_dir / "protein.pdb")

        n_written += 1

    # Per-molecule metric arrays, indexed in gen_mols order (i <-> mols/{i:04d}); null where invalid.
    if per_mol is not None:
        n = len(gen_mols)
        per_mol_out = {
            "index": list(range(n)),
            "smiles": [Chem.MolToSmiles(m) if m is not None else None for m in gen_mols]
        }
        per_mol_out.update({k: list(v) for k, v in per_mol.items()})
        with open(output_dir / "per_mol.json", "w") as f:
            json.dump(per_mol_out, f)

    print(f"Wrote {n_written} systems and results to {output_dir}")


def align_gen_confs(mols, data_mols, ref_profiles, align_weight=0.5):
    """Align raw generated conformers directly to references without MMFF optimisation.

    For each mol:
    1. Remove Hs
    2. Align to ref using shape + conditioning pharmacophore features

    Returns list[Chem.Mol | None] of aligned molecules.
    """

    aligned = []
    for mol, ref, profile in zip(mols, data_mols, ref_profiles):
        if mol is None or ref is None:
            aligned.append(None)
            continue

        try:
            mol_clean = Chem.RemoveAllHs(mol)
        except Exception:
            aligned.append(None)
            continue

        if mol_clean is None:
            aligned.append(None)
            continue

        try:
            ref_clean = Chem.RemoveAllHs(ref)
        except Exception:
            aligned.append(None)
            continue

        if ref_clean is None:
            aligned.append(None)
            continue

        if mol_clean.GetNumAtoms() < 4 or ref_clean.GetNumAtoms() < 4:
            aligned.append(None)
            continue

        if mol_clean.GetNumConformers() == 0 or ref_clean.GetNumConformers() == 0:
            aligned.append(None)
            continue

        aligned_mol, _, _ = Geom.align_conf(mol_clean, ref_clean, align_weight=align_weight, ref_profile=profile)
        aligned.append(aligned_mol)

    return aligned


def score_alignments(
    mols,
    data_mols,
    ref_profiles,
    align=False,
    align_weight=0.5,
    optimise=False,
    max_opt_iters=100,
    recovery_dist=2.0,
    return_per_mol=False
):
    """Score shape/colour overlaps, optionally with xTB optimisation and/or alignment.

    Returns dict with shape-tanimoto, colour-tanimoto, interaction-recovery, and
    (when optimise=True) xtb-energy and xtb-strain in kcal/mol.
    """

    assert len(mols) == len(data_mols) == len(ref_profiles)

    scored = Metrics.score_confs(
        mols,
        data_mols,
        ref_profiles,
        align=align,
        align_weight=align_weight,
        optimise=optimise,
        max_opt_iters=max_opt_iters,
        n_workers=16
    )

    metrics = {
        "shape-tanimoto": Metrics.ShapeTanimoto(),
        "colour-tanimoto": Metrics.ColourTanimoto(),
        "interaction-recovery": Metrics.InteractionRecovery(dist_threshold=recovery_dist)
    }

    if optimise:
        metrics["xtb-strain"] = Metrics.XTBLocalStrain()

    group = MetricCollection(metrics, compute_groups=False)
    group.update(mols, data_mols, scored)
    agg = group.compute()
    if not return_per_mol:
        return agg

    def _safe(v):
        return None if (v is None or np.isnan(v)) else float(v)

    per_mol = {
        "shape-tanimoto": [None if a is None else _safe(a.shape_tani) for a in scored],
        "colour-tanimoto": [None if a is None else _safe(a.colour_tani) for a in scored],
        "interaction-recovery": _per_mol_metric(
            lambda: Metrics.InteractionRecovery(dist_threshold=recovery_dist),
            [([m], [r], [s]) for m, r, s in zip(mols, data_mols, scored)]
        )
    }
    return agg, per_mol


def score_ensembles(
    mols,
    n_confs=128,
    max_opt_iters=1000,
    strain_threshold=6.0,
    n_threads=8,
    return_per_mol=False
):
    """Scoring for ensemble metrics - PSA3D and mean pairwise RMSD"""

    ensemble_metrics = {
        "ensemble-pair-rmsd": Metrics.EnsembleMeanPairwiseRMSD(),
        "ensemble-psa3d": Metrics.EnsemblePSA3D()
    }
    ensemble_group = MetricCollection(ensemble_metrics, compute_groups=False)

    # Sample conformer ensembles and compute ensemble metrics
    print("Sampling conformer ensembles...")
    emb_fn = partial(
        Geom.sample_ensemble,
        max_confs=n_confs,
        max_conf_attempts=100,
        max_opt_iters=max_opt_iters,
        strain_filter=strain_threshold,
        dedup_rmsd_threshold=0.5,
        n_threads=n_threads
    )

    ensembles = [emb_fn(mol) if mol is not None else None for mol in tqdm(mols, desc="MMFF ensembles")]

    # PSA and pairwise RMSD don't require aligned confs so just pass None here
    ensemble_group.update(ensembles, None)

    ensemble_results = ensemble_group.compute()
    if not return_per_mol:
        return ensemble_results

    per_mol = {
        "ensemble-psa3d": _per_mol_metric(lambda: Metrics.EnsemblePSA3D(), [([e], None) for e in ensembles]),
        "ensemble-pair-rmsd": _per_mol_metric(lambda: Metrics.EnsembleMeanPairwiseRMSD(), [([e], None) for e in ensembles])
    }
    return ensemble_results, per_mol


def score_ensemble_overlaps(
    gen_mols,
    ref_compact,
    ref_extended,
    n_confs=128,
    max_opt_iters=1000,
    strain_filter=6.0,
    n_threads=8
):
    """Sample MMFF ensembles once per generated mol and score against both refs."""

    emb_fn = partial(
        Geom.sample_ensemble,
        max_confs=n_confs,
        max_conf_attempts=100,
        max_opt_iters=max_opt_iters,
        strain_filter=strain_filter,
        n_threads=n_threads
    )

    def score_one(gen_mol, ref_a, ref_b):
        if gen_mol is None or ref_a is None or ref_b is None:
            return None, None, None, None

        try:
            result = emb_fn(gen_mol)
        except Exception:
            return None, None, None, None

        if result is None:
            return None, None, None, None

        emb_mol = Chem.RemoveAllHs(result[0])
        aligned_a, shape_a, _ = Geom.align_best_conf(emb_mol, ref_a, align_weight=1.0)
        aligned_b, shape_b, _ = Geom.align_best_conf(emb_mol, ref_b, align_weight=1.0)
        return shape_a, shape_b, aligned_a, aligned_b

    zipped = zip(gen_mols, ref_compact, ref_extended)
    scored = [score_one(g, a, b) for g, a, b in tqdm(zipped, total=len(gen_mols), desc="Scoring ensembles")]
    compact_tanis, extended_tanis, compact_aligned, extended_aligned = zip(*scored)
    return list(compact_tanis), list(extended_tanis), list(compact_aligned), list(extended_aligned)
