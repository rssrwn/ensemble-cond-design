"""Multi-condition shape evaluation script.

Evaluates the model's ability to generate molecules conditioned on two shape profiles simultaneously.
Reads cross-mol pairs (compact_A from one mol, extended_B from another) prepared by prep_pairs.py.

The positive cond gets equivariant features, the negative cond gets invariant features (rotated=True).
Mode is inferred from negate flags: no flag → pos_pos, --negate_extended → compact pos / extended neg,
--negate_compact → extended pos / compact neg.

Generated molecules are scored by sampling MMFF ensembles and measuring shape overlap against both refs,
plus ECFP similarity against both source mols.
"""

import json
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import torch
import lightning as L
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import enscondflow.scriptutil as util
import enscondflow.data.features as Features
from enscondflow.repr import GraphBatch
from enscondflow.data import GraphNoise, GraphInterpolant, GraphDataset, InterpolantDM


# FM sampling args
DEFAULT_BATCH_COST = 128
DEFAULT_N_INF_STEPS = 100
DEFAULT_STEP_SIZE = "decay"
DEFAULT_CAT_NOISE_LEVEL = 3
DEFAULT_CAT_STRATEGY = "sample"

# Conditioning args
DEFAULT_CFG_GAMMA = 1.0
DEFAULT_SHAPE_STD_DEV = 0.3

# Ensemble scoring args
DEFAULT_N_ENSEMBLE_CONFS = 128
DEFAULT_MAX_OPT_ITERS = 1000


def build_dm(hparams, data_path, batch_cost, pos_std_dev, n_mols, inv_feats=False):
    rotate_prob = 1.0 if inv_feats else 0.0

    eval_profile = Features.StochasticProfile(
        Features.InteractionProfile(),
        pos_std_dev=pos_std_dev,
        rotate_prob=rotate_prob,
        shape_resample=hparams["shape_resample"],
        global_pharm_dropout=1.0
    )

    batch = GraphBatch.load_hdf5_shard(Path(data_path))
    batch = GraphBatch(batch[:n_mols]) if n_mols is not None else batch

    # NOTE rand_rot here ensures we don't see the raw conformer (which could add bias)
    # It's not related to whether the rotated flag will be set in the encoder input
    eval_transform = partial(util.mol_transform, rand_rot=True, profile_feat=eval_profile, feat_dropout=1.0)
    test_dataset = GraphDataset(batch, transform=eval_transform)

    eval_interpolant = GraphInterpolant(
        GraphNoise(cat_noise="uniform", zero_com=True),
        perm_ot=False,
        pad_to=hparams["max_size"],
        pad_coord_mode=hparams["train-pad-coord-mode"]
    )

    dm = InterpolantDM(
        None,
        None,
        test_dataset,
        batch_cost,
        train_interpolant=None,
        val_interpolant=eval_interpolant,
        test_interpolant=eval_interpolant
    )
    return dm.test_dataloader()


def sample_multi_cond(model, a_batch, b_batch, negate_b, cfg_gamma):
    a_cond = util.to_device(a_batch["features"], model.device)
    b_cond = util.to_device(b_batch["features"], model.device)
    prior_batch = util.to_device(a_batch["prior_mols"], model.device)

    enc_a, enc_a_mask, ada_ls = model.create_latents(
        a_cond,
        profile_cond="shape",
        drop_props=True,
        training=False
    )
    enc_b, enc_b_mask, _ = model.create_latents(
        b_cond,
        profile_cond="shape",
        drop_props=True,
        training=False
    )

    gen_batch = model.generate_multi_cond(
        prior_batch,
        enc_a,
        enc_a_mask,
        enc_b,
        enc_b_mask,
        ada_ls,
        model.integrator.steps,
        step_strategy=model.integrator.step_size,
        negate_b=negate_b,
        cfg_gamma=cfg_gamma
    )

    mols = model.generate_mols(gen_batch, sanitise=True)
    return mols


def sample_uncond(model, compact_batch):
    return model.predict(compact_batch, cfg_gamma=0.0, profile_cond="none", drop_props=True, sanitise=True)


def print_sampling_args(args):
    print("\n***** Sampling Setup *****\n")

    sampling_args = {
        "Uncond": args.uncond,
        "Negate compact": args.negate_compact,
        "Negate extended": args.negate_extended,
        "CFG gamma": args.cfg_gamma,
        "Shape std dev": args.shape_std_dev
    }

    for hparam, val in sampling_args.items():
        print(f"{hparam:<25} {val}")
    print()

    print("\n**************************\n")


def per_system_ecfp_tanis(gen_mols, ref_mols, radius=2, n_bits=2048):
    """Per-system ECFP tanimoto between paired (gen, ref) mols. None for any failed pair."""

    def fp(mol):
        if mol is None:
            return None
        try:
            return AllChem.GetMorganFingerprintAsBitVect(Chem.RemoveAllHs(mol), radius=radius, nBits=n_bits)
        except Exception:
            return None

    tanis = []
    for gen, ref in zip(gen_mols, ref_mols):
        gen_fp = fp(gen)
        ref_fp = fp(ref)
        if gen_fp is None or ref_fp is None:
            tanis.append(None)
            continue
        tanis.append(DataStructs.TanimotoSimilarity(gen_fp, ref_fp))

    return tanis


def _smiles_or_none(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveAllHs(mol))
    except Exception:
        return None


def save_results(
    results_path,
    args,
    mol_results,
    sampled_mols,
    ref_a_mols,
    ref_b_mols,
    baseline_compact,
    baseline_extended,
    compact_tanis,
    extended_tanis,
    ecfp_a,
    ecfp_b
):
    def _floats(values):
        return [None if v is None else float(v) for v in values]

    payload = {
        "config": {**vars(args), "ckpt_path": str(args.ckpt_path), "multi_path": str(args.multi_path)},
        "mol_results": {k: float(v) for k, v in mol_results.items()},
        "smiles": {
            "gen": [_smiles_or_none(m) for m in sampled_mols],
            "ref_a": [_smiles_or_none(m) for m in ref_a_mols],
            "ref_b": [_smiles_or_none(m) for m in ref_b_mols]
        },
        "shape_tani": {
            "gen_compact": _floats(compact_tanis),
            "gen_extended": _floats(extended_tanis),
            "baseline_compact": _floats(baseline_compact),
            "baseline_extended": _floats(baseline_extended)
        },
        "ecfp_tani": {
            "vs_a": _floats(ecfp_a),
            "vs_b": _floats(ecfp_b)
        }
    }

    path = Path(results_path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved per-system results to {path}")


def print_results(
    mol_results,
    compact_tanis,
    extended_tanis,
    baseline_compact,
    baseline_extended,
    ecfp_a,
    ecfp_b
):
    print("\n******** Multi-Cond Evaluation Results ********\n")

    print("Standard generation results:")
    for metric, val in mol_results.items():
        print(f"{metric:<18}{val:.4f}")
    print()

    print("Shape tanimoto (raw + Δ vs random baseline):")
    sides = [
        ("compact (A)", compact_tanis, baseline_compact),
        ("extended (B)", extended_tanis, baseline_extended)
    ]
    for label, tanis, bases in sides:
        valid_idxs = [i for i, t in enumerate(tanis) if t is not None]
        if not valid_idxs:
            print(f"  {label}: no valid scores")
            continue

        gen = np.array([tanis[i] for i in valid_idxs])
        print(f"  {label} gen      — mean: {gen.mean():.4f}, median: {np.median(gen):.4f}, "
              f"std: {gen.std():.4f}")

        # Baseline is intrinsic to the test pool — mean over all valid baselines, not filtered by gen validity
        valid_bas = np.array([b for b in bases if b is not None])
        if len(valid_bas):
            print(f"  {label} baseline — mean: {valid_bas.mean():.4f}, median: {np.median(valid_bas):.4f}")

        # Δ is per-row; paired over rows where both gen and baseline are valid
        bas_pairs = [(tanis[i], bases[i]) for i in valid_idxs if bases[i] is not None]
        if bas_pairs:
            gen_for_bas = np.array([t for t, _ in bas_pairs])
            bas = np.array([b for _, b in bas_pairs])
            d = gen_for_bas - bas
            print(f"  {label} Δ base   — mean: {d.mean():+.4f}, median: {np.median(d):+.4f}")
        print()

    print("ECFP tanimoto vs source mols (per-system; None = invalid pair):")
    for label, vals in [("vs mol_A (compact source)", ecfp_a), ("vs mol_B (extended source)", ecfp_b)]:
        valid = np.array([v for v in vals if v is not None])
        if len(valid) == 0:
            print(f"  {label}: no valid scores")
            continue

        print(f"  {label:<32} mean: {valid.mean():.4f}, median: {np.median(valid):.4f}, "
              f"max: {valid.max():.4f}")

    paired = [(a, b) for a, b in zip(ecfp_a, ecfp_b) if a is not None and b is not None]
    if paired:
        avg = np.array([(a + b) / 2.0 for a, b in paired])
        print(f"  {'avg of A & B':<32} mean: {avg.mean():.4f}, median: {np.median(avg):.4f}, max: {avg.max():.4f}")
    print()


def main(args):
    print("Running multi-condition evaluation script...")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 128
    torch.set_float32_matmul_precision("high")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    multi_path = Path(args.multi_path)
    if args.mode is not None:
        mode = args.mode
    else:
        mode = "pos_neg" if (args.negate_compact or args.negate_extended) else "pos_pos"

    data_path = multi_path / "pairs" / mode
    compact_path = data_path / "compact.hdf5"
    extended_path = data_path / "extended.hdf5"
    print(f"Evaluating mode '{mode}' from {data_path}")

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    hparams = checkpoint["hyper_parameters"]

    print("Building integrator...")
    integrator = util.load_integrator(
        hparams,
        integration_steps=args.n_inf_steps,
        step_size=args.step_size,
        cat_strategy=args.cat_strategy,
        cat_noise_level=args.cat_noise_level
    )

    print("Loading model...")
    model = util.load_model(args.ckpt_path, integrator, arch="hybrid")
    model = model.eval().to(util.get_device())

    if args.negate_compact and args.negate_extended:
        raise ValueError("Cannot set both --negate_compact and --negate_extended")

    # The positive-conditioned conformer gets equivariant features, the negative gets invariant.
    # For pos/pos we default to compact equi features and extended inv features; --swap_feats flips this.
    if args.negate_compact:
        compact_inv, extended_inv = True, False
    else:
        compact_inv, extended_inv = False, True

    if args.swap_feats:
        compact_inv, extended_inv = extended_inv, compact_inv

    print("Building dataloaders...")
    compact_dl = build_dm(
        hparams,
        compact_path,
        args.batch_cost,
        args.shape_std_dev,
        args.n_mols,
        inv_feats=compact_inv
    )
    extended_dl = build_dm(
        hparams,
        extended_path,
        args.batch_cost,
        args.shape_std_dev,
        args.n_mols,
        inv_feats=extended_inv
    )

    # Baselines (avg size-matched random shape tani) loaded from sidecar JSON. Witnesses are also stored
    # there but not used in eval — they're a feasibility certificate from prep_pairs, not an upper bound.
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Expected metadata.json in {data_path}. Regenerate with prep_pairs.py.")

    with open(metadata_path) as f:
        pair_metadata = json.load(f)

    metadata_mode = pair_metadata["config"]["mode"]
    if metadata_mode != mode:
        raise ValueError(
            f"Metadata at {metadata_path} declares mode '{metadata_mode}' but eval inferred '{mode}' "
            f"from negate flags. The pairs/{mode}/ directory may be stale or mislabeled."
        )

    pair_records = pair_metadata["pairs"]
    if args.n_mols is not None:
        pair_records = pair_records[:args.n_mols]

    baseline_compact = [r["baseline"]["compact_tani"] for r in pair_records]
    baseline_extended = [r["baseline"]["extended_tani"] for r in pair_records]

    print_sampling_args(args)
    print("Sampling molecules...")

    compact_mols = []
    extended_mols = []
    sampled_mols = []

    negate_b = args.negate_compact or args.negate_extended

    for compact_batch, extended_batch in tqdm(zip(compact_dl, extended_dl), desc="Generating"):
        compact_mols.extend(model.generate_mols(compact_batch["data_mols"], sanitise=True))
        extended_mols.extend(model.generate_mols(extended_batch["data_mols"], sanitise=True))

        if args.uncond:
            mols = sample_uncond(model, compact_batch)
        else:
            # A is the positive (equi) cond, B is the negative (inv) cond
            if args.negate_compact:
                a_batch, b_batch = extended_batch, compact_batch
            else:
                a_batch, b_batch = compact_batch, extended_batch

            mols = sample_multi_cond(model, a_batch, b_batch, negate_b, args.cfg_gamma)

        sampled_mols.extend(mols)

    print("Sampling complete.")
    print("Scoring molecules...")

    # Generation-only metrics (validity, qed, etc.) — no ref mol comparison since cross-mol pairs have no
    # single "correct" reference. Per-source ECFP similarities are computed below as ecfp_a / ecfp_b.
    mol_results = util.score_molecules(sampled_mols)

    compact_tanis, extended_tanis, _, _ = util.score_ensemble_overlaps(
        sampled_mols,
        compact_mols,
        extended_mols,
        n_confs=args.n_ensemble_confs,
        max_opt_iters=args.max_opt_iters,
        n_threads=16
    )

    ecfp_a = per_system_ecfp_tanis(sampled_mols, compact_mols)
    ecfp_b = per_system_ecfp_tanis(sampled_mols, extended_mols)

    print_results(
        mol_results,
        compact_tanis,
        extended_tanis,
        baseline_compact,
        baseline_extended,
        ecfp_a,
        ecfp_b
    )

    if args.results_path is not None:
        save_results(
            args.results_path,
            args,
            mol_results,
            sampled_mols,
            compact_mols,
            extended_mols,
            baseline_compact,
            baseline_extended,
            compact_tanis,
            extended_tanis,
            ecfp_a,
            ecfp_b
        )

    print("***** Multi-cond evaluation complete *****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--multi_path", type=str, required=True, help="Dir with pairs/{pos_pos|pos_neg}/ inside")
    parser.add_argument("--mode", type=str, choices=["pos_pos", "pos_neg"], default=None)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_mols", type=int, default=None)
    parser.add_argument("--results_path", type=str, default=None, help="Path to save per-system results JSON")

    # FM sampling args
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--step_size", type=str, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--n_inf_steps", type=int, default=DEFAULT_N_INF_STEPS)
    parser.add_argument("--cat_strategy", type=str, default=DEFAULT_CAT_STRATEGY)
    parser.add_argument("--cat_noise_level", type=int, default=DEFAULT_CAT_NOISE_LEVEL)

    # Conditioning args
    parser.add_argument("--uncond", action="store_true", default=False)
    parser.add_argument("--negate_compact", action="store_true", default=False)
    parser.add_argument("--negate_extended", action="store_true", default=False)
    parser.add_argument("--swap_feats", action="store_true", default=False,
                        help="Swap which conditioner gets equi vs inv features (for the pos_pos symmetry check)")
    parser.add_argument("--cfg_gamma", type=float, default=DEFAULT_CFG_GAMMA)
    parser.add_argument("--shape_std_dev", type=float, default=DEFAULT_SHAPE_STD_DEV)

    # Ensemble scoring args
    parser.add_argument("--n_ensemble_confs", type=int, default=DEFAULT_N_ENSEMBLE_CONFS)
    parser.add_argument("--max_opt_iters", type=int, default=DEFAULT_MAX_OPT_ITERS)

    args = parser.parse_args()
    main(args)
