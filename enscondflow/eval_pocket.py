"""Pocket conditioned evaluation script

This script is focused on evaluating how well the model can generate molecules which fit the shape and pharmacophore
profile of a ligand bound to a protein pocket. It is designed to evaluate models on the SPINDR dataset.
"""

import argparse
from pathlib import Path
from functools import partial

import torch
import numpy as np
import lightning as L

import enscondflow.scriptutil as util
import enscondflow.data.features as Features
from enscondflow.repr import ComplexBatch
from enscondflow.data import GraphNoise, GraphInterpolant, ComplexDataset, InterpolantDM
from enscondflow.data.profile import ConfProfile


# FM sampling default args
DEFAULT_BATCH_COST = 256   # number of molecules in the batch
DEFAULT_N_INF_STEPS = 100
DEFAULT_STEP_SIZE = "decay"
DEFAULT_CAT_NOISE_LEVEL = 3
DEFAULT_CAT_STRATEGY = "sample"

# Conditioning and CFG args
DEFAULT_LIGAND_COND = "profile"
DEFAULT_PAIR_RMSD_COND = None
DEFAULT_PSA3D_COND = None
DEFAULT_CFG_GAMMA = 1.0

# Evaluation args
DEFAULT_SHAPE_STD_DEV = 0.3
DEFAULT_RECOVERY_DIST = 2.0
DEFAULT_DOCK_BOX_PADDING = 8.0


PAIR_RMSD_STD_DEV = 0.1
PSA3D_STD_DEV = 10.0

# Matches MIN_QED in train.py — applied to training systems, so apply to test systems too
MIN_QED = 0.3


def prep_complex(
    system,
    profile_feat=None,
    pocket_cond=False,
    pair_rmsd_cond=None,
    psa3d_cond=None
):
    """Prepare a BindingComplex for evaluation.

    When pocket_cond=True, centers at pocket COM (matching complex_transform training).
    When pocket_cond=False, centers at ligand COM (matching mol_transform training) so that
    the profile's rotated=0 flag correctly indicates the profile is in the molecule's frame
    at the origin, consistent with how the model was trained on GEOM-drugs.
    """

    assert system.ligand.n_conformers == 1

    # Compute binding profile from the original system (needs Hs and interactions)
    profile = None
    if profile_feat is not None:
        result = profile_feat(system)
        if isinstance(result, ConfProfile):
            profile = result

    # Remove Hs from both protein and ligand, drop interactions
    system = system.remove_hs()
    system.interactions = None

    # Either centre at pocket or ligand COM, depending on whether we are cond on pocket or not
    if pocket_cond:
        com_shift = -system.protein.coords.mean(axis=0)
        system = system.zero_pocket_com()
        system.protein.meta["rotated"] = 0
    else:
        com_shift = -system.ligand.get_conformer(0).mean(axis=0)
        system = system.zero_ligand_com()

    # Transform profile positions into the chosen COM frame (no augmentation rotation at eval)
    if profile is not None:
        profile = ConfProfile(
            positions=profile.positions + com_shift,
            types=profile.types,
            atom_ids=profile.atom_ids,
            directions=profile.directions.copy(),
            rotated=profile.rotated,
            profile_mode=profile.profile_mode,
            pos_noise_std=profile.pos_noise_std
        )

    # Build features dict
    features = {}
    if profile is not None:
        features["profile"] = profile

    prop_conds = [
        (
            "standardised-mean-pairwise-rmsd",
            pair_rmsd_cond,
            PAIR_RMSD_STD_DEV,
            util.MEAN_PAIRWISE_RMSD_MEAN,
            util.MEAN_PAIRWISE_RMSD_STD,
        ),
        ("standardised-psa3d", psa3d_cond, PSA3D_STD_DEV, util.PSA3D_MEAN, util.PSA3D_STD),
    ]

    for name, val, std, feat_mean, feat_std in prop_conds:
        if val is not None:
            sample = max(np.random.normal(val, std), 0.0)
            features[name] = (sample - feat_mean) / feat_std
            features[f"{name}-mask"] = np.array(1, dtype=np.int64)
        else:
            features[name] = 0.0
            features[f"{name}-mask"] = np.array(0, dtype=np.int64)

    system.ligand.meta["features"] = features
    return system


def load_dataset(args, hparams):
    print("Initialising datasets...")

    profile_feat = Features.StochasticProfile(
        Features.BindingProfile(),
        pos_std_dev=args.shape_std_dev,
        rotate_prob=0.0,
        shape_resample=hparams["shape_resample"]
    )

    transform = partial(
        prep_complex,
        profile_feat=profile_feat,
        pocket_cond=args.pocket_cond,
        pair_rmsd_cond=args.pair_rmsd_cond,
        psa3d_cond=args.psa3d_cond
    )

    # Load the data and prepare the structures, then we can close the files
    hdf5_dataset = ComplexDataset.load(Path(args.data_path), transform=transform)
    complexes = ComplexBatch([system for system in hdf5_dataset])
    hdf5_dataset.close()

    # When pocket_cond is enabled, keep full complexes so the interpolant includes pocket data
    # Otherwise strip to ligands only
    dataset_transform = None if args.pocket_cond else lambda system: system.ligand
    dataset = ComplexDataset(complexes, transform=dataset_transform)
    if hparams["max_size"] is not None:
        dataset = dataset.select(lambda system: len(system.ligand) <= hparams["max_size"])

    qed_feat = Features.Qed()
    n_before = len(dataset)
    dataset = dataset.select(lambda system: qed_feat(system.ligand) >= MIN_QED)
    n_filtered = n_before - len(dataset)
    if n_filtered > 0:
        print(f"Filtered {n_filtered}/{n_before} systems with ligand QED less than {MIN_QED}")

    if args.n_mols is not None:
        dataset = dataset.sample(min(args.n_mols, len(dataset)))

    if args.n_replicates > 1:
        dataset = dataset.replicate(args.n_replicates)

    print(f"Remaining systems: {len(dataset)}")

    return dataset, dataset._data


def build_dm(args, hparams, dataset):
    eval_interpolant = GraphInterpolant(
        GraphNoise(cat_noise="uniform", zero_com=True),
        pad_to=hparams["max_size"],
        pad_coord_mode=hparams["train-pad-coord-mode"]
    )

    dm = InterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        train_interpolant=None,
        val_interpolant=None,
        test_interpolant=eval_interpolant
    )
    return dm


def print_sampling_args(args, integrator):
    print("\n***** Sampling Setup *****\n")

    print(f"Running sampling on SPINDR dataset for model at {args.ckpt_path}\n")

    print("Molecular integrator hyperparameters:")
    for hparam, val in integrator.hparams.items():
        print(f"{hparam:<30} {val}")
    print()

    conditioning_args = {
        "Pocket conditioning": args.pocket_cond,
        "Ligand conditioning type": args.ligand_cond,
        "Shape standard dev": args.shape_std_dev,
        "Mean pairwise RMSD conditioning": args.pair_rmsd_cond,
        "PSA3D conditioning": args.psa3d_cond,
        "CFG gamma": args.cfg_gamma
    }

    print("Conditioning arguments:")
    for hparam, val in conditioning_args.items():
        print(f"{hparam:<25} {val}")
    print()

    print("\n**************************\n")


def print_results(results, alignment_results, ensemble_results):
    print("\n******** Evaluation Results ********\n")

    print("Standard generation results:")
    for metric, val in results.items():
        print(f"{metric:<18}{val:.4f}")
    print()

    print("Alignment results:")
    for metric, val in alignment_results.items():
        print(f"{metric:<24}{val:.4f}")
    print()

    print("Ensemble results:")
    for metric, val in ensemble_results.items():
        print(f"{metric:<24}{val:.4f}")
    print()


def main(args):
    print("Running enscondflow evaluation script...")

    # Set some useful torch properties
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 128
    torch.set_float32_matmul_precision("high")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    if args.ligand_cond not in {"shape", "pharma", "profile", "none"}:
        raise ValueError("Conditioning type must be in ['shape', 'pharma', 'profile', 'none']")

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    hparams = checkpoint["hyper_parameters"]

    print("Building evaluation integrator...")

    integrator = util.load_integrator(
        hparams,
        integration_steps=args.n_inf_steps,
        step_size=args.step_size,
        cat_strategy=args.cat_strategy,
        cat_noise_level=args.cat_noise_level
    )

    print("Integrator complete.")
    print("Loading model weights...")

    model = util.load_model(args.ckpt_path, integrator, arch="hybrid")
    model = model.eval().to(util.get_device())

    if args.pocket_cond and not model.encoder.include_pocket:
        raise ValueError("--pocket_cond requires a checkpoint with a pocket encoder")

    print("Model complete.")
    print("Loading datamodule...")

    dataset, complexes = load_dataset(args, hparams)
    datamodule = build_dm(args, hparams, dataset)
    test_dl = datamodule.test_dataloader()

    # Extract reference data — profiles and data mols from prepared complexes
    ref_profiles = [system.ligand.meta["features"]["profile"] for system in complexes]
    data_mols = [system.ligand.to_rdkit(sanitise=True) for system in complexes]

    print("Datamodule complete.")
    print_sampling_args(args, integrator)
    print("Sampling molecules...")

    mols, _ = util.generate_molecules(
        model,
        test_dl,
        args.cfg_gamma,
        profile_cond=args.ligand_cond,
        sanitise=True
    )

    assert len(mols) == len(data_mols)

    print("Sampling complete.")
    print("Scoring molecules...")

    results, results_pm = util.score_molecules(mols, data_mols, return_per_mol=True)
    align_results, align_pm = util.score_alignments(
        mols,
        data_mols,
        ref_profiles,
        optimise=True,
        max_opt_iters=100,
        recovery_dist=args.recovery_dist,
        return_per_mol=True
    )
    ensemble_results, ensemble_pm = util.score_ensembles(mols, n_threads=16, max_opt_iters=1000, return_per_mol=True)

    all_results = {**results, **align_results, **ensemble_results}
    per_mol = {**results_pm, **align_pm, **ensemble_pm}
    print_results(results, align_results, ensemble_results)

    if args.dock:
        dock_results, dock_pm = util.score_pose_quality(
            mols, data_mols, complexes, box_padding=args.dock_box_padding, return_per_mol=True
        )
        all_results.update(dock_results)
        per_mol.update(dock_pm)

        print("Docking results:")
        for metric, val in dock_results.items():
            if isinstance(val, float):
                print(f"{metric:<24}{val:.4f}")
        print()

    if args.output_dir:
        systems = complexes if (args.dock or args.pocket_cond) else None
        util.write_output(args.output_dir, mols, data_mols, all_results, systems=systems, per_mol=per_mol)

    print("***** Evaluation script complete *****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_mols", type=int, default=None)
    parser.add_argument("--n_replicates", type=int, default=1)

    # Flow matching sampling args
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--step_size", type=str, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--n_inf_steps", type=int, default=DEFAULT_N_INF_STEPS)
    parser.add_argument("--cat_strategy", type=str, default=DEFAULT_CAT_STRATEGY)
    parser.add_argument("--cat_noise_level", type=int, default=DEFAULT_CAT_NOISE_LEVEL)

    # Conditioning and CFG args
    # Note pocket_cond is a true/false flag
    # ligand_cond can be 'none' (pocket only), 'shape', 'pharma', 'profile' (default, both shape and pharmacophores)
    parser.add_argument("--pocket_cond", action="store_true", default=False)
    parser.add_argument("--ligand_cond", type=str, default=DEFAULT_LIGAND_COND)
    parser.add_argument("--shape_std_dev", type=float, default=DEFAULT_SHAPE_STD_DEV)
    parser.add_argument("--pair_rmsd_cond", type=float, default=DEFAULT_PAIR_RMSD_COND)
    parser.add_argument("--psa3d_cond", type=float, default=DEFAULT_PSA3D_COND)
    parser.add_argument("--cfg_gamma", type=float, default=DEFAULT_CFG_GAMMA)

    # Evaluation args
    parser.add_argument("--recovery_dist", type=float, default=DEFAULT_RECOVERY_DIST)

    # Docking args
    parser.add_argument("--dock", action="store_true", default=False)
    parser.add_argument("--dock_box_padding", type=float, default=DEFAULT_DOCK_BOX_PADDING)

    # Output args
    help_str = "Directory to write generated mols, results JSON, and structures"
    parser.add_argument("--output_dir", type=str, default=None, help=help_str)

    args = parser.parse_args()
    main(args)
