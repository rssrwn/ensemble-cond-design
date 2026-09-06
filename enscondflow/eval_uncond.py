"""Unconditional generation evaluation script.

Generates molecules from the prior with all conditioning dropped and reports the size distribution
alongside standard generation quality metrics. Used for the size-learning ablation, comparing models
trained with different couplings between the prior and the padded data molecule.

Molecular size is read from the model's own pad-token predictions rather than from RDKit, so it is
defined even for molecules which fail sanitisation. The prior is a plain Gaussian at the model's
padded size and is built directly rather than through a dataloader, so no dataset information can
enter the samples.

Generated molecules are independent draws, so uncertainty is estimated by bootstrapping over the
saved per-molecule records rather than by running seeded replicates.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import lightning as L
from tqdm import tqdm
from rdkit import Chem

import enscondflow.scriptutil as util
from enscondflow.repr import GraphBatch, AtomVocab
from enscondflow.repr.util import PAD_TOKEN, MASK_TOKEN
from enscondflow.data import GraphNoise
from enscondflow.data.datamodules import graph_batch_to_dict


DEFAULT_N_MOLS = 1000
DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 12345


def sample_prior(hparams, n_mols, n_atoms):
    """Sample prior molecules at the model's padded size, with no dependence on any dataset."""

    prior_sampler = GraphNoise(
        cat_noise=hparams["val-prior-cat-noise"],
        zero_com=hparams["val-prior-zero-com"]
    )
    mols = [prior_sampler.sample_molecule(n_atoms, 1) for _ in range(n_mols)]
    return graph_batch_to_dict(GraphBatch(mols))


def generated_sizes(gen):
    """Number of atoms the model predicted as real, per molecule.

    Matches MolBuilder's token extraction, so this is the size of the molecule the builder will try
    to construct, whether or not RDKit can then sanitise it.
    """

    atomics = gen["atomics"]
    indices = torch.argmax(atomics, dim=-1) if atomics.dim() == 3 else atomics
    pad_idx = AtomVocab.get_index(PAD_TOKEN)
    mask_idx = AtomVocab.get_index(MASK_TOKEN)

    real = (indices != pad_idx) & (indices != mask_idx) & (gen["mask"] == 1)
    return real.sum(dim=-1).cpu().tolist()


def generate_uncond(model, hparams, n_mols, n_atoms, batch_size):
    mols = []
    sizes = []

    for start in tqdm(range(0, n_mols, batch_size), desc="Generating"):
        n_batch = min(batch_size, n_mols - start)
        prior = util.to_device(sample_prior(hparams, n_batch, n_atoms), model.device)
        ada_latents = model.null_ada_latents(n_batch, model.device)

        gen = model.generate(
            prior,
            None,
            None,
            ada_latents,
            model.integrator.steps,
            step_strategy=model.integrator.step_size,
            cfg_gamma=0.0
        )

        sizes.extend(generated_sizes(gen))
        mols.extend(model.generate_mols(gen, sanitise=True))

    return mols, sizes


def _smiles_or_none(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveAllHs(mol))
    except Exception:
        return None


def score_mols(mols, sizes):
    """Aggregate metrics plus one record per generated molecule."""

    results, per_mol = util.score_molecules(mols, return_per_mol=True)

    records = []
    for i, size in enumerate(sizes):
        records.append({
            "size": int(size),
            "valid": per_mol["validity"][i],
            "connected": per_mol["fc-validity"][i],
            "qed": per_mol["qed"][i],
            "smiles": _smiles_or_none(mols[i])
        })

    return {k: float(v) for k, v in results.items()}, records


def print_results(metrics, sizes, n_boot=1000):
    """Print aggregate metrics, with a bootstrap 95% interval on the mean size as a sanity check.

    The full uncertainty analysis is done in the notebook from the saved per-molecule records.
    """

    sizes = np.array(sizes)
    boot_means = [np.random.choice(sizes, size=len(sizes), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    print("\n***** Unconditional Generation Results *****\n")

    for metric, val in metrics.items():
        print(f"{metric:<18}{val:.4f}")
    print()

    print(f"{'size mean':<18}{sizes.mean():.2f}  [{lo:.2f}, {hi:.2f}]")
    print(f"{'size median':<18}{np.median(sizes):.1f}")
    print(f"{'size std':<18}{sizes.std():.2f}")
    print(f"{'size min/max':<18}{sizes.min()} / {sizes.max()}")
    print(f"{'frac at ceiling':<18}{(sizes >= 48).mean():.4f}")
    print()


def save_results(results_path, args, hparams, metrics, records):
    payload = {
        "config": {**vars(args), "ckpt_path": str(args.ckpt_path)},
        "model": {
            "max_size": hparams["max_size"],
            "perm_ot": hparams["train-perm-ot"],
            "pad_coord_mode": hparams["train-pad-coord-mode"],
            "include_pocket": hparams.get("enc-include_pocket", False)
        },
        "metrics": metrics,
        "mols": records
    }

    path = Path(results_path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved per-molecule results to {path}")


def main(args):
    print("Running unconditional evaluation script...")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 128
    torch.set_float32_matmul_precision("high")

    L.seed_everything(args.seed)
    util.disable_lib_stdout()
    util.configure_fs()

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    hparams = checkpoint["hyper_parameters"]

    n_atoms = hparams["max_size"]
    if n_atoms is None:
        raise ValueError("Checkpoint was trained without a padded size, so it cannot generate unconditionally.")

    print("Building integrator...")
    integrator = util.load_integrator(hparams)

    print("Loading model...")
    model = util.load_model(args.ckpt_path, integrator, arch="hybrid")
    model = model.eval().to(util.get_device())

    print("\n***** Sampling Setup *****\n")
    setup = {
        "Checkpoint": args.ckpt_path,
        "Perm OT": hparams["train-perm-ot"],
        "Pad coord mode": hparams["train-pad-coord-mode"],
        "Prior size": n_atoms,
        "Num mols": args.n_mols,
        "Seed": args.seed,
        "Integration steps": integrator.steps,
        "Step size": integrator.step_size
    }
    for hparam, val in setup.items():
        print(f"{hparam:<25} {val}")
    print("\n**************************\n")

    print("Sampling molecules...")
    mols, sizes = generate_uncond(model, hparams, args.n_mols, n_atoms, args.batch_size)

    print("Scoring molecules...")
    metrics, records = score_mols(mols, sizes)
    print_results(metrics, sizes)

    if args.results_path is not None:
        save_results(args.results_path, args, hparams, metrics, records)

    print("***** Unconditional evaluation complete *****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, default=None, help="Path to save per-molecule results JSON")

    parser.add_argument("--n_mols", type=int, default=DEFAULT_N_MOLS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()
    main(args)
