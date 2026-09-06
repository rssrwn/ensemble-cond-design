"""Sample a witness pool of train molecules for the multi-cond benchmark.

Groups train molecules by exact heavy-atom count and samples a fixed number per count, then runs MMFF ensemble
sampling on each. Saves the result as a single GraphBatch with all confs per mol.

Heavy-atom counts considered: [16, 36). Anything outside this range is dropped.
"""

import warnings
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem

import enscondflow.scriptutil as util
import enscondflow.util.geometry as Geom
from enscondflow.repr.mol import GraphMol, GraphBatch
from enscondflow.repr.confs import ConfSet
from enscondflow.data import GraphDataset


N_HEAVY_ATOMS_LO = 16
N_HEAVY_ATOMS_HI = 36
DEFAULT_N_PER_COUNT = 100
DEFAULT_N_CONFS = 128
DEFAULT_MAX_OPT_ITERS = 1000
DEFAULT_STRAIN_FILTER = 6.0
DEFAULT_N_WORKERS = 8
SAMPLE_SEED = 42


def sample_ensemble_worker(mol, n_confs, max_opt_iters, strain_filter):
    warnings.filterwarnings("ignore")
    util.disable_lib_stdout()

    rdkit_mol = mol.to_rdkit(sanitise=True)
    if rdkit_mol is None:
        return None

    try:
        result = Geom.sample_ensemble(
            rdkit_mol,
            max_confs=n_confs,
            max_opt_iters=max_opt_iters,
            strain_filter=strain_filter,
            n_threads=1
        )
    except Exception:
        return None

    if result is None:
        return None

    emb_mol, _, _ = result
    if emb_mol.GetNumConformers() == 0:
        return None

    emb_mol = Chem.RemoveAllHs(emb_mol)
    graph_mol = GraphMol.from_rdkit(emb_mol)
    graph_mol.meta = dict(mol.meta) if mol.meta else {}
    graph_mol.meta["n_confs"] = emb_mol.GetNumConformers()
    graph_mol.confs = ConfSet(graph_mol.confs.coords)
    return graph_mol


def main(args):
    warnings.filterwarnings("ignore")
    util.disable_lib_stdout()
    util.configure_fs()

    data_path = Path(args.data_path)
    multi_path = Path(args.multi_path)
    multi_path.mkdir(exist_ok=True, parents=True)
    save_path = multi_path / "witnesses.hdf5"

    print("Loading dataset...")
    dataset = GraphDataset.load(data_path)
    train_mols = [mol for mol in dataset._data if mol.meta.get("split") == "train"]
    print(f"Found {len(train_mols)} train molecules.")

    print("Grouping train mols by heavy-atom count...")
    by_count = {n: [] for n in range(N_HEAVY_ATOMS_LO, N_HEAVY_ATOMS_HI)}
    for idx, mol in enumerate(train_mols):
        n = mol.n_heavy_atoms
        if n in by_count:
            by_count[n].append(idx)

    for n in sorted(by_count.keys()):
        print(f"  n={n:>2}  {len(by_count[n]):>6} train mols")

    rng = np.random.default_rng(SAMPLE_SEED)
    selected_idxs = []
    for n in sorted(by_count.keys()):
        idxs = by_count[n]
        if len(idxs) == 0:
            print(f"  Warning: no train mols with n_heavy_atoms={n}, skipping.")
            continue

        k = min(args.n_per_count, len(idxs))
        chosen = rng.choice(len(idxs), k, replace=False)
        selected_idxs.extend([idxs[i] for i in chosen])

    print(f"\nSelected {len(selected_idxs)} witness mols across {N_HEAVY_ATOMS_HI - N_HEAVY_ATOMS_LO} atom counts.")

    print("Reading mols into memory...")
    selected_mols = [train_mols[i].read() for i in tqdm(selected_idxs, desc="Loading")]

    print("Sampling MMFF ensembles...")
    executor = ProcessPoolExecutor(args.n_workers)
    futures = [
        executor.submit(
            sample_ensemble_worker,
            mol,
            args.n_confs,
            args.max_opt_iters,
            args.strain_filter
        )
        for mol in selected_mols
    ]

    witness_mols = []
    n_failed = 0
    for future in tqdm(as_completed(futures), total=len(futures), desc="Ensembles"):
        result = future.result()
        if result is None:
            n_failed += 1
            continue

        witness_mols.append(result)

    executor.shutdown()

    print(f"\nSampled ensembles for {len(witness_mols)} / {len(selected_mols)} witnesses ({n_failed} failed).")

    if len(witness_mols) == 0:
        print("No witnesses produced. Aborting.")
        return

    n_confs_per = [m.n_conformers for m in witness_mols]
    print(f"Confs/mol — mean: {np.mean(n_confs_per):.1f}, median: {int(np.median(n_confs_per))}, "
          f"min: {min(n_confs_per)}, max: {max(n_confs_per)}")

    by_count = {}
    for mol in witness_mols:
        by_count.setdefault(mol.n_heavy_atoms, []).append(mol)

    for n in sorted(by_count.keys()):
        print(f"  n={n:>2}  {len(by_count[n])} witnesses")

    batch = GraphBatch(witness_mols)
    batch.save_hdf5_shard(save_path)
    print(f"\nSaved {len(witness_mols)} witnesses to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True, help="Path to source dataset (with train/test splits)")
    parser.add_argument("--multi_path", type=str, required=True,
                        help="Multi-cond eval dir; witnesses.hdf5 will be written inside it")

    parser.add_argument("--n_per_count", type=int, default=DEFAULT_N_PER_COUNT)
    parser.add_argument("--n_confs", type=int, default=DEFAULT_N_CONFS)
    parser.add_argument("--max_opt_iters", type=int, default=DEFAULT_MAX_OPT_ITERS)
    parser.add_argument("--strain_filter", type=float, default=DEFAULT_STRAIN_FILTER)
    parser.add_argument("--n_workers", type=int, default=DEFAULT_N_WORKERS)

    args = parser.parse_args()
    main(args)
