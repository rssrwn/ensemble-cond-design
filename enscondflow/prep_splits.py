"""Create reproducible GEOM or QM9 splits from processed HDF5 shards."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold

from enscondflow.data import GraphDataset
from enscondflow.repr import GraphBatch


def assign_splits(mols, dataset="geom", n_test=1000, n_val=10000, seed=12345):
    """Return labels, using the source notebook's filters and an explicit RNG.

    GEOM test candidates have singleton scaffolds, 16–35 heavy atoms, logP < 5,
    and at least 10 conformers. Validation is random among remaining molecules.
    """
    if dataset not in ("geom", "qm9"):
        raise ValueError("dataset must be geom or qm9")
    if n_test < 0 or n_val < 0 or n_test + n_val > len(mols):
        raise ValueError("Requested split sizes exceed the dataset or are negative")
    candidates = np.arange(len(mols))
    if dataset == "geom":
        rd_mols = [m.drop_3d().to_rdkit(sanitise=True) for m in mols]
        if any(m is None for m in rd_mols):
            raise ValueError("Dataset contains invalid molecules")
        scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in rd_mols]
        counts = Counter(scaffolds)
        candidates = np.array([
            i for i, (mol, scaffold) in enumerate(zip(rd_mols, scaffolds))
            if counts[scaffold] == 1 and 16 <= mol.GetNumHeavyAtoms() <= 35
            and Crippen.MolLogP(mol) < 5.0 and mols[i].n_conformers >= 10
        ], dtype=int)
    if n_test > len(candidates):
        raise ValueError(f"Only {len(candidates)} eligible test molecules; requested {n_test}")
    rng = np.random.default_rng(seed)
    test = rng.choice(candidates, n_test, replace=False)
    remaining = np.setdiff1d(np.arange(len(mols)), test)
    val = rng.choice(remaining, n_val, replace=False)
    labels = np.full(len(mols), "train", dtype="<U5")
    labels[test] = "test"
    labels[val] = "val"
    return labels.tolist()


def main(args):
    if args.shard_size < 1:
        raise ValueError("shard_size must be positive")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Use a new output directory: {destination}")
    dataset = GraphDataset.load(Path(args.data_path))
    try:
        labels = assign_splits(dataset, args.dataset, args.n_test, args.n_val, args.seed)
        destination.mkdir(parents=True)
        manifest = []
        for start in range(0, len(dataset), args.shard_size):
            mols = []
            for i in range(start, min(start + args.shard_size, len(dataset))):
                mol = dataset[i]
                mol.meta = dict(mol.meta or {})
                mol.meta["split"] = labels[i]
                mols.append(mol)
                manifest.append({"index": i, "smiles": mol.meta.get("smiles"), "split": labels[i]})
            GraphBatch(mols).save_hdf5_shard(destination / f"{start // args.shard_size:05d}.hdf5")
        (destination / "splits.json").write_text(json.dumps(
            {"config": vars(args), "counts": dict(Counter(labels)), "molecules": manifest}, indent=2) + "\n")
        print(dict(Counter(labels)))
    finally:
        dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=["geom", "qm9"], default="geom")
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--shard_size", type=int, default=20000)
    main(parser.parse_args())
