import pickle
import warnings
import argparse
from rdkit import Chem
from pathlib import Path
from more_itertools import grouper
from concurrent.futures import ProcessPoolExecutor
from rdkit.Chem import rdMolTransforms, rdMolAlign

import enscondflow.scriptutil as util
import enscondflow.util.rdkit as smolRD
from enscondflow.repr.mol import GraphMol, GraphBatch


DEFAULT_N_WORKERS = 16
DEFAULT_SHARD_SIZE = 20000
DEFAULT_DROP_CONF_THRESHOLD = 0.0


def process_mol(filepath, conf_threshold, return_metadata=True):
    warnings.filterwarnings("ignore")
    util.disable_lib_stdout()

    file_bytes = filepath.read_bytes()
    mol_data = pickle.loads(file_bytes)
    confs = mol_data["conformers"]

    ref_mol = None
    ref_atoms = None
    ref_smiles = None

    metadata = {
        "total_confs": 0,
        "invalid conf": 0,
        "missmatch": 0,
        "missing weight": 0,
        "low boltzmann weight": 0
    }

    for conf in confs:
        mol = conf["rd_mol"]
        weight = conf.get("boltzmannweight")

        assert mol.GetNumConformers() == 1

        metadata["total_confs"] += 1

        # Reject missing weights before comparing them with the threshold.
        if weight is None:
            metadata["missing weight"] += 1
            continue

        # Keep the historical strict cutoff (zero is dropped at the default).
        if weight <= conf_threshold:
            metadata["low boltzmann weight"] += 1
            continue

        # Skip any conf that is disconnected or invalid
        if not smolRD.mol_is_valid(mol, with_hs=True, connected=True):
            metadata["invalid conf"] += 1
            continue

        mol_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        mol_smiles = Chem.MolToSmiles(mol, canonical=True)

        # Keep the first valid mol as the reference to compare to
        if ref_mol is None:
            ref_mol = mol
            ref_atoms = mol_atoms
            ref_smiles = mol_smiles

            mol_conf = mol.GetConformer()
            mol_conf.SetProp("weight", str(weight))
            rdMolTransforms.CanonicalizeConformer(mol_conf)

        # If we find a match, align the conf (to first in ref) and add that conf to the ref mol
        elif (ref_smiles == mol_smiles) and (ref_atoms == mol_atoms):
            rdMolAlign.AlignMol(prbMol=mol, refMol=ref_mol)
            mol_conf = mol.GetConformer()
            mol_conf.SetProp("weight", str(weight))
            ref_mol.AddConformer(mol_conf, assignId=True)

        else:
            metadata["missmatch"] += 1

    mol = None
    if ref_mol is not None:
        # Try to create the GraphMol object
        # Create the smiles by loading back to rdkit so it will be consistent with downstream use
        try:
            mol = GraphMol.from_rdkit(ref_mol)
            regen_mol = Chem.RemoveHs(mol.to_rdkit())
            smi = Chem.MolToSmiles(regen_mol)

            assert regen_mol is not None
            assert smi not in [None, ""]
            assert mol.confs.weights is not None

            mol.meta["smiles"] = smi
            mol.meta["n_atoms"] = ref_mol.GetNumAtoms()
            mol.meta["n_heavy_atoms"] = ref_mol.GetNumHeavyAtoms()

        except:
            mol = None

    if return_metadata:
        return mol, metadata

    return mol


def process_shard(filepaths, n_workers, conf_threshold):
    executor = ProcessPoolExecutor(n_workers)
    futures = [executor.submit(process_mol, mol_path, conf_threshold) for mol_path in filepaths]

    mols = []
    metas = []

    # Get the result of the processing and lookup the split within the main process
    for future in futures:
        mol, meta = future.result()

        if mol is None:
            meta["dropped reason"] = "invalid confs"
            metas.append(meta)
            continue

        mols.append(mol)
        metas.append(meta)

    executor.shutdown()

    batch = GraphBatch(mols)
    return batch, metas


def save_shard(save_path, batch, index):
    shard_path = save_path / f"{index}.hdf5"
    batch.save_hdf5_shard(shard_path)


def print_metadata(metadatas):
    dropped_mols = {
        "invalid confs": 0
    }
    dropped_confs = {
        "invalid conf": 0,
        "missmatch": 0,
        "missing weight": 0,
        "low boltzmann weight": 0
    }
    splits = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    for meta in metadatas:
        dropped_reason = meta.get("dropped reason")
        if dropped_reason is None:
            continue

        dropped_mols[dropped_reason] += 1

        for conf_reason in dropped_confs.keys():
            dropped_confs[conf_reason] += meta[conf_reason]

        if "split" in meta:
            splits[meta["split"]] += 1

    total_mols = len(metadatas)
    total_confs = sum([meta["total_confs"] for meta in metadatas])

    print(dropped_mols)

    n_drop_mols = sum(dropped_mols.values())
    n_drop_confs = sum(dropped_confs.values())

    print(f"Total dropped molecules - {n_drop_mols} / {total_mols}. Reasons:")
    for key, val in dropped_mols.items():
        print(f"{key} -- {val}")

    print(f"Total dropped conformers - {n_drop_confs} / {total_confs}. Reasons:")
    for key, val in dropped_confs.items():
        print(f"{key} -- {val}")

    print("Molecule splits within shard:")
    for key, val in splits.items():
        print(f"{key} -- {val}")


def main(args):
    warnings.filterwarnings("ignore")
    util.disable_lib_stdout()

    data_path = Path(args.data_path)
    pickles_path = (data_path / "raw") / "pickles"
    save_path = data_path / "processed"

    print("Running data preprocessing script...")
    print(f"Processing data from", str(pickles_path))

    mol_paths = sorted(p for p in pickles_path.iterdir() if p.suffix == ".pickle")
    print(f"Found {len(mol_paths)} molecule pickle files.")

    splits = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    save_path.mkdir(exist_ok=True, parents=False)

    for idx, group_paths in enumerate(list(grouper(mol_paths, args.shard_size))):
        group_paths = [p for p in group_paths if p is not None]

        print(f"\nProcessing shard {idx}...")
        batch, proc_metadatas = process_shard(
            group_paths,
            args.n_workers,
            args.drop_conf_threshold
        )
        save_shard(save_path, batch, idx)

        for mol in batch:
            if "split" in mol.meta:
                splits[mol.meta["split"]] += 1

        print(f"Shard {idx} processing complete. Stats for shard {idx}:")
        print_metadata(proc_metadatas)

    print("Dataset processing complete.")
    print(f"Data saved to", str(save_path))

    print("Dataset splits:")
    for split, val in splits.items():
        print(f"{split} -- {val}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--n_workers", type=int, default=DEFAULT_N_WORKERS)
    parser.add_argument("--shard_size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--drop_conf_threshold", type=float, default=DEFAULT_DROP_CONF_THRESHOLD)

    args = parser.parse_args()
    main(args)
