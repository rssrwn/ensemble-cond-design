"""Prepare the multi-cond benchmark from scratch (replaces prep_multi.py).

Pipeline (witnesses must be pre-computed via prep_witnesses.py):
  1. Load test mols, filter to N_HEAVY_ATOMS_LO <= n_heavy_atoms < N_HEAVY_ATOMS_HI
  2. Sample MMFF ensembles per mol (parallel) — or load cached pool if --reuse_test_pool
  3. Pick compact (low-percentile RoG) and extended (high-percentile RoG) conf per mol; save snapshot to test_pool/
  4. Load witness pool
  5. Compute [n_test, n_wit] shape tani matrices (compact and extended targets vs witness ensembles)
  6. Compute [n_test, n_wit] ECFP tani matrix (source mols vs witnesses)
  7. Compute per-target shape baselines = mean over ±n_atoms_tol witnesses (no chem filter)
  8. Find cross-mol partners with ECFP guard (drop witnesses with ecfp tani > tau_chem to either source mol):
     - pos_pos: best witness with high joint shape tani to compact_A and extended_B
     - pos_neg bidirectional: w1 (high-A / low-B) AND w2 (high-B / low-A) for the same partner
  9. Save pairs/ with witness_tanis.npz (raw [n_test, n_wit] matrices) and pos_pos/, pos_neg/ subdirs
     (compact.hdf5, extended.hdf5, metadata.json with witness + baseline scores)
"""

import json
import warnings
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import enscondflow.scriptutil as util
import enscondflow.util.geometry as Geom
from enscondflow.repr.mol import GraphMol, GraphBatch
from enscondflow.repr.confs import ConfSet
from enscondflow.data import GraphDataset


N_HEAVY_ATOMS_LO = 16
N_HEAVY_ATOMS_HI = 36

DEFAULT_N_CONFS = 128
DEFAULT_MAX_OPT_ITERS = 1000
DEFAULT_STRAIN_FILTER = 6.0

DEFAULT_ROG_LO_PCT = 20.0
DEFAULT_ROG_HI_PCT = 80.0

DEFAULT_TAU_HIGH = 0.75
DEFAULT_TAU_LOW = 0.5
DEFAULT_TAU_CHEM = 0.5
DEFAULT_N_ATOMS_TOL = 1

DEFAULT_ECFP_RADIUS = 2
DEFAULT_ECFP_BITS = 2048

DEFAULT_N_WORKERS = 10
DEFAULT_TANI_CHUNK_SIZE = 500
PAIR_SEED = 123


def _radius_of_gyration(coords):
    center = coords.mean(axis=0)
    return np.sqrt(((coords - center) ** 2).sum(axis=1).mean())


def _heavy_atom_coords(rdkit_mol, conf_idx):
    conf = rdkit_mol.GetConformer(conf_idx)
    positions = np.array(conf.GetPositions())
    heavy_idxs = [a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetAtomicNum() != 1]
    return positions[heavy_idxs]


def sample_test_mol(mol, n_confs, max_opt_iters, strain_filter, rog_lo_pct, rog_hi_pct):
    """Worker: sample MMFF ensemble for a test mol, pick compact and extended confs by RoG percentile.

    Returns (compact_graph, extended_graph, info) on success, or string status on failure.
    """

    warnings.filterwarnings("ignore")
    util.disable_lib_stdout()

    rdkit_mol = mol.to_rdkit(sanitise=True)
    if rdkit_mol is None:
        return "ensemble_fail"

    try:
        result = Geom.sample_ensemble(
            rdkit_mol,
            max_confs=n_confs,
            max_opt_iters=max_opt_iters,
            strain_filter=strain_filter,
            n_threads=1
        )
    except Exception:
        return "ensemble_fail"

    if result is None:
        return "ensemble_fail"

    emb_mol, _, _ = result
    n_emb = emb_mol.GetNumConformers()
    if n_emb < 2:
        return "ensemble_fail"

    rogs = [_radius_of_gyration(_heavy_atom_coords(emb_mol, i)) for i in range(n_emb)]
    order = np.argsort(rogs)
    compact_idx = int(order[int(round((rog_lo_pct / 100.0) * (n_emb - 1)))])
    extended_idx = int(order[int(round((rog_hi_pct / 100.0) * (n_emb - 1)))])
    if compact_idx == extended_idx:
        return "ensemble_fail"

    compact_mol = Chem.Mol(emb_mol)
    compact_mol.RemoveAllConformers()
    compact_mol.AddConformer(emb_mol.GetConformer(compact_idx), assignId=True)

    extended_mol = Chem.Mol(emb_mol)
    extended_mol.RemoveAllConformers()
    extended_mol.AddConformer(emb_mol.GetConformer(extended_idx), assignId=True)

    def make_graph(rdkit_single_conf, label):
        graph_mol = GraphMol.from_rdkit(rdkit_single_conf)
        graph_mol.meta = dict(mol.meta) if mol.meta else {}
        graph_mol.meta["conf_label"] = label
        graph_mol.confs = ConfSet(graph_mol.confs.coords)
        return graph_mol

    compact_graph = make_graph(compact_mol, "compact")
    extended_graph = make_graph(extended_mol, "extended")

    info = {
        "smiles": mol.meta.get("smiles", ""),
        "rog_compact": rogs[compact_idx],
        "rog_extended": rogs[extended_idx],
        "n_ensemble_confs": n_emb
    }
    return compact_graph, extended_graph, info


def _to_rdkit_no_h(graph_mol):
    rdkit_mol = graph_mol.to_rdkit(sanitise=True)
    if rdkit_mol is None:
        return None

    return Chem.RemoveAllHs(rdkit_mol)


def _shape_tani_chunk(witness_pickles, target_pickles):
    """Worker — process a chunk of (witness, target) pairs, return list of best-conf shape tanis."""

    warnings.filterwarnings("ignore")
    util.disable_lib_stdout()

    results = []
    for w_pickle, t_pickle in zip(witness_pickles, target_pickles):
        try:
            witness = Chem.Mol(w_pickle)
            target = Chem.Mol(t_pickle)
            _, shape_tani, _ = Geom.align_best_conf(witness, target, align_weight=1.0)
        except Exception:
            shape_tani = 0.0

        results.append(float(shape_tani))

    return results


def compute_shape_tani_matrix(test_mols, witnesses, n_atoms_tol, n_workers, chunk_size, label):
    """Compute [n_test, n_witnesses] shape tanimoto matrix, only filling entries within ±tol heavy atoms.

    Entries outside the tolerance window are NaN. Jobs are batched in chunks to amortize pool dispatch overhead.
    """

    print(f"Preparing rdkit mols for {label}...")
    test_rdkit = [_to_rdkit_no_h(m) for m in test_mols]
    witness_rdkit = [_to_rdkit_no_h(m) for m in witnesses]

    n_test_failed = sum(1 for m in test_rdkit if m is None)
    n_wit_failed = sum(1 for m in witness_rdkit if m is None)
    if n_test_failed:
        print(f"  Warning: {n_test_failed}/{len(test_mols)} {label} test mols failed rdkit conversion (will be NaN).")

    if n_wit_failed:
        print(f"  Warning: {n_wit_failed}/{len(witnesses)} witnesses failed rdkit conversion (will be NaN).")

    test_n_atoms = np.array([m.n_heavy_atoms for m in test_mols])
    witness_n_atoms = np.array([m.n_heavy_atoms for m in witnesses])

    n_test = len(test_mols)
    n_wit = len(witnesses)

    matrix = np.full((n_test, n_wit), np.nan, dtype=np.float32)
    jobs = []
    for i in range(n_test):
        if test_rdkit[i] is None:
            continue

        for k in range(n_wit):
            if witness_rdkit[k] is None:
                continue

            if abs(int(test_n_atoms[i]) - int(witness_n_atoms[k])) > n_atoms_tol:
                continue

            jobs.append((i, k))

    print(f"  {len(jobs)} alignments in chunks of {chunk_size} (±{n_atoms_tol} heavy atoms)...")

    chunks = [jobs[start:start + chunk_size] for start in range(0, len(jobs), chunk_size)]

    executor = ProcessPoolExecutor(n_workers)
    futures = {}
    for chunk in chunks:
        w_pickles = [witness_rdkit[k].ToBinary() for _, k in chunk]
        t_pickles = [test_rdkit[i].ToBinary() for i, _ in chunk]
        future = executor.submit(_shape_tani_chunk, w_pickles, t_pickles)
        futures[future] = chunk

    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Tani {label}"):
        chunk = futures[future]
        results = future.result()
        for (i, k), tani in zip(chunk, results):
            matrix[i, k] = tani

    executor.shutdown()

    return matrix


def compute_ecfp_tani_matrix(test_mols, witnesses, radius, n_bits):
    """Compute [n_test, n_witnesses] ECFP tanimoto matrix between test source mols and witness mols."""

    def _fp(graph_mol):
        rdkit_mol = graph_mol.to_rdkit(sanitise=True)
        if rdkit_mol is None:
            return None

        try:
            return AllChem.GetMorganFingerprintAsBitVect(Chem.RemoveAllHs(rdkit_mol), radius=radius, nBits=n_bits)
        except Exception:
            return None

    test_fps = [_fp(m) for m in test_mols]
    wit_fps = [_fp(m) for m in witnesses]

    n_test_failed = sum(1 for fp in test_fps if fp is None)
    n_wit_failed = sum(1 for fp in wit_fps if fp is None)
    if n_test_failed:
        print(f"  Warning: {n_test_failed}/{len(test_mols)} test mols failed ECFP fingerprint.")

    if n_wit_failed:
        print(f"  Warning: {n_wit_failed}/{len(witnesses)} witnesses failed ECFP fingerprint.")

    n_test = len(test_mols)
    n_wit = len(witnesses)

    matrix = np.full((n_test, n_wit), np.nan, dtype=np.float32)
    valid_wit = [k for k, fp in enumerate(wit_fps) if fp is not None]
    valid_wit_fps = [wit_fps[k] for k in valid_wit]
    if not valid_wit_fps:
        return matrix

    for i, fp_i in enumerate(test_fps):
        if fp_i is None:
            continue

        sims = DataStructs.BulkTanimotoSimilarity(fp_i, valid_wit_fps)
        for k, s in zip(valid_wit, sims):
            matrix[i, k] = float(s)

    return matrix


def find_pos_pos_partner(
    i,
    n_test,
    compact_tani,
    extended_tani,
    ecfp_test_wit,
    tau_high,
    tau_chem,
    n_atoms,
    n_atoms_tol,
    exclude_self=True
):
    n_size_matched = 0
    n_with_valid_w = 0
    n_after_chem_guard = 0
    best_score_seen = -np.inf
    best = None
    for j in range(n_test):
        if exclude_self and j == i:
            continue

        if abs(int(n_atoms[i]) - int(n_atoms[j])) > n_atoms_tol:
            continue

        n_size_matched += 1

        ca = compact_tani[i]
        eb = extended_tani[j]
        valid = ~np.isnan(ca) & ~np.isnan(eb)
        if not valid.any():
            continue

        n_with_valid_w += 1

        ecfp_a = ecfp_test_wit[i]
        ecfp_b = ecfp_test_wit[j]
        ecfp_ok = (~np.isnan(ecfp_a)) & (~np.isnan(ecfp_b)) & (ecfp_a <= tau_chem) & (ecfp_b <= tau_chem)
        valid = valid & ecfp_ok
        if not valid.any():
            continue

        n_after_chem_guard += 1

        scores = np.minimum(ca, eb)
        scores = np.where(valid, scores, -np.inf)
        k = int(np.argmax(scores))
        s = float(scores[k])
        best_score_seen = max(best_score_seen, s)
        if s < tau_high:
            continue

        if best is None or s > best[2]:
            best = (j, k, s)

    return {
        "best": best,
        "n_size_matched": n_size_matched,
        "n_with_valid_w": n_with_valid_w,
        "n_after_chem_guard": n_after_chem_guard,
        "best_score_seen": best_score_seen
    }


def find_pos_neg_partner(
    i,
    n_test,
    compact_tani,
    extended_tani,
    ecfp_test_wit,
    tau_high,
    tau_low,
    tau_chem,
    n_atoms,
    n_atoms_tol
):
    n_size_matched = 0
    n_with_valid_w = 0
    n_after_chem_guard = 0
    n_d1_pass = 0
    n_d2_pass = 0
    n_both_pass = 0
    best = None
    for j in range(n_test):
        if j == i:
            continue

        if abs(int(n_atoms[i]) - int(n_atoms[j])) > n_atoms_tol:
            continue

        n_size_matched += 1

        ca = compact_tani[i]
        eb = extended_tani[j]
        valid = ~np.isnan(ca) & ~np.isnan(eb)
        if not valid.any():
            continue

        n_with_valid_w += 1

        ecfp_a = ecfp_test_wit[i]
        ecfp_b = ecfp_test_wit[j]
        ecfp_ok = (~np.isnan(ecfp_a)) & (~np.isnan(ecfp_b)) & (ecfp_a <= tau_chem) & (ecfp_b <= tau_chem)
        valid = valid & ecfp_ok
        if not valid.any():
            continue

        n_after_chem_guard += 1

        d1_mask = valid & (ca >= tau_high) & (eb <= tau_low)
        d2_mask = valid & (eb >= tau_high) & (ca <= tau_low)
        d1_ok = bool(d1_mask.any())
        d2_ok = bool(d2_mask.any())
        if d1_ok:
            n_d1_pass += 1

        if d2_ok:
            n_d2_pass += 1

        if not (d1_ok and d2_ok):
            continue

        n_both_pass += 1

        d1_gap = np.where(d1_mask, ca - eb, -np.inf)
        w1 = int(np.argmax(d1_gap))

        d2_gap = np.where(d2_mask, eb - ca, -np.inf)
        w2 = int(np.argmax(d2_gap))

        score = float(min(d1_gap[w1], d2_gap[w2]))

        if best is None or score > best[3]:
            best = (j, w1, w2, score)

    return {
        "best": best,
        "n_size_matched": n_size_matched,
        "n_with_valid_w": n_with_valid_w,
        "n_after_chem_guard": n_after_chem_guard,
        "n_d1_pass": n_d1_pass,
        "n_d2_pass": n_d2_pass,
        "n_both_pass": n_both_pass
    }


def _witness_record(witness, partner, ecfp_test_wit, role, key_compact, key_extended):
    k = partner[role]
    i = partner["i"]
    j = partner["j"]
    return {
        "role": role,
        "idx": int(k),
        "smiles": witness.meta.get("smiles", "") if witness.meta else "",
        "compact_tani": float(partner[key_compact]),
        "extended_tani": float(partner[key_extended]),
        "ecfp_a": float(ecfp_test_wit[i, k]),
        "ecfp_b": float(ecfp_test_wit[j, k])
    }


def build_pair_data(
    compact_mols,
    extended_mols,
    partners,
    witnesses,
    baseline_compact,
    baseline_extended,
    ecfp_test_wit,
    mode
):
    """Build paired GraphBatches with lean per-mol meta plus a list of pair records for sidecar JSON."""

    out_compact = []
    out_extended = []
    pair_records = []
    for row, (i, partner) in enumerate(partners):
        partner = {**partner, "i": int(i)}
        j = partner["j"]
        partner_b = extended_mols[j]

        compact_a = compact_mols[i].copy()
        extended_b = partner_b.copy()
        compact_smiles = compact_a.meta.get("smiles", "") if compact_a.meta else ""
        extended_smiles = partner_b.meta.get("smiles", "") if partner_b.meta else ""
        compact_a.meta = {"smiles": compact_smiles, "conf_label": "compact", "row": row, "source_idx": int(i)}
        extended_b.meta = {"smiles": extended_smiles, "conf_label": "extended", "row": row, "source_idx": int(j)}
        out_compact.append(compact_a)
        out_extended.append(extended_b)

        record = {
            "row": row,
            "compact": {
                "source_idx": int(i),
                "smiles": compact_smiles,
                "n_heavy_atoms": int(compact_mols[i].n_heavy_atoms)
            },
            "extended": {
                "source_idx": int(j),
                "smiles": extended_smiles,
                "n_heavy_atoms": int(partner_b.n_heavy_atoms)
            },
            "baseline": {
                "compact_tani": float(baseline_compact[i]),
                "extended_tani": float(baseline_extended[j])
            }
        }

        if mode == "pos_pos":
            partner_renamed = {**partner, "primary": partner["k"]}
            record["witnesses"] = [
                _witness_record(
                    witnesses[partner["k"]],
                    partner_renamed,
                    ecfp_test_wit,
                    "primary",
                    "w_compact_tani",
                    "w_extended_tani"
                )
            ]
        else:
            partner_renamed = {**partner, "compact_pos": partner["w1"], "extended_pos": partner["w2"]}
            record["witnesses"] = [
                _witness_record(
                    witnesses[partner["w1"]],
                    partner_renamed,
                    ecfp_test_wit,
                    "compact_pos",
                    "w1_compact_tani",
                    "w1_extended_tani"
                ),
                _witness_record(
                    witnesses[partner["w2"]],
                    partner_renamed,
                    ecfp_test_wit,
                    "extended_pos",
                    "w2_compact_tani",
                    "w2_extended_tani"
                )
            ]

        pair_records.append(record)

    return GraphBatch(out_compact), GraphBatch(out_extended), pair_records


def save_pair_set(pair_dir, compact_batch, extended_batch, pair_records, mode, args):
    pair_dir.mkdir(exist_ok=True, parents=True)
    compact_batch.save_hdf5_shard(pair_dir / "compact.hdf5")
    extended_batch.save_hdf5_shard(pair_dir / "extended.hdf5")

    metadata = {
        "config": {
            "mode": mode,
            "tau_high": args.tau_high,
            "tau_low": args.tau_low if mode == "pos_neg" else None,
            "tau_chem": args.tau_chem,
            "n_atoms_tol": args.n_atoms_tol
        },
        "pairs": pair_records
    }
    with open(pair_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main(args):
    warnings.filterwarnings("ignore")
    util.disable_lib_stdout()
    util.configure_fs()

    data_path = Path(args.data_path)
    multi_path = Path(args.multi_path)
    witness_path = multi_path / "witnesses.hdf5"
    save_path = multi_path / "pairs"
    save_path.mkdir(exist_ok=True, parents=True)

    test_pool_dir = save_path / "test_pool"
    cached_compact = test_pool_dir / "compact.hdf5"
    cached_extended = test_pool_dir / "extended.hdf5"
    cache_available = cached_compact.exists() and cached_extended.exists()

    if args.reuse_test_pool and cache_available:
        print(f"Reusing cached test pool from {test_pool_dir}/")
        compact_batch = GraphBatch.load_hdf5_shard(cached_compact)
        extended_batch = GraphBatch.load_hdf5_shard(cached_extended)
        compact_mols = [m.read() for m in compact_batch[:]]
        extended_mols = [m.read() for m in extended_batch[:]]
        n_pairs = len(compact_mols)
        print(f"  Loaded {n_pairs} test mol pairs from cache.")

        if args.n_test_mols is not None and args.n_test_mols < n_pairs:
            rng = np.random.default_rng(PAIR_SEED)
            idxs = sorted(rng.choice(n_pairs, args.n_test_mols, replace=False))
            compact_mols = [compact_mols[i] for i in idxs]
            extended_mols = [extended_mols[i] for i in idxs]
            n_pairs = len(compact_mols)
            print(f"  Subsampled cache to {n_pairs} mol pairs.")
    else:
        if args.reuse_test_pool and not cache_available:
            print(f"--reuse_test_pool set but cache missing at {test_pool_dir}/; falling back to MMFF.")

        print("Loading dataset...")
        dataset = GraphDataset.load(data_path)
        test_mols_all = [m for m in dataset._data if m.meta.get("split") == "test"]
        test_mols = [m for m in test_mols_all if N_HEAVY_ATOMS_LO <= m.n_heavy_atoms < N_HEAVY_ATOMS_HI]
        print(f"  {len(test_mols_all)} test mols total, {len(test_mols)} after filter "
              f"[{N_HEAVY_ATOMS_LO}, {N_HEAVY_ATOMS_HI}).")

        if args.n_test_mols is not None and args.n_test_mols < len(test_mols):
            rng = np.random.default_rng(PAIR_SEED)
            idxs = rng.choice(len(test_mols), args.n_test_mols, replace=False)
            test_mols = [test_mols[i] for i in idxs]
            print(f"  Subsampled to {len(test_mols)} mols.")

        print("Reading test mols into memory...")
        test_mols = [m.read() for m in tqdm(test_mols, desc="Loading")]

        print(f"\nSampling MMFF ensembles ({args.n_workers} workers)...")
        executor = ProcessPoolExecutor(args.n_workers)
        futures = [
            executor.submit(
                sample_test_mol,
                mol,
                args.n_confs,
                args.max_opt_iters,
                args.strain_filter,
                args.rog_lo_pct,
                args.rog_hi_pct
            )
            for mol in test_mols
        ]

        compact_mols = []
        extended_mols = []
        infos = []
        drops = {"ensemble_fail": 0}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ensembles"):
            result = future.result()
            if isinstance(result, str):
                drops[result] = drops.get(result, 0) + 1
                continue

            compact_graph, extended_graph, info = result
            compact_mols.append(compact_graph)
            extended_mols.append(extended_graph)
            infos.append(info)

        executor.shutdown()

        n_pairs = len(compact_mols)
        print(f"\nKept {n_pairs} / {len(test_mols)} test mol pairs.")
        print(f"  Drops — ensemble fail: {drops.get('ensemble_fail', 0)}")

        if n_pairs == 0:
            print("No valid pairs. Aborting.")
            return

        rog_compact = [info["rog_compact"] for info in infos]
        rog_extended = [info["rog_extended"] for info in infos]
        print(f"  RoG — compact mean: {np.mean(rog_compact):.3f}, "
              f"extended mean: {np.mean(rog_extended):.3f}")

        test_pool_dir.mkdir(exist_ok=True, parents=True)
        GraphBatch(compact_mols).save_hdf5_shard(cached_compact)
        GraphBatch(extended_mols).save_hdf5_shard(cached_extended)
        print(f"  Saved test pool snapshot to {test_pool_dir}/")

    if n_pairs == 0:
        print("No valid pairs. Aborting.")
        return

    print("\nLoading witness pool...")
    witness_batch = GraphBatch.load_hdf5_shard(witness_path)
    witnesses = [m for m in witness_batch[:]]
    print(f"  {len(witnesses)} witnesses loaded.")

    print("\nComputing witness shape tani vs compact targets...")
    compact_tani = compute_shape_tani_matrix(
        compact_mols,
        witnesses,
        args.n_atoms_tol,
        args.n_workers,
        args.tani_chunk_size,
        "compact"
    )

    print("\nComputing witness shape tani vs extended targets...")
    extended_tani = compute_shape_tani_matrix(
        extended_mols,
        witnesses,
        args.n_atoms_tol,
        args.n_workers,
        args.tani_chunk_size,
        "extended"
    )

    matrix_path = save_path / "witness_tanis.npz"
    np.savez_compressed(matrix_path, compact=compact_tani, extended=extended_tani)
    print(f"Saved witness shape tani matrices to {matrix_path}")

    print("\nComputing ECFP tani (test source mol vs witness)...")
    ecfp_test_wit = compute_ecfp_tani_matrix(compact_mols, witnesses, args.ecfp_radius, args.ecfp_bits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline_compact = np.nanmean(compact_tani, axis=1)
        baseline_extended = np.nanmean(extended_tani, axis=1)

    print(f"Baseline (avg ±{args.n_atoms_tol}-atom witness tani) — "
          f"compact mean: {np.nanmean(baseline_compact):.3f}, "
          f"extended mean: {np.nanmean(baseline_extended):.3f}")

    n_atoms = np.array([m.n_heavy_atoms for m in compact_mols])

    valid_compact = (~np.isnan(compact_tani)).sum(axis=1)
    valid_extended = (~np.isnan(extended_tani)).sum(axis=1)
    print(f"\nWitnesses considered per test mol — compact mean: {valid_compact.mean():.1f}, "
          f"extended mean: {valid_extended.mean():.1f}")
    n_test_no_compact_w = int((valid_compact == 0).sum())
    n_test_no_extended_w = int((valid_extended == 0).sum())
    if n_test_no_compact_w:
        print(f"  Warning: {n_test_no_compact_w} test mols have no in-range compact witnesses.")

    if n_test_no_extended_w:
        print(f"  Warning: {n_test_no_extended_w} test mols have no in-range extended witnesses.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_compact = np.nanmax(compact_tani, axis=1)
        best_extended = np.nanmax(extended_tani, axis=1)

    def _summary(arr, label):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            print(f"  {label}: no valid scores")
            return

        print(f"  {label} best-witness tani — mean: {arr.mean():.3f}, "
              f"median: {np.median(arr):.3f}, p25: {np.percentile(arr, 25):.3f}, "
              f"p75: {np.percentile(arr, 75):.3f}, max: {arr.max():.3f}, "
              f">= τ_high ({args.tau_high}): {(arr >= args.tau_high).sum()}/{len(arr)}")

    _summary(best_compact, "compact_A")
    _summary(best_extended, "extended_A")

    print("\nFinding pos-pos partners...")
    pos_pos_partners = []
    pp_drops = {"no_size_match": 0, "no_valid_w": 0, "no_chem_pass": 0, "below_tau_high": 0}
    pp_best_scores = []
    for i in tqdm(range(n_pairs)):
        diag = find_pos_pos_partner(
            i,
            n_pairs,
            compact_tani,
            extended_tani,
            ecfp_test_wit,
            args.tau_high,
            args.tau_chem,
            n_atoms,
            args.n_atoms_tol
        )
        pp_best_scores.append(diag["best_score_seen"])
        if diag["best"] is not None:
            j, k, score = diag["best"]
            pos_pos_partners.append((i, {
                "j": j,
                "k": k,
                "w_compact_tani": compact_tani[i, k],
                "w_extended_tani": extended_tani[j, k],
                "score": score
            }))
            continue

        if diag["n_size_matched"] == 0:
            pp_drops["no_size_match"] += 1
        elif diag["n_with_valid_w"] == 0:
            pp_drops["no_valid_w"] += 1
        elif diag["n_after_chem_guard"] == 0:
            pp_drops["no_chem_pass"] += 1
        else:
            pp_drops["below_tau_high"] += 1

    print(f"  Kept {len(pos_pos_partners)} / {n_pairs} test mols.")
    print(f"  Drops — no size-matched B: {pp_drops['no_size_match']}, "
          f"no valid witness: {pp_drops['no_valid_w']}, "
          f"no witness passed τ_chem: {pp_drops['no_chem_pass']}, "
          f"best joint score < τ_high: {pp_drops['below_tau_high']}")
    pp_scores_seen = np.array([s for s in pp_best_scores if s > -np.inf])
    if len(pp_scores_seen):
        print(f"  Best joint score per test mol — mean: {pp_scores_seen.mean():.3f}, "
              f"median: {np.median(pp_scores_seen):.3f}, max: {pp_scores_seen.max():.3f}")

    print("\nFinding pos-neg bidirectional partners...")
    pos_neg_partners = []
    pn_drops = {
        "no_size_match": 0,
        "no_valid_w": 0,
        "no_chem_pass": 0,
        "no_either_dir": 0,
        "no_d1_only": 0,
        "no_d2_only": 0,
        "no_joint": 0
    }
    for i in tqdm(range(n_pairs)):
        diag = find_pos_neg_partner(
            i,
            n_pairs,
            compact_tani,
            extended_tani,
            ecfp_test_wit,
            args.tau_high,
            args.tau_low,
            args.tau_chem,
            n_atoms,
            args.n_atoms_tol
        )
        if diag["best"] is not None:
            j, w1, w2, score = diag["best"]
            pos_neg_partners.append((i, {
                "j": j,
                "w1": w1,
                "w1_compact_tani": compact_tani[i, w1],
                "w1_extended_tani": extended_tani[j, w1],
                "w2": w2,
                "w2_compact_tani": compact_tani[i, w2],
                "w2_extended_tani": extended_tani[j, w2],
                "score": score
            }))
            continue

        if diag["n_size_matched"] == 0:
            pn_drops["no_size_match"] += 1
        elif diag["n_with_valid_w"] == 0:
            pn_drops["no_valid_w"] += 1
        elif diag["n_after_chem_guard"] == 0:
            pn_drops["no_chem_pass"] += 1
        elif diag["n_d1_pass"] == 0 and diag["n_d2_pass"] == 0:
            pn_drops["no_either_dir"] += 1
        elif diag["n_d1_pass"] == 0:
            pn_drops["no_d1_only"] += 1
        elif diag["n_d2_pass"] == 0:
            pn_drops["no_d2_only"] += 1
        else:
            pn_drops["no_joint"] += 1

    print(f"  Kept {len(pos_neg_partners)} / {n_pairs} test mols.")
    print(f"  Drops — no size-matched B: {pn_drops['no_size_match']}, "
          f"no valid witness: {pn_drops['no_valid_w']}, "
          f"no witness passed τ_chem: {pn_drops['no_chem_pass']}, "
          f"neither D1 nor D2 for any B: {pn_drops['no_either_dir']}, "
          f"only D2 found (no D1): {pn_drops['no_d1_only']}, "
          f"only D1 found (no D2): {pn_drops['no_d2_only']}, "
          f"both dirs pass but never for same B: {pn_drops['no_joint']}")

    if args.sample_mols is not None:
        rng_sample = np.random.default_rng(PAIR_SEED + 1)
        if len(pos_pos_partners) > args.sample_mols:
            idxs = sorted(rng_sample.choice(len(pos_pos_partners), args.sample_mols, replace=False))
            pos_pos_partners = [pos_pos_partners[i] for i in idxs]
            print(f"\nSubsampled pos-pos to {len(pos_pos_partners)} pairs.")

        if len(pos_neg_partners) > args.sample_mols:
            idxs = sorted(rng_sample.choice(len(pos_neg_partners), args.sample_mols, replace=False))
            pos_neg_partners = [pos_neg_partners[i] for i in idxs]
            print(f"Subsampled pos-neg to {len(pos_neg_partners)} pairs.")

    if pos_pos_partners:
        pp_compact, pp_extended, pp_records = build_pair_data(
            compact_mols,
            extended_mols,
            pos_pos_partners,
            witnesses,
            baseline_compact,
            baseline_extended,
            ecfp_test_wit,
            "pos_pos"
        )
        save_pair_set(save_path / "pos_pos", pp_compact, pp_extended, pp_records, "pos_pos", args)
        print(f"\nSaved {len(pos_pos_partners)} pos-pos pairs to {save_path / 'pos_pos'}")

    if pos_neg_partners:
        pn_compact, pn_extended, pn_records = build_pair_data(
            compact_mols,
            extended_mols,
            pos_neg_partners,
            witnesses,
            baseline_compact,
            baseline_extended,
            ecfp_test_wit,
            "pos_neg"
        )
        save_pair_set(save_path / "pos_neg", pn_compact, pn_extended, pn_records, "pos_neg", args)
        print(f"Saved {len(pos_neg_partners)} pos-neg pairs to {save_path / 'pos_neg'}")

    summary = {
        "n_test_pairs": n_pairs,
        "n_witnesses": len(witnesses),
        "n_pos_pos": len(pos_pos_partners),
        "n_pos_neg": len(pos_neg_partners),
        "tau_high": args.tau_high,
        "tau_low": args.tau_low,
        "tau_chem": args.tau_chem,
        "n_atoms_tol": args.n_atoms_tol
    }
    with open(save_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {save_path / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True, help="Path to source dataset (with train/test splits)")
    parser.add_argument("--multi_path", type=str, required=True,
                        help="Multi-cond eval dir. Reads witnesses.hdf5 and writes pairs/ subtree inside")

    parser.add_argument("--n_test_mols", type=int, default=None,
                        help="Subsample the test mol pool before MMFF (or from cache when --reuse_test_pool)")
    parser.add_argument("--sample_mols", type=int, default=None,
                        help="Subsample each output pool (pos_pos and pos_neg) to this size before saving")
    parser.add_argument("--reuse_test_pool", action="store_true",
                        help="Skip MMFF if pairs/test_pool/{compact,extended}.hdf5 already exist")

    parser.add_argument("--n_confs", type=int, default=DEFAULT_N_CONFS)
    parser.add_argument("--max_opt_iters", type=int, default=DEFAULT_MAX_OPT_ITERS)
    parser.add_argument("--strain_filter", type=float, default=DEFAULT_STRAIN_FILTER)

    parser.add_argument("--rog_lo_pct", type=float, default=DEFAULT_ROG_LO_PCT,
                        help="RoG percentile for the compact target conformer (0 = min RoG)")
    parser.add_argument("--rog_hi_pct", type=float, default=DEFAULT_ROG_HI_PCT,
                        help="RoG percentile for the extended target conformer (100 = max RoG)")

    parser.add_argument("--tau_high", type=float, default=DEFAULT_TAU_HIGH)
    parser.add_argument("--tau_low", type=float, default=DEFAULT_TAU_LOW)
    parser.add_argument("--tau_chem", type=float, default=DEFAULT_TAU_CHEM)
    parser.add_argument("--n_atoms_tol", type=int, default=DEFAULT_N_ATOMS_TOL)

    parser.add_argument("--ecfp_radius", type=int, default=DEFAULT_ECFP_RADIUS)
    parser.add_argument("--ecfp_bits", type=int, default=DEFAULT_ECFP_BITS)

    parser.add_argument("--n_workers", type=int, default=DEFAULT_N_WORKERS)
    parser.add_argument("--tani_chunk_size", type=int, default=DEFAULT_TANI_CHUNK_SIZE,
                        help="Number of alignments per process pool job; larger = less dispatch overhead")

    args = parser.parse_args()
    main(args)
