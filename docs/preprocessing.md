# Data preparation

## Molecular conformers

The raw input expected by `preprocess.py` is a directory of trusted GEOM-style
pickles under `DATA/raw/pickles/`. Each pickle contains a `conformers` list; each
entry has an RDKit `rd_mol` with one conformer and a numeric `boltzmannweight`.
The script checks molecular validity, aligns matching conformers, records weights,
and writes numbered molecular HDF5 shards under `DATA/processed/`.

```bash
python -m enscondflow.preprocess --data_path /path/to/DATA --n_workers 8
python -m enscondflow.prep_splits --data_path /path/to/DATA/processed \
  --output /path/to/DATA/splits --dataset geom --seed 12345
```

The default cutoff drops conformers with weight <= 0, matching the original code.
Missing weights are rejected and counted. Molecules with no surviving conformers
are dropped. Atom ordering and canonical SMILES must agree across conformers.
Raw processing does not assign splits.

`prep_splits` extracts the original split methodology into a script. GEOM test
candidates have singleton Murcko scaffolds, 16–35 heavy atoms, logP < 5, and at
least 10 conformers; the default test size is 1,000. Validation contains 10,000
random molecules from the remaining dataset. This is not a fully scaffold-disjoint
three-way split. Use `--dataset qm9 --n_test 5000 --n_val 5000` for random QM9 splits
on compatible already-processed shards.

The script requires a new output directory, overwrites split labels in the output
copies, and saves `splits.json` with the seed, input indices, SMILES, and assignments.
The original notebook was unseeded: newly generated splits cannot be assumed to
match the historical benchmark. Original assignments would be needed for exact
reproduction. Keep the same source shards and ordering when reproducing a seeded run.

## Representations and conditioning

`GraphBatch` stores atoms, bonds, conformer coordinates, weights, and molecule
metadata in numbered `.hdf5` shards. Training selects `mol.meta['split']` values
`train` and `val`. Explicit hydrogens are retained through preprocessing; transforms
compute pharmacophore directions before removing hydrogens and centering coordinates.
HDF5 metadata contains pickled data, so use trusted datasets.

Inspect `repr/` for serialization, `data/features.py` for molecular/ensemble
features, and `data/profile.py` for conditioning profiles. Shared normalization
constants in `scriptutil.py` are part of model input conventions:
mean pairwise RMSD uses mean 2.3 and standard deviation 0.5; PSA3D uses mean 105
and standard deviation 50. Do not change them when using an existing checkpoint.

## Multi-condition benchmark

```bash
python -m enscondflow.prep_witnesses --data_path /path/to/DATA/splits \
  --multi_path /path/to/multi --n_workers 8
python -m enscondflow.prep_pairs --data_path /path/to/DATA/splits \
  --multi_path /path/to/multi --n_workers 8
```

Witness preparation samples training molecules by heavy-atom count and generates
MMFF ensembles. Pair preparation constructs compact/extended cross-molecule targets
with shape and chemical similarity filters. The resulting `pairs/pos_pos/` and
`pairs/pos_neg/` directories hold `compact.hdf5`, `extended.hdf5`, and `metadata.json`.
Use `--help` for pool sizes, conformer counts, thresholds, and reuse options. Full
preparation can be expensive; these are benchmark construction commands, not
prerequisites for sampling from an ordinary SDF.

## Pocket data

Pocket training and evaluation expect preprocessed SPINDR `ComplexBatch` shards.
Each `BindingComplex` holds a ligand, protein, and interaction annotations used by
binding-profile features. Training expects `SPINDR/train/` and `SPINDR/val/`;
evaluation takes the directory of test shards directly.

A raw SPINDR download/conversion pipeline is not included in this source tree.
Supply compatible preprocessed complexes; arbitrary PDB files cannot be substituted
for these HDF5 inputs. Data distribution and the external preparation recipe remain
to be documented when available. The representation and feature code is included
for inspection and custom conversion.
