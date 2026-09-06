# Evaluations

Run from the repository root in the README environment. Replace paths with your
checkpoint and prepared datasets. Checkpoint architecture comes from its saved
hyperparameters. Default evaluation sizes and ensemble sampling can be expensive;
start with `--n_mols 2` where supported. Small runs check execution, not benchmark
quality. Use each command's `--help` for the complete parameter list.

## Unconditional generation

```bash
python -m enscondflow.eval_uncond --ckpt_path /path/to/model.ckpt \
  --n_mols 1000 --batch_size 128 --results_path results/unconditional.json
```

Measures generation quality and predicted size distribution. This benchmark
requires a model trained with size padding (`max_size` in its checkpoint). For
unpadded models, use the standalone sampling command with an explicit atom count.
The JSON contains aggregate and per-molecule results.

## Pocket and ligand conditioning

```bash
python -m enscondflow.eval_pocket --ckpt_path /path/to/model.ckpt \
  --data_path /path/to/spindr/test --pocket_cond --ligand_cond profile \
  --n_mols 100 --output_dir results/pocket
```

Use a checkpoint with a pocket encoder when passing `--pocket_cond`. Omit the flag
for ligand-only conditioning. `--pair_rmsd_cond` and `--psa3d_cond` set ensemble
property targets. Output includes aggregate/per-molecule JSON and generated and
reference molecules, with protein structures where available.

Add `--dock` to compute Vina scores. Receptor preparation uses Meeko's
`mk_prepare_receptor.py`, which must be on PATH (activate the environment or use
`mamba run`). Geometry optimization/scoring also uses xTB in the evaluation path.
These calculations are substantially more expensive than drawing samples.

## Multiple conditions

Prepare pairs as described in [preprocessing](preprocessing.md), then run:

```bash
python -m enscondflow.eval_multi --ckpt_path /path/to/model.ckpt \
  --multi_path /path/to/multi --n_mols 100 --results_path results/pos_pos.json
python -m enscondflow.eval_multi --ckpt_path /path/to/model.ckpt \
  --multi_path /path/to/multi --negate_extended --n_mols 100 \
  --results_path results/pos_neg.json
```

Positive/positive uses both targets; `--negate_extended` requests the compact shape
while avoiding the extended shape. `--negate_compact` reverses the roles. Mode is
inferred from negate flags and validated against pair metadata. Generated molecules
are scored by sampling MMFF ensembles and comparing overlap with both references.
`--uncond` provides an unconditional comparison on the same target pairs.

## Symmetry

```bash
python -m enscondflow.eval_symmetry --ckpt_path /path/to/model.ckpt \
  --geom_path /path/to/geom-splits --spindr_path /path/to/spindr/test \
  --n_systems 64 --results_path results/symmetry.json
```

Measures invariance/equivariance of model outputs under rotations, using states
from generation trajectories. This is the numerical evaluation only; plotting and
paper-figure notebooks are intentionally excluded. Both dataset paths are required
by the current script. See its module docstring for the precise comparisons.

## Reproducibility

Record the checkpoint, split/pair manifests, seed, integration settings, guidance,
and scoring parameters with results. Defaults are preserved from the research
scripts; unconditional and symmetry expose `--seed`, while pocket and multi use
12345. The new sampling CLI also exposes a seed. Stochastic results may differ
across devices and library versions. Full benchmark reproduction requires the
original data assignments, not just freshly generated splits.
