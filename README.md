# Ensemble-conditioned molecular design

Research code for flow-matching molecular generation conditioned on ligand shape,
pharmacophores, ensemble properties, and protein pockets. The Python module is
`enscondflow`. The code includes training, sampling, evaluations, and preprocessing.

Run commands from the repository root. No `pip install -e .` or package build is
needed. Checkpoints, datasets, and generated results are not included in this repo.

## Environment

Create a fresh mamba environment in the ignored `.env/` directory:

```bash
mamba create -y -p ./.env -c conda-forge python=3.13 pip cxx-compiler vina=1.2.6 xtb-python=22.1
mamba activate ./.env
python -m pip install -r requirements.txt
python -m pip install -r extra_requirements.txt
python -m pip check
python -m pytest -q
```

The environment includes docking and xTB dependencies used by the evaluation code.
W&B logging is opt-in; training otherwise logs locally. A GPU is recommended for
full training and large evaluations. Sampling selects CUDA, then MPS, then CPU;
use `--device cpu` to choose explicitly.

## Load a model and sample

Supply a Lightning `.ckpt` produced by this model architecture. Our trained
model is available at ...

Unconditional sampling:

```bash
python -m enscondflow.sample --ckpt_path /path/to/model.ckpt \
  --n_mols 16 --batch_size 16 --output outputs/unconditional.sdf
```

Condition on a 3D ligand's shape and pharmacophore profile:

```bash
python examples/make_reference.py --output outputs/reference.sdf
python -m enscondflow.sample --ckpt_path /path/to/model.ckpt \
  --reference outputs/reference.sdf --condition profile \
  --cfg_gamma 1.0 --n_mols 16 --output outputs/conditioned.sdf
```

Use `--condition shape` for shape-only conditioning. The SDF must contain one
valid 3D molecule. Generated coordinates are centered on the reference heavy-atom
centroid; these are not docked poses. Property conditioning is dropped in this
simple ligand quickstart. Pocket and multiple-condition workflows are described
in [evaluation](docs/evaluation.md).

Each sampling run writes an SDF and adjacent JSON with settings, valid count,
and failed sample indices. Invalid builds are omitted from the SDF; `sample_index`
preserves their original indexing. Models trained with padding automatically
predict size and generate arbitrarily-sized molecules.

The same loader is available in Python:

```python
from enscondflow.sample import load_pretrained, sample_molecules

model = load_pretrained("/path/to/model.ckpt", device="cpu")
molecules = sample_molecules(model, n_mols=4, batch_size=4)
```

The loader preserves saved architecture and EMA weights and disables whole-model
compilation for portable inference. It does not load arbitrary unrelated model
architectures. See [sampling details](docs/sampling.md) for conformer ensembles
and lower-level conditioning methods.

## Evaluation and training

There are four evaluation scripts: unconditional generation, pocket-conditioned
generation, multiple conditions, and symmetry. Commands, input layouts, and
optional scoring flags are in [docs/evaluation.md](docs/evaluation.md).

For training, first prepare molecular HDF5 shards with train/validation/test
assignments as described in [docs/preprocessing.md](docs/preprocessing.md):

```bash
python -m enscondflow.train --data_path /path/to/geom-splits
```

Size learning is enabled by default; `--no_learn_size` disables padding.
Add `--spindr_path /path/to/spindr` to train with pocket complexes (`train/` and
`val/` directories). Add `--wandb` to use W&B logging. `--trial_run` is a small
pipeline check, not a reproduction of the full training run. Run
`python -m enscondflow.train --help` for architecture and training parameters.

## Code map

| Path | Contents |
| --- | --- |
| `enscondflow/models/` | Feature encoder, molecular decoder, `EnsCondFlow` training and sampling |
| `enscondflow/data/` | Conditioning features/profiles, datasets, batching, priors, interpolation |
| `enscondflow/repr/` | Molecules, conformers, proteins, interactions, vocabularies, HDF5 I/O |
| `enscondflow/util/` | RDKit helpers, alignment, energies, conformer sampling |
| `enscondflow/eval/` | Integrator, molecule construction, metrics, docking |
| `enscondflow/preprocess.py`, `prep_splits.py` | Raw conformer processing and split generation |
| `enscondflow/prep_witnesses.py`, `prep_pairs.py` | Multi-condition benchmark preparation |
| `defs/pharmacophore.fdef` | Required pharmacophore definitions; keep with the source tree |
| `tests/` | Feature and workflow regression tests |
| `examples/` | Small runnable examples without external datasets |
