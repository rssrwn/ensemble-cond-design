# Sampling details

The quickstart in the README covers unconditional and single-ligand sampling.
`enscondflow.sample.load_pretrained` accepts local Lightning checkpoints created by
this code's architecture, including pocket/no-pocket and user-trained variants.
The checkpoint provides encoder/decoder dimensions, EMA settings, vocab-dependent
weights, and integration configuration. Filenames are unrestricted.

Changing the Python namespace does not change learned tensor keys or feature
ordering. Vocabulary and pharmacophore definitions must remain consistent with
training. Keep `defs/pharmacophore.fdef` with the repository.

## Conditioning API

`EnsCondFlow.predict` accepts a collated batch with prior molecules, conditioning
features, and optionally a pocket. `create_latents` produces encoded conditions;
`generate` integrates a prior; `generate_mols` reconstructs RDKit molecules.
The standalone sampling helper disables gradients and drops ensemble property
conditioning unless you build a batch specifying those features explicitly.

For advanced workflows, inspect the working evaluation implementations:

- `eval_pocket.py`: binding profiles, pocket coordinates, and ensemble property targets.
- `eval_multi.py`: positive/positive and positive/negative shape conditions.
- `models/fm.py`: `generate_multi_cond`, `generate_multi_pos_neg`,
  `generate_multi_pocket`, and `generate_multi_latent` for custom experiments.

A ligand-only checkpoint cannot acquire pocket conditioning merely by loading it
through the same API. Use the checkpoint's actual trained capabilities.

## Conformer ensembles

To sample conformers for a molecular graph, independent of the flow model:

```python
from rdkit import Chem
from enscondflow.util.geometry import sample_ensemble

molecule = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
result = sample_ensemble(molecule, max_confs=16, n_threads=1)
if result is not None:
    ensemble, weights, minimum_energy = result
    print(ensemble.GetNumConformers(), weights)
```

This uses RDKit embedding, MMFF optimization, RMSD deduplication, strain filtering,
and Boltzmann weighting. It is separate from generating new molecular graphs.
See the function for parameters and the feature tests for examples of computing
shape, pharmacophore, RMSD, and PSA3D descriptors.
