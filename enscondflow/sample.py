"""Draw molecules from a compatible checkpoint, optionally conditioned on an SDF ligand."""

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from rdkit import Chem

from enscondflow import scriptutil as util
from enscondflow.data import GraphDataset, GraphInterpolant, GraphNoise, InterpolantDM
from enscondflow.data.features import InteractionProfile, StochasticProfile
from enscondflow.eval_uncond import generate_uncond
from enscondflow.repr import GraphBatch, GraphMol


def load_pretrained(checkpoint, device="auto", steps=None):
    """Load pocket or ligand-only models trained with this code's architecture.

    Checkpoints are Lightning .ckpt files with hyper_parameters and state_dict.
    No particular filename or download service is required.
    """
    model = util.load_model(checkpoint)
    if steps is not None:
        if steps < 1:
            raise ValueError("steps must be positive")
        model.integrator = util.load_integrator(model.hparams, integration_steps=steps)
    return model.to(util.get_device() if device == "auto" else device).eval()


def read_reference(path):
    """Read one 3D ligand, preserving explicit H coordinates when supplied."""
    mols = list(Chem.SDMolSupplier(str(path), removeHs=False))
    if len(mols) != 1 or mols[0] is None:
        raise ValueError("Reference SDF must contain exactly one valid molecule")
    mol = mols[0]
    if not mol.GetNumConformers() or not mol.GetConformer().Is3D():
        raise ValueError("Reference SDF must contain a 3D conformer")
    return GraphMol.from_rdkit(Chem.AddHs(mol, addCoords=True))


@torch.inference_mode()
def sample_molecules(model, n_mols=16, batch_size=16, reference=None,
                     profile_cond="profile", cfg_gamma=1.0, n_atoms=None,
                     shape_std_dev=0.3):
    """Return RDKit molecules (None for failed builds), without running metrics.

    A reference is a GraphMol with one 3D conformer. Generated coordinates are
    centered on the reference heavy-atom centroid; they are not docked poses.
    For an unpadded model, unconditional sampling requires n_atoms explicitly.
    """
    if n_mols < 1 or batch_size < 1:
        raise ValueError("n_mols and batch_size must be positive")
    if n_atoms is not None and n_atoms < 1:
        raise ValueError("n_atoms must be positive")
    if shape_std_dev < 0:
        raise ValueError("shape_std_dev cannot be negative")
    if profile_cond not in ("shape", "profile"):
        raise ValueError("profile_cond must be shape or profile")
    padded_size = model.hparams.get("max_size")
    if padded_size is not None and n_atoms not in (None, padded_size):
        raise ValueError(f"This model uses {padded_size} prior slots; omit n_atoms")
    if reference is None:
        size = padded_size if padded_size is not None else n_atoms
        if size is None:
            raise ValueError("Unpadded models require n_atoms for unconditional sampling")
        return generate_uncond(model, model.hparams, n_mols, size, batch_size)[0]
    if reference.n_conformers != 1:
        raise ValueError("Reference must have exactly one conformer")
    if n_atoms is not None:
        raise ValueError("n_atoms is only supported for unconditional sampling")
    if padded_size is not None and reference.remove_hs().n_atoms > padded_size:
        raise ValueError("Reference has more heavy atoms than the model's padded size")
    profile = StochasticProfile(
        InteractionProfile(raise_on_err=True), pos_std_dev=shape_std_dev,
        shape_resample=model.hparams.get("shape_resample", 0.0), raise_on_err=True,
    )
    # Prepare on demand; don't construct a large repeated dataset in memory.
    from functools import partial
    transform = partial(util.mol_transform, conf_sample=True, profile_feat=profile, feat_dropout=1.0)
    dataset = GraphDataset(GraphBatch([reference] * n_mols), transform=transform)
    interpolant = GraphInterpolant(
        GraphNoise(cat_noise=model.hparams["val-prior-cat-noise"],
                   zero_com=model.hparams["val-prior-zero-com"]),
        pad_to=padded_size, pad_coord_mode=model.hparams["train-pad-coord-mode"],
    )
    dm = InterpolantDM(None, None, dataset, batch_size,
                      test_interpolant=interpolant, max_workers=0)
    return util.generate_molecules(model, dm.test_dataloader(), cfg_gamma,
                                   profile_cond=profile_cond, drop_props=True)[0]


def main(args):
    L.seed_everything(args.seed)
    model = load_pretrained(args.ckpt_path, args.device, args.steps)
    reference = read_reference(args.reference) if args.reference else None
    mols = sample_molecules(model, args.n_mols, args.batch_size, reference,
                            args.condition, args.cfg_gamma, args.n_atoms, args.shape_std_dev)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(output)) as writer:
        for i, mol in enumerate(mols):
            if mol is not None:
                mol.SetIntProp("sample_index", i)
                writer.write(mol)
    metadata = {**vars(args), "device": str(model.device),
                "integration_steps": model.integrator.steps,
                "valid": sum(m is not None for m in mols),
                "failed_indices": [i for i, m in enumerate(mols) if m is None]}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {metadata['valid']}/{len(mols)} molecules to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output", required=True, help="Generated SDF; run metadata goes alongside as JSON")
    parser.add_argument("--reference", help="One 3D ligand in SDF format")
    parser.add_argument("--condition", choices=["shape", "profile"], default="profile")
    parser.add_argument("--n_mols", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_atoms", type=int, help="Prior size for unpadded models, unconditional only")
    parser.add_argument("--cfg_gamma", type=float, default=1.0)
    parser.add_argument("--shape_std_dev", type=float, default=0.3)
    parser.add_argument("--steps", type=int, help="Defaults to checkpoint integration settings")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=12345)
    main(parser.parse_args())
