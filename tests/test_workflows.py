"""Small CPU workflows: user-trained checkpoint loading, sampling, and batching."""
from functools import partial

import lightning as L
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow import scriptutil as util
from enscondflow.data import GraphDataset, GraphInterpolant, GraphNoise, InterpolantDM
from enscondflow.data.features import InteractionProfile
from enscondflow.eval.integrator import Integrator
from enscondflow.models import EnsCondFlow, FeatureEncoder, HybridGenerator
from enscondflow.repr import AtomVocab, BondVocab, GraphBatch, GraphMol
from enscondflow.sample import load_pretrained, read_reference, sample_molecules


@pytest.fixture
def ligand():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return GraphMol.from_rdkit(mol)


def tiny_model(pocket=False, padded=True):
    generator = HybridGenerator(32, 4, 8, 2, 1, len(AtomVocab), len(BondVocab),
                                dropout=0.0, d_emb=16)
    encoder = FeatureEncoder(32, 4, 1, 2, include_pocket=pocket)
    return EnsCondFlow(generator, encoder, 0.001, Integrator(2), ema_decay=0.9,
                       max_size=8 if padded else None, shape_resample=0.0,
                       **{"val-prior-cat-noise": "uniform", "val-prior-zero-com": True,
                          "train-pad-coord-mode": "com"})


def make_dm(ligand):
    dataset = GraphDataset(GraphBatch([ligand, ligand]), transform=partial(
        util.mol_transform, profile_feat=InteractionProfile(raise_on_err=True)))
    interp = GraphInterpolant(GraphNoise(), pad_to=8)
    return InterpolantDM(dataset, None, dataset, 2, train_interpolant=interp,
                         test_interpolant=interp, max_workers=0)


@pytest.mark.parametrize("pocket,padded", [(False, False), (True, True)])
def test_user_training_checkpoint_and_sampling(tmp_path, ligand, pocket, padded):
    L.seed_everything(7)
    model = tiny_model(pocket, padded)
    dm = make_dm(ligand)
    trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=1,
                        limit_train_batches=1, limit_val_batches=0,
                        logger=L.pytorch.loggers.CSVLogger(tmp_path, name="train"),
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False)
    trainer.fit(model, datamodule=dm)
    assert trainer.global_step == 1
    checkpoint = tmp_path / "custom-name.ckpt"
    trainer.save_checkpoint(checkpoint)
    loaded = load_pretrained(checkpoint, device="cpu")
    assert loaded.encoder.include_pocket == pocket
    assert not loaded.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    # A random tiny model need not generate valid chemistry; it must execute both paths.
    assert len(sample_molecules(loaded, n_mols=1, n_atoms=None if padded else 3)) == 1
    assert len(sample_molecules(loaded, n_mols=1, reference=ligand)) == 1
    if not padded:
        with pytest.raises(ValueError, match="n_atoms"):
            sample_molecules(loaded, n_mols=1)
    # A compiled training model introduces wrappers in checkpoint key paths.
    saved = torch.load(checkpoint, weights_only=True)
    saved["hyper_parameters"]["compile_model"] = True
    saved["state_dict"] = {
        key.replace("generator.", "generator._orig_mod.", 1) if key.startswith("generator.")
        else key.replace("ema_gen.", "ema_gen._orig_mod.", 1) if key.startswith("ema_gen.")
        else key: value for key, value in saved["state_dict"].items()
    }
    torch.save(saved, checkpoint)
    compiled_loaded = load_pretrained(checkpoint, device="cpu")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(compiled_loaded.state_dict()[key], value, rtol=0, atol=0)


def test_sdf_reference_without_weights(tmp_path, ligand):
    path = tmp_path / "reference.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(ligand.to_rdkit())
    reference = read_reference(path)
    assert reference.n_conformers == 1
    assert reference.confs.weights is None
    assert len(sample_molecules(tiny_model().eval(), n_mols=1, reference=reference)) == 1
