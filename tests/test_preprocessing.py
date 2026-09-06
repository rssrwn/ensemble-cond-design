import pickle

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from enscondflow.preprocess import process_mol
from enscondflow.prep_splits import assign_splits
from enscondflow.repr import GraphBatch, GraphMol


def embedded(smiles="CCO"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    return mol


def test_preprocess_missing_and_zero_weights(tmp_path):
    mol = embedded()
    path = tmp_path / "raw.pickle"
    path.write_bytes(pickle.dumps({"conformers": [
        {"rd_mol": mol}, {"rd_mol": mol, "boltzmannweight": 0.0},
        {"rd_mol": mol, "boltzmannweight": 0.7},
    ]}))
    result, metadata = process_mol(path, conf_threshold=0.0)
    assert result.n_conformers == 1
    assert metadata["missing weight"] == 1
    assert metadata["low boltzmann weight"] == 1
    assert result.confs.weights[0] == pytest.approx(0.7)


def test_seeded_splits_and_hdf5_roundtrip(tmp_path):
    mols = [GraphMol.from_rdkit(embedded()) for _ in range(12)]
    labels = assign_splits(mols, "qm9", n_test=3, n_val=2, seed=7)
    assert labels == assign_splits(mols, "qm9", n_test=3, n_val=2, seed=7)
    assert labels.count("test") == 3 and labels.count("val") == 2
    for mol, label in zip(mols, labels):
        mol.meta["split"] = label
    path = tmp_path / "0.hdf5"
    GraphBatch(mols).save_hdf5_shard(path)
    loaded = GraphBatch.load_hdf5_shard(path)
    try:
        assert [m.meta["split"] for m in loaded] == labels
        np.testing.assert_allclose(loaded[0].confs.coords, mols[0].confs.coords)
    finally:
        loaded.close_hdf5()


def test_impossible_split_fails():
    with pytest.raises(ValueError, match="split sizes"):
        assign_splits([], "qm9", n_test=1, n_val=0)
    with pytest.raises(ValueError, match="eligible"):
        assign_splits([GraphMol.from_rdkit(embedded())], "geom", n_test=1, n_val=0)
