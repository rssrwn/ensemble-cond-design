"""Create a small 3D aspirin SDF for the sampling quickstart."""
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/reference.sdf")
    args = parser.parse_args()
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    if AllChem.EmbedMolecule(mol, randomSeed=12345) != 0:
        raise RuntimeError("Could not embed reference")
    AllChem.MMFFOptimizeMolecule(mol)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    print(path)


if __name__ == "__main__":
    main()
