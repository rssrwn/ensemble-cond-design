from __future__ import annotations

import tqdm
import shutil
import tempfile
import subprocess
import numpy as np
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

import biotite.structure.io.pdb as pdb
from biotite.structure import AtomArray

import enscondflow.util.rdkit as smolRD
from enscondflow.repr.protein import Protein


class DockingMode(Enum):
    DOCK = "dock"
    SCORE = "score"
    MINIMIZE = "minimize"


@dataclass
class DockingResult:
    """Result from a single docking run."""

    poses: list[Chem.rdchem.Mol]
    affinities: np.ndarray
    mode: DockingMode

    @property
    def n_poses(self) -> int:
        return len(self.poses)

    @property
    def best_affinity(self) -> float:
        if len(self.affinities) == 0:
            return float("nan")
        return float(self.affinities[0, 0])

    @property
    def best_pose(self) -> Chem.rdchem.Mol | None:
        if len(self.poses) == 0:
            return None
        return self.poses[0]


class VinaDock:
    """AutoDock Vina wrapper for docking, scoring, and local minimization.

    Uses meeko for ligand/receptor preparation and the vina Python package for execution.

    Args:
        mode: One of "dock", "score", or "minimize".
        sf_name: Scoring function, either "vina" or "vinardo".
        seed: Random seed for reproducibility. 0 = random.
        cpu: Number of CPUs to use. 0 = all available.
        exhaustiveness: Number of Monte Carlo runs for docking mode.
        n_poses: Maximum number of poses to return.
        energy_range: Max energy difference (kcal/mol) from best pose for returned poses.
        box_padding: Angstroms of padding around reference ligand for auto box sizing.
        verbosity: Vina verbosity level (0=silent, 1=normal, 2=verbose).
    """

    def __init__(
        self,
        mode: str = "dock",
        sf_name: str = "vina",
        seed: int = 0,
        cpu: int = 0,
        exhaustiveness: int = 32,
        n_poses: int = 10,
        energy_range: float = 3.0,
        box_padding: float = 5.0,
        verbosity: int = 0,
    ):
        self.mode = DockingMode(mode)
        self.sf_name = sf_name
        self.seed = seed
        self.cpu = cpu
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.energy_range = energy_range
        self.box_padding = box_padding
        self.verbosity = verbosity

        self._ligand_preparator = MoleculePreparation()

    def prepare_ligand(self, mol: Chem.rdchem.Mol, conf_id: int = -1) -> str:
        """Convert an RDKit molecule to a PDBQT string for Vina.

        The molecule must have at least one conformer with 3D coordinates. Hydrogens will be
        added if not already present.

        Args:
            mol: RDKit molecule with 3D coordinates.
            conf_id: Which conformer to use. -1 uses the first conformer.

        Returns:
            PDBQT string ready for Vina.
        """

        mol = Chem.AddHs(mol, addCoords=True)
        mol_setups = self._ligand_preparator.prepare(mol, conformer_id=conf_id)

        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            raise RuntimeError(f"Meeko ligand preparation failed: {error_msg}")

        return pdbqt_string

    def prepare_receptor(self, protein: Protein, output_path: str | Path | None = None, receptor_cmd: str | None = None) -> str:
        """Convert a Protein to a receptor PDBQT file using meeko's mk_prepare_receptor.

        Writes a temporary PDB from the Protein, then calls mk_prepare_receptor.py to produce
        the PDBQT. If output_path is given, the PDBQT is written there for caching/reuse.

        Args:
            protein: Protein object from the binding complex.
            output_path: Optional path to write the receptor PDBQT file to. If None, a temporary
                file is used and the PDBQT string is returned.

        Returns:
            Path to the receptor PDBQT file.
        """

        n_atoms = len(protein)
        atoms = AtomArray(n_atoms)
        elements = [smolRD.PT.symbol_from_atomic(a) for a in protein.atomics.tolist()]

        atoms.coord = protein.coords
        atoms.element = np.array(elements)
        atoms.res_name = protein.res_names
        atoms.atom_name = protein.atom_names
        atoms.res_id = protein.res_ids
        atoms.chain_id = np.array(["A"] * n_atoms)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            pdb_path = tmp_dir / "receptor.pdb"

            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, atoms)
            pdb_file.write(pdb_path)

            if output_path is not None:
                output_path = Path(output_path)
                out_stem = str(output_path.with_suffix(""))
            else:
                out_stem = str(tmp_dir / "receptor")

            cmd = receptor_cmd or "mk_prepare_receptor.py"
            result = subprocess.run(
                [
                    cmd,
                    "--read_pdb", str(pdb_path),
                    "-o", out_stem,
                    "-p",
                    "--allow_bad_res",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"mk_prepare_receptor failed (exit {result.returncode}):\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            pdbqt_path = out_stem + ".pdbqt"

            if output_path is not None:
                return str(output_path.with_suffix(".pdbqt"))

            # If no output path given, read the temp file content and write to a new temp file
            # that persists beyond this context manager. The caller is responsible for cleanup.
            # Actually, just return the path — but it's in a temp dir that's about to be deleted.
            # So if no output_path, we need to copy it somewhere persistent.
            persistent_path = Path(tempfile.mktemp(suffix=".pdbqt"))
            persistent_path.write_text(Path(pdbqt_path).read_text())
            return str(persistent_path)

    def compute_box(
        self,
        ref_coords: np.ndarray,
        padding: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """Compute a docking box centered on reference coordinates.

        Args:
            ref_coords: Array of shape [N, 3] with reference atom positions (e.g. co-crystal ligand).
            padding: Angstroms of padding on each side. Uses self.box_padding if None.

        Returns:
            Tuple of (center, size) where each is a list of 3 floats in Angstroms.
        """

        padding = self.box_padding if padding is None else padding

        center = ref_coords.mean(axis=0).tolist()
        span = ref_coords.max(axis=0) - ref_coords.min(axis=0)
        size = (span + 2 * padding).tolist()

        return center, size

    def dock(
        self,
        mol: Chem.rdchem.Mol,
        receptor_pdbqt_path: str,
        box_center: list[float],
        box_size: list[float],
        conf_id: int = -1,
    ) -> DockingResult:
        """Run Vina on a single molecule against a prepared receptor.

        Args:
            mol: RDKit molecule with 3D coordinates.
            receptor_pdbqt_path: Path to the receptor PDBQT file.
            box_center: Docking box center [x, y, z] in Angstroms.
            box_size: Docking box size [x, y, z] in Angstroms.
            conf_id: Which conformer to prepare. -1 uses the first.

        Returns:
            DockingResult with poses, affinities, and mode.
        """

        ligand_pdbqt = self.prepare_ligand(mol, conf_id=conf_id)

        v = Vina(sf_name=self.sf_name, cpu=self.cpu, seed=self.seed, verbosity=self.verbosity)
        v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_path)
        v.set_ligand_from_string(ligand_pdbqt)
        v.compute_vina_maps(center=box_center, box_size=box_size)

        if self.mode == DockingMode.DOCK:
            v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)
            energies = v.energies(n_poses=self.n_poses, energy_range=self.energy_range)
            poses_pdbqt = v.poses(n_poses=self.n_poses, energy_range=self.energy_range)
            poses = self._pdbqt_to_rdkit(poses_pdbqt)
        elif self.mode == DockingMode.SCORE:
            energies = np.array([v.score()])
            poses = []
        elif self.mode == DockingMode.MINIMIZE:
            energies = np.array([v.optimize()])
            poses = []

        return DockingResult(poses=poses, affinities=energies, mode=self.mode)

    def dock_batch(
        self,
        mols: list[Chem.rdchem.Mol | None],
        receptor_pdbqt_path: str,
        box_center: list[float],
        box_size: list[float],
        conf_ids: list[int] | None = None,
    ) -> list[DockingResult | None]:
        """Run docking on a batch of molecules.

        Skips None molecules and molecules that fail preparation. Returns None in the
        corresponding position for failed molecules.

        Args:
            mols: List of RDKit molecules (may contain None).
            receptor_pdbqt_path: Path to the receptor PDBQT file.
            box_center: Docking box center [x, y, z].
            box_size: Docking box size [x, y, z].
            conf_ids: Optional per-molecule conformer IDs. Defaults to -1 for all.

        Returns:
            List of DockingResult or None, one per input molecule.
        """

        if conf_ids is None:
            conf_ids = [-1] * len(mols)

        results = []
        for mol, conf_id in zip(mols, conf_ids):
            if mol is None:
                results.append(None)
                continue

            try:
                result = self.dock(mol, receptor_pdbqt_path, box_center, box_size, conf_id=conf_id)
                results.append(result)
            except Exception as e:
                print(f"Docking failed for molecule: {e}")
                results.append(None)

        return results

    @staticmethod
    def _pdbqt_to_rdkit(poses_pdbqt: str) -> list[Chem.rdchem.Mol]:
        """Convert Vina output PDBQT string to a list of RDKit molecules."""

        if not poses_pdbqt or poses_pdbqt.strip() == "":
            return []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
            f.write(poses_pdbqt)
            f.flush()
            pdbqt_mol = PDBQTMolecule.from_file(f.name, skip_typing=True)

        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        return [mol for mol in rdkit_mols if mol is not None]


# ************************************************
# ***** Parallel multi-system docking utils *****
# ************************************************


def _prepare_single(protein, output_path, box_padding, receptor_cmd):
    """Prepare a single receptor PDBQT. Runs in a worker process."""

    try:
        dock = VinaDock(box_padding=box_padding)
        dock.prepare_receptor(protein, output_path=output_path, receptor_cmd=receptor_cmd)
        return str(output_path)
    except Exception as e:
        print(f"Receptor preparation failed: {e}")
        return None


def prepare_receptor_batch(
    proteins: list[Protein],
    output_dir: str | Path,
    box_padding: float = 5.0,
    n_workers: int = 8,
) -> list[str | None]:
    """Prepare receptor PDBQT files in parallel.

    Args:
        proteins: List of Protein objects.
        output_dir: Directory to write receptor PDBQT files.
        box_padding: Angstroms of padding for docking box.
        n_workers: Number of parallel workers.

    Returns:
        List of receptor PDBQT paths (or None for failures).
    """

    receptor_cmd = shutil.which("mk_prepare_receptor.py")
    if receptor_cmd is None:
        raise FileNotFoundError("mk_prepare_receptor not found on PATH. Is meeko installed?")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / f"{i}.pdbqt" for i in range(len(proteins))]
    n = len(proteins)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_prepare_single, prot, out, box_padding, receptor_cmd): i
            for i, (prot, out) in enumerate(zip(proteins, output_paths))
        }
        results = [None] * n
        for future in tqdm.tqdm(as_completed(futures), total=n, desc="Preparing receptors"):
            results[futures[future]] = future.result()

    return results


def _dock_single(mol, receptor_path, ref_coords, mode, exhaustiveness, box_padding):
    """Dock a single molecule against a single receptor. Runs in a worker process."""

    if mol is None or receptor_path is None:
        return None

    try:
        dock = VinaDock(
            mode=mode,
            exhaustiveness=exhaustiveness,
            box_padding=box_padding,
            cpu=1,
            verbosity=0,
        )
        center, size = dock.compute_box(ref_coords)
        return dock.dock(mol, receptor_path, center, size)
    except Exception as e:
        print(f"Docking failed: {e}")
        return None


def dock_batch_parallel(
    mols: list[Chem.rdchem.Mol | None],
    receptor_paths: list[str | None],
    ref_coords_list: list[np.ndarray],
    mode: str = "dock",
    exhaustiveness: int = 32,
    box_padding: float = 5.0,
    n_workers: int = 8,
) -> list[DockingResult | None]:
    """Dock molecules against different receptors in parallel.

    Each molecule is docked against its corresponding receptor using reference
    coordinates to define the docking box.

    Args:
        mols: List of RDKit molecules (may contain None).
        receptor_paths: List of receptor PDBQT paths (may contain None).
        ref_coords_list: List of reference coordinate arrays for box computation.
        mode: Vina docking mode ("dock", "score", or "minimize").
        exhaustiveness: Number of Monte Carlo runs.
        box_padding: Angstroms of padding for docking box.
        n_workers: Number of parallel workers.

    Returns:
        List of DockingResult or None, one per input molecule.
    """

    n = len(mols)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_dock_single, mol, rpath, rcoords, mode, exhaustiveness, box_padding): i
            for i, (mol, rpath, rcoords) in enumerate(zip(mols, receptor_paths, ref_coords_list))
        }
        results = [None] * n
        for future in tqdm.tqdm(as_completed(futures), total=n, desc="Docking"):
            results[futures[future]] = future.result()

    return results
