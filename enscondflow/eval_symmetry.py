"""Encoder symmetry evaluation script.

Section 3.3 claims the encoders learn adaptive symmetries: with the rotated flag set the conditioning
should carry no information about the input frame, and with it unset the frame should be preserved.
Nothing in training enforces either property on the encoder embeddings themselves. The loss only
constrains end-to-end behaviour, so the encoder is free to pass the frame through and let the decoder
discard it. Both properties are therefore measured at the model output, where they are actually
claimed, by treating the model as a function and comparing its predictions directly.

Two separate experiments, each stated as the exact symmetry it tests, so the ideal is zero in both
cases and neither needs the other as a reference:

  Invariance (flag=1)    f(x_t, c) should equal f(x_t, Rc). Rotating the condition alone must not
                         move the prediction, since a flagged condition carries no frame.
  Equivariance (flag=0)  f(Rx_t, Rc) should equal R f(x_t, c) for coordinates, and should leave the
                         atom and bond distributions unchanged, since those are invariant channels.

States x_t come from the model's own generation trajectory rather than from interpolating towards a
known molecule. Interpolated states carry the true molecule's frame, which would let the model ignore
the condition entirely and still denoise correctly, so the invariance test would report success
without having tested anything. Each experiment probes a trajectory generated under its own flag, so
the state carries whatever frame that flag actually gives the model. Driving both from a flag=0
trajectory would leave the invariance states aligned to the unrotated condition, which reinstates the
same free frame that ruled interpolated states out, and increasingly so as the state commits.
Integration uses constant step sizes so snapshots are evenly spaced over t.

Comparisons are single forward passes, so there is no sampling, no integration cascade from an early
discrete flip, and no molecule building. Slot correspondence is exact because the prior is rotated
rather than permuted.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import enscondflow.scriptutil as util
from enscondflow.repr import GraphBatch
from enscondflow.data import GraphNoise, GraphDataset, ComplexDataset
from enscondflow.data.features import InteractionProfile
from enscondflow.data.interpolate import GraphInterpolant
from enscondflow.data.datamodules import graph_batch_to_dict


DEFAULT_N_SYSTEMS = 64
DEFAULT_N_ROTATIONS = 8
DEFAULT_N_STEPS = 100
DEFAULT_SNAPSHOT_EVERY = 10
DEFAULT_CFG_GAMMA = 1.0
DEFAULT_SEED = 12345

# Rotated flag each experiment tests, and whether the state rotates along with the condition
EXPERIMENTS = {"invariance": (1, False), "equivariance": (0, True)}


def random_rotations(n_rotations):
    """Random SO(3) rotations with the identity first, so index 0 is always the reference."""

    return [Rotation.identity()] + list(Rotation.random(n_rotations))


def rotate_batch_coords(coords, rotations):
    """Apply rotation k to row k of a [K, N, 3] coordinate tensor."""

    rotated = [rot.apply(coords[k].cpu().numpy()) for k, rot in enumerate(rotations)]
    return torch.from_numpy(np.stack(rotated)).float().to(coords.device)


def build_profile_cond(profile, rotations):
    """Stack rotations of one profile into an encoder input batch."""

    positions = []
    directions = []
    for rot in rotations:
        dirs = profile.directions.copy()
        nonzero = np.linalg.norm(dirs, axis=1) > 0.5
        if nonzero.any():
            dirs[nonzero] = rot.apply(dirs[nonzero])

        positions.append(rot.apply(profile.positions))
        directions.append(dirs)

    n_points = len(profile.positions)
    types = np.tile(profile.types.astype(np.int64), (len(rotations), 1))

    return {
        "coords": torch.from_numpy(np.stack(positions)).float(),
        "types": torch.from_numpy(types).long(),
        "directions": torch.from_numpy(np.stack(directions)).float(),
        "mask": torch.from_numpy(np.ones((len(rotations), n_points), dtype=np.int64)).long()
    }


def build_pocket_cond(protein, rotations):
    """Stack rotations of one pocket into an encoder input batch."""

    converted = GraphInterpolant.convert_pocket_atoms(protein)
    coords = np.stack([rot.apply(converted.coords) for rot in rotations])
    atomics = np.tile(converted.atomics, (len(rotations), 1))

    return {
        "atomics": torch.from_numpy(atomics).long(),
        "coords": torch.from_numpy(coords).float(),
        "mask": torch.from_numpy(np.ones_like(atomics)).long()
    }


def encode_profile(model, batch, rotated_flag):
    """Encode a profile batch with a fixed rotated flag, returning tokens and their mask."""

    n_batch = batch["coords"].size(0)
    device = batch["coords"].device
    rotated = torch.full((n_batch,), rotated_flag, dtype=torch.long, device=device)
    profile_mode = torch.zeros(n_batch, dtype=torch.long, device=device)
    pos_noise_std = torch.zeros(n_batch, dtype=torch.float, device=device)

    with torch.inference_mode():
        tokens = model.encoder.profile_emb(
            batch["coords"],
            batch["types"],
            batch["directions"],
            rotated,
            profile_mode,
            pos_noise_std,
            batch["mask"]
        )

    return tokens, batch["mask"]


def encode_pocket(model, batch, rotated_flag):
    """Encode a pocket batch with a fixed rotated flag, returning tokens and their mask."""

    n_batch = batch["coords"].size(0)
    device = batch["coords"].device
    rotated = torch.full((n_batch,), rotated_flag, dtype=torch.long, device=device)

    with torch.inference_mode():
        tokens = model.encoder.pocket_emb(batch["atomics"], batch["coords"], rotated, batch["mask"])

    return tokens, batch["mask"]


def sample_prior(hparams, n_mols, n_atoms):
    """Prior molecules at the model's padded size."""

    sampler = GraphNoise(cat_noise=hparams["val-prior-cat-noise"], zero_com=hparams["val-prior-zero-com"])
    return graph_batch_to_dict(GraphBatch([sampler.sample_molecule(n_atoms, 1) for _ in range(n_mols)]))


def predict(model, state, time, cond_tokens, cond_mask):
    """One forward pass, returning the endpoint coordinate prediction and the type/bond distributions."""

    n_batch = state["coords"].size(0)
    times = torch.full((n_batch,), time, device=model.device)
    ada_latents = model.null_ada_latents(n_batch, model.device)

    with torch.inference_mode():
        coords, type_logits, bond_logits = model(
            state,
            ada_latents,
            times,
            cond_latents=cond_tokens,
            cond_mask=cond_mask,
            training=False
        )

    return {
        "coords": coords.float(),
        "atomics": F.softmax(type_logits.float(), dim=-1),
        "bonds": F.softmax(bond_logits.float(), dim=-1)
    }


def coord_rmsd(a, b):
    """RMSD between two [N, 3] coordinate sets, over every slot."""

    return float(((a - b) ** 2).sum(dim=-1).mean().sqrt().item())


def tv_distance(p, q):
    """Mean total variation distance between two sets of categorical distributions."""

    return float((0.5 * (p - q).abs().sum(dim=-1)).mean().item())


def compare(ref, other, rotation):
    """Symmetry errors between a reference prediction and its rotated counterpart.

    The reference coordinates are compared to the rotated run both directly and after applying the
    rotation. One of the two is the symmetry being tested and the other is the scale of failing it,
    which experiment decides. In the invariance experiment the direct comparison is the ideal and the
    rotated one says what following the frame would have cost; in the equivariance experiment it is
    the other way round. Atom and bond distributions are invariant channels in both experiments, so
    they are always compared directly.
    """

    rotated_ref = torch.from_numpy(rotation.apply(ref["coords"].cpu().numpy())).float().to(ref["coords"].device)

    return {
        "rmsd_direct": coord_rmsd(ref["coords"], other["coords"]),
        "rmsd_rotated": coord_rmsd(rotated_ref, other["coords"]),
        "atom_tv": tv_distance(ref["atomics"], other["atomics"]),
        "bond_tv": tv_distance(ref["bonds"], other["bonds"])
    }


def run_system(model, hparams, cond_batch, encode, rotations, args):
    """Run both symmetry experiments, each probing snapshots from a trajectory driven by its own flag.

    Invariance rotates the condition alone, so the prediction must not move: rmsd_direct is the
    symmetry error and rmsd_rotated the cost of having followed the frame instead. Equivariance
    rotates the state with the condition, so the prediction must rotate too, and the two swap roles.
    """

    n_atoms = hparams["max_size"]
    cond_batch = util.to_device(cond_batch, model.device)
    ref_cond = {k: v[:1] for k, v in cond_batch.items()}

    # Both trajectories start from the same noise, so the two experiments differ only in the flag
    prior = util.to_device(sample_prior(hparams, 1, n_atoms), model.device)

    n_rot = len(rotations)
    records = {}

    for experiment, (flag, rotate_state) in EXPERIMENTS.items():
        traj_tokens, traj_mask = encode(model, ref_cond, flag)
        _, snapshots = model.generate(
            prior,
            traj_tokens,
            traj_mask,
            model.null_ada_latents(1, model.device),
            args.n_steps,
            step_strategy="constant",
            cfg_gamma=args.cfg_gamma,
            snapshot_every=args.snapshot_every
        )

        # One encoder pass covers every rotation, since the batch holds them all
        tokens, mask = encode(model, cond_batch, flag)

        rows = []
        for time, state in snapshots:
            state = {k: v.expand(n_rot, *v.shape[1:]).contiguous() for k, v in state.items()}
            state = util.to_device(state, model.device)
            if rotate_state:
                state = {**state, "coords": rotate_batch_coords(state["coords"], rotations)}

            preds = predict(model, state, time, tokens, mask)
            errors = [compare(index_pred(preds, 0), index_pred(preds, k), rotations[k]) for k in range(1, n_rot)]
            rows.append({"time": time, **mean_errors(errors)})

        records[experiment] = rows

    return records


def index_pred(preds, idx):
    return {k: v[idx] for k, v in preds.items()}


def mean_errors(errors):
    """Average each error term over rotations."""

    return {key: float(np.mean([e[key] for e in errors])) for key in errors[0]}


def run_encoder(model, hparams, sources, build_cond, encode, args, label):
    results = []
    for source in tqdm(sources, desc=label):
        rotations = random_rotations(args.n_rotations)
        cond_batch = build_cond(source, rotations)
        results.append(run_system(model, hparams, cond_batch, encode, rotations, args))

    return results


def load_profiles(mols, label):
    profiles = []
    for mol in tqdm(mols, desc=f"Profiles ({label})"):
        try:
            profile = InteractionProfile()(mol)
        except Exception:
            continue

        if hasattr(profile, "positions") and len(profile.positions) >= 3:
            profiles.append(profile)

    print(f"Built {len(profiles)}/{len(mols)} profiles for {label}")
    return profiles


def summarise(label, results):
    """Print each experiment's error averaged over systems, at the first, middle and last snapshot."""

    # Which comparison is the symmetry error, and which is the scale of failing it
    ideal_key = {"invariance": "rmsd_direct", "equivariance": "rmsd_rotated"}
    scale_key = {"invariance": "rmsd_rotated", "equivariance": "rmsd_direct"}

    def mean_at(per_system, idx, key):
        vals = [s[idx][key] for s in per_system if s[idx][key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n*** {label} ***")
    for experiment in EXPERIMENTS:
        per_system = [r[experiment] for r in results]
        n_times = len(per_system[0])

        print(f"  {experiment}")
        for idx in [0, n_times // 2, n_times - 1]:
            time = per_system[0][idx]["time"]
            error = mean_at(per_system, idx, ideal_key[experiment])
            scale = mean_at(per_system, idx, scale_key[experiment])
            atom = mean_at(per_system, idx, "atom_tv")
            print(f"    t={time:.2f}   error {error:.4f} A   (broken-symmetry scale {scale:.4f} A)   atom TV {atom:.4f}")


def main(args):
    print("Running encoder symmetry evaluation script...")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    L.seed_everything(args.seed)
    util.disable_lib_stdout()
    util.configure_fs()

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    hparams = checkpoint["hyper_parameters"]

    print("Loading model...")
    integrator = util.load_integrator(hparams)
    model = util.load_model(args.ckpt_path, integrator, arch="hybrid")
    model = model.eval().to(util.get_device())

    if not model.encoder.include_pocket:
        raise ValueError("Checkpoint has no pocket encoder, so the pocket comparison cannot be run.")

    print("Loading GEOM test molecules...")
    geom_dataset = GraphDataset.load(Path(args.geom_path))
    geom_test = geom_dataset.select(lambda m: m.meta.get("split") == "test")
    geom_subset, _ = geom_test.split(min(args.n_systems, len(geom_test)))
    geom_mols = [geom_subset._data[i].read() for i in range(len(geom_subset))]

    print("Loading SPINDR test complexes...")
    spindr_dataset = ComplexDataset.load(Path(args.spindr_path))
    systems = [system.read() for system in spindr_dataset]
    spindr_dataset.close()
    if len(systems) > args.n_systems:
        idxs = np.random.choice(len(systems), args.n_systems, replace=False)
        systems = [systems[i] for i in idxs]

    ligands = [s.zero_ligand_com().ligand for s in systems]
    pockets = [s.remove_hs(include_ligand=False).zero_pocket_com().protein for s in systems]

    print("\n***** Setup *****\n")
    setup = {
        "Checkpoint": args.ckpt_path,
        "GEOM molecules": len(geom_mols),
        "SPINDR systems": len(systems),
        "Rotations per system": args.n_rotations,
        "Integration steps": args.n_steps,
        "Snapshot every": args.snapshot_every,
        "Trajectory CFG gamma": args.cfg_gamma
    }
    for name, val in setup.items():
        print(f"{name:<25} {val}")
    print("\n*****************\n")

    encoders = {}
    for key, mols, name in [("profile-geom", geom_mols, "geom"), ("profile-spindr", ligands, "spindr ligands")]:
        profiles = load_profiles(mols, name)
        encoders[key] = run_encoder(model, hparams, profiles, build_profile_cond, encode_profile, args, key)

    encoders["pocket-spindr"] = run_encoder(
        model,
        hparams,
        pockets,
        build_pocket_cond,
        encode_pocket,
        args,
        "pocket-spindr"
    )

    for label, results in encoders.items():
        summarise(label, results)

    if args.results_path is not None:
        payload = {"config": {**vars(args), "ckpt_path": str(args.ckpt_path)}, "encoders": encoders}
        path = Path(args.results_path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"\nSaved per-system results to {path}")

    print("***** Encoder symmetry evaluation complete *****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--geom_path", type=str, required=True)
    parser.add_argument("--spindr_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, default=None)

    parser.add_argument("--n_systems", type=int, default=DEFAULT_N_SYSTEMS)
    parser.add_argument("--n_rotations", type=int, default=DEFAULT_N_ROTATIONS)
    parser.add_argument("--n_steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--snapshot_every", type=int, default=DEFAULT_SNAPSHOT_EVERY)
    parser.add_argument("--cfg_gamma", type=float, default=DEFAULT_CFG_GAMMA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()
    main(args)
