"""BCMG training script"""

import argparse
from pathlib import Path

import torch
import lightning as L
from functools import partial
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import enscondflow.scriptutil as util
import enscondflow.data.features as Features
from enscondflow.repr import AtomVocab, BondVocab
from enscondflow.eval.integrator import Integrator
from enscondflow.data import GraphNoise, GraphInterpolant, GraphDataset, ComplexDataset, InterpolantDM
from enscondflow.models import EnsCondFlow, FeatureEncoder, HybridGenerator


# Model default hparams
DEFAULT_D_MODEL = 384
DEFAULT_N_LAYERS = 16
DEFAULT_D_EDGE = 64
DEFAULT_N_HEADS = 16
DEFAULT_DROPOUT = 0.1
DEFAULT_N_BLOCKS = 4
DEFAULT_ARCH = "hybrid"

DEFAULT_ENC_D_MODEL = 128
DEFAULT_ENC_N_HEADS = 8
DEFAULT_ENC_N_LAYERS = 8

# Training default hparams
DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.001
DEFAULT_BATCH_COST = 384
DEFAULT_GRAD_CLIP_VAL = 1.0
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_WARM_UP_STEPS = 10000
DEFAULT_EMA_DECAY = 0.999
DEFAULT_COORD_SHIFT_STD = 3.0
DEFAULT_PAD_COORD_MODE = "com"
DEFAULT_N_POCKET_REPLICATES = 8

# CFG freq corresponds to the frequency of seeing the global CFG cond embeddings - shape profile and/or property embeddings
# CFG feat freq is the frequency of seeing individual components of embeddings - all pharmacophores, logp, qed, etc.
DEFAULT_CFG_FREQ = 0.9
DEFAULT_CFG_FEAT_FREQ = 0.7

# Profile augmentation hparams
DEFAULT_SHAPE_MAX_STD_DEV = 1.0
DEFAULT_ROTATE_PROB = 0.5
DEFAULT_SHAPE_RESAMPLE = 0.2
DEFAULT_MOL_PHARMACO_DROPOUT = 0.5
DEFAULT_XTAL_PHARMACO_DROPOUT = 0.2

# Loss weightings
DEFAULT_TYPE_LOSS_WEIGHT = 0.3
DEFAULT_BOND_LOSS_WEIGHT = 5.0

# FM training default args
DEFAULT_COORD_NOISE_STD = 0.2
DEFAULT_TIME_ALPHA = 1.5
DEFAULT_TIME_BETA = 1.0

# FM sampling default args
DEFAULT_N_VAL_MOLS = 1024
DEFAULT_N_INF_STEPS = 100
DEFAULT_STEP_SIZE = "decay"
DEFAULT_CAT_NOISE_LEVEL = 3
DEFAULT_CAT_STRATEGY = "sample"


N_MOL_PROPS = 2
MIN_QED = 0.3


def get_precision(args):
    return "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"


def get_max_size(args):
    if not args.learn_size:
        return None

    return 48


def build_model(args, dm):
    # Get hyperparameters from the datamodule, pass these into the model to be saved
    hparams = {
        "epochs": args.epochs,
        "architecture": args.arch,
        "grad_clip_val": args.grad_clip_val,
        "precision": get_precision(args),
        "max_size": get_max_size(args),
        "learn_size": args.learn_size,
        "shape_max_std_dev": args.shape_max_std_dev,
        "shape_resample": args.shape_resample,
        "mol_pharmaco_dropout": args.mol_pharmaco_dropout,
        "xtal_pharmaco_dropout": args.xtal_pharmaco_dropout,
        "rotate_prob": args.rotate_prob,
        **dm.hparams
    }

    train_steps = util.calc_train_steps(dm, args.epochs, 1)
    print(f"Total training steps {train_steps}")

    n_atom_types = len(AtomVocab)
    n_bond_types = len(BondVocab)

    if args.arch == "hybrid":
        generator = HybridGenerator(
            args.d_model,
            args.n_heads,
            args.d_edge,
            args.n_layers,
            args.n_blocks,
            n_atom_types,
            n_bond_types,
            ff_factor=4,
            dropout=args.dropout
        )

    else:
        raise ValueError(f"Unknown architecture {args.arch}")

    integrator = Integrator(
        args.n_inf_steps,
        step_size=args.step_size,
        cat_strategy=args.cat_strategy,
        cat_noise_level=args.cat_noise_level
    )

    encoder = FeatureEncoder(
        args.enc_d_model,
        args.enc_n_heads,
        args.enc_n_layers,
        N_MOL_PROPS,
        d_out=args.d_model,
        include_pocket=args.spindr_path is not None
    )

    fm_model = EnsCondFlow(
        generator,
        encoder,
        args.lr,
        integrator,
        type_loss_weight=args.type_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        lr_schedule=args.lr_schedule,
        warm_up_steps=args.warm_up_steps,
        total_steps=train_steps,
        compile_model=False,
        ema_decay=args.ema_decay,
        cfg_freq=args.cfg_freq,
        **hparams
    )
    return fm_model


def load_mol_dataset(args):
    print("Initialising molecule dataset...")

    pad_to = get_max_size(args)
    feat_dropout = 1 - args.cfg_feat_freq

    # Apply between 0.0 and max_std_dev noise during training to allow arbitrary noising at inference
    train_profile = Features.StochasticProfile(
        Features.InteractionProfile(),
        pos_std_dev=(0.1, args.shape_max_std_dev),
        rotate_prob=args.rotate_prob,
        shape_resample=args.shape_resample,
        local_pharm_dropout=args.mol_pharmaco_dropout,
        global_pharm_dropout=feat_dropout,
        global_shape_dropout=feat_dropout
    )

    # Simulate a binding profile during validation
    # Dropout some pharmacophores and add a fixed amount of noise to the shape
    # Always pass a non-rotated profile so that the model should produce an aligned output
    eval_profile = Features.StochasticProfile(
        Features.InteractionProfile(),
        pos_std_dev=0.3,
        rotate_prob=0.0,
        shape_resample=args.shape_resample,
        local_pharm_dropout=args.mol_pharmaco_dropout
    )

    train_transform = partial(
        util.mol_transform,
        rand_rot=True,
        profile_feat=train_profile,
        feat_dropout=feat_dropout,
        shift_std=args.coord_shift_std
    )
    eval_transform = partial(util.mol_transform, rand_rot=False, profile_feat=eval_profile, feat_dropout=0.0)

    dataset = GraphDataset.load(Path(args.data_path), transform=train_transform)

    # Filter out molecules which are too big or have fewer than 2 conformers
    mol_size_fn = lambda mol: mol.meta["n_heavy_atoms"]
    dataset = dataset.select(lambda mol: mol_size_fn(mol) <= pad_to) if pad_to is not None else dataset
    dataset = dataset.select(lambda mol: mol.confs is not None and len(mol.confs) >= 2)

    # Precompute features and subsample conformers, then release HDF5 file handles
    if not args.trial_run:
        print("Preloading dataset into memory...")
        feats = [
            Features.StandardisedFeature(
                Features.MeanPairwiseRMSD(),
                util.MEAN_PAIRWISE_RMSD_MEAN,
                util.MEAN_PAIRWISE_RMSD_STD
            ),
            Features.StandardisedFeature(Features.PSA3D(), util.PSA3D_MEAN, util.PSA3D_STD),
            Features.Qed(),
            Features.LogP(),
            Features.SavedPharmacophores()
        ]
        dataset.preload(features=feats, n_confs=10, max_workers=32)

    # Use pre-defined dataset splits and cap val dataset to n_val_mols for efficiency
    train_dataset = dataset.select(lambda mol: mol.meta["split"] == "train")

    val_dataset = dataset.select(lambda mol: mol.meta["split"] == "val")
    val_dataset = val_dataset.sample(args.n_val_mols)
    val_dataset.transform = eval_transform

    print("Datasets initialisation complete.")
    print(f"Num training molecules: {len(train_dataset)}")
    print(f"Num validation molecules: {len(val_dataset)}")

    return train_dataset, val_dataset


def load_pocket_train_dataset(args):
    """Load and prepare the pocket (complex) dataset for training."""

    print("Loading pocket dataset...")

    pad_to = get_max_size(args)
    feat_dropout = 1 - args.cfg_feat_freq

    # Note no local pharm dropout applied here since these are PL interactions
    # So already a subset of the full ligand pharmacophore profile
    profile_feat = Features.StochasticProfile(
        Features.BindingProfile(),
        pos_std_dev=(0.1, args.shape_max_std_dev),
        rotate_prob=args.rotate_prob,
        shape_resample=args.shape_resample,
        global_pharm_dropout=feat_dropout,
        global_shape_dropout=feat_dropout,
        local_pharm_dropout=args.xtal_pharmaco_dropout
    )

    # NOTE feat_dropout needs to be 1.0 here (always dropped) since we don't actually have feats for pocket data
    transform = partial(
        util.complex_transform,
        profile_feat=profile_feat,
        feat_dropout=1.0,
        pocket_rotate_prob=args.rotate_prob
    )

    spindr_path = Path(args.spindr_path) / "train"
    dataset = ComplexDataset.load(spindr_path, transform=transform)

    n_total = len(dataset)
    dataset = dataset.select(lambda s: (s.ligand.atomics == 1).any())
    n_no_hs = n_total - len(dataset)
    if n_no_hs > 0:
        print(f"Filtered {n_no_hs}/{n_total} complexes with no explicit Hs on ligand")

    dataset = dataset.select(lambda s: len(s.ligand.remove_hs()) <= pad_to)

    # Precompute binding profiles and ligand properties, then filter out failures
    if not args.trial_run:
        print("Preloading pocket dataset into memory...")
        dataset.preload(max_workers=32)

        n_before = len(dataset)
        dataset = dataset.select(lambda s: Features.BindingProfile.CACHE_KEY in s.ligand.meta)
        n_failed = n_before - len(dataset)
        if n_failed > 0:
            print(f"Filtered {n_failed}/{n_before} complexes with failed binding profiles")

        n_before = len(dataset)
        dataset = dataset.select(lambda s: s.ligand.meta["qed"] >= MIN_QED)
        n_filtered = n_before - len(dataset)
        if n_filtered > 0:
            print(f"Filtered {n_filtered}/{n_before} complexes with ligand QED less than {MIN_QED}")

    print(f"Num pocket training complexes: {len(dataset)}, replicated x{args.n_pocket_replicates}")

    dataset = dataset.replicate(args.n_pocket_replicates)
    return dataset


def load_pocket_val_dataset(args):
    """Load and prepare the pocket (complex) validation dataset."""

    print("Loading pocket validation dataset...")

    pad_to = get_max_size(args)

    eval_profile = Features.StochasticProfile(
        Features.BindingProfile(),
        pos_std_dev=0.3,
        rotate_prob=0.0,
        shape_resample=args.shape_resample
    )

    transform = partial(
        util.complex_transform,
        rand_rot=False,
        profile_feat=eval_profile,
        feat_dropout=1.0,
        pocket_rotate_prob=0.0
    )

    spindr_path = Path(args.spindr_path) / "val"
    dataset = ComplexDataset.load(spindr_path, transform=transform)

    n_total = len(dataset)
    dataset = dataset.select(lambda s: (s.ligand.atomics == 1).any())
    n_no_hs = n_total - len(dataset)
    if n_no_hs > 0:
        print(f"Filtered {n_no_hs}/{n_total} pocket val complexes with no explicit Hs on ligand")

    dataset = dataset.select(lambda s: len(s.ligand.remove_hs()) <= pad_to)

    if not args.trial_run:
        print("Preloading pocket validation dataset into memory...")
        dataset.preload(max_workers=32)

        n_before = len(dataset)
        dataset = dataset.select(lambda s: Features.BindingProfile.CACHE_KEY in s.ligand.meta)
        n_failed = n_before - len(dataset)
        if n_failed > 0:
            print(f"Filtered {n_failed}/{n_before} pocket val complexes with failed binding profiles")

    print(f"Num pocket validation complexes: {len(dataset)}")

    return dataset


def build_dm(args):
    if args.cat_strategy not in ["sample", "velocity", "mask"]:
        raise ValueError(f"Unsupported value of cat_strategy, '{args.cat_strategy}'")

    pad_to = get_max_size(args)
    cat_noise = "mask" if args.cat_strategy == "mask" else "uniform"
    prior_sampler = GraphNoise(cat_noise=cat_noise, zero_com=True)

    interpolant = GraphInterpolant(
        prior_sampler,
        coord_noise_std=args.coord_noise_std,
        perm_ot=args.perm_ot,
        time_alpha=args.time_alpha,
        time_beta=args.time_beta,
        pad_to=pad_to,
        pad_coord_mode=args.pad_coord_mode
    )

    eval_interpolant = GraphInterpolant(
        prior_sampler,
        perm_ot=False,
        pad_to=pad_to,
        pad_coord_mode=args.pad_coord_mode
    )

    train_mol_dataset, val_mol_dataset = load_mol_dataset(args)

    train_datasets = [train_mol_dataset]
    val_datasets = [val_mol_dataset]
    if args.spindr_path is not None:
        train_datasets.append(load_pocket_train_dataset(args))
        val_datasets.append(load_pocket_val_dataset(args))

    print("Creating datamodule...")

    dm = InterpolantDM(
        train_datasets,
        val_datasets,
        None,
        args.batch_cost,
        train_interpolant=interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=None,
        max_workers=16
    )

    print("Datamodule complete.")

    return dm


def build_trainer(args):
    epochs = 1 if args.trial_run else args.epochs
    log_steps = 1 if args.trial_run else 50

    val_check_epochs = 20

    project_name = f"{util.PROJECT_PREFIX}-geom-drugs"
    precision = get_precision(args)

    logger = (WandbLogger(project=project_name, save_dir="wandb", log_model=True)
              if args.wandb else CSVLogger("lightning_logs", name=project_name))
    devices = [args.device] if args.device is not None else "auto"

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpointing = ModelCheckpoint(
        every_n_epochs=val_check_epochs,
        monitor="val_uncond_validity",
        mode="max",
        save_last=True
    )

    # Overwrite if doing a trial run
    val_check_epochs = 1 if args.trial_run else val_check_epochs
    logger = None if args.trial_run else logger

    trainer = L.Trainer(
        min_epochs=epochs,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        gradient_clip_val=args.grad_clip_val,
        check_val_every_n_epoch=val_check_epochs,
        callbacks=[lr_monitor, checkpointing],
        precision=precision,
        devices=devices,
        # num_sanity_val_steps=0,
        # detect_anomaly=True
    )
    return trainer


def main(args):
    # Set some useful torch properties
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 128
    torch.set_float32_matmul_precision("high")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Loading datamodule...")
    dm = build_dm(args)
    print("Datamodule complete.")

    print(f"Building model architecture...")
    model = build_model(args, dm)
    print("Model complete.")

    trainer = build_trainer(args)

    print("Fitting datamodule to model...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")

    # Close any remaining open file pointers
    dm.close_datasets()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--wandb", action="store_true", help="Log to W&B instead of local CSV files")
    parser.add_argument("--spindr_path", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--trial_run", action="store_true")

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--d_edge", type=int, default=DEFAULT_D_EDGE)
    parser.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--n_blocks", type=int, default=DEFAULT_N_BLOCKS)
    parser.add_argument("--arch", type=str, default=DEFAULT_ARCH)

    parser.add_argument("--enc_d_model", type=int, default=DEFAULT_ENC_D_MODEL)
    parser.add_argument("--enc_n_heads", type=int, default=DEFAULT_ENC_N_HEADS)
    parser.add_argument("--enc_n_layers", type=int, default=DEFAULT_ENC_N_LAYERS)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--grad_clip_val", type=float, default=DEFAULT_GRAD_CLIP_VAL)
    parser.add_argument("--type_loss_weight", type=float, default=DEFAULT_TYPE_LOSS_WEIGHT)
    parser.add_argument("--bond_loss_weight", type=float, default=DEFAULT_BOND_LOSS_WEIGHT)
    parser.add_argument("--lr_schedule", type=str, default=DEFAULT_LR_SCHEDULE)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--ema_decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--coord_shift_std", type=float, default=DEFAULT_COORD_SHIFT_STD)
    parser.add_argument("--pad_coord_mode", type=str, default=DEFAULT_PAD_COORD_MODE)
    parser.add_argument("--n_pocket_replicates", type=int, default=DEFAULT_N_POCKET_REPLICATES)

    parser.add_argument("--no_learn_size", action="store_false", dest="learn_size")
    parser.add_argument("--no_perm_ot", action="store_false", dest="perm_ot")

    # CFG freq corresponds to the frequency of seeing the global CFG cond embeddings
    # CFG feat freq is the frequency of seeing individual components of embeddings
    parser.add_argument("--cfg_freq", type=float, default=DEFAULT_CFG_FREQ)
    parser.add_argument("--cfg_feat_freq", type=float, default=DEFAULT_CFG_FEAT_FREQ)

    # Profile augmentation args
    parser.add_argument("--shape_max_std_dev", type=float, default=DEFAULT_SHAPE_MAX_STD_DEV)
    parser.add_argument("--rotate_prob", type=float, default=DEFAULT_ROTATE_PROB)
    parser.add_argument("--shape_resample", type=float, default=DEFAULT_SHAPE_RESAMPLE)
    parser.add_argument("--mol_pharmaco_dropout", type=float, default=DEFAULT_MOL_PHARMACO_DROPOUT)
    parser.add_argument("--xtal_pharmaco_dropout", type=float, default=DEFAULT_XTAL_PHARMACO_DROPOUT)

    # Flow matching training args
    parser.add_argument("--coord_noise_std", type=float, default=DEFAULT_COORD_NOISE_STD)
    parser.add_argument("--time_alpha", type=float, default=DEFAULT_TIME_ALPHA)
    parser.add_argument("--time_beta", type=float, default=DEFAULT_TIME_BETA)

    # Flow matching sampling args
    parser.add_argument("--n_val_mols", type=int, default=DEFAULT_N_VAL_MOLS)
    parser.add_argument("--step_size", type=str, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--n_inf_steps", type=int, default=DEFAULT_N_INF_STEPS)
    parser.add_argument("--cat_noise_level", type=int, default=DEFAULT_CAT_NOISE_LEVEL)
    parser.add_argument("--cat_strategy", type=str, default=DEFAULT_CAT_STRATEGY)

    parser.set_defaults(
        trial_run=False,
        learn_size=True,
        perm_ot=True
    )

    args = parser.parse_args()
    main(args)
