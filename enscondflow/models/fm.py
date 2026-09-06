import tempfile
from typing import Optional

import torch
import numpy as np
import lightning as L
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import LinearLR

import enscondflow.util.functional as smolF
import enscondflow.eval.metrics as Metrics
from enscondflow.eval.builder import MolBuilder
from enscondflow.eval.integrator import Integrator
from enscondflow.eval.docking import prepare_receptor_batch, dock_batch_parallel
from enscondflow.repr import AtomVocab, BondVocab


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


class EnsCondFlow(L.LightningModule):
    def __init__(
        self,
        generator: torch.nn.Module,
        encoder: torch.nn.Module,
        lr: float,
        integrator: Integrator,
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        compile_model: bool = False,
        lr_schedule: str = "constant",
        warm_up_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        ema_decay: Optional[float] = None,
        cfg_freq: Optional[float] = 0.9,
        **kwargs
    ):
        super().__init__()

        if lr_schedule not in ["constant"]:
            raise ValueError(f"LR scheduler {lr_schedule} not supported.")

        enc_hparams = {f"enc-{name}": val for name, val in encoder.hparams.items()}

        self.generator = generator
        self.encoder = encoder
        self.lr = lr
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.compile_model = compile_model
        self.lr_schedule = lr_schedule
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.cfg_freq = cfg_freq

        self.n_atom_names = len(AtomVocab)
        self.n_bond_types = len(BondVocab)

        self.init_params_()

        if ema_decay is not None:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay)
            ema_gen = torch.optim.swa_utils.AveragedModel(generator, multi_avg_fn=avg_fn)

        if compile_model:
            self.generator = self._compile_model(generator)
            ema_gen = self._compile_model(ema_gen) if ema_decay is not None else None

        self.integrator = integrator
        self.ema_gen = ema_gen if ema_decay is not None else None

        # Anything else passed into kwargs will also be saved
        hparams = {
            "lr": lr,
            "type_loss_weight": type_loss_weight,
            "bond_loss_weight": bond_loss_weight,
            "lr_schedule": lr_schedule,
            "compile_model": compile_model,
            "warm_up_steps": warm_up_steps,
            "ema_decay": ema_decay,
            "cfg_freq": cfg_freq,
            **generator.hparams,
            **enc_hparams,
            **integrator.hparams,
            **kwargs
        }
        self.save_hyperparameters(hparams)

        uncond_metrics = self._create_metrics()
        self.uncond_gen_metrics = uncond_metrics[0]
        self.uncond_pair_metrics = uncond_metrics[1]
        self.uncond_align_metrics = uncond_metrics[2]

        shape_metrics = self._create_metrics()
        self.shape_gen_metrics = shape_metrics[0]
        self.shape_pair_metrics = shape_metrics[1]
        self.shape_align_metrics = shape_metrics[2]

        profile_metrics = self._create_metrics()
        self.profile_gen_metrics = profile_metrics[0]
        self.profile_pair_metrics = profile_metrics[1]
        self.profile_align_metrics = profile_metrics[2]

        pocket_metrics = self._create_metrics()
        self.pocket_gen_metrics = pocket_metrics[0]
        self.pocket_pair_metrics = pocket_metrics[1]
        self.pocket_align_metrics = pocket_metrics[2]
        self._pocket_val_updated = False
        self._pocket_vina_scores = []


    # ***************************************************************************
    # **************************** Core functions *******************************
    # ***************************************************************************


    def forward(self, batch, ada_latents, times, cond_latents=None, cond_mask=None, training=False):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            ada_latents (Tensor): Latent embeddings for AdaLN conditioning (props + mode flags), shape [B, d_model]
            times (Tensor): Interpolation times between 0 and 1, shape [B]
            cond_latents (Tensor): Conditioning embeddings, shape [B, L, d_model], None means uncond generation
            cond_mask (Tensor): Mask for cond tokens, shape [B, L], 1 for real tokens, 0 otherwise
            training (bool): Whether to run forward in training mode

        Returns:
            (Tensor, Tensor, Tensor): Coords, atom types, bond predicted (unormalised) distributions
        """

        coords = batch["coords"]
        atomics = batch["atomics"]
        bonds = batch["bonds"]
        mask = batch["mask"]

        if not training and self.ema_gen is not None:
            model = self.ema_gen
        else:
            model = self.generator

        out = model(
            coords,
            atomics,
            bonds,
            ada_latents,
            times.unsqueeze(-1),
            mask=mask,
            conds=cond_latents,
            cond_mask=cond_mask
        )
        return out

    def encode_profiles(self, features, mask_pharmas=False, mask_shapes=False, training=False):
        if mask_pharmas and mask_shapes:
            raise ValueError("At least one of shape and pharmacophore features must be unmasked.")

        points = features["profile"]["coords"]
        types = features["profile"]["types"]
        directions = features["profile"]["directions"]
        rotated = features["profile"]["rotated"]
        pos_noise_std = features["profile"]["pos_noise_std"]
        base_mask = features["profile"]["mask"]

        # Build encoder mask: start from explicit pad mask, then filter by type if requested
        if mask_pharmas:
            mask = base_mask * (types == 1).long()
        elif mask_shapes:
            mask = base_mask * (types >= 2).long()
        else:
            mask = base_mask

        if training:
            profile_mode = features["profile"]["profile_mode"]
            latent = self.encoder.encode_profiles(
                points,
                types,
                directions,
                rotated,
                profile_mode,
                pos_noise_std,
                mask
            )
            return latent, mask

        if mask_pharmas:
            mode_val = 1
        elif mask_shapes:
            mode_val = 2
        else:
            mode_val = 0

        profile_mode = torch.full((points.size(0),), mode_val, dtype=torch.long, device=points.device)

        with torch.inference_mode():
            latent = self.encoder.encode_profiles(
                points,
                types,
                directions,
                rotated,
                profile_mode,
                pos_noise_std,
                mask
            )

        return latent, mask

    def encode_properties(self, features, training=False):
        pairwise_rmsds = features["standardised-mean-pairwise-rmsd"]
        psa3ds = features["standardised-psa3d"]

        pairwise_rmsd_mask = features["standardised-mean-pairwise-rmsd-mask"]
        psa3d_mask = features["standardised-psa3d-mask"]

        feats = torch.stack((pairwise_rmsds, psa3ds), dim=-1)
        mask = torch.stack((pairwise_rmsd_mask, psa3d_mask), dim=-1)

        if training:
            return self.encoder.encode_properties(feats, mask)

        with torch.inference_mode():
            return self.encoder.encode_properties(feats, mask)

    def encode_pocket(self, pocket, training=False):
        proteins = pocket["proteins"]
        rotated = pocket["rotated"]

        atoms = proteins["atomics"]
        coords = proteins["coords"]
        mask = proteins["mask"]

        if training:
            return self.encoder.encode_pocket(atoms, coords, rotated, mask)

        with torch.inference_mode():
            return self.encoder.encode_pocket(atoms, coords, rotated, mask)

    def create_latents(
        self,
        features,
        pocket=None,
        profile_cond: str = "profile",
        drop_props: bool = False,
        training: bool = False
    ):
        """Build cond tokens + ada_latents for the generator.

        At training time, CFG dropout is sampled per source with a shared global drop. At inference
        time, profile_cond and drop_props directly control the per-source drops. Pass pocket=None
        to drop pocket conditioning entirely.

        Cross-attn sources (profile, pocket) are dropped via the cond_mask only — the decoder's
        attention mask sends -inf to those positions, so their latent values don't affect output.
        Props feed directly into ada_latents, so dropped samples swap in the encoder's learned
        null_props vector.
        """

        if profile_cond not in {"profile", "shape", "pharma", "none"}:
            raise ValueError(f"profile_cond must be one of 'profile','shape','pharma','none', got '{profile_cond}'")

        B = features["profile"]["coords"].size(0)
        device = features["profile"]["coords"].device
        has_pocket = pocket is not None

        if training:
            prof_keep, props_keep, pocket_keep = self._sample_cfg_keep(B, device, has_pocket)
        else:
            prof_keep = torch.full((B,), profile_cond != "none", dtype=torch.bool, device=device)
            props_keep = torch.full((B,), not drop_props, dtype=torch.bool, device=device)
            pocket_keep = torch.full((B,), has_pocket, dtype=torch.bool, device=device)

        profile_latent, profile_mask = None, None
        if training or profile_cond != "none":
            profile_latent, profile_mask = self.encode_profiles(
                features,
                mask_pharmas=(profile_cond == "shape"),
                mask_shapes=(profile_cond == "pharma"),
                training=training
            )
            profile_mask = profile_mask * prof_keep.view(-1, 1).long()

        props_latent = self.encode_properties(features, training=training)
        null = self.encoder.null_props.unsqueeze(0).expand_as(props_latent)
        ada_latents = torch.where(props_keep.view(-1, 1), props_latent, null)

        pocket_latent, pocket_mask = None, None
        if has_pocket:
            pocket_latent = self.encode_pocket(pocket, training=training)
            pocket_mask = pocket["proteins"]["mask"] * pocket_keep.view(-1, 1).long()

        latents = [t for t in [profile_latent, pocket_latent] if t is not None]
        masks = [t for t in [profile_mask, pocket_mask] if t is not None]
        cond_tokens = torch.cat(latents, dim=1) if latents else None
        cond_mask = torch.cat(masks, dim=1) if masks else None

        return cond_tokens, cond_mask, ada_latents

    def _sample_cfg_keep(self, batch_size, device, has_pocket):
        """Sample CFG dropout keep flags: per-source with a shared global drop."""

        global_keep = torch.rand(batch_size, device=device) < self.cfg_freq
        prof_keep = (torch.rand(batch_size, device=device) < self.cfg_freq) & global_keep
        props_keep = (torch.rand(batch_size, device=device) < self.cfg_freq) & global_keep

        if has_pocket:
            pocket_keep = (torch.rand(batch_size, device=device) < self.cfg_freq) & global_keep
        else:
            pocket_keep = torch.zeros(batch_size, dtype=torch.bool, device=device)

        return prof_keep, props_keep, pocket_keep

    def null_ada_latents(self, batch_size, device):
        """Ada_latents for the fully-uncond state — used as the null prediction in CFG.

        Returns the encoder's learned null_props vector, matching the CFG-dropped training state.
        """

        return self.encoder.null_props.unsqueeze(0).expand(batch_size, -1)


    # ********************************************************************************
    # **************************** Lightning functions *******************************
    # ********************************************************************************


    def training_step(self, batch, b_idx):
        interp = batch["interp_mols"]
        data = batch["data_mols"]
        times = batch["times"]
        pocket = batch.get("pocket")

        cond_tokens, cond_mask, ada_latents = self.create_latents(batch["features"], pocket=pocket, training=True)
        coords, types, bonds = self(
            interp,
            ada_latents,
            times,
            cond_latents=cond_tokens,
            cond_mask=cond_mask,
            training=True
        )

        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds
        }

        losses = self._loss(data, predicted, times)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss

    def on_train_batch_end(self, outputs, batch, b_idx):
        if self.ema_gen is not None:
            self.ema_gen.update_parameters(self.generator)

    def validation_step(self, batch, b_idx):
        if batch.get("pocket") is not None:
            self._val_step_pocket(batch)
        else:
            self._val_step_mol(batch)

    def _val_step_mol(self, batch):
        data_mols = self.generate_mols(batch["data_mols"], sanitise=True)
        ref_profiles = batch["features"]["profile"]["raw"]

        uncond_gen_mols = self.predict(batch, cfg_gamma=0.0, drop_props=True, sanitise=True)
        shape_gen_mols = self.predict(batch, cfg_gamma=1.0, profile_cond="shape", drop_props=True, sanitise=True)
        profile_gen_mols = self.predict(batch, cfg_gamma=1.0, drop_props=True, sanitise=True)

        self.uncond_gen_metrics.update(uncond_gen_mols)
        self.shape_gen_metrics.update(shape_gen_mols)
        self.profile_gen_metrics.update(profile_gen_mols)

        # Score overlaps directly (gen mols are already in the reference frame)
        uncond_aligned = Metrics.score_confs(uncond_gen_mols, data_mols, ref_profiles)
        shape_aligned = Metrics.score_confs(shape_gen_mols, data_mols, ref_profiles)
        profile_aligned = Metrics.score_confs(profile_gen_mols, data_mols, ref_profiles)

        self.uncond_pair_metrics.update(uncond_gen_mols, data_mols)
        self.shape_pair_metrics.update(shape_gen_mols, data_mols)
        self.profile_pair_metrics.update(profile_gen_mols, data_mols)

        self.uncond_align_metrics.update(uncond_gen_mols, data_mols, uncond_aligned)
        self.shape_align_metrics.update(shape_gen_mols, data_mols, shape_aligned)
        self.profile_align_metrics.update(profile_gen_mols, data_mols, profile_aligned)

    def _val_step_pocket(self, batch):
        data_mols = self.generate_mols(batch["data_mols"], sanitise=True)
        ref_profiles = batch["features"]["profile"]["raw"]

        pocket_gen_mols = self.predict(batch, cfg_gamma=1.0, profile_cond="pharma", drop_props=True, sanitise=True)

        self._pocket_val_updated = True
        self.pocket_gen_metrics.update(pocket_gen_mols)

        pocket_aligned = Metrics.score_confs(pocket_gen_mols, data_mols, ref_profiles)
        self.pocket_pair_metrics.update(pocket_gen_mols, data_mols)
        self.pocket_align_metrics.update(pocket_gen_mols, data_mols, pocket_aligned)

        self._score_pocket_docking(pocket_gen_mols, batch)

    def _score_pocket_docking(self, gen_mols, batch):
        raw_proteins = batch["pocket"]["raw_proteins"]
        ref_coords = batch["data_mols"]["coords"].cpu().numpy()
        real_atoms = batch["data_mols"]["atomics"].cpu().numpy() != 0
        ref_coords_list = [coords[m] for coords, m in zip(ref_coords, real_atoms)]

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                receptor_paths = prepare_receptor_batch([p.read() for p in raw_proteins], tmp_dir, n_workers=8)
                n_receptors = sum(1 for p in receptor_paths if p is not None)
                results = dock_batch_parallel(gen_mols, receptor_paths, ref_coords_list, mode="score", n_workers=8)

            n_scored = sum(1 for r in results if r is not None)
            scores = [r.best_affinity for r in results if r is not None and not np.isnan(r.best_affinity)]
            self._pocket_vina_scores.extend(scores)
            print(f"Docking: {n_receptors}/{len(raw_proteins)} receptors, {n_scored}/{len(gen_mols)} scored, mean={np.mean(scores):.1f}, median={np.median(scores):.1f}" if scores else "Docking: no scores")
        except Exception as e:
            print(f"Docking scoring failed: {e}")

    def on_validation_epoch_end(self):
        uncond_metrics = {
            **self.uncond_gen_metrics.compute(),
            **self.uncond_pair_metrics.compute(),
            **self.uncond_align_metrics.compute()
        }
        shape_metrics = {
            **self.shape_gen_metrics.compute(),
            **self.shape_pair_metrics.compute(),
            **self.shape_align_metrics.compute()
        }
        profile_metrics = {
            **self.profile_gen_metrics.compute(),
            **self.profile_pair_metrics.compute(),
            **self.profile_align_metrics.compute()
        }

        uncond_metrics = {f"uncond_{name}": val for name, val in uncond_metrics.items()}
        shape_metrics = {f"shapecond_{name}": val for name, val in shape_metrics.items()}
        profile_metrics = {f"profilecond_{name}": val for name, val in profile_metrics.items()}
        metrics = {**uncond_metrics, **profile_metrics, **shape_metrics}

        if self._pocket_val_updated:
            pocket_metrics = {
                **self.pocket_gen_metrics.compute(),
                **self.pocket_pair_metrics.compute(),
                **self.pocket_align_metrics.compute()
            }
            if self._pocket_vina_scores:
                pocket_metrics["vina-score"] = float(np.mean(self._pocket_vina_scores))

            pocket_metrics = {f"pocketcond_{name}": val for name, val in pocket_metrics.items()}
            metrics.update(pocket_metrics)

        for metric, value in metrics.items():
            progbar = True if metric == "uncond_validity" else False
            self.log(f"val_{metric}", value, on_epoch=True, logger=True, prog_bar=progbar)

        self.uncond_gen_metrics.reset()
        self.uncond_pair_metrics.reset()
        self.uncond_align_metrics.reset()

        self.shape_gen_metrics.reset()
        self.shape_pair_metrics.reset()
        self.shape_align_metrics.reset()

        self.profile_gen_metrics.reset()
        self.profile_pair_metrics.reset()
        self.profile_align_metrics.reset()

        self.pocket_gen_metrics.reset()
        self.pocket_pair_metrics.reset()
        self.pocket_align_metrics.reset()
        self._pocket_val_updated = False
        self._pocket_vina_scores = []

    def test_step(self, batch, b_idx):
        return self.validation_step(batch, b_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict(
        self,
        batch,
        cfg_gamma=0.0,
        profile_cond: str = "profile",
        drop_pocket: bool = False,
        drop_props: bool = False,
        sanitise: bool = True
    ):
        prior = batch["prior_mols"]
        pocket = None if drop_pocket else batch.get("pocket")

        cond_tokens, cond_mask, ada_ls = self.create_latents(
            batch["features"],
            pocket=pocket,
            profile_cond=profile_cond,
            drop_props=drop_props,
            training=False
        )

        gen_batch = self.generate(
            prior,
            cond_tokens,
            cond_mask,
            ada_ls,
            self.integrator.steps,
            step_strategy=self.integrator.step_size,
            cfg_gamma=cfg_gamma
        )

        gen_mols = self.generate_mols(gen_batch, sanitise=sanitise)
        return gen_mols

    def configure_optimizers(self):
        params = [
            {"params": self.generator.parameters()},
            {"params": self.encoder.parameters()}
        ]

        opt = torch.optim.Adam(params, lr=self.lr, amsgrad=True, foreach=True, weight_decay=0.0)

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warm_up_steps)
        else:
            raise ValueError(f"LR schedule {self.lr_schedule} is not supported.")

        config = {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
        return config


    # ************************************************************************
    # **************************** Loss functions ****************************
    # ************************************************************************


    def _loss(self, data, predicted, t):
        pred_coords = predicted["coords"]
        coords = data["coords"]
        mask = data["mask"].unsqueeze(-1)

        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        coord_loss = (coord_loss * mask).mean(dim=-1).sum(dim=1)

        type_loss = self._type_loss(data, predicted)
        bond_loss = self._bond_loss(data, predicted)

        t_weight = (t / (1 - t)).clamp(max=10.0)

        coord_loss = (coord_loss * t_weight).mean()
        type_loss = (type_loss * t_weight).mean() * self.type_loss_weight
        bond_loss = (bond_loss * t_weight).mean() * self.bond_loss_weight

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss
        }
        return losses

    def _type_loss(self, data, predicted, eps=1e-3):
        bs, n_atoms = data["atomics"].size()

        pred_logits = predicted["atomics"].flatten(0, 1)
        atomics = data["atomics"].flatten(0, 1)
        mask = data["mask"]

        type_loss = F.cross_entropy(pred_logits, atomics, reduction="none")
        type_loss = type_loss.unflatten(0, (bs, n_atoms))
        type_loss = (type_loss * mask).sum(dim=1)

        return type_loss

    def _bond_loss(self, data, predicted, eps=1e-3):
        bs, n_atoms = data["atomics"].size()

        pred_logits = predicted["bonds"].flatten(0, 2)
        bonds = data["bonds"].flatten(0, 2)

        bond_loss = F.cross_entropy(pred_logits, bonds, reduction="none")
        bond_loss = bond_loss.unflatten(0, (bs, n_atoms, n_atoms))

        adj_matrix = smolF.adj_from_node_mask(data["mask"], self_connect=True)
        bond_loss_per_atom = (bond_loss * adj_matrix).sum(dim=2) / (adj_matrix.sum(dim=2) + eps)
        bond_loss = bond_loss_per_atom.sum(dim=1)

        return bond_loss


    # **************************************************************************
    # ************************* Molecule sampling ******************************
    # **************************************************************************


    def generate_mols(self, generated, sanitise=True):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        masks = generated["mask"]

        # Set n_workers=0 to avoid deadlock with DataLoader multiprocessing
        mols = MolBuilder.mols_from_tensors(
            atom_dists,
            bond_dists=bond_dists,
            coords=coords,
            mask=masks,
            sanitise=sanitise,
            n_workers=None
        )
        return mols

    def generate(
        self,
        prior,
        cond_tokens,
        cond_mask,
        ada_latents,
        n_steps,
        step_strategy="decay",
        cfg_gamma=0.0,
        snapshot_every=None
    ):
        """Integrate the flow from the prior to t=1.

        Pass snapshot_every to also return the intermediate states, taken every n steps, as a list of
        (time, state) pairs. Used by the symmetry evaluation to probe the model along its own
        trajectory rather than along an interpolation towards a known molecule.
        """

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = self._create_step_sizes(n_steps, step_strategy)
        snapshots = []

        # Skip predictions that would be weighted by 0
        use_cond = cfg_gamma != 0.0
        use_null = cfg_gamma != 1.0
        assert use_cond or use_null

        null_ada_ls = self.null_ada_latents(prior["coords"].size(0), self.device) if use_null else None

        device = "cpu" if self.device.type == "mps" else self.device
        curr = {k: v.clone().to(device) for k, v in prior.items()}

        for step_idx, step in enumerate(step_sizes):
            device_curr = {k: v.to(self.device) for k, v in curr.items()}

            if snapshot_every is not None and step_idx % snapshot_every == 0:
                snapshots.append((float(times[0].item()), {k: v.clone() for k, v in curr.items()}))

            cond_pred = self._model_pred(device_curr, times, cond_tokens, cond_mask, ada_latents) if use_cond else None
            null_pred = self._model_pred(device_curr, times, None, None, null_ada_ls) if use_null else None

            if use_cond and use_null:
                coord_vel = self._cfg_combine_continuous(cond_pred["coord_vel"], null_pred["coord_vel"], cfg_gamma)
                atomics = self._cfg_combine_discrete(cond_pred["atomics"], null_pred["atomics"], cfg_gamma)
                bonds = self._cfg_combine_discrete(cond_pred["bonds"], null_pred["bonds"], cfg_gamma)

                # Provide a prediction for coords here too so generate can access a prediction on the final step
                pred_coords = cond_pred["coords"] if cfg_gamma >= 0.5 else null_pred["coords"]

                pred = {
                    "coords": pred_coords,
                    "coord_vel": coord_vel,
                    "atomics": atomics,
                    "bonds": bonds,
                    "mask": curr["mask"]
                }
            else:
                pred = cond_pred if use_cond else null_pred

            curr = self.integrator.step(curr, pred, times.to(device), step, coord_vel=pred["coord_vel"])
            times = times + step

        if snapshot_every is not None:
            return pred, snapshots

        return pred

    def generate_multi_cond(
        self,
        prior,
        enc_cond_a,
        cond_a_mask,
        enc_cond_b,
        cond_b_mask,
        ada_latents,
        n_steps,
        step_strategy="decay",
        negate_b=False,
        cfg_gamma=1.0
    ):
        """Generate molecules conditioned on two shape profiles.

        When cfg_gamma != 1.0, the combined pred is amplified against an uncond pred using standard CFG.

        ada_latents should be the cond-side vector for both A and B (they share mode flags);
        the fully-uncond null is computed internally via null_ada_latents.
        """

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = self._create_step_sizes(n_steps, step_strategy)

        use_uncond = cfg_gamma != 1.0
        null_ada_ls = self.null_ada_latents(prior["coords"].size(0), self.device) if use_uncond else None

        gamma = 1.5 if negate_b else 0.5
        device = "cpu" if self.device.type == "mps" else self.device
        curr = {k: v.clone().to(device) for k, v in prior.items()}

        for step in step_sizes:
            device_curr = {k: v.to(self.device) for k, v in curr.items()}

            pred_uncond = self._model_pred(device_curr, times, None, None, null_ada_ls) if use_uncond else None
            pred_a = self._model_pred(device_curr, times, enc_cond_a, cond_a_mask, ada_latents)
            pred_b = self._model_pred(device_curr, times, enc_cond_b, cond_b_mask, ada_latents)

            coord_vel = self._cfg_combine_continuous(pred_a["coord_vel"], pred_b["coord_vel"], gamma)
            atomics = self._cfg_combine_discrete(pred_a["atomics"], pred_b["atomics"], gamma)
            bonds = self._cfg_combine_discrete(pred_a["bonds"], pred_b["bonds"], gamma)

            if cfg_gamma != 1.0:
                coord_vel = self._cfg_combine_continuous(coord_vel, pred_uncond["coord_vel"], cfg_gamma)
                atomics = self._cfg_combine_discrete(atomics, pred_uncond["atomics"], cfg_gamma)
                bonds = self._cfg_combine_discrete(bonds, pred_uncond["bonds"], cfg_gamma)

            pred = {
                "coords": pred_a["coords"],
                "coord_vel": coord_vel,
                "atomics": atomics,
                "bonds": bonds,
                "mask": curr["mask"]
            }

            curr = self.integrator.step(curr, pred, times.to(device), step, coord_vel=coord_vel)
            times = times + step

        return pred

    def generate_multi_pocket(
        self,
        prior,
        on_pocket,
        off_pockets,
        ada_latents,
        n_steps,
        step_strategy="decay",
        neg_weight=0.5,
        reduce="mean",
        cfg_gamma=1.0
    ):
        """Generate molecules conditioned on an on-target pocket while steering away from off-target pockets.

        The on-target pocket is the positive (equivariant, rotated=0) conditioner that fixes the frame; each
        off-target pocket is a negative conditioner (typically invariant, rotated=1). At every step the model
        is run once per pocket and the predictions are combined with contrastive guidance:

            coord_vel   = on + (neg_weight / norm) * sum_k (on - off_k)
            p(atomic)  ~= p_on * prod_k (p_on / p_off_k) ** (neg_weight / norm)

        With a single off-target and neg_weight=0.5 this reduces exactly to generate_multi_cond(negate_b=True),
        i.e. 1.5*on - 0.5*off. reduce="mean" sets norm=len(off_pockets) so the negative contribution stays a
        stable magnitude as more off-targets are added; reduce="sum" (norm=1) pushes away from each one
        independently. An empty off_pockets list gives plain on-target pocket generation.

        on_pocket / off_pockets are pocket dicts of the form the interpolant produces:
        {"proteins": {"atomics", "coords", "mask"}, "rotated": ...}. ada_latents is shared across all pockets.
        """

        enc_on = self.encode_pocket(on_pocket)
        mask_on = on_pocket["proteins"]["mask"]
        enc_offs = [self.encode_pocket(pocket) for pocket in off_pockets]
        mask_offs = [pocket["proteins"]["mask"] for pocket in off_pockets]

        return self.generate_multi_latent(
            prior,
            enc_on,
            mask_on,
            enc_offs,
            mask_offs,
            ada_latents,
            n_steps,
            step_strategy=step_strategy,
            neg_weight=neg_weight,
            reduce=reduce,
            cfg_gamma=cfg_gamma
        )

    def generate_multi_latent(
        self,
        prior,
        enc_on,
        mask_on,
        enc_offs,
        mask_offs,
        ada_latents,
        n_steps,
        step_strategy="decay",
        neg_weight=0.5,
        reduce="mean",
        cfg_gamma=1.0
    ):
        """Contrastive generation from pre-encoded conditioning latents.

        Same contrastive guidance as generate_multi_pocket but operating on already-encoded (cond_tokens, mask)
        pairs rather than raw pockets. This lets each conditioner mix pocket and profile tokens — e.g. an
        off-target encoded as its pocket concatenated with the ref-ligand pharmacophores via create_latents.
        enc_on is the positive; each enc_offs[k] is a negative. Reduces to generate_multi_pocket exactly when
        the latents are pure pocket encodings.
        """

        if reduce not in {"mean", "sum"}:
            raise ValueError(f"reduce must be 'mean' or 'sum', got '{reduce}'")

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = self._create_step_sizes(n_steps, step_strategy)

        use_uncond = cfg_gamma != 1.0
        null_ada_ls = self.null_ada_latents(prior["coords"].size(0), self.device) if use_uncond else None

        k = len(enc_offs)
        norm = k if (reduce == "mean" and k > 0) else 1

        device = "cpu" if self.device.type == "mps" else self.device
        curr = {key: val.clone().to(device) for key, val in prior.items()}

        for step in step_sizes:
            device_curr = {key: val.to(self.device) for key, val in curr.items()}

            pred_uncond = self._model_pred(device_curr, times, None, None, null_ada_ls) if use_uncond else None
            pred_on = self._model_pred(device_curr, times, enc_on, mask_on, ada_latents)
            preds_off = [self._model_pred(device_curr, times, e, m, ada_latents) for e, m in zip(enc_offs, mask_offs)]

            coord_vel = pred_on["coord_vel"]
            atomics = pred_on["atomics"]
            bonds = pred_on["bonds"]

            for pred_off in preds_off:
                coord_vel = coord_vel + (neg_weight / norm) * (pred_on["coord_vel"] - pred_off["coord_vel"])
                atom_ratio = pred_on["atomics"].clamp(min=1e-10) / pred_off["atomics"].clamp(min=1e-10)
                bond_ratio = pred_on["bonds"].clamp(min=1e-10) / pred_off["bonds"].clamp(min=1e-10)
                atomics = atomics * torch.pow(atom_ratio, neg_weight / norm)
                bonds = bonds * torch.pow(bond_ratio, neg_weight / norm)

            if cfg_gamma != 1.0:
                coord_vel = self._cfg_combine_continuous(coord_vel, pred_uncond["coord_vel"], cfg_gamma)
                atomics = self._cfg_combine_discrete(atomics, pred_uncond["atomics"], cfg_gamma)
                bonds = self._cfg_combine_discrete(bonds, pred_uncond["bonds"], cfg_gamma)

            pred = {
                "coords": pred_on["coords"],
                "coord_vel": coord_vel,
                "atomics": atomics,
                "bonds": bonds,
                "mask": curr["mask"]
            }

            curr = self.integrator.step(curr, pred, times.to(device), step, coord_vel=coord_vel)
            times = times + step

        return pred

    def generate_multi_pos_neg(
        self,
        prior,
        pos_latents,
        pos_masks,
        neg_latents,
        neg_masks,
        ada_latents,
        n_steps,
        step_strategy="decay",
        neg_weight=0.5,
        reduce="mean",
        cfg_gamma=1.0,
        pos_weights=None
    ):
        """Generate from MULTIPLE positive conditioners (pooled -> multi-target binding) while steering away
        from negative conditioners (contrastive). Positives are pooled per step — weighted-mean velocity,
        weighted-geometric-mean distributions — then each negative contributes (pos - neg) like
        generate_multi_latent:

            coord_vel   = pos + (neg_weight / norm) * sum_k (pos - neg_k)
            p(atomic)  ~= p_pos * prod_k (p_pos / p_neg_k) ** (neg_weight / norm)

        where pos is the pooled positive. This is the multi-positive generalisation of generate_multi_latent:
        with a single positive and the given negatives it matches generate_multi_latent exactly; with two
        positives and no negatives it matches generate_multi_cond(negate_b=False) (0.5*A + 0.5*B). Every
        conditioner is a pre-encoded (cond_tokens, mask) pair, so each may mix pocket and profile tokens.
        ada_latents is shared across all conditioners.

        pos_weights optionally weights the positive pool (e.g. to upweight a weaker invariant target); it is
        normalised to sum to 1 and defaults to equal weights, which recovers the plain mean / geometric mean.
        """

        if reduce not in {"mean", "sum"}:
            raise ValueError(f"reduce must be 'mean' or 'sum', got '{reduce}'")
        if len(pos_latents) == 0:
            raise ValueError("generate_multi_pos_neg needs at least one positive conditioner")

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = self._create_step_sizes(n_steps, step_strategy)

        use_uncond = cfg_gamma != 1.0
        null_ada_ls = self.null_ada_latents(prior["coords"].size(0), self.device) if use_uncond else None

        n_pos = len(pos_latents)
        if pos_weights is None:
            pos_weights = [1.0 / n_pos] * n_pos
        else:
            if len(pos_weights) != n_pos:
                raise ValueError(f"pos_weights must have {n_pos} entries, got {len(pos_weights)}")

            total = float(sum(pos_weights))
            pos_weights = [w / total for w in pos_weights]

        k = len(neg_latents)
        norm = k if (reduce == "mean" and k > 0) else 1

        device = "cpu" if self.device.type == "mps" else self.device
        curr = {key: val.clone().to(device) for key, val in prior.items()}

        for step in step_sizes:
            device_curr = {key: val.to(self.device) for key, val in curr.items()}

            pred_uncond = self._model_pred(device_curr, times, None, None, null_ada_ls) if use_uncond else None
            preds_pos = [self._model_pred(device_curr, times, e, m, ada_latents) for e, m in zip(pos_latents, pos_masks)]
            preds_neg = [self._model_pred(device_curr, times, e, m, ada_latents) for e, m in zip(neg_latents, neg_masks)]

            # Pool positives: weighted-mean velocity, weighted-geometric-mean distributions
            pos_vel = sum(w * p["coord_vel"] for w, p in zip(pos_weights, preds_pos))
            pos_atom = torch.ones_like(preds_pos[0]["atomics"])
            pos_bond = torch.ones_like(preds_pos[0]["bonds"])
            for w, p in zip(pos_weights, preds_pos):
                pos_atom = pos_atom * torch.pow(p["atomics"].clamp(min=1e-10), w)
                pos_bond = pos_bond * torch.pow(p["bonds"].clamp(min=1e-10), w)

            coord_vel, atomics, bonds = pos_vel, pos_atom, pos_bond
            for pred_neg in preds_neg:
                coord_vel = coord_vel + (neg_weight / norm) * (pos_vel - pred_neg["coord_vel"])
                atom_ratio = pos_atom / pred_neg["atomics"].clamp(min=1e-10)
                bond_ratio = pos_bond / pred_neg["bonds"].clamp(min=1e-10)
                atomics = atomics * torch.pow(atom_ratio, neg_weight / norm)
                bonds = bonds * torch.pow(bond_ratio, neg_weight / norm)

            if cfg_gamma != 1.0:
                coord_vel = self._cfg_combine_continuous(coord_vel, pred_uncond["coord_vel"], cfg_gamma)
                atomics = self._cfg_combine_discrete(atomics, pred_uncond["atomics"], cfg_gamma)
                bonds = self._cfg_combine_discrete(bonds, pred_uncond["bonds"], cfg_gamma)

            pred = {
                "coords": preds_pos[0]["coords"],
                "coord_vel": coord_vel,
                "atomics": atomics,
                "bonds": bonds,
                "mask": curr["mask"]
            }

            curr = self.integrator.step(curr, pred, times.to(device), step, coord_vel=coord_vel)
            times = times + step

        return pred

    def _model_pred(self, curr, times, cond_tokens, cond_mask, ada_latents):
        device = "cpu" if self.device.type == "mps" else self.device

        with torch.inference_mode():
            out = self(curr, ada_latents, times, cond_latents=cond_tokens, cond_mask=cond_mask, training=False)
            coords, type_logits, bond_logits = out

            coord_vel = coords - curr["coords"]
            type_probs = F.softmax(type_logits.to(device).to(torch.float64), dim=-1)
            bond_probs = F.softmax(bond_logits.to(device).to(torch.float64), dim=-1)

            # NOTE we define the coord vel as the non-time-normalised velocity
            # This makes it simpler to define time schedules in integrator
            # But this still leads to equivalent behaviour since the CFG is just a linear combination
            pred = {
                "coords": coords.to(device),
                "coord_vel": coord_vel.to(device),
                "atomics": type_probs.to(device),
                "bonds": bond_probs.to(device),
                "mask": curr["mask"].to(device)
            }

        return pred

    @staticmethod
    def _create_step_sizes(n_steps, step_strategy):
        if step_strategy == "constant":
            time_points = np.linspace(0, 0.999, n_steps + 1).tolist()

        elif step_strategy == "decay":
            time_points = (1 - np.geomspace(0.01, 0.999, n_steps + 1)).tolist()
            time_points.reverse()

        else:
            raise ValueError(f"Unknown ODE integration step size '{step_strategy}'")

        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        return step_sizes

    @staticmethod
    def _cfg_combine_continuous(cond_vel, null_vel, cfg_gamma):
        return (cond_vel * cfg_gamma) + ((1 - cfg_gamma) * null_vel)

    @staticmethod
    def _cfg_combine_discrete(cond_pred, null_pred, cfg_gamma):
        cond_pred_pow = torch.pow(cond_pred.clamp(min=1e-10), cfg_gamma)
        null_pred_pow = torch.pow(null_pred.clamp(min=1e-10), 1 - cfg_gamma)
        pred_dist = cond_pred_pow * null_pred_pow

        # Note we turn off dist normalisation since the integrator does technically handle rate matrices directly
        # The unormalised rate is the true integration method from the guided discrete FM framework
        # pred_dist = pred_dist / pred_dist.sum(dim=-1, keepdim=True)

        return pred_dist

    @staticmethod
    def _dombi_conjunction(pred1, pred2, lambda_):
        p1_neg = torch.pow(pred1.clamp(min=1e-10), -lambda_)
        p2_neg = torch.pow(pred2.clamp(min=1e-10), -lambda_)
        pred_dist = torch.pow(p1_neg + p2_neg, -1.0 / lambda_)
        pred_dist = pred_dist / pred_dist.sum(dim=-1, keepdim=True)
        return pred_dist

    @staticmethod
    def _dombi_negation(pred, null_pred):
        neg_pred = null_pred.clamp(min=1e-10) ** 2 / pred.clamp(min=1e-10)
        neg_pred = neg_pred / neg_pred.sum(dim=-1, keepdim=True)
        return neg_pred


    # ******************************************************************************
    # ****************************** Other helpers *********************************
    # ******************************************************************************


    def init_params_(self):
        # Xavier for Linear weights only. Everything else (Linear biases, Embeddings, LayerNorms,
        # bare Parameters) keeps PyTorch defaults — matches the pre-refactor behaviour except
        # Embeddings are no longer shrunk by Xavier.
        with torch.no_grad():
            for root in [self.generator, self.encoder]:
                for module in root.modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight)

        # Then call module-specific init_params_ if it exists
        for root in [self.generator, self.encoder]:
            for module in root.modules():
                if hasattr(module, "init_params_"):
                    module.init_params_()

    def _compile_model(self, model):
        return torch.compile(model, dynamic=False, fullgraph=True)

    def _create_metrics(self):
        gen_metrics = {
            "validity": Metrics.Validity(),
            "fc-validity": Metrics.Validity(connected=True),
            "uniqueness": Metrics.Uniqueness(),
            "energy": Metrics.AverageEnergy(),
            "energy-validity": Metrics.EnergyValidity(),
            "average-size": Metrics.AverageSize()
        }

        pair_metrics = {
            "ref-match": Metrics.ReconstructionAccuracy(),
            "ecfp-tanimoto": Metrics.ECFPTanimoto()
        }

        alignment_metrics = {
            "shape-tanimoto": Metrics.ShapeTanimoto(),
            "colour-tanimoto": Metrics.ColourTanimoto(),
            "interaction-recovery": Metrics.InteractionRecovery()
        }

        gen_group = MetricCollection(gen_metrics, compute_groups=False)
        pair_group = MetricCollection(pair_metrics, compute_groups=False)
        alignment_group = MetricCollection(alignment_metrics, compute_groups=False)

        return gen_group, pair_group, alignment_group
