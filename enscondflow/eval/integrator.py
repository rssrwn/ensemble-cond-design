import torch

import enscondflow.util.functional as smolF
from enscondflow.repr.vocab import AtomVocab, BondVocab


class Integrator:
    def __init__(
        self,
        steps,
        step_size="decay",
        coord_noise_std=0.0,
        cat_strategy="sample",
        cat_noise_level=0,
        coord_sch="identity",
        atom_sch="identity",
        bond_sch="identity",
        corrector_sch_a=0.0,
        corrector_sch_b=0.0,
        eps=1e-5
    ):
        if cat_strategy not in ["sample", "mask", "velocity"]:
            raise ValueError(f"Categorical sampling strategy '{cat_strategy}' is not supported.")

        if cat_strategy == "mask" and (atom_sch != "identity" or bond_sch != "identity"):
            raise ValueError("Scheduling is not supported with mask sampling.")

        atom_mask_idx = AtomVocab.get_index("MASK")
        bond_mask_idx = BondVocab.get_index("MASK")

        self.steps = steps
        self.step_size = step_size
        self.coord_noise_std = coord_noise_std

        self.cat_strategy = cat_strategy
        self.cat_noise_level = cat_noise_level

        self.coord_sch = coord_sch
        self.atom_sch = atom_sch
        self.bond_sch = bond_sch

        self.corrector_sch_a = corrector_sch_a
        self.corrector_sch_b = corrector_sch_b

        self.atom_mask_idx = atom_mask_idx
        self.bond_mask_idx = bond_mask_idx
        self.eps = eps

    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "step-size": self.step_size,
            "integration-coord-noise-std": self.coord_noise_std,
            "integration-cat-strategy": self.cat_strategy,
            "integration-cat-noise-level": self.cat_noise_level,
            "integration-coord-schedule": self.coord_sch,
            "integration-atom-schedule": self.atom_sch,
            "integration-bond-schedule": self.bond_sch,
            "corrector-schedule-a": self.corrector_sch_a,
            "corrector-schedule-b": self.corrector_sch_b
        }

    def step(self, curr, pred_dists, t, step_size, coord_vel=None):
        """Take one integration step based on the prediction and current state.

        Args:
            curr (dict[str, Tensor]): Current state
            pred_dists (dict[str, Tensor]): Predicted state (in normalised distributions for discrete types)
            t (Tensor): Current time tensor, shape [batch_size]
            step_size (float): Step size for the integration step
            coord_vel (Tensor, optional): Optionally pass in the calculated velocity field for coords

        Returns:
            dict[str, Tensor]: Updated state after the integration step
        """

        curr_cs = curr["coords"]
        pred_cs = pred_dists["coords"]

        updated = self.step_graph(curr, pred_dists, t, step_size)
        coords = self.step_coords(curr_cs, pred_cs, t, step_size, coord_vel=coord_vel)

        updated["coords"] = coords

        if curr.get("res_names") is not None:
            updated["res_names"] = curr["res_names"]

        return updated

    def step_coords(self, curr_coords, pred_coords, t, step_size, coord_vel=None):
        coord_vel = pred_coords - curr_coords if coord_vel is None else coord_vel

        time_sch = self.schedule_time(self.coord_sch, t.view(-1, 1, 1))
        sch_deriv = self.schedule_time_derivative(self.coord_sch, t.view(-1, 1, 1))

        coord_vel = (coord_vel * sch_deriv) / ((1 - time_sch) + self.eps)
        coord_vel += torch.randn_like(coord_vel) * self.coord_noise_std
        coords = curr_coords + (step_size * coord_vel)

        return coords

    def step_graph(self, curr, preds, t, step_size):
        atom_mask = self.atom_mask_idx
        bond_mask = self.bond_mask_idx

        curr_as = curr["atomics"]
        curr_bs = curr["bonds"]

        pred_as = preds["atomics"]
        pred_bs = preds["bonds"]

        # Uniform sampling strategy from multiflow paper
        if self.cat_strategy == "sample":

            # For now use velocity sampling if we are using scheduling
            if self.atom_sch != "identity":
                atomics = self._velocity_sample_step(curr_as, pred_as, t, step_size, self.atom_sch)
            else:
                atomics = self._uniform_sample_step(curr_as, pred_as, t, step_size)

            if self.bond_sch != "identity":
                bonds = self._velocity_sample_step(curr_bs, pred_bs, t, step_size, self.bond_sch)
            else:
                bonds = self._uniform_sample_step(curr_bs, pred_bs, t, step_size)

        # Mask sampling strategy from multiflow paper
        elif self.cat_strategy == "mask":
            atomics = self._mask_sampling_step(curr_as, pred_as, t, step_size, atom_mask)
            bonds = self._mask_sampling_step(curr_bs, pred_bs, t, step_size, bond_mask)

        # Velocity sampling strategy from discrete FM paper
        elif self.cat_strategy == "velocity":
            atomics = self._velocity_sample_step(curr_as, pred_as, t, step_size, self.atom_sch)
            bonds = self._velocity_sample_step(curr_bs, pred_bs, t, step_size, self.bond_sch)

        updated = {
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"]
        }

        if curr.get("charges") is not None:
            updated["charges"] = curr["charges"]

        return updated

    def _uniform_sample_step(self, curr, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)
        curr = curr.unsqueeze(-1)

        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = self.cat_noise_level

        # Off-diagonal step probs
        mult = ((1 + noise + (noise * (n_categories - 1) * times)) / (1 - times))
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        samples = torch.distributions.Categorical(step_probs).sample()
        return samples

    def _mask_sampling_step(self, curr, pred_dist, t, step_size, mask_index):
        updated = curr.clone()
        pred = torch.distributions.Categorical(pred_dist).sample()

        ones = [1] * (len(pred.shape) - 1)
        times = t.view(-1, *ones)

        # Choose elements to unmask
        limit = (step_size * (1 + (self.cat_noise_level * times)) / (1 - times))
        unmask = torch.rand_like(pred.float()) < limit
        unmask = unmask * (updated == mask_index)

        # Choose elements to mask
        mask = torch.rand_like(pred.float()) < (step_size * self.cat_noise_level)
        mask = mask * (updated != mask_index)
        mask[t + step_size >= 1.0] = 0.0

        # Applying unmasking and re-masking
        updated[unmask] = pred[unmask]
        updated[mask] = mask_index 

        return updated

    def _velocity_sample_step(self, curr, pred_dist, t, step_size, sch):
        n_categories = pred_dist.size(-1)
        curr_dist = smolF.one_hot_encode_tensor(curr, n_categories)

        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)

        beta_t = torch.pow(times, self.corrector_sch_a) * torch.pow(1 - times, self.corrector_sch_b)
        beta_t = self.cat_noise_level * beta_t
        alpha_t = beta_t + 1

        time_sch = self.schedule_time(sch, times)
        sch_deriv = self.schedule_time_derivative(sch, times)

        # Backward velocity assumes uniform dist for prior
        forward_vel = (pred_dist - curr_dist) * (sch_deriv / (1 - time_sch))
        backward_vel = (curr_dist - (1 / n_categories)) * (sch_deriv / time_sch)

        prob_vel = (alpha_t * forward_vel) - (beta_t * backward_vel)
        step_dist = curr_dist + (step_size * prob_vel)
        step_dist = step_dist.clamp(min=0.0)

        samples = torch.distributions.Categorical(step_dist).sample()
        return samples

    @staticmethod
    def schedule_time(sch: str, t: torch.Tensor) -> torch.Tensor:
        if sch == "identity":
            return t
        elif sch == "polydec":
            return (2.0 * t) - (t ** 2)
        elif sch == "polyinc":
            return t ** 2
        else:
            raise ValueError(f"Unknown time schedule {sch}")

    @staticmethod
    def schedule_time_derivative(sch: str, t: torch.Tensor) -> torch.Tensor:
        if sch == "identity":
            return torch.ones_like(t)
        elif sch == "polydec":
            return 2.0 * (1 - t)
        elif sch == "polyinc":
            return 2.0 * t
        else:
            raise ValueError(f"Unknown time schedule {sch}")
