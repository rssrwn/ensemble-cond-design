import copy
import torch

from enscondflow.repr import AtomVocab
from enscondflow.util.rdkit import PharmacophoreFinder
from enscondflow.models.util import SelfAttention, NormParams, AdaLN, create_attention_mask


def node_attn_mask(node_mask, dtype=None):
    dtype = torch.float if dtype is None else dtype
    attn_mask = torch.zeros_like(node_mask, dtype=dtype)
    attn_mask[node_mask < 0.1] = float("-inf")
    return attn_mask


def get_clones(module, n):
    return [copy.deepcopy(module) for _ in range(n)]


# *****************************************************************************
# ****************************** Helper modules *******************************
# *****************************************************************************


class _PropEmbedding(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model)
        )

    def forward(self, features):
        return self.mlp(features.unsqueeze(-1))


class _ShapeLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_cond, ff_factor=4):
        super().__init__()

        self.ada_params = NormParams(d_model, n_blocks=2, cond=True, d_cond=d_cond)

        self.attn_norm = AdaLN(d_model)
        self.attn = SelfAttention(d_model, n_heads)

        self.mlp_norm = AdaLN(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * ff_factor),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model * ff_factor, d_model)
        )

    def forward(self, nodes, cond, attn_mask=None):
        """Pass data through shape encoder layer

        Args:
            nodes (Tensor): Node features, shape [B, N, d_model]
            cond (Tensor): Conditioning features, shape [B, d_cond]
            attn_mask (Tensor): Pre-computed attention mask, shape [B, N, N], 0 for attended, -inf for padded

        Returns:
            Tensor: Updated node features, shape [B, N, d_model]
        """

        adas = self.ada_params(cond)

        norm_nodes = self.attn_norm(nodes, adas[0].gamma, adas[0].beta)
        attn_out = self.attn(norm_nodes, attn_mask=attn_mask)
        nodes = (attn_out * adas[0].alpha.unsqueeze(1)) + nodes

        norm_nodes = self.mlp_norm(nodes, adas[1].gamma, adas[1].beta)
        mlp_out = self.mlp(norm_nodes)
        nodes = (mlp_out * adas[1].alpha.unsqueeze(1)) + nodes

        return nodes


# *****************************************************************************
# ***************************** Encoder modules *******************************
# *****************************************************************************


class _PropEncoder(torch.nn.Module):
    def __init__(self, d_model, n_props, d_out=None):
        super().__init__()

        d_out = d_model if d_out is None else d_out

        embs = [_PropEmbedding(d_model) for _ in range(n_props)]

        self.embs = torch.nn.ModuleList(embs)
        self.out_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_out)
        )

    def forward(self, props, mask):
        assert props.size(1) == len(self.embs)
        assert mask.size(1) == props.size(1)

        # Embed each feature into its own vector, then zero out masked features
        # This stops accidentally encoding a feature as having a value of 0
        embs = [emb(props[:, idx]) for idx, emb in enumerate(self.embs)]
        embs = torch.stack(embs).transpose(0, 1)
        embs = embs * mask.unsqueeze(-1)

        # Add embedded features together and pass through final MLP
        embs = embs.sum(dim=1)
        out = self.out_mlp(embs)

        return out


class _ProfileEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        n_types,
        d_out=None,
        d_emb=64,
        eps=1e-5
    ):
        super().__init__()

        d_out = d_model if d_out is None else d_out

        self.eps = eps

        # Node input embeddings (coords, types, directions)
        self.coord_emb = torch.nn.Linear(3, d_emb)
        self.type_emb = torch.nn.Embedding(n_types, d_emb)
        self.dir_emb = torch.nn.Linear(3, d_emb)
        self.node_emb = torch.nn.Linear(d_emb * 3, d_model)

        # Conditioning embeddings for adaptive norms
        # Rotations: 1 = rotated, 0 otherwise
        # Cond mode: 0 = both, 1 = shape-only, 2 = pharma-only
        # Sigma: continuous noise std dev applied to positions
        self.rot_emb = torch.nn.Embedding(2, d_model)
        self.mode_emb = torch.nn.Embedding(3, d_model)
        self.sigma_emb = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model)
        )
        self.cond_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model)
        )

        layer = _ShapeLayer(d_model, n_heads, d_cond=d_model)
        self.layers = torch.nn.ModuleList(get_clones(layer, n_layers))

        self.out_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_out)
        )

    def forward(self, points, types, directions, rotated, profile_mode, pos_noise_std, mask):
        """Produce a vector embedding for each conformer profile

        Args:
            points (Tensor): Shape and/or pharmacophore points, shape [B, N, 3]
            types (Tensor): Point types, shape [B, N], max must be no more than n_pharmacophores + 1
            directions (Tensor): Direction vectors, shape [B, N, 3]
            rotated (Tensor): Rotation flags, shape [B], int 0 or 1
            profile_mode (Tensor): Profile mode flags, shape [B], int 0=both, 1=shape-only, 2=pharma-only
            pos_noise_std (Tensor): Position noise std dev, shape [B]
            mask (Tensor): Mask for real points, shape [B, N], 1 for real, 0 for padded

        Returns:
            Tensor: Embeddings, shape [B, n_encs, d_out]
        """

        attn_mask = create_attention_mask(mask)

        # Embed points, types and directions into node embeddings
        coords = self.coord_emb(points)
        types = self.type_emb(types)
        dirs = self.dir_emb(directions)

        nodes = torch.cat((coords, types, dirs), dim=-1)
        nodes = self.node_emb(nodes)

        # Build conditioning vector from rotation flag, profile mode, and noise level
        rot_cond = self.rot_emb(rotated.long())
        mode_cond = self.mode_emb(profile_mode)
        sigma_cond = self.sigma_emb(pos_noise_std.unsqueeze(-1))
        cond = self.cond_proj(rot_cond + mode_cond + sigma_cond)

        # Pass data through layers
        for layer in self.layers:
            nodes = layer(nodes, cond=cond, attn_mask=attn_mask)

        nodes = self.out_proj(nodes)
        nodes = nodes * mask.unsqueeze(-1)
        return nodes


class _PocketEncoder(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_layers, n_atom_types, d_out=None, d_emb=64):
        super().__init__()

        d_out = d_model if d_out is None else d_out

        self.coord_emb = torch.nn.Linear(3, d_emb)
        self.type_emb = torch.nn.Embedding(n_atom_types, d_emb)
        self.node_emb = torch.nn.Linear(d_emb * 2, d_model)

        self.rot_emb = torch.nn.Embedding(2, d_model)
        self.cond_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model)
        )

        layer = _ShapeLayer(d_model, n_heads, d_cond=d_model)
        self.layers = torch.nn.ModuleList(get_clones(layer, n_layers))

        self.out_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_out)
        )

    def forward(self, atoms, coords, rotated, mask):
        """Encode pocket atoms into per-atom embeddings.

        Args:
            atoms: Pocket atom vocab indices, shape [B, M]
            coords: Pocket atom coords, shape [B, M, 3]
            rotated: Rotation flags, shape [B], int 0 or 1
            mask: 1 for real atoms, 0 for pad, shape [B, M]

        Returns:
            Tensor: Pocket embeddings, shape [B, M, d_out]
        """

        attn_mask = create_attention_mask(mask)

        coord_embs = self.coord_emb(coords)
        type_embs = self.type_emb(atoms)
        nodes = torch.cat((coord_embs, type_embs), dim=-1)
        nodes = self.node_emb(nodes)

        cond = self.cond_proj(self.rot_emb(rotated.long()))

        for layer in self.layers:
            nodes = layer(nodes, cond=cond, attn_mask=attn_mask)

        nodes = self.out_proj(nodes)
        nodes = nodes * mask.unsqueeze(-1)
        return nodes


# *****************************************************************************
# ************************** Unified encoder module ***************************
# *****************************************************************************


class FeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        n_props,
        d_out=None,
        include_pocket=False,
        eps=1e-5
    ):
        super().__init__()

        d_out = d_model if d_out is None else d_out

        # Take pharmaco vocab size directly from PharmacoFinder
        # Add 2 - one for pad/mask type and another for shape point type
        n_point_types = PharmacophoreFinder.get_vocab_size() + 2
        n_atom_types = len(AtomVocab)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_props = n_props
        self.d_out = d_out
        self.include_pocket = include_pocket
        self.eps = eps

        pocket_emb = None
        if include_pocket:
            pocket_emb = _PocketEncoder(
                d_model,
                n_heads,
                n_layers,
                n_atom_types,
                d_out=d_out
            )

        profile_emb = _ProfileEncoder(
            d_model,
            n_heads,
            n_layers,
            n_point_types,
            d_out=d_out,
            eps=eps
        )

        self.prop_emb = _PropEncoder(d_model, n_props, d_out=d_out)
        self.profile_emb = profile_emb
        self.pocket_emb = pocket_emb

        # Learned "no props" vector — used when all per-sample prop features are masked (pocket
        # training, CFG-dropped samples, and the CFG null prediction). Replaces the zero-vector
        # uncond convention so AdaLN sees a consistent non-collapsed magnitude.
        self.null_props = torch.nn.Parameter(torch.randn(d_out))

    @property
    def hparams(self):
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "n_props": self.n_props,
            "d_out": self.d_out,
            "include_pocket": self.include_pocket
        }

    def encode_profiles(self, points, types, directions, rotated, profile_mode, pos_noise_std, mask):
        """Produce a vector embedding for each conformer profile

        Args:
            points (Tensor): Shape and/or pharmacophore points, shape [B, N, 3]
            types (Tensor): Point types, shape [B, N], max must be no more than n_pharmacophores + 1
            directions (Tensor): Direction vectors, shape [B, N, 3]
            rotated (Tensor): Rotation flags, shape [B], int 0 or 1
            profile_mode (Tensor): Profile mode flags, shape [B], int 0=both, 1=shape-only, 2=pharma-only
            pos_noise_std (Tensor): Position noise std dev, shape [B]
            mask (Tensor): Mask for real points, shape [B, N], 1 for real, 0 for padded

        Returns:
            Tensor: Profile embeddings, shape [B, n_encs, d_out]
        """

        return self.profile_emb(points, types, directions, rotated, profile_mode, pos_noise_std, mask)

    def encode_properties(self, props, mask):
        """Produce embeddings for molecular/boltzmann properties

        Args:
            props (Tensor): Molecular properties, shape [B, n_feats]
            mask (Tensor): Masked features, shape [B, n_feats], 1 for kept, 0 for masked

        Returns:
            Tensor: Prop embeddings, shape [B, d_out]
        """

        out = self.prop_emb(props, mask)
        all_masked = mask.sum(dim=-1) == 0
        null = self.null_props.unsqueeze(0).expand_as(out)
        return torch.where(all_masked.unsqueeze(-1), null, out)

    def encode_pocket(self, atoms, coords, rotated, mask):
        """Encode pocket atoms into per-atom embeddings.

        Args:
            atoms: Pocket atom vocab indices, shape [B, M]
            coords: Pocket atom coords, shape [B, M, 3]
            rotated: Rotation flags, shape [B], int 0 or 1
            mask: 1 for real, 0 for pad, shape [B, M]

        Returns:
            Tensor: Pocket embeddings, shape [B, M, d_out]
        """

        if self.pocket_emb is None:
            raise RuntimeError("FeatureEncoder was created without pocket embedding support.")

        return self.pocket_emb(atoms, coords, rotated, mask)
