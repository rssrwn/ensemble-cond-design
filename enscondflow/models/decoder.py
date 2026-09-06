import copy
import torch

import enscondflow.util.functional as smolF
from enscondflow.models.util import (
    AdaLN,
    NormParams,
    NodeEmbedding,
    EdgeEmbedding,
    CondEmbedding,
    CondAttention,
    FeatureMix,
    AtomPrediction,
    BondPrediction
)


# *****************************************************************************
# ******************************* Core Components *****************************
# *****************************************************************************


class GraphAttention(torch.nn.Module):
    def __init__(self, d_model, d_edge, n_heads):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")

        d_node_proj = (d_edge * 2) + d_model
        d_edge_out = d_edge + n_heads

        self.d_model = d_model
        self.d_edge = d_edge
        self.n_heads = n_heads

        self.node_proj = torch.nn.Linear(d_model, d_node_proj)
        self.message_proj = torch.nn.Linear(d_edge, d_edge_out)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, nodes, edges, attn_mask=None):
        """Apply self attention along the node dimension

        Args:
            nodes (Tensor): Node features, shape [B, N, d_model]
            edges (Tensor): Pairwise features, shape [B, N, N, d_edge]
            attn_mask (Tensor): Pre-computed attention mask, shape [B, N, N], 0 for attended, -inf for padded

        Returns:
            Tensor: Node and edge features, shape [B, N, d_model], [B, N, N, d_edge]
        """

        proj_nodes = self.node_proj(nodes)
        q, k, v = proj_nodes.split((self.d_edge, self.d_edge, self.d_model), dim=-1)
        v_heads = v.unflatten(-1, (-1, self.n_heads))

        # Produce pairwise attention logits and next edge features
        pairwise = q.unsqueeze(2) * k.unsqueeze(1)
        pairwise = pairwise * torch.sigmoid(edges)
        heads, edge_out = self.message_proj(pairwise).split((self.n_heads, self.d_edge), dim=-1)

        # Calculate the attention output
        scores = torch.softmax(heads + attn_mask.unsqueeze(-1), dim=2)
        attn_out = torch.einsum("bnmh,bmdh->bndh", scores, v_heads).flatten(2,3)
        node_out = self.out_proj(attn_out)

        return node_out, edge_out


class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, ada_cond=False, ff_factor=4, dropout=0.1):
        super().__init__()

        self.ada_cond = ada_cond

        self.ada_params = NormParams(d_model, cond=ada_cond, n_blocks=2, two_layer=False)
        self.attn_norm = AdaLN(d_model)
        self.ff_norm = AdaLN(d_model)

        # Separate norm for cond features
        self.cond_norm = torch.nn.LayerNorm(d_model)

        self.attn = CondAttention(d_model, n_heads)
        self.ff = FeatureMix(d_model, ff_factor=ff_factor, dropout=dropout)

    def forward(self, nodes, ada_cond=None, attn_cond=None, bias=None, attn_mask=None):
        """Apply one transformer layer with optional conditioning tokens concatenated into self-attention.

        Args:
            nodes (Tensor): Node features, shape [B, N, d_model]
            ada_cond (Tensor): Adaptive conditioning features, shape [B, 1, d_model]
            attn_cond (Tensor): Attention conditioning features, shape [B, L, d_model]
            bias (Tensor): Pairwise features, shape [B, N, N + L, n_heads]
            attn_mask (Tensor): Pre-computed attention mask, shape [B, N, N + L], 0 for attended, -inf for padded

        Returns:
            Tensor: Updated node features, shape [B, N, d_model]
        """

        adas = self.ada_params(ada_cond)
        attn_cond = self.cond_norm(attn_cond) if attn_cond is not None else None

        # Self attention with optional conditioning concatenated
        norm_nodes = self.attn_norm(nodes, adas[0].gamma, adas[0].beta)
        attn_out = self.attn(norm_nodes, conds=attn_cond, bias=bias, attn_mask=attn_mask)
        nodes = (attn_out * adas[0].alpha.unsqueeze(1)) + nodes

        # Node feedforward
        norm_nodes = self.ff_norm(nodes, adas[1].gamma, adas[1].beta)
        ff_out = self.ff(norm_nodes)
        nodes = (ff_out * adas[1].alpha.unsqueeze(1)) + nodes

        return nodes


class GraphLayer(torch.nn.Module):
    def __init__(self, d_model, d_edge, n_heads, ada_cond=True, ff_factor=4, dropout=0.1):
        super().__init__()

        self.node_ada = NormParams(d_model, cond=ada_cond, two_layer=True)
        self.edge_ada = NormParams(d_edge, cond=ada_cond, d_cond=d_model, two_layer=True)

        self.node_attn_norm = AdaLN(d_model)
        self.node_ff_norm = AdaLN(d_model)
        self.edge_attn_norm = AdaLN(d_edge)
        self.edge_ff_norm = AdaLN(d_edge)

        self.attention = GraphAttention(d_model, d_edge, n_heads)
        self.node_ff = FeatureMix(d_model, ff_factor=ff_factor, dropout=dropout)
        self.edge_ff = FeatureMix(d_edge, ff_factor=2, dropout=dropout)

    def forward(self, nodes, edges, ada_cond=None, attn_mask=None):
        """Apply one layer of the Geometric Transformer

        Args:
            nodes (Tensor): Node features, shape [B, N, d_model]
            edges (Tensor): Edge features, shape [B, N, N, d_edge]
            cond (Tensor): Conditioning features, shape [B, d_model]
            attn_mask (Tensor): Pre-computed attention mask, shape [B, N, N], 0 for attended, -inf for padded

        Returns:
            (Tensor, Tensor): Updated node and edge features, shape [B, N, d_model], [B, N, N, d_edge]
        """

        # Adaptive norm params
        node_adas = self.node_ada(ada_cond)
        edge_adas = self.edge_ada(ada_cond)

        norm_nodes = self.node_attn_norm(nodes, node_adas[0].gamma, node_adas[0].beta)
        norm_edges = self.edge_attn_norm(edges, edge_adas[0].gamma, edge_adas[0].beta)

        # Self attention
        node_out, edge_out = self.attention(norm_nodes, norm_edges, attn_mask=attn_mask)
        nodes = (node_out * node_adas[0].alpha.unsqueeze(1)) + nodes
        edges = (edge_out * edge_adas[0].alpha.unsqueeze(1).unsqueeze(2)) + edges

        # Node feedforward
        norm_nodes = self.node_ff_norm(nodes, node_adas[1].gamma, node_adas[1].beta)
        node_out = self.node_ff(norm_nodes)
        nodes = (node_out * node_adas[1].alpha.unsqueeze(1)) + nodes

        # Edge feedforward
        norm_edges = self.edge_ff_norm(edges, edge_adas[1].gamma, edge_adas[1].beta)
        edge_out = self.edge_ff(norm_edges)
        edges = (edge_out * edge_adas[1].alpha.unsqueeze(1).unsqueeze(2)) + edges

        return nodes, edges


# *****************************************************************************
# ********************************* Main Classes ******************************
# *****************************************************************************


class HybridBlock(torch.nn.Module):
    def __init__(self, d_model, d_edge, n_heads, n_layers, ada_cond=False, ff_factor=4, dropout=0.1):
        super().__init__()

        if n_layers < 2:
            raise ValueError(f"n_layers in Hybrid block must be at least 2, got {n_layers}")

        d_bias = (n_layers - 1) * n_heads

        self.n_heads = n_heads
        self._d_bias = d_bias

        layer = TransformerLayer(
            d_model,
            n_heads,
            ada_cond=ada_cond,
            ff_factor=ff_factor,
            dropout=dropout
        )

        self.bias_proj = torch.nn.Linear(d_edge, d_bias)
        self.layers = torch.nn.ModuleList(self._get_clones(layer, n_layers - 1))

        self.graph_layer = GraphLayer(
            d_model,
            d_edge,
            n_heads,
            ada_cond=ada_cond,
            ff_factor=ff_factor,
            dropout=dropout
        )

    def forward(self, nodes, edges, ada_cond=None, attn_cond=None, attn_mask=None):
        """Apply one hybrid block of layers

        Args:
            nodes (Tensor): Node features, shape [B, N, d_model]
            edges (Tensor): Edge features, shape [B, N, N, d_edge]
            ada_cond (Tensor): Adaptive conditioning features, shape [B, S, d_model]
            attn_cond (Tensor): Attention conditioning features, shape [B, L, d_model]
            attn_mask (Tensor): Pre-computed attention mask, shape [B, N, N + L], 0 for attended, -inf for padded

        Returns:
            (Tensor, Tensor): Updated node and edge features, shape [B, N, d_model], [B, N, N, d_edge]
        """

        # Produce attention biases from pairwise features
        biases = self.bias_proj(edges)
        if attn_cond is not None:
            zero_bias = torch.zeros((*nodes.shape[:2], attn_cond.size(1), self._d_bias), device=biases.device)
            biases = torch.cat((biases, zero_bias), dim=2)

        biases = biases.split(self.n_heads, dim=-1)

        # Pass through transformer layers in this block
        for idx, layer in enumerate(self.layers):
            nodes = layer(nodes, ada_cond=ada_cond, attn_cond=attn_cond, bias=biases[idx], attn_mask=attn_mask)

        # Final graph transformer layer, cut out correctly sized attn_mask for this
        node_attn_mask = attn_mask[:, :, :nodes.size(1)]
        nodes, edges = self.graph_layer(nodes, edges, ada_cond=ada_cond, attn_mask=node_attn_mask)
        return nodes, edges

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


class HybridGenerator(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_edge,
        n_layers,
        n_blocks,
        n_atom_types,
        n_bond_types,
        ff_factor=4,
        dropout=0.1,
        d_emb=64
    ):
        super().__init__()

        if (n_layers % n_blocks) != 0:
            raise ValueError(f"n_layers must be divisible by n_blocks, got {n_layers} and {n_blocks}")

        layers_per_block = n_layers // n_blocks

        hparams = {
            "d_model": d_model,
            "n_heads": n_heads,
            "d_edge": d_edge,
            "n_layers": n_layers,
            "n_blocks": n_blocks,
            "ff_factor": ff_factor,
            "dropout": dropout,
            "d_emb": d_emb
        }

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_edge = d_edge

        self._hparams = hparams

        # *** Embedding modules ***
        self.node_emb = NodeEmbedding(d_model, n_atom_types)
        self.edge_emb = EdgeEmbedding(d_model, d_edge, n_bond_types, d_emb=d_emb)
        self.cond_emb = CondEmbedding(d_model, d_latent=d_model)

        self.attn_cond_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model * ff_factor),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model * ff_factor, d_model)
        )

        # *** Stack of blocks ***
        block = HybridBlock(
            d_model,
            d_edge,
            n_heads,
            layers_per_block,
            ada_cond=True,
            ff_factor=ff_factor,
            dropout=dropout
        )
        self.blocks = torch.nn.ModuleList(self._get_clones(block, n_blocks))

        # *** Prediction modules ***
        self.coord_pred = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, 3)
        )

        self.atom_pred = AtomPrediction(d_model, n_atom_types)
        self.bond_pred = BondPrediction(d_model, d_edge, n_bond_types)

    @property
    def hparams(self):
        return self._hparams

    def forward(self, coords, atoms, bonds, ada_latents, times, mask=None, conds=None, cond_mask=None):
        """Pass data through the hybrid encoder.

        Args:
            coords (Tensor): Coordinates in complex, shape [B, N, 3]
            atoms (Tensor): Atom types/names, shape [B, N]
            bonds (Tensor): Adjacency matrix of the complex, shape [B, N, N]
            ada_latents (Tensor): Latent embeddings for adaptive normalisation, shape [B, d_model]
            times (Tensor): Flow matching times for atoms, shape [B, N, 1]
            mask (Tensor, optional): Mask for real atoms, shape [B, N], 1 for non-pad atom, 0 otherwise
            conds (Tensor): Attention conditioning embeddings, shape [B, L, d_model], None means uncond generation
            cond_mask (Tensor): Mask for cond, shape [B, L], 1 for non-pad atom, 0 otherwise

        Returns:
            (Tensor, Tensor, Tensor): Predicted coords, atom types and bond types for whole complex.
        """

        if conds is not None and cond_mask is None:
            raise ValueError("cond_mask must be provided if conditioning is provided.")

        mask = torch.ones_like(atoms) if mask is None else mask
        attn_mask = self._create_attn_mask(mask, cond_mask)

        nodes = self.node_emb(coords, atoms)
        edges = self.edge_emb(nodes, bonds)

        adas = self.cond_emb(times, latents=ada_latents)
        adas = adas.unsqueeze(1) if len(adas.shape) == 2 else adas
        attn_cond = self.attn_cond_mlp(conds) + conds if conds is not None else None

        for block in self.blocks:
            nodes, edges = block(nodes, edges, ada_cond=adas, attn_cond=attn_cond, attn_mask=attn_mask)

        pred_coords = self.coord_pred(nodes)
        pred_atoms = self.atom_pred(nodes)
        pred_bonds = self.bond_pred(nodes, edges)

        return pred_coords, pred_atoms, pred_bonds

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]

    def _create_attn_mask(self, node_mask, cond_mask):
        adj_matrix = smolF.adj_from_node_mask(node_mask, self_connect=True)

        # Set ones along the diagonal to ensure that softmax attention does not have instabilities
        diag = torch.ones_like(node_mask, dtype=torch.long)
        node_idxs = torch.arange(node_mask.size(1))
        adj_matrix[:, node_idxs, node_idxs] = diag

        if cond_mask is not None:
            cond_adj_matrix = cond_mask.unsqueeze(1).expand(-1, node_mask.size(1), -1)
            adj_matrix = torch.cat((adj_matrix, cond_adj_matrix), dim=2)

        attn_mask = torch.zeros_like(adj_matrix.float())
        attn_mask[adj_matrix == 0] = float("-inf")
        return attn_mask
