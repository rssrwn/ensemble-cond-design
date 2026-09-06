import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

import enscondflow.util.functional as smolF


# *******************************************************************************
# ******************************* Helper Functions ******************************
# *******************************************************************************


def adj_from_bonds(bonds):
    """Comnputes a binary adj matrix from a bond tensor"""

    return (bonds != 0).long()


def create_attention_mask(node_mask):
    adj_matrix = smolF.adj_from_node_mask(node_mask, self_connect=True)

    # Set ones along the diagonal to ensure that softmax attention does not have instabilities
    diag = torch.ones_like(node_mask, dtype=torch.long)
    node_idxs = torch.arange(node_mask.size(1))
    adj_matrix[:, node_idxs, node_idxs] = diag

    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = float("-inf")
    return attn_mask


def adj_to_attn_mask(adj_matrix, pos_inf=False):
    """Assumes adj_matrix is only 0s and 1s"""

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    n_nodes = adj_matrix.sum(dim=-1)
    attn_mask[n_nodes == 0] = 0.0

    return attn_mask


def compute_rrwps(adj_matrix, iters):
    """Compute Relative Random Walk Probabilities (RRWP) for a graph.

    Args:
        adj_matrix (Tensor): Adjacency matrix of shape [B, N, N] with 0s and 1s
        iters (int): Number of random walk steps to simulate

    Returns:
        Tensor: RRWP features, shape [B, N, N, iters]
    """

    assert adj_matrix.size(1) == adj_matrix.size(2)

    device = adj_matrix.device
    B, N, _ = adj_matrix.shape

    adj_clone = adj_matrix.clone()
    adj_clone[:, range(N), range(N)] = 1

    # Compute transition matrix
    degrees = adj_clone.sum(dim=1)
    degree_inv = 1.0 / degrees
    P = adj_clone * degree_inv.unsqueeze(1)

    # Setup tensor for storing probabilities
    P_i = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
    rrwps = torch.zeros((B, N, N, iters), device=device, dtype=torch.float)

    # Skip the identity matrix as the first feature
    for i in range(iters):
        P_i = torch.bmm(P_i, P)
        rrwps[:, :, :, i] = P_i

    return rrwps


# *****************************************************************************
# ****************************** Embedding Modules ****************************
# *****************************************************************************


class NodeEmbedding(torch.nn.Module):
    def __init__(self, d_model, n_atom_types, emb_pos=False):
        super().__init__()

        self.d_model = d_model
        self.emb_pos = emb_pos

        self.atom_emb = torch.nn.Embedding(n_atom_types, d_model)
        self.coord_emb = torch.nn.Linear(3, d_model)
        self.node_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model * 2),
            torch.nn.Linear(d_model * 2, d_model)
        )

    def forward(self, coords, atoms):
        atom_emb = self.atom_emb(atoms)
        coord_emb = self.coord_emb(coords)
        node_emb = torch.cat((atom_emb, coord_emb), dim=-1)
        node_emb = self.node_proj(node_emb)

        if self.emb_pos:
            pos_embs = self._sinusoidal_embs(node_emb.size(1), atoms.device)
            node_emb = node_emb + pos_embs.unsqueeze(0)

        return node_emb

    def _sinusoidal_embs(self, seq_len, device):
        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)], device=device)
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs


class EdgeEmbedding(torch.nn.Module):
    def __init__(self, d_model, d_out, n_bond_types, d_emb=64, n_rrwps=None):
        super().__init__()

        n_rrwps = n_rrwps if n_rrwps is not None and n_rrwps > 0 else 0
        d_edge_in = (d_emb * 2) + n_rrwps

        self.d_emb = d_emb
        self.n_rrwps = n_rrwps

        self.node_proj = torch.nn.Linear(d_model, d_emb * 2)
        self.bond_emb = torch.nn.Embedding(n_bond_types, d_emb)
        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(d_edge_in, d_out * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(d_out * 2, d_out)
        )

    def forward(self, nodes, bonds):
        # Take outer product from node projection
        nodes_A, nodes_B = self.node_proj(nodes).split(self.d_emb, dim=-1)
        pairwise = nodes_A.unsqueeze(2) * nodes_B.unsqueeze(1)

        # Concat bond embeddings
        bond_embs = self.bond_emb(bonds)
        pairwise = torch.cat((pairwise, bond_embs), dim=-1)

        # Extra pairwise features
        if self.n_rrwps > 0:
            adj = adj_from_bonds(bonds)
            rrwps = compute_rrwps(adj, self.n_rrwps)
            pairwise = torch.cat((pairwise, rrwps), dim=-1)

        return self.edge_proj(pairwise)


class CondEmbedding(torch.nn.Module):
    "Handles embeding the CFG conditioning, FM time, and latent embedding"

    def __init__(self, d_model, d_latent=None, n_freqs=128, max_period=10000):
        super().__init__()

        if n_freqs % 2 != 0:
            raise ValueError(f"n_freqs must be even, got {n_freqs}")

        d_time_emb = d_model if d_latent is None else d_latent

        self.d_latent = d_latent
        self.n_freqs = n_freqs
        self.max_period = max_period

        self.time_proj = torch.nn.Sequential(
            torch.nn.Linear(n_freqs, d_time_emb),
            torch.nn.LayerNorm(d_time_emb)
        )

        self.emb = torch.nn.Sequential(
            torch.nn.Linear(d_time_emb, d_model * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model * 4, d_model),
            torch.nn.LayerNorm(d_model)
        )

    def forward(self, times, latents=None):
        """Create conditioning vector

        Args:
            times (Tensor): FM times, shape [B, 1]
            latents (Tensor, optional): Latent conditioning, shape [B, *, d_latent]

        Returns:
            Tensor: Conditioning vector, shape [B, *, d_model]
        """

        # Check for simple failure cases
        if self.d_latent is not None and latents is None:
            raise ValueError("Latent conditioning was set but no conditioning was provided.")
        if self.d_latent is None and latents is not None:
            raise ValueError("Latent conditioning was not set but conditioning was provided.")

        embs = self.time_emb(times)
        embs = self.time_proj(embs)

        # Add latents to time embs if they are provided
        if latents is not None:
            embs = embs.unsqueeze(1) if len(latents.shape) == 3 else embs
            embs = embs + latents

        return self.emb(embs)

    def time_emb(self, times):
        if len(times.shape) == 1:
            times = times.unsqueeze(-1)

        half_dim = self.n_freqs // 2
        log_period = torch.log(torch.tensor(self.max_period))

        indices = torch.arange(half_dim, dtype=torch.float32, device=times.device)
        freqs = torch.exp(indices * -(log_period / half_dim))
        embs = times * freqs.unsqueeze(0)

        emb = torch.cat([torch.sin(embs), torch.cos(embs)], dim=-1)
        return emb


# *****************************************************************************
# ***************************** Prediction Modules ****************************
# *****************************************************************************


class AtomPrediction(torch.nn.Module):
    def __init__(self, d_model, n_atom_types):
        super().__init__()

        self.atom_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, n_atom_types)
        )

    def forward(self, nodes):
        return self.atom_proj(nodes)


class BondPrediction(torch.nn.Module):
    def __init__(self, d_model, d_edge, n_bond_types):
        super().__init__()

        self.d_edge = d_edge

        self.node_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_edge * 2)
        )

        self.pairwise_proj = torch.nn.Linear(d_edge * 2, d_edge)
        self.out_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_edge),
            torch.nn.Linear(d_edge, d_edge * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(d_edge * 2, n_bond_types)
        )

    def forward(self, nodes, edges):
        # Compute node features and take outer product
        nodes_A, nodes_B = self.node_proj(nodes).split(self.d_edge, dim=-1)
        pairwise = nodes_A.unsqueeze(2) * nodes_B.unsqueeze(1)

        # Concat pairwise features and predict bond types
        pairwise = torch.cat((edges, pairwise), dim=-1)
        pairwise = self.pairwise_proj(pairwise)
        bond_logits = self.out_proj(pairwise + pairwise.transpose(1, 2))

        return bond_logits


# ************************************************************************************
# ******************************* Normalisation Modules ******************************
# ************************************************************************************


class AdaParams:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


class NormParams(torch.nn.Module):
    """Wrapper for normalisation parameters.

    This class can handle both Adapative Normalisation and standard layer norm params with zero output scaling.
    In both cases parameter sets get wrapped in the same object (AdaParams instances) to make downstream use neater.
    """

    def __init__(self, d_model, n_blocks=2, cond=False, two_layer=False, d_cond=None):
        super().__init__()

        if not cond and d_cond is not None:
            raise ValueError("If cond is not set for NormParams d_cond must be None.")

        d_cond = d_model if d_cond is None else d_cond
        d_proj_in = d_model * 2 if two_layer else d_cond

        self.d_model = d_model
        self.n_blocks = n_blocks
        self.cond = cond
        self.two_layer = two_layer

        if cond:
            # Optional first projection
            if two_layer:
                self.in_proj = torch.nn.Sequential(
                    torch.nn.Linear(d_cond, d_model * 2),
                    torch.nn.SiLU()
                )

            # Then separate matmuls to make it easier to zero init final scales
            self.shift_scale_proj = torch.nn.Linear(d_proj_in, d_model * n_blocks * 2)
            self.out_scale_proj = torch.nn.Linear(d_proj_in, d_model * n_blocks)

        else:
            self.gammas = torch.nn.Parameter(torch.ones(n_blocks, 1, d_model))
            self.betas = torch.nn.Parameter(torch.zeros(n_blocks, 1, d_model))
            self.alphas = torch.nn.Parameter(torch.zeros(n_blocks, 1, d_model))

    def forward(self, cond=None):
        """Produce norm parameters.

        Args:
            cond (Tensor): Conditioning features, shape [B, d_model] or [B, seq_len, d_model]

        Returns:
            list[AdaParams]: List of adapative params, one per block
        """

        if cond is None and self.cond:
            raise ValueError("Conditioning was set but cond was not provided.")
        if cond is not None and not self.cond:
            raise ValueError("Conditioning was not set but cond was provided.")

        # If no conditioning is provided just use the learned parameters
        if not self.cond:
            param_sets = []
            for b_idx in range(self.n_blocks):
                params = (self.alphas[b_idx], self.betas[b_idx], self.gammas[b_idx])
                param_sets.append(AdaParams(*params))

            return param_sets

        # Otherwise apply an MLP to the input conditioning to produce the norm params
        cond = self.in_proj(cond) if self.two_layer else cond
        ada_params = cond.sum(dim=1) if len(cond.shape) == 3 else cond

        shift_scales = self.shift_scale_proj(ada_params).split(self.d_model * 2, dim=-1)
        out_scales = self.out_scale_proj(ada_params).split(self.d_model, dim=-1)

        param_sets = []
        for b_idx in range(self.n_blocks):
            beta, gamma = shift_scales[b_idx].split(self.d_model, dim=-1)
            param_set = AdaParams(out_scales[b_idx], beta, gamma)
            param_sets.append(param_set)

        return param_sets

    def init_params_(self):
        # Zero-init out_scale_proj so AdaLN starts at identity. No-cond branch needs no override —
        # gammas/betas/alphas are bare Parameters constructed with the correct values.
        if self.cond:
            torch.nn.init.zeros_(self.out_scale_proj.weight)
            torch.nn.init.zeros_(self.out_scale_proj.bias)


class AdaLN(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.norm = torch.nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, nodes, gamma, beta):
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        if len(nodes.shape) != len(gamma.shape):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return (self.norm(nodes) * gamma) + beta


# *****************************************************************************
# ******************* Reusable Transformer Components *************************
# *****************************************************************************


class BiasedAttention(torch.nn.Module):
    def __init__(self, d_qk, d_v, n_heads, mult_bias=False):
        super().__init__()

        if d_qk % n_heads != 0:
            raise ValueError(f"d_qk must be divisible by n_heads, got {d_qk} and {n_heads}")

        if d_v % n_heads != 0:
            raise ValueError(f"d_v must be divisible by n_heads, got {d_v} and {n_heads}")

        if mult_bias and (d_qk // n_heads) < 16:
            raise ValueError(f"Query and key head dim must be at least 16 if using multiplicative bias.")

        self.d_qk = d_qk
        self.d_v = d_v
        self.n_heads = n_heads
        self.mult_bias = mult_bias

        if mult_bias:
            self.flex_attention = torch.compile(flex_attention)

    def forward(self, query, key, value, attn_bias=None, attn_mask=None):
        """Apply dot product attention function

        NOTE if applying multiplicative bias, this function assumes the bias has already had sigmoid applied, if
        required, and applies the given attn_bias directly.

        Args:
            query (Tensor): Query features, shape [B, N_q, d_qk]
            key (Tensor): Key features, shape [B, N_kv, d_qk]
            value (Tensor): Value features, shape [B, N_kv, d_v]
            attn_bias (Tensor): Pairwise attention bias features, shape [B, N_q, N_kv, n_heads]
            attn_mask (Tensor): Attention mask, shape [B, N_q, N_kv], must be 0 for attended, -inf for padded

        Returns:
            Tensor: Accumulated features, shape [B, N_q, d_v]
        """

        assert query.size(-1) == self.d_qk
        assert key.size(-1) == self.d_qk
        assert value.size(-1) == self.d_v

        # Unflatten the head features into separate heads
        query = query.unflatten(-1, (-1, self.n_heads)).movedim(-1, 1).contiguous()
        key = key.unflatten(-1, (-1, self.n_heads)).movedim(-1, 1).contiguous()
        value = value.unflatten(-1, (-1, self.n_heads)).movedim(-1, 1).contiguous()

        # Apply multiplicative bias if requested and attn_bias is provided
        if self.mult_bias and attn_bias is not None:
            score_mod = self._create_score_mod(attn_bias, attn_mask)
            attn_out = self.flex_attention(query, key, value, score_mod=score_mod)
            out = attn_out.movedim(1, -1).flatten(2, 3)
            return out

        bias = None

        # Otherwise use additive bias by accumulating attn_bias and attn_mask
        if attn_bias is not None and attn_mask is not None:
            bias = attn_bias.movedim(-1, 1) + attn_mask.unsqueeze(1)
        elif attn_bias is not None:
            bias = attn_bias.movedim(-1, 1)
        elif attn_mask is not None:
            bias = attn_mask.unsqueeze(1)

        # Run SDPA and recombine the head features
        attn_out = F.scaled_dot_product_attention(query, key, value, attn_mask=bias)
        out = attn_out.movedim(1, -1).flatten(2, 3)

        return out

    # This doesn't seem to work with partial functions, so let's do it like this instead
    def _create_score_mod(self, attn_bias, mask):
        def score_mod(score, batch, head, q_idx, kv_idx):
            return (score * attn_bias[batch, q_idx, kv_idx, head]) + mask[batch, q_idx, kv_idx]

        return score_mod


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, attn_gate=False):
        super().__init__()

        d_node_proj = d_model * 4 if attn_gate else 3 * d_model

        self.d_model = d_model
        self.attn_gate = attn_gate

        self.node_proj = torch.nn.Linear(d_model, d_node_proj)
        self.attention = BiasedAttention(d_model, d_model, n_heads)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, nodes, bias=None, attn_mask=None):
        """Apply self attention along the node dimension

        Args:
            nodes (Tensor): Node features, shape [B, N, d_model]
            bias (Tensor): Pairwise features, shape [B, N, N, n_heads]
            attn_mask (Tensor): Pre-computed attention mask, shape [B, N, N], 0 for attended, -inf for padded

        Returns:
            Tensor: Accumulated node features, shape [B, N, d_model]
        """

        splits = self.node_proj(nodes).split(self.d_model, dim=-1)
        q, k, v = splits[:3]

        out = self.attention(q, k, v, attn_bias=bias, attn_mask=attn_mask)
        out = out * torch.sigmoid(splits[3]) if self.attn_gate else out
        return self.out_proj(out)


class CondAttention(torch.nn.Module):
    """Attention between nodes and conditioning (includes self-attention nodes->nodes)"""

    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model

        # Nodes get projected as queries and as keys/values since self-attn is included
        self.q_node_proj = torch.nn.Linear(d_model, d_model)
        self.kv_node_proj = torch.nn.Linear(d_model, d_model * 2)

        self.kv_cond_proj = torch.nn.Linear(d_model, d_model * 2)
        self.attn = BiasedAttention(d_model, d_model, n_heads)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, nodes, conds=None, bias=None, attn_mask=None):
        """Apply conditional attention along the node dimension

        NOTE nodes will also be self-attended to, so the attention output is a weighted sum over nodes and cond.
        If conds is None then this is just self attention.

        Args:
            nodes (Tensor): Node features, shape [B, N, d_model]
            conds (Tensor): Cond features, shape [B, L, d_model]
            bias (Tensor): Rectanular bias features, shape [B, N, N + L, n_heads]
            attn_mask (Tensor): Pre-computed attention mask, shape [B, N, N + L], 0 for attended, -inf for padded

        Returns:
            Tensor: Node features, shape [B, N, d_model]
        """

        q = self.q_node_proj(nodes)
        k, v = self.kv_node_proj(nodes).split(self.d_model, dim=-1)

        if conds is not None:
            k_cond, v_cond = self.kv_cond_proj(conds).split(self.d_model, dim=-1)
            k = torch.cat((k, k_cond), dim=1)
            v = torch.cat((v, v_cond), dim=1)

        attn_out = self.attn(q, k, v, attn_bias=bias, attn_mask=attn_mask)
        out = self.out_proj(attn_out)
        return out


class FeatureMix(torch.nn.Module):
    def __init__(self, d_model, d_out=None, ff_factor=4, dropout=0.1):
        super().__init__()

        d_out = d_model if d_out is None else d_out

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * ff_factor),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model * ff_factor, d_out)
        )

    def forward(self, nodes):
        """Apply a two-layer MLP feedforward module to node features

        Args:
            nodes (Tensor): Node features, shape [B, N, d_model]

        Returns:
            Tensor: Updated node features, shape [B, N, d_model]
        """

        return self.mlp(nodes)
