from typing import Union

import torch
import numpy as np
from scipy.spatial.transform import Rotation


_T = torch.Tensor
TArr = np.ndarray
TupleRot = tuple[float, float, float]


# *************************************************************************************************
# ********************************** Tensor Util Functions ****************************************
# *************************************************************************************************


def pad_tensors(tensors: list[_T]) -> _T:
    """Pad a list of tensors with zeros

    All dimensions other than pad_dim must have the same shape. A single tensor is returned with the batch dimension
    first, where the batch dimension is the length of the tensors list.

    Args:
        tensors (list[torch.Tensor]): List of tensors

    Returns:
        torch.Tensor: Batched, padded tensor, if pad_dim is 0 then shape [B, L, *] where L is length of longest tensor.
    """

    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded


def pad_arrays(arrays: list[TArr]) -> TArr:
    if len(arrays) == 0:
        return np.array([])

    max_len = max(arr.shape[0] for arr in arrays)
    batch_shape = (len(arrays), max_len, *arrays[0].shape[1:])
    padded = np.zeros(batch_shape, dtype=arrays[0].dtype)

    for i, arr in enumerate(arrays):
        padded[i, :arr.shape[0]] = arr

    return padded


def one_hot_encode_tensor(indices: _T, vocab_size: int) -> _T:
    """Create one-hot encodings from indices

    Args:
        indices (torch.Tensor): Indices into one-hot vectors, shape [*, L]
        vocab_size (int): Length of returned vectors

    Returns:
        torch.Tensor: One-hot encoded vectors, shape [*, L, vocab_size]
    """

    one_hot_shape = (*indices.shape, vocab_size)
    one_hots = torch.zeros(one_hot_shape, dtype=indices.dtype, device=indices.device)
    one_hots.scatter_(-1, indices.long().unsqueeze(-1), 1)
    return one_hots


def one_hot_encode_array(indices: TArr, vocab_size: int) -> TArr:
    """Create one-hot encodings from indices

    Args:
        indices (np.ndarray): Indices into one-hot vectors, shape [*]
        vocab_size (int): Length of returned vectors

    Returns:
        np.ndarray: One-hot encoded vectors, shape [*, vocab_size]
    """

    one_hots = np.zeros((*indices.shape, vocab_size), dtype=np.long)
    np.put_along_axis(one_hots, np.expand_dims(indices, -1), 1.0, axis=-1)
    return one_hots


# *************************************************************************************************
# ******************************* Functions for handling edges ************************************
# *************************************************************************************************


def cleanup_bonds(indices, types):
    """Tidy up a bond list by removing duplicates, removing null bonds and ensuring an upper tri adjacency
    
    This function performs a number of useful tasks:
        1. Removes bonds in the list which are set as 0 (no bond)
        2. Removes duplicate bonds
        3. Throws an error if the adjacency cannot be made upper triangular

    Args:
        indices (torch.Tensor): Bond indices, shape [E, 2]
        types (torch.Tensor): Bond types, shape [E]

    Returns:
        (Tensor, Tensor): Tidied up bond indices and types, shapes [E*, 2], [E*]
    """

    assert len(indices.shape) == 2
    assert len(types.shape) == 1
    assert indices.size(1) == 2

    if indices.size(0) != types.size(0):
        raise ValueError("Lengths of indices and type bond tensors must be equal.")
    
    # Always assume that bond type 0 means no bond so we can exclude it
    edge_mask = types.nonzero().squeeze(-1)
    indices = indices[edge_mask, :]
    types = types[edge_mask]

    if len(indices) == 0:
        return indices, types

    try:
        from_indices = indices[:, 0]
        to_indices = indices[:, 1]
    except:
        print("edge mask", edge_mask)
        print("indices", indices)
        print("types", types)

    # First swap the indices where required to make the adj upper triangular
    # Note this doesn't change the semantics of the bond list
    swap_mask = from_indices > to_indices
    tmp = from_indices.clone()
    from_indices[swap_mask] = to_indices[swap_mask]
    to_indices[swap_mask] = tmp[swap_mask]

    # Remove surplus bonds where the indices and type are duplicated
    # Note these duplicates are benign because they have the same bond type
    bonds = torch.stack((from_indices, to_indices, types)).T
    unique_bonds = bonds.unique(dim=0, sorted=True)

    # Then look for duplicates in bond indices and raise an error
    # Note we can use the consecutive version here since the bonds are now sorted
    unique_bond_idxs = unique_bonds[:, :2].unique_consecutive(dim=0)
    if unique_bonds.size(0) != unique_bond_idxs.size(0):
        raise ValueError("Found at least one pair of atoms which are connected with multiple bond types.")

    bond_indices = unique_bonds[:, :2]
    bond_types = unique_bonds[:, 2]

    # Double check that the adj matrix will be upper triangular only
    is_upper_tri = (bond_indices[:, 0] <= bond_indices[:, 1]).all().item()
    assert is_upper_tri

    return bond_indices, bond_types


def adj_from_node_mask(node_mask, self_connect=False):
    """Creates an edge mask from a given node mask assuming all nodes are fully connected excluding self-connections

    Args:
        node_mask (torch.Tensor): Node mask tensor, shape [batch_size, num_nodes], 1 for real node 0 otherwise
        self_connect (bool): Whether to include self connections in the adjacency. This only applies to real nodes, 
                fake nodes will still have a zero self connection

    Returns:
        torch.Tensor: Adjacency tensor, shape [batch_size, num_nodes, num_nodes], 1 for real edge 0 otherwise
    """

    num_nodes = node_mask.size()[1]

    # Calculate outer product on the node mask
    mask = node_mask.float()
    adjacency = mask.unsqueeze(2) * mask.unsqueeze(1)
    adjacency = adjacency.long()

    # Set diagonal connections for only real nodes
    diag = node_mask if self_connect else torch.zeros_like(node_mask, dtype=torch.long)
    node_idxs = torch.arange(num_nodes)
    adjacency[:, node_idxs, node_idxs] = diag

    return adjacency


def bonds_from_adj(adj_matrix, upper_tri=True):
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix, can be batched or not, shape [batch_size, num_nodes, num_nodes].
            Each item in the matrix corrsponds to the bond type and will be placed into index 2 on dim 1 in bonds.
        upper_tri (bool): Whether to only consider bonds which sit in the upper triangular of adj_matrix.

    Returns:
        An bond list tensor, shape [batch_size, num_bonds, 3], 0 on final dim for padded (ie. no bond)
    """

    batched = True
    if len(adj_matrix.shape) == 2:
        adj_matrix = adj_matrix.unsqueeze(0)
        batched = False

    if upper_tri:
        adj_matrix = torch.triu(adj_matrix, diagonal=1)

    bonds = []
    for adj in list(adj_matrix):
        bond_indices = adj.nonzero()
        bond_types = adj[bond_indices[:, 0], bond_indices[:, 1]]
        bond_list = torch.cat((bond_indices, bond_types.unsqueeze(-1)), dim=-1)
        bonds.append(bond_list)

    # Bonds will be padded with 0s so the bond type will tell whether the bond is real or not
    bonds = pad_tensors(bonds)
    if not batched:
        bonds = bonds.squeeze(0)

    return bonds


def adj_from_edges(edge_indices: _T, edge_types: _T, n_nodes: int, symmetric: bool = False):
    """Create adjacency matrix from a list of edge indices and types

    If an edge pair appears multiple times with different edge types, the adj element for that edge is undefined.

    Args:
        edge_indices (torch.Tensor): Edge list tensor, shape [n_edges, 2]. Pairs of (from_idx, to_idx).
        edge_types (torch.Tensor): Edge types, shape [n_edges].
        n_nodes (int): Number of nodes in the adjacency matrix. This must be >= to the max node index in edges.
        symmetric (bool): Whether edges are considered symmetric. If True the adjacency matrix will also be symmetric,
                otherwise only the exact node indices within edges will be used to create the adjacency.

    Returns:
        torch.Tensor: Adjacency matrix tensor, shape [n_nodes, n_nodes] or 
                [n_nodes, n_nodes, edge_types] if distributions over edge types are provided.
    """

    assert len(edge_indices.shape) == 2
    assert len(edge_types.shape) == 1
    assert edge_indices.shape[0] == edge_types.shape[0]
    assert edge_indices.size(1) == 2

    adj = torch.zeros((n_nodes, n_nodes), device=edge_indices.device, dtype=edge_indices.dtype)

    # If symmetry is not requested we just set the adj with whatever node indices were provided
    if not symmetric:
        adj[edge_indices[:, 0], edge_indices[:, 1]] = edge_types
        return adj

    # If symmetry is requested check that the edges are upper triangular and try to rectify them if not
    if not (edge_indices[:, 0] <= edge_indices[:, 1]).all().item():
        edge_indices, edge_types = cleanup_bonds(edge_indices, edge_types)

    from_indices = edge_indices[:, 0]
    to_indices = edge_indices[:, 1]

    adj[from_indices, to_indices] = edge_types
    adj[to_indices, from_indices] = edge_types

    return adj


# *************************************************************************************************
# ********************************* Geometric Util Functions **************************************
# *************************************************************************************************


# TODO rename? Maybe also merge with inter_distances
# TODO test unbatched and coord sets inputs
def calc_distances(coords, edges=None, sqrd=False, eps=1e-6):
    """Computes distances between connected nodes

    Takes an optional edges argument. If edges is None this will calculate distances between all nodes and return the
    distances in a batched square matrix [batch_size, num_nodes, num_nodes]. If edges is provided the distances are
    returned for each edge in a batched 1D format [batch_size, num_edges].

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [batch_size, num_nodes, 3]
        edges (tuple): Two-tuple of connected node indices, each tensor has shape [batch_size, num_edges]
        sqrd (bool): Whether to return the squared distances
        eps (float): Epsilon to add before taking the square root for numical stability in the gradients

    Returns:
        torch.Tensor: Distances tensor, the shape depends on whether edges is provided (see above).
    """

    # TODO add checks

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords.size()) == 2:
        coords = coords.unsqueeze(0)
        unbatched = True

    if edges is None:
        coord_diffs = coords.unsqueeze(-2) - coords.unsqueeze(-3)
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=-1)

    else:
        edge_is, edge_js = edges
        batch_index = torch.arange(coords.size(0)).unsqueeze(1)
        coord_diffs = coords[batch_index, edge_js, :] - coords[batch_index, edge_is, :]
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=2)

    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def inter_distances(coords1, coords2, sqrd=False, eps=1e-6):
    # TODO add checks and doc

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords1.size()) == 2:
        coords1 = coords1.unsqueeze(0)
        coords2 = coords2.unsqueeze(0)
        unbatched = True

    coord_diffs = coords1.unsqueeze(2) - coords2.unsqueeze(1)
    sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=3)
    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def calc_com(coords, node_mask=None):
    """Calculates the centre of mass of a pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM of pointclouds with imaginary nodes excluded, shape [*, 1, 3]
    """

    node_mask = torch.ones_like(coords[..., 0]) if node_mask is None else node_mask

    assert node_mask.shape == coords[..., 0].shape

    num_nodes = node_mask.sum(dim=-1)
    real_coords = coords * node_mask.unsqueeze(-1)
    com = real_coords.sum(dim=-2) / num_nodes.unsqueeze(-1)
    return com.unsqueeze(-2)


def zero_com(coords, node_mask=None):
    """Sets the centre of mass for a batch of pointclouds to zero for each pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM-free coordinates, where imaginary nodes are excluded from CoM calculation
    """

    com = calc_com(coords, node_mask=node_mask)
    shifted = coords - com
    return shifted


def standardise_coords(coords, node_mask=None):
    """Convert coords into a standard normal distribution

    This will first remove the centre of mass from all pointclouds in the batch, then calculate the (biased) variance
    of the shifted coords and use this to produce a standard normal distribution.

    Args:
        coords (torch.Tensor):  Coordinate tensor, shape [batch_size, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [batch_size, num_nodes], 1 for real node, 0 otherwise

    Returns:
        Tuple[torch.Tensor, float]: The standardised coords and the variance of the original coords
    """

    if node_mask is None:
        node_mask = torch.ones_like(coords)[:, :, 0]

    coord_idxs = node_mask.nonzero()
    real_coords = coords[coord_idxs[:, 0], coord_idxs[:, 1], :]

    variance = torch.var(real_coords, correction=0)
    std_dev = torch.sqrt(variance)

    result = (coords / std_dev) * node_mask.unsqueeze(2)
    return result, std_dev.item()


def rotate(coords: torch.Tensor, rotation: Union[Rotation, TupleRot]):
    """Rotate coordinates for a single molecule

    Args:
        coords (torch.Tensor): Unbatched coordinate tensor, shape [num_atoms, 3]
        rotation (Union[Rotation, Tuple[float, float, float]]): Can be either a scipy Rotation object or a tuple of
                rotation values in radians, (x, y, z). These are treated as extrinsic rotations. See the scipy docs
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) for info.

    Returns:
        torch.Tensor: Rotated coordinates
    """

    if not isinstance(rotation, Rotation):
        rotation = Rotation.from_euler("xyz", rotation)

    device = coords.device
    coords = coords.cpu().numpy()

    rotated = rotation.apply(coords)
    rotated = torch.tensor(rotated, device=device)
    return rotated
