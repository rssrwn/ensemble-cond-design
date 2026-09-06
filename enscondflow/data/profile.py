from __future__ import annotations

import numpy as np
from typing import Optional

import enscondflow.util.functional as smolF


TArr = np.ndarray


class ConfProfile:
    """Container for a conformer's shape and pharmacophore profile"""

    def __init__(
        self,
        positions: TArr,
        types: TArr,
        atom_ids: list[tuple[int, ...]],
        directions: Optional[TArr] = None,
        rotated: bool = False,
        profile_mode: int = 0,
        pos_noise_std: float = 0.0,
    ):
        if not (len(positions) == len(types) == len(atom_ids)):
            raise ValueError("All arrays must have the same number of points.")

        self.positions = positions
        self.types = types
        self.atom_ids = atom_ids
        self.directions = directions if directions is not None else np.zeros_like(positions)
        self.rotated = rotated
        self.profile_mode = profile_mode
        self.pos_noise_std = pos_noise_std

    @staticmethod
    def batch_profiles(profiles: list[ConfProfile]) -> tuple[TArr, TArr, TArr, TArr, TArr, TArr, TArr]:
        """Batch a list of profiles into np arrays, padded to the max length in the batch.

        Returns 7 np arrays:
        1. Positions [B, N, 3], 2. Types [B, N], 3. Directions [B, N, 3], 4. Mask [B, N],
        5. Rotated flags [B], 6. Profile modes [B], 7. Position noise std devs [B]
        """

        positions_arr = smolF.pad_arrays([p.positions for p in profiles])
        types_arr = smolF.pad_arrays([p.types for p in profiles]).astype(np.long)
        directions_arr = smolF.pad_arrays([p.directions for p in profiles])
        mask_arr = smolF.pad_arrays([np.ones(len(p.positions)) for p in profiles])

        rotated_arr = np.array([int(p.rotated) for p in profiles], dtype=np.int64)
        mode_arr = np.array([p.profile_mode for p in profiles], dtype=np.int64)
        noise_std_arr = np.array([p.pos_noise_std for p in profiles], dtype=np.float32)

        return positions_arr, types_arr, directions_arr, mask_arr, rotated_arr, mode_arr, noise_std_arr
