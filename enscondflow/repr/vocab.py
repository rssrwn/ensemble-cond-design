from __future__ import annotations

import pickle
from rdkit import Chem
from typing import Iterator, Union
from collections.abc import Mapping

from enscondflow.repr.util import (
    PICKLE_PROTOCOL,
    PAD_TOKEN,
    MASK_TOKEN,
    CHARGED_ATOM_TYPES,
    check_unique,
    check_type_all
)


# Useful type declarations
_BondT = Union[Chem.BondType, str]
_StereoT = Union[Chem.BondStereo, str]


# *********************************
# ***** Arbitrary Vocab class *****
# *********************************


class Vocabulary[T](Mapping):
    """Vocabulary class which maps tokens <--> indices"""

    def __init__(self, tokens: list[T]):
        check_unique(tokens, "tokens list")

        token_idx_map = {token: idx for idx, token in enumerate(tokens)}
        idx_token_map = {idx: token for idx, token in enumerate(tokens)}

        self.token_idx_map = token_idx_map
        self.idx_token_map = idx_token_map


    # *** Mapping Collection methods ***

    # Note getitem, contains and iter map from tokens to indices, 
    # but this is a bit arbitrary

    def __len__(self) -> int:
        return len(self.token_idx_map)

    def __getitem__(self, token: T) -> int:
        return self.get_index(token)

    def __contains__(self, token: T) -> bool:
        return self.contains_token(token)

    def __iter__(self) -> Iterator[T]:
        return self.iter_tokens()


    # *** Mapping functions ***

    def get_token(self, index: int) -> T:
        return self.idx_token_map[index]

    def get_index(self, token: T) -> int:
        return self.token_idx_map[token]

    def tokens_from_indices(self, indices: list[int]) -> list[T]:
        check_type_all(indices, int, "indices list")

        return [self.get_token(idx) for idx in indices]

    def indices_from_tokens(self, tokens: list[T]) -> list[int]:
        return [self.get_index(token) for token in tokens]


    # *** Check contents of vocab map ***

    def contains_token(self, token: T) -> bool:
        return token in self.token_idx_map

    def contains_index(self, index: int) -> bool:
        return index in self.idx_token_map


    # *** Iter functions ***

    def iter_tokens(self) -> Iterator[T]:
        return iter(self.token_idx_map.keys())

    def iter_indices(self) -> Iterator[int]:
        return iter(self.idx_token_map.keys())


    # *** Saving and loading functionality ***

    def to_bytes(self) -> bytes:
        tokens = list(self.token_idx_map.keys())
        obj_bytes = pickle.dumps(tokens, protocol=PICKLE_PROTOCOL)
        return obj_bytes

    @staticmethod
    def from_bytes(data: bytes) -> Vocabulary:
        tokens = pickle.loads(data)
        return Vocabulary(tokens)


# ***************************************
# ***** Internal Vocabulary Classes *****
# ***************************************


# TODO this prob needs refactored to remove all the internal class state
class _BondVocab(Mapping):
    """Global vocabulary object for manipulating bond types"""

    # Supported bond types including no bond and masked bond
    # Note aromatic bonds are supported but use an additional feature on top of bond type
    # All bonds must be kekulised first, hence an aromatic bond is represented by 1 or 2 with the aromatic flag set
    _enum_bond_map = {
        0: "NONE",
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        -1: "MASK"
    }
    _bond_enum_map = {bond: idx for idx, bond in _enum_bond_map.items()}

    # Bond stereochemistry supported options
    _stereo_str_map = {
        Chem.BondStereo.STEREONONE: None,
        Chem.BondStereo.STEREOE: "E",
        Chem.BondStereo.STEREOZ: "Z"
    }
    _str_stereo_map = {s: stereo for stereo, s in _stereo_str_map.items()}

    # Vocabulary of all possible bonds supported
    _bond_strs = [
        "0_F",
        "1_F",
        "2_F",
        "3_F",
        "1_T",
        "2_T",
        "3_T",
        "2_F_E",
        "2_F_Z",
        "-1_F"
    ]

    def __init__(self):
        self._vocab = Vocabulary[str](_BondVocab._bond_strs)

    def __len__(self) -> int:
        return len(self._vocab)

    def __contains__(self, index: int) -> bool:
        return self._vocab.contains_index(index)

    def __getitem__(self, index: int) -> tuple[_BondT, bool, _StereoT]:
        bond_type = self.get_bond_type(index)
        is_arom = self.get_is_aromatic(index)
        stereo = self.get_stereo(index)
        return bond_type, is_arom, stereo

    def __iter__(self) -> Iterator[int]:
        return self._vocab.iter_indices()

    def get_index(self, bond: _BondT, is_aromatic: bool = False, stereo: _StereoT = None) -> int:
        bond_enum = self._bond_to_enum(bond)
        bond_str = self._bond_str_from_info(bond_enum, is_aromatic, stereo)

        if bond_str in self._vocab:
            return self._vocab[bond_str]

        # Provide error messages for common mistakes
        if bond_enum in [0, 1] and is_aromatic:
            raise ValueError("is_aromatic must be False for NONE and MASK bonds.")

        if stereo is not None and bond_enum != 2:
            raise ValueError("stero_type can only be provided for non-aromatic double bonds.")

        # Generic error vocab lookup message for anything else
        return self._vocab[bond_str]

    def get_bond_type(self, index: int) -> _BondT:
        token = self._vocab.get_token(index)
        enum = int(token.split("_")[0])
        return self._enum_to_bond(enum)

    def get_is_aromatic(self, index: int) -> bool:
        token = self._vocab.get_token(index)
        is_arom = token.split("_")[1] == "T"
        return is_arom

    def get_stereo(self, index: int) -> Union[str, None]:
        token = self._vocab.get_token(index)
        splits = token.split("_")

        if len(splits) not in [2, 3]:
            raise RuntimeError(f"Unknown token in bond vocabulary {token}")

        stereo_str = None if len(splits) == 2 else splits[-1]
        stereo = self._str_stereo_map[stereo_str]
        return stereo

    @staticmethod
    def _bond_to_enum(bond: _BondT) -> int:
        return _BondVocab._bond_enum_map[bond]

    @staticmethod
    def _enum_to_bond(enum: int) -> _BondT:
        return _BondVocab._enum_bond_map[enum]

    @staticmethod
    def _stereo_to_str(stereo: _StereoT) -> str:
        if isinstance(stereo, Chem.BondStereo):
            if stereo not in _BondVocab._stereo_str_map:
                raise NotImplementedError(f"None, 'E', and 'Z' are the only supported RDKit stereo types.")

            stereo = _BondVocab._stereo_str_map[stereo]

        if stereo not in [None, "E", "Z", "e", "z"]:
            raise ValueError(f"stereo_type must be either 'E', 'Z', or None, got {stereo}")

        return stereo

    @staticmethod
    def _bond_str_from_info(bond_enum: int, is_aromatic: bool, stereo: _StereoT) -> str:
        stereo_str = _BondVocab._stereo_to_str(stereo)
        is_arom_str = "T" if is_aromatic else "F"

        bond_str = f"{bond_enum}_{is_arom_str}"
        if stereo_str is not None:
            bond_str = f"{bond_str}_{stereo_str.upper()}"

        return bond_str


# *****************************************************************************
# *********************** Globally Accessible Vocabularies ********************
# *****************************************************************************


# NOTE the PAD token is added to the front to ensure that zero padding corresponds to the PAD token
# This is not required for charges since zero padding will automatically correspond to a 0 charge
# We create a unified vocabulary for ligand atom types and pocket atom names

AtomVocab = Vocabulary[str]([PAD_TOKEN] + CHARGED_ATOM_TYPES[:] + [MASK_TOKEN])

BondVocab = _BondVocab()
