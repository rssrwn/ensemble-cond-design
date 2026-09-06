import threading
from rdkit import Chem
from typing import Union


class _PeriodicTable:
    """Wrapper for the RDKit periodic table providing a neater interface"""

    def __init__(self):
        self._table = Chem.GetPeriodicTable()

        # Just to be certain that vocab objects are thread safe
        self._pt_lock = threading.Lock()

    def atomic_from_symbol(self, symbol: str) -> int:
        with self._pt_lock:
            symbol = symbol.upper() if len(symbol) == 1 else symbol
            atomic = self._table.GetAtomicNumber(symbol)

        return atomic

    def symbol_from_atomic(self, atomic_num: int) -> str:
        with self._pt_lock:
            token = self._table.GetElementSymbol(atomic_num)

        return token
    
    def valence(self, atom: Union[str, int]) -> int:
        with self._pt_lock:
            valence = self._table.GetDefaultValence(atom)

        return valence


PT = _PeriodicTable()
