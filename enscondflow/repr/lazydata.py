import copy
import h5py
import numpy as np
from functools import reduce
from typing import Union, Optional

from enscondflow.repr.util import check_type


TArr = np.ndarray


class LazyData:
    """A thin wrapper for an h5py dataset.

    By allowing a shared h5py Dataset to be used and read from, this class abstracts reading from a segment of an h5py
    Dataset and allows end users to treat the data segment as a very simple np array with a read function for reading
    the data segment from h5py into a real np array.
    """

    def __init__(self, arr: h5py.Dataset, start_idx: int, n_items: Union[int, tuple]):
        check_type(start_idx, int, "start_idx")
        check_type(n_items, [int, tuple], "n_items")

        n_items = (n_items,) if isinstance(n_items, int) else copy.deepcopy(n_items)
        shape = self._calc_shape(arr.shape, n_items)

        self._arr = arr
        self._start_idx = start_idx
        self._n_items = n_items
        self._shape = shape

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._arr.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def read(self, index: Optional[int] = None) -> TArr:
        # Index reads the corresponding index into the first dim of the array
        # TODO allow index to be a tuple when the data shape is multidimensional

        # Read the whole array by default
        if index is None:
            end_idx = self._start_idx + reduce(lambda i,j: i*j, self._n_items)
            data = self._arr[self._start_idx : end_idx]
            return np.array(data).reshape(self.shape)

        if index > self.shape[0]:
            raise ValueError(f"Tried to access index {index} of array with shape {self.shape}")

        # Otherwise read the index along axis = 0
        n_items = self._n_items[1:] if len(self._n_items) > 1 else (1,)
        flat_size = reduce(lambda i,j: i*j, n_items)

        start_idx = self._start_idx + (index * flat_size)
        data = self._arr[start_idx : start_idx + flat_size]

        item_shape = self.shape[1:] if len(self.shape) > 1 else ()
        return np.array(data).reshape(item_shape)

    def _calc_shape(self, arr_shape, n_items):
        if len(arr_shape) == 1:
            return n_items

        return (*n_items, *arr_shape[1:])
