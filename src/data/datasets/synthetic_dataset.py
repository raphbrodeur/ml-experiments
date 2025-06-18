"""
    @file:              synthetic_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 04/2025

    @Description:       This file contains the class SyntheticDataset that is used as a Torch Dataset for Torch models.
"""

from typing import List, NamedTuple

from torch import (
    float32,
    Tensor,
    tensor
)
from torch.utils.data import Dataset

from src.data.generation import SyntheticData


class DataExample(NamedTuple):
    """
    Stores an example's features and labels.

    Elements
    --------
    x : Tensor
        The example's features.
    y : Tensor
        The example's labels.
    """
    x: Tensor
    y: Tensor


class SyntheticDataset(Dataset):
    """
    This class is used to create a Torch Dataset for generated data.
    """

    def __init__(self, data: List[SyntheticData]):
        """
        Creates a dataset from given data.

        Parameters
        ----------
        data : List[SyntheticData]
        """
        super().__init__()

        self._data: List[SyntheticData] = data

    def __len__(self) -> int:
        """
        The length of the dataset.

        Returns
        -------
        length : int
            The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> DataExample:
        """
        Gets an item from the dataset.

        Parameters
        ----------
        index : int
            The index of the item to get.

        Returns
        -------
        item : DataExample
            The data example of the given index.
        """
        item = DataExample(
            x=tensor(self._data[index].x, dtype=float32).unsqueeze(1),
            y=tensor(self._data[index].y, dtype=float32).unsqueeze(1)
        )

        return item
