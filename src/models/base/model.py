"""
    @file:              model.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains the base class for all Torch models.
"""

from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

from src.models.base.utils import check_if_built


class Model(Module, ABC):
    """
    This class serves as a base class for all implemented Torch models.
    """

    def __init__(self):
        """
        Initializes the base model.
        """
        super().__init__()

        self._is_built: bool = False

    @property
    def is_built(self) -> bool:
        """
        Whether the model has been built with the build() method.
        """
        return self._is_built

    def build(self):
        """
        Builds the model and initializes weights.
        """
        self._is_built = True   # Sets the _is_built attribute to True

    @abstractmethod
    @check_if_built
    def forward(self, x: Tensor) -> Tensor:
        """
        Gets output tensors for input tensors.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        y : Tensor
            The output tensor.
        """
        raise NotImplementedError
