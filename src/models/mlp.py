"""
    @file:              mlp.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains a simple MLP model.
"""

from typing import Optional, Sequence

from torch import Tensor
from torch.nn import Module

from src.models.base import (
    check_if_built,
    MLPBlock,
    Model
)


class MLP(Model):
    """
    This class is an MLP model.
    """

    def __init__(
            self,
            hidden_channels_width: Sequence[int],
            activation: str,
            dropout: Optional[float],
            normalization: Optional[str]
    ):
        """
        Initializes the MLP.
        """
        super().__init__()

        self._hidden_channels_width: Sequence[int] = hidden_channels_width
        self._activation: Optional[str] = activation
        self._dropout: Optional[float] = dropout
        self._normalization: Optional[str] = normalization

        self._mlp: Optional[Module] = None

    def build(self):
        """
        Builds the model and initializes the weights.
        """
        super().build()

        self._mlp = MLPBlock(
            hidden_channels_width=self._hidden_channels_width,
            activation=self._activation,
            dropout=self._dropout,
            normalization=self._normalization
        )

    @check_if_built
    def forward(self, x: Tensor) -> Tensor:
        """
        Gets an output tensor for an input tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        y : Tensor
            The output tensor.
        """
        return self._mlp(x)
