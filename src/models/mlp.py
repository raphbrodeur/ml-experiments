"""
    @file:              mlp.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains a simple MLP model with an arbitrary number of layers.
"""

from typing import (
    Dict,
    Optional,
    Sequence,
    Tuple
)

from torch import Tensor
from torch.nn import Module

from src.models.base import (
    check_if_built,
    MLPBlock,
    Model
)


class MLP(Model):
    """
    This class is an MLP model with an arbitrary number of layers.
    """

    def __init__(
            self,
            input_features: int,
            hidden_channels_width: Sequence[int],
            output_features: int,
            activation: Optional[str | Tuple[str, Dict]],
            dropout: Optional[float | Tuple[str, Dict]] = None,
            normalization: Optional[str | Tuple[str, Dict]] = None,
            ordering: str = "NDA"
    ):
        """
        Initializes the MLP.

        Parameters
        ----------
        input_features : int
            The number of features of the MLP's input.
        hidden_channels_width : Sequence[int]
            The width of each hidden layer in the MLP.
        activation : Optional[str | Tuple[str, Dict]]
            The activation layer to use. Optional. Either the activation's name (str) or a tuple of the name and a
            dictionary of keyword arguments.
        dropout :  Optional[float | Tuple[str, Dict]]
            The dropout layer to use. Optional. Either the dropout probability (float) for default dropout or a tuple of
            the dropout layer's name and a dictionary of keyword arguments. Defaults to None.
        normalization : Optional[str | Tuple[str, Dict]]
            The normalization layer to use. Optional. Either the normalization's name (str) or a tuple of the name
            and a dictionary of keyword arguments. Defaults to None.
        ordering : str
            The ordering of the activation (A), dropout (D) and normalization (N) layers. Defaults to "NDA".
        """
        super().__init__()

        self._input_features = input_features
        self._hidden_channels_width: Sequence[int] = hidden_channels_width
        self._output_features: int = output_features
        self._activation: Optional[str | Tuple[str, Dict]] = activation
        self._dropout: Optional[float | Tuple[str, Dict]] = dropout
        self._normalization: Optional[str | Tuple[str, Dict]] = normalization
        self._ordering: str = ordering

        self._mlp: Optional[Module] = None

    def build(self):
        """
        Builds the model and initializes the weights.
        """
        super().build()

        self._mlp = MLPBlock(
            input_features=self._input_features,
            hidden_channels_width=self._hidden_channels_width,
            output_features=self._output_features,
            activation=self._activation,
            dropout=self._dropout,
            normalization=self._normalization,
            ordering=self._ordering
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
