"""
    @file:              blocks.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 02/2025

    @Description:       This file contains multiple blocks and modules used to build torch models.
"""

from typing import Optional, Sequence

from torch.nn import Dropout, Linear, Sequential

from src.models.base.utils import get_activation_layer


class NAD(Sequential):
    """
    This class constructs a sequential module of optional normalization (N), activation (A), and dropout (D) layers with
    given ordering. Inspired by the ADN module from MONAI.
    """

    def __init__(
            self,
            ordering: str = "NAD",
            norm: Optional[str] = None,
            act: Optional[str] = None,
            dropout: Optional[float] = None
    ):
        """
        Initializes the NAD module.

        Parameters
        ----------
        ordering : str
            The ordering of the normalization, activation, and dropout layers. Defaults to "NAD".
        norm : Optional[str]
            The normalization layer to use. Defaults to None.
        act : Optional[str]
            The activation layer to use. Defaults to None.
        dropout : Optional[float]
            The dropout probability to use. Defaults to None.
        """
        super().__init__()

        module_dict = {"N": None, "A": None, "D": None}

        if norm is not None:
            pass

        if act is not None:
            pass

        if dropout is not None:
            pass


class MLPBlock(Sequential):
    """
    This class constructs an MLP with optional normalization, activation, and dropout.
    """

    def __init__(
            self,
            hidden_channels_width: Sequence[int],
            activation: str = "PReLU",
            dropout: float = 0.0
    ):
        """
        Initializes the MLP module.

        Parameters
        ----------
        hidden_channels_width : Sequence[int]
            The width of the hidden layers.
        activation : str
            The activation function to use. Defaults to "PReLU".
        dropout : float
            The dropout probability to use. Defaults to 0.0.
        """
        super().__init__()

        input_width: int = 1
        for i, width in enumerate(hidden_channels_width):
            layer = Sequential()

            # Linear layer
            layer.add_module("Linear", Linear(in_features=input_width, out_features=width))

            # NAD
            layer.add_module(activation, get_activation_layer(name=activation))
            layer.add_module("Dropout", Dropout(p=dropout))

            self.add_module(f"Layer_{i}", layer)

            input_width = width

        final_layer = Sequential()
        final_layer.add_module("Linear", Linear(in_features=input_width, out_features=1))
        self.add_module("Layer_final", final_layer)
