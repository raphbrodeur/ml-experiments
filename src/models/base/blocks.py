"""
    @file:              blocks.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains blocks and modules used to build torch models.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

from torch.nn import Linear, Sequential

from src.models.base.utils import (
    get_activation_layer,
    get_dropout_layer,
    get_normalization_layer
)


class ADN(Sequential):
    """
    This class constructs a sequential module of optional activation, dropout, and normalization layers with
    a given ordering. Inspired by the ADN module from MONAI.
    """

    def __init__(
            self,
            ordering: str = "NDA",
            activation: Optional[Union[str, Tuple[str, Dict]]] = None,
            dropout: Optional[Union[float, Tuple[str, Dict]]] = None,
            normalization: Optional[Union[str, Tuple[str, Dict]]] = None
    ):
        """
        Initializes the NAD module.

        Parameters
        ----------
        ordering : str
            The ordering of the activation (A), dropout (D) and normalization (N) layers. Defaults to "NAD".
        activation : Optional[Union[str, Tuple[str, Dict]]]
            The activation layer to add. Optional. Either the activation's name (str) or a tuple of the name (str) and
            a dictionary of keyword arguments. Defaults to None.
        dropout : Optional[Union[float, Tuple[str, Dict]]]
            The dropout layer to add. Optional. Either the dropout probability (float) for default dropout or a tuple of
            the dropout layer's name and a dictionary of keyword arguments. Defaults to None.
        normalization : Optional[Union[str, Tuple[str, Dict]]]
            The normalization layer to use. Optional Defaults to None.
        """
        super().__init__()

        module_dict = {"A": None, "D": None, "N": None}

        if activation is not None:
            if isinstance(activation, str):
                act_name = activation
                act_kwargs = {}
            else:
                act_name, act_kwargs = activation

            module_dict["A"] = get_activation_layer(name=act_name, **act_kwargs)

        if dropout is not None:
            if isinstance(dropout, float):
                dropout_name = "dropout"
                dropout_kwargs = {"p": float(dropout)}
            else:
                dropout_name, dropout_kwargs = dropout

            module_dict["D"] = get_dropout_layer(name=dropout_name, dropout_dim=1, **dropout_kwargs)

        if normalization is not None:
            if isinstance(normalization, str):
                norm_name = normalization
                norm_kwargs = {}
            else:
                norm_name, norm_kwargs = normalization

            module_dict["N"] = get_normalization_layer(name=norm_name, spatial_dim=1, **norm_kwargs)

        for module in ordering:
            if module_dict[module] is not None:
                self.add_module(name=module, module=module_dict[module])


class MLPBlock(Sequential):
    """
    This class constructs an MLP with optional normalization, activation, and dropout.
    """

    def __init__(
            self,
            hidden_channels_width: Sequence[int],
            activation: str = "prelu",
            dropout: float = 0.2,
            normalization: str = "instance"
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

            # ADN
            layer.add_module(
                name="ADN",
                module=ADN(
                    ordering="NDA",
                    activation=activation,
                    dropout=dropout,
                    normalization=normalization
                )
            )

            self.add_module(f"Layer_{i}", layer)

            input_width = width

        final_layer = Sequential()
        final_layer.add_module("Linear", Linear(in_features=input_width, out_features=1))
        self.add_module("Layer_final", final_layer)
