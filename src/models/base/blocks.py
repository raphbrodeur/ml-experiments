"""
    @file:              blocks.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains blocks and modules used to build torch models.
"""

from typing import (
    Dict,
    Optional,
    Sequence,
    Tuple
)

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
            activation: Optional[str | Tuple[str, Dict]] = None,
            dropout: Optional[float | Tuple[str, Dict]] = None,
            normalization: Optional[str | Tuple[str, Dict]] = None
    ):
        """
        Initializes the ADN module.

        Parameters
        ----------
        ordering : str
            The ordering of the activation (A), dropout (D) and normalization (N) layers. Defaults to "NDA".
        activation : Optional[str | Tuple[str, Dict]]
            The activation layer to add. Optional. Either the activation's name (str) or a tuple of the name and a
            dictionary of keyword arguments. Defaults to None.
        dropout : Optional[float | Tuple[str, Dict]]
            The dropout layer to add. Optional. Either the dropout probability (float) for default dropout or a tuple of
            the dropout layer's name (str) and a dictionary of keyword arguments. Defaults to None.
        normalization : Optional[str | Tuple[str, Dict]]
            The normalization layer to add. Optional. Either the normalization's name (str) or a tuple of the name
            and a dictionary of keyword arguments. Defaults to None.

        Raises
        ------
        Exception
            If activation, dropout and normalization are all set to None.
        """
        if activation is None and dropout is None and normalization is None:
            raise Exception("Initializing an empty ADN block.")

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
    This class constructs an MLP with optional activation, dropout and normalization.
    """

    def __init__(
            self,
            hidden_channels_width: Sequence[int],
            activation: Optional[str | Tuple[str, Dict]] = "PReLU",
            dropout: Optional[float | Tuple[str, Dict]] = None,
            normalization: Optional[str | Tuple[str, Dict]] = None,
    ):
        """
        Initializes the MLP module.

        Parameters
        ----------
        hidden_channels_width : Sequence[int]
            The width of the hidden layers.
        activation : Optional[str | Tuple[str, Dict]]
            The activation layer to use. Optional. Either the activation's name (str) or a tuple of the name and a
            dictionary of keyword arguments. Defaults to "PReLU".
        dropout :  Optional[float | Tuple[str, Dict]]
            The dropout layer to use. Optional. Either the dropout probability (float) for default dropout or a tuple of
            the dropout layer's name and a dictionary of keyword arguments. Defaults to None.
        normalization : Optional[str | Tuple[str, Dict]]
            The normalization layer to use. Optional. Either the normalization's name (str) or a tuple of the name
            and a dictionary of keyword arguments. Defaults to None.
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
