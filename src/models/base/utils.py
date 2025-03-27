"""
    @file:              utils.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains utility functions for creating and using models.
"""

from typing import Callable

from torch.nn import Module

from src.models.base.layer_factory import (
    Act,
    Dropout,
    Norm
)


def check_if_built(func: Callable) -> Callable:
    """
    This decorator ensures that the model has been built before calling the decorated method.

    Parameters
    ----------
    func : Callable
        The function to decorate.

    Returns
    -------
    _wrapper : Callable
        The decorated function.

    Raises
    ------
    Exception
        If model has not been built with .build() method.
    """

    def _wrapper(*args, **kwargs):
        self = args[0]

        if not self.is_built:
            raise Exception(
                f"The model has to be built using the 'build' method prior to calling the method {func.__name__}."
            )

        return func(*args, **kwargs)

    return _wrapper


def enable_dropout(module: Module):
    """
    This function enables dropout for a Torch Module.

    Parameters
    ----------
    module : Module
        The module for which to enable dropout.
    """
    for m in module.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def get_activation_layer(name: str, **kwargs) -> Module:
    """
    This function creates an activation layer given its name and arguments.

    Parameters
    ----------
    name : str
        The name of the activation function.
    **kwargs
        The parameters of the activation function.

    Returns
    -------
    layer : Module
        The activation layer.
    """
    act_type = Act[name]
    layer = act_type(**kwargs)

    return layer


def get_dropout_layer(
        name: str,
        dropout_dim: int,
        **kwargs
) -> Module:
    """
    This function creates a dropout layer.

    Parameters
    ----------
    name : str
        The name of the dropout type.
    dropout_dim : int
        The dropout dimension.
    **kwargs
        The parameters of the dropout module.

    Returns
    -------
    layer : Module
        The dropout layer.
    """
    dropout_type = Dropout[name, dropout_dim]
    layer = dropout_type(**kwargs)

    return layer


def get_normalization_layer(
        name: str,
        spatial_dim: int,
        **kwargs
) -> Module:
    """
    This function creates a normalization layer.

    Parameters
    ----------
    name : str
        The name of the normalization type.
    spatial_dim : int
        The number of spatial dimensions of the normalization's input
    **kwargs
        The parameters of the normalization module.

    Returns
    -------
    layer : Module
        The normalization layer.
    """
    norm_type = Norm[name, spatial_dim]
    layer = norm_type(**kwargs)

    return layer
