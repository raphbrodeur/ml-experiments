"""
    @file:              utils.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 02/2025

    @Description:       This file contains utility functions for creating models.
"""

from torch.nn import LeakyReLU, Module, PReLU, ReLU, Sigmoid, Tanh


def check_if_built(_func):
    """
    This decorator is used to ensure that the model has been built before calling the decorated method.

    Parameters
    ----------
    _func : function
        The function to decorate.

    Returns
    -------
    wrapper : function
        The decorated function.
    """
    def wrapper(*args, **kwargs):
        self = args[0]

        assert self._is_built, (
            f"The model has to be built using the 'build' method prior to calling the method {_func.__name__}."
        )

        return _func(*args, **kwargs)

    return wrapper


def get_activation_layer(name: str, **kwargs):
    """
    This function gets an activation module given its name and arguments.

    Parameters
    ----------
    name : str
        The name of the activation function.
    **kwargs
        The parameters of the activation function.

    Returns
    -------
    layer : Module
        The activation module.
    """
    if name.upper() == "RELU":
        return ReLU(**kwargs)
    elif name.upper() == "LEAKYRELU":
        return LeakyReLU(**kwargs)
    elif name.upper() == "TANH":
        return Tanh()
    elif name.upper() == "SIGMOID":
        return Sigmoid()
    elif name.upper() == "PRELU":
        return PReLU(**kwargs)

    else:
        raise ValueError(f"Activation function {name} not supported.")


def get_normalization_layer(name: str, **kwargs):
    """
    This function gets a normalization module given its name and arguments.

    Parameters
    ----------
    name : str
        The name of the normalization function.
    **kwargs
        The parameters of the normalization function.

    Returns
    -------
    layer : Module
        The normalization module.
    """
    pass


def get_dropout_layer(prob: float):
    """
    This function gets a dropout layer.

    Parameters
    ----------
    prob : float
        The dropout probability.

    Returns
    -------
    layer : Module
        The dropout module.
    """
    pass
