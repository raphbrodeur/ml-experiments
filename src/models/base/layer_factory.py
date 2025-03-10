"""
    @file:              layer_factory.py
    @Author:            Raphael Brodeur

    @Creation Date:     03/2025
    @Last modification: 03/2025

    @Description:       This file contains layer factories for activation, convolution, dropout, normalization, padding
                        and pooling layers. Factory functions are registered to instances of the class LayerFactory().
                        Function registration is done with a decorator. Inspired by MONAI.
"""

from typing import Any, Callable, Dict, Tuple, Type, Union

import torch.nn as nn


class LayerFactory:
    """
    This class serves as a base for layer factories. Factory functions can be registered to instances of this class with
    the decorator register_factory_function().

    Examples
    --------
        Norm = LayerFactory()

        @Norm.register_factory_function("batch")
        def batch_norm_factory(dim) -> BatchNorm:
            types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            return types[dim - 1]

        batch_norm_layer = Norm["batch", norm_dim]

        @Norm.register_factory_function("layer")
        def layer_norm_factory(dim) -> GroupNorm:
            return nn.GroupNo
    """

    def __init__(self):
        """
        Initializes the layer factory.
        """
        self._factories: Dict[str, Callable] = {}

    def __getitem__(self, args: str | Tuple) -> Any:
        """
        Gets a layer type for a given name or tuple of name and arguments.

        Parameters
        ----------
        args : str | Tuple
            The arguments specifying which layer class to get. Either a string of the factory's name or a tuple whose
            first element is the name of the factory and whose remaining elements are positional arguments of the
            factory function.

        Returns
        -------
        layer_type : Any
            The layer's class. That is, not an instance of the layer but its actual class. e.g. nn.Conv3d
        """
        if isinstance(args, str):
            factory_name = args
            args = ()   # *args is empty tuple

        else:
            factory_name, *args = args  # Remaining elements of args are unpacked into the new list args

        factory_function = self._factories[factory_name.upper()]    # Gets the factory function
        layer_type = factory_function(*args)                        # Gets the layer type from the factory

        return layer_type

    @property
    def factories(self) -> Tuple[str, ...]:
        """
        Names of factory functions registered to the class.
        """
        return tuple(self._factories)

    def _add_factory_function(
            self,
            name: str,
            func: Callable
    ):
        """
        Registers a new factory function to the LayerFactory() instance under a given name.

        Parameters
        ----------
        name : str
            The name of the factory function.
        func : Callable
            The factory function to add.
        """
        self._factories[name.upper()] = func

    def register_factory_function(self, name: str) -> Callable:
        """
        This decorator registers the decorated factory function to a LayerFactory() instance under a given name.

        Parameters
        ----------
        name : str
            The name of the factory function.

        Returns
        -------
        _wrapper : Callable
            The decorated function.
        """

        def _wrapper(func: Callable) -> Callable:
            self._add_factory_function(name=name, func=func)
            return func

        return _wrapper


# Create an activation factory and some factory functions
Act = LayerFactory()


@Act.register_factory_function("elu")
def elu_factory() -> Type[nn.ELU]:
    return nn.ELU


@Act.register_factory_function("relu")
def relu_factory() -> Type[nn.ReLU]:
    return nn.ReLU


@Act.register_factory_function("leakyrelu")
def leakyrelu_factory() -> Type[nn.LeakyReLU]:
    return nn.LeakyReLU


@Act.register_factory_function("prelu")
def prelu_factory() -> Type[nn.PReLU]:
    return nn.PReLU


@Act.register_factory_function("relu6")
def relu6_factory() -> Type[nn.ReLU6]:
    return nn.ReLU6


@Act.register_factory_function("selu")
def selu_factory() -> Type[nn.SELU]:
    return nn.SELU


@Act.register_factory_function("celu")
def celu_factory() -> Type[nn.CELU]:
    return nn.CELU


@Act.register_factory_function("gelu")
def gelu_factory() -> Type[nn.GELU]:
    return nn.GELU


@Act.register_factory_function("sigmoid")
def sigmoid_factory() -> Type[nn.Sigmoid]:
    return nn.Sigmoid


@Act.register_factory_function("tanh")
def tanh_factory() -> Type[nn.Tanh]:
    return nn.Tanh


@Act.register_factory_function("softmax")
def softmax_factory() -> Type[nn.Softmax]:
    return nn.Softmax


@Act.register_factory_function("logsoftmax")
def logsoftmax_factory() -> Type[nn.LogSoftmax]:
    return nn.LogSoftmax


# Create a dropout factory and some factory functions
Dropout = LayerFactory()


@Dropout.register_factory_function("dropout")
def dropout_factory(dim: int) -> Type[Union[nn.Dropout, nn.Dropout2d, nn.Dropout3d]]:
    types = [nn.Dropout, nn.Dropout2d, nn.Dropout3d]
    return types[dim - 1]


@Dropout.register_factory_function("alphadropout")
def alpha_dropout_factory(_dim) -> Type[nn.AlphaDropout]:
    return nn.AlphaDropout


# Create a normalization factory and some factory functions
Norm = LayerFactory()


@Norm.register_factory_function("instance")
def instance_norm_factory(dim: int) -> Type[Union[nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]]:
    types = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    return types[dim - 1]


@Norm.register_factory_function("batch")
def batch_norm_factory(dim: int) -> Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]]:
    types = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    return types[dim - 1]


@Norm.register_factory_function("group")
def group_norm_factory(_dim) -> Type[nn.GroupNorm]:
    return nn.GroupNorm


@Norm.register_factory_function("layer")
def layer_norm_factory(_dim) -> Type[nn.LayerNorm]:
    return nn.LayerNorm


@Norm.register_factory_function("localresponse")
def local_response_norm_factory(_dim) -> Type[nn.LocalResponseNorm]:
    return nn.LocalResponseNorm


@Norm.register_factory_function("syncbatch")
def sync_batch_norm_factory(_dim) -> Type[nn.SyncBatchNorm]:
    return nn.SyncBatchNorm
