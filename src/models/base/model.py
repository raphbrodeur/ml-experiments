"""
    @file:              model.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains the class Model which serves as a base class for all Torch models.
"""

from abc import ABC, abstractmethod
from typing import (
    Callable,
    Dict,
    Tuple
)

from torch import Tensor
from torch.nn import Module

from src.models.base.utils import check_if_built


class _LearningAlgorithmRegistry:
    """
    This class is used to register and access the learning algorithms of a model.
    """

    def __init__(self):
        """
        Initializes the learning algorithm registry.
        """
        self._learning_algorithms: Dict[str, Callable] = {}

    def __getitem__(self, name: str) -> Callable:
        """
        Gets a learning algorithm given its name.

        Parameters
        ----------
        name : str
            The name of the learning algorithm to get.

        Returns
        -------
        learning_algorithm : Callable
            The learning algorithm.
        """
        return self._learning_algorithms[name.upper()]

    def __setitem__(self, name: str, func: Callable):
        """
        Sets a learning algorithm given its name. Allows setting the learning algorithm via the same logic as for a
        dictionary within the Model class.

        Parameters
        ----------
        name : str
            The name of the learning algorithm to set.
        func : Callable
            The learning algorithm.
        """
        self._learning_algorithms[name.upper()] = func

    @property
    def learning_algorithms(self) -> Tuple[str, ...]:
        """
        The names of all the learning algorithms in the registry.

        Returns
        -------
        learning_algorithms : Tuple[str, ...]
            The names of the learning algorithms.
        """
        return tuple(self._learning_algorithms)


class Model(Module, ABC):
    """
    This class serves as a base class for all implemented Torch models.

    Notes
    -----
    The purpose of using this class, rather than simply using torch.nn.Module, is to enforce the use of the .build()
    method and the .learn["algorithm"](args) method. This is done so that models can share the same architecture yet
    have different initialization weights and training procedures. This is useful for experiments in which multiple
    trainings of the same model are done.
    """

    def __init__(self):
        """
        Initializes the base model.
        """
        super().__init__()

        self._is_built: bool = False
        self._learning_algorithms: _LearningAlgorithmRegistry = _LearningAlgorithmRegistry()

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

    @property
    def is_built(self) -> bool:
        """
        Whether the model has been built with the build() method.

        Returns
        -------
        is_built : bool
            Whether the model has been built.
        """
        return self._is_built

    def build(self):
        """
        Builds the model and initializes weights. Children classes need to call this method in their build() method.
        """
        self._is_built = True   # Sets the _is_built attribute to True

    @property
    def learning_algorithms(self) -> Tuple[str, ...]:
        """
        The names of all the learning algorithms implemented for the model.

        Returns
        -------
        learning_algorithms : Tuple[str, ...]
            The names of the learning algorithms.
        """
        return self._learning_algorithms.learning_algorithms

    def _add_learning_algorithm(self, name: str, func: Callable):
        """
        Registers a new learning algorithm to the registry of an instance of a Model's child class under a given name.

        Parameters
        ----------
        name : str
            The name of the learning algorithm.
        func : Callable
            The learning algorithm to add.

        Raises
        ------
        ValueError
            If a learning algorithm is already registered under the same name.
        """
        if name.upper() in self.learning_algorithms:
            raise ValueError(f"Learning algorithm already registered under the name {name}.")

        self._learning_algorithms[name] = func

    def register_learning_algorithm(self, name: str) -> Callable:
        """
        This decorator registers the decorated learning algorithm to an instance of a Model's child class under a given
        name.

        Parameters
        ----------
        name : str
            The name of the learning algorithm.

        Returns
        -------
        _wrapper : Callable
            The decorated function.
        """
        def _wrapper(func: Callable) -> Callable:
            self._add_learning_algorithm(name=name, func=func)
            return func

        return _wrapper

    @property
    @check_if_built
    def learn(self) -> _LearningAlgorithmRegistry:
        """
        Not intended to be used as an actual property! This property is used to get and call a specific learning
        algorithm registered to a child class instance.

        Returns
        -------
        learning_algorithms_registry : _LearningAlgorithmRegistry
            The learning algorithm registry of the model.

        Examples
        --------
            # Define a model inheriting from Model
            class MyModel(Model):
                ...

            net = MyModel()     # Create an instance of the model

            # Define and register a learning algorithm under the name "adam-mse"
            @net.register_learning_algorithm("adam-mse")
            def Adam_MSE(args):
                ...

            net.build()     # Build the model

            # Call the learning algorithm to train the model
            net.learn["adam-mse"](args)
        """
        return self._learning_algorithms

    @check_if_built
    def eval(self):
        """
        Overrides the .eval() method of torch.nn.Module to ensure that the model is built before calling. Calls the
        underlying torch.nn.Module method with the provided arguments.
        """
        return super().eval()

    @check_if_built
    def load_state_dict(self, *args, **kwargs):
        """
        Overrides the .load_state_dict() method of torch.nn.Module to ensure that the model is built before calling.
        Calls the underlying torch.nn.Module method with the provided arguments.
        """
        return super().load_state_dict(*args, **kwargs)

    @check_if_built
    def to(self, *args, **kwargs):
        """
        Overrides the .to() method of torch.nn.Module to ensure that the model is built before calling. Calls the
        underlying torch.nn.Module method with the provided arguments.
        """
        return super().to(*args, **kwargs)

    @check_if_built
    def train(self, *args, **kwargs):
        """
        Overrides the .train() method of torch.nn.Module to ensure that the model is built before calling. Calls the
        underlying torch.nn.Module method with the provided arguments.
        """
        return super().train(*args, **kwargs)
