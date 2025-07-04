"""
    @file:              simple_linear_regression.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 04/2025

    @Description:       This file contains the generation process of simple linear regression data examples. The
                        underlying deterministic function is defined as:
                            R^1 -> R^1
                            x ↦ a * x + b
"""

from typing import Optional

from numpy import ndarray

from src.data.generation.base import AleatoricUncertainty, DataGenerationProcess


class SimpleLinearRegression(DataGenerationProcess):
    """
    This class generates simple linear regression data examples.

    The data generation process (DGP) is decomposed into a deterministic function and stochastic terms representing
    aleatoric uncertainty. The underlying deterministic function is defined as:
        R^1 -> R^1
        x ↦ a * x + b
    """

    def __init__(
            self,
            a: float = 0.5,
            b: float = 1.0,
            aleatoric_uncertainty: Optional[AleatoricUncertainty] = None
    ):
        """
        Initializes the data generation process.

        Parameters
        ----------
        a : float
            The value of coefficient a. Defaults to 0.5.
        b : float
            The value of coefficient b. Defaults to 1.0.
        aleatoric_uncertainty : Optional[AleatoricUncertainty]
            The aleatoric uncertainty of the DGP. Defaults to None.
        """
        super().__init__(aleatoric_uncertainty=aleatoric_uncertainty)

        self._a: float = a
        self._b: float = b

    def deterministic_function(self, x: ndarray) -> ndarray:
        """
        Gets the deterministic component of the labels for given examples' features according to the DGP's underlying
        deterministic function.

        Parameters
        ----------
        x : np.ndarray
            The examples' features. Has shape (num_examples, ...).

        Returns
        -------
        y : np.ndarray
            The examples' labels. Has shape (num_examples, ...).
        """
        return self._a * x + self._b
