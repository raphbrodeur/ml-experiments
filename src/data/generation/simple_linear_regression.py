"""
    @file:              simple_linear_regression.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 02/2025

    @Description:       This file contains the generation process of simple linear regression data. The underlying
                        deterministic function is defined as:
                            R^1 -> R^1
                            x ↦ a * x + b
"""

from typing import Optional

from numpy import ndarray

from src.data.generation.base import AleatoricUncertainty, DataGenerationProcess


class SimpleLinearRegression(DataGenerationProcess):
    """
    This class generates simple linear regression data. The data generation process (DGP) is decomposed into a
    deterministic function and stochastic terms representing aleatoric uncertainty. The underlying deterministic
    function is defined as:
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
            The value of coefficient a.
        b : float
            The value of coefficient b.
        aleatoric_uncertainty : Optional[AleatoricUncertainty]
            The aleatoric uncertainty of the DGP.
        """
        super().__init__(aleatoric_uncertainty=aleatoric_uncertainty)

        self.a: float = a
        self.b: float = b

    def _deterministic_function(self, x: ndarray) -> ndarray:
        """
        Gets the deterministic component of target value y for a given observation's features x according to the DGP's
        underlying deterministic function.

        Parameters
        ----------
        x : np.ndarray
            The observations' features x. Has shape (num_obs, ...).

        Returns
        -------
        y : np.ndarray
            The observations' targets y. Has shape (num_obs, ...).
        """
        return self.a * x + self.b
