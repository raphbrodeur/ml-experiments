"""
    @file:              simple_wiggly_regression.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains the generation process of simple regression data. The underlying
                        deterministic function is defined as:
                            R^1 -> R^1
                            x ↦ a * x^i + b * sin^j(c * x) * cos^k(d * x)
"""

from typing import Optional

from numpy import cos, ndarray, sin

from src.data.generation.base import AleatoricUncertainty, DataGenerationProcess


class SimpleWigglyRegression(DataGenerationProcess):
    """
    This class generates simple regression data. The data generation process (DGP) is decomposed into a
    deterministic function and stochastic terms representing aleatoric uncertainty. The underlying deterministic
    function is defined as:
        R^1 -> R^1
        x ↦ a * x^i + b * sin^j(c * x) * cos^k(d * x)
    """

    def __init__(
            self,
            a: float = 1.0,
            b: float = -10.0,
            c: float = 1.0,
            d: float = 1.0,
            i: int = 2,
            j: int = 1,
            k: int = 1,
            aleatoric_uncertainty: Optional[AleatoricUncertainty] = None
    ):
        """
        Initializes the data generation process.

        Parameters
        ----------
        a : float
            The value of coefficient a. Defaults to 1.0.
        b : float
            The value of coefficient b. Defaults to -10.0.
        c : float
            The value of coefficient c. Defaults to 1.0.
        d : float
            The value of coefficient d. Defaults to 1.0.
        i : int
            The value of exponent i. Defaults to 2.
        j : int
            The value of exponent j. Defaults to 1.
        k : int
            The value of exponent k. Defaults to 1.
        aleatoric_uncertainty : Optional[AleatoricUncertainty]
            The aleatoric uncertainty of the DGP. Defaults to None.
        """
        super().__init__(aleatoric_uncertainty=aleatoric_uncertainty)

        self._a: float = a
        self._b: float = b
        self._c: float = c
        self._d: float = d

        self._i: int = i
        self._j: int = j
        self._k: int = k

    def deterministic_function(self, x: ndarray) -> ndarray:
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
        return self._a * (x ** self._i) + self._b * (sin(self._c * x) ** self._j) * (cos(self._d * x) ** self._k)
