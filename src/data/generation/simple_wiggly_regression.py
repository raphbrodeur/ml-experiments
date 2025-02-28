"""
    @file:              simple_wiggly_regression.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 02/2025

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
            The value of coefficient a.
        b : float
            The value of coefficient b.
        c : float
            The value of coefficient c.
        d : float
            The value of coefficient d.
        i : int
            The value of exponent i.
        j : int
            The value of exponent j.
        k : int
            The value of exponent k.
        aleatoric_uncertainty : Optional[AleatoricUncertainty]
            The aleatoric uncertainty of the DGP.
        """
        super().__init__(aleatoric_uncertainty=aleatoric_uncertainty)

        self.a: float = a
        self.b: float = b
        self.c: float = c
        self.d: float = d

        self.i: int = i
        self.j: int = j
        self.k: int = k

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
        return self.a * (x ** self.i) + self.b * (sin(self.c * x) ** self.j) * (cos(self.d * x) ** self.k)
