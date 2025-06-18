"""
    @file:              aleatoric_uncertainty.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 04/2025

    @Description:       This file contains multiple aleatoric uncertainty types and their base class
                        UncertaintyDistribution. This file also contains the AleatoricUncertainty NamedTuple used to
                        store the aleatoric uncertainty to apply on features (see error-in-variables) and labels of
                        examples produced by a data generation process.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

from numpy import ndarray
from numpy.random import normal


class UncertaintyDistribution(ABC):
    """
    This class serves as a base class for all random distributions used as aleatoric uncertainty by data generation
    processes.
    """

    def __init__(self):
        """
        Initializes the abstract class.
        """
        super().__init__()

    @abstractmethod
    def sample_noise(self, signal: ndarray) -> ndarray:
        """
        Samples noise from a random distribution for given signals.

        Parameters
        ----------
        signal : ndarray
            The signals for which noise is sampled. Has shape (num_signals, ...).

        Returns
        -------
        noise : ndarray
            The sampled noise for each input signal. Has shape (num_signals, ...).
        """
        raise NotImplementedError


class NormalUncertainty(UncertaintyDistribution):
    """
    This class defines a normal (gaussian) random distribution for aleatoric uncertainty.
    """

    def __init__(
            self,
            mean: float = 0.0,
            standard_deviation: float = 1.0
    ):
        """
        Initializes the class.

        Parameters
        ----------
        mean : float
            The mean of the normal distribution. Defaults to 0.0.
        standard_deviation : float
            The standard deviation of the normal distribution. Defaults to 1.0.
        """
        super().__init__()

        self._mean: float = mean
        self._std: float = standard_deviation

    def sample_noise(self, signal: ndarray) -> ndarray:
        """
        Samples noise from a normal (gaussian) distribution for given signals.

        Parameters
        ----------
        signal : ndarray
            The signals for which noise is sampled. Has shape (num_signals, ...)

        Returns
        -------
        noise : ndarray
            The sampled noise for each input signal. Has shape (num_signals, ...).
        """
        noise = normal(self._mean, self._std, signal.shape)

        return noise


class AleatoricUncertainty(NamedTuple):
    """
    Stores the aleatoric uncertainty to apply to features (x) (for error-in-variable simulations) and labels (y).

    Elements
    --------
    x_uncertainty : Optional[UncertaintyDistribution]
        The uncertainty on features (x). See error-in-variables, measurement error. Either None, NormalUncertainty, ...
    y_uncertainty : Optional[UncertaintyDistribution]
        The uncertainty on labels (y). Either None, NormalUncertainty, ...
    """
    x_uncertainty: Optional[UncertaintyDistribution] = None
    y_uncertainty: Optional[UncertaintyDistribution] = None
