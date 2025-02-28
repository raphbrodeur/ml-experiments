"""
    @file:              aleatoric_uncertainty.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 02/2025

    @Description:       This file contains different aleatoric uncertainties types and a function to treat them. This is
                        used for data generation.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

from numpy import ndarray
from numpy.random import normal


class UncertaintyDistribution(ABC):
    """
    This class defines the abstract class UncertaintyDistribution. This class serves as a base class for all random
    distributions used as aleatoric uncertainty by the data generation processes.
    """

    def __init__(self):
        """
        Initializes the abstract class.
        """
        super().__init__()

    @abstractmethod
    def sample(self, signal: ndarray) -> ndarray:
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
    This class defines a normal random distribution to be used as normal (gaussian) aleatoric uncertainty.
    """

    def __init__(
            self,
            mean: float = 0.0,
            std: float = 1.0
    ):
        super().__init__()

        self.mean = mean
        self.std = std

    def sample(self, signal: ndarray) -> ndarray:
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
        noise = normal(self.mean, self.std, signal.shape)

        return noise


class AleatoricUncertainty(NamedTuple):
    """
    Stores the aleatoric uncertainty to apply to targets y and features x (for error-in-variable simulations).

    Elements
    --------
    feature_uncertainty : Optional[UncertaintyDistribution]
        The uncertainty on features x. See error-in-variables, measurement error. Either None,
        NormalUncertainty, ...
    target_uncertainty : Optional[UncertaintyDistribution]
        The uncertainty on targets y. Either None, NormalUncertainty, ...
    """
    feature_uncertainty: Optional[UncertaintyDistribution] = None
    target_uncertainty: Optional[UncertaintyDistribution] = None
