"""
    @file:              data_generation_process.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 02/2025

    @Description:       This file contains the abstract class DataGenerationProcess that serves as a base class for all
                        data generation processes.
"""

from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Tuple

from numpy import atleast_1d, ndarray, zeros

from src.data.generation.base.aleatoric_uncertainty import AleatoricUncertainty, UncertaintyDistribution


class SyntheticData(NamedTuple):
    """
    Stores a single observation of synthetic data.

    Elements
    --------
    x : ndarray
        The observation's features x.
    y : ndarray
        The observation's target y.
    """
    x: ndarray
    y: ndarray


class DataGenerationProcess(ABC):
    """
    This class serves as a base class for all data generation processes (DGPs). The DGP is decomposed into a
    deterministic function and stochastic terms representing aleatoric uncertainty.
    """

    def __init__(self, aleatoric_uncertainty: Optional[AleatoricUncertainty] = None):
        """
        Initializes the base DGP.

        Parameters
        ----------
        aleatoric_uncertainty : Optional[AleatoricUncertainty]
            The aleatoric uncertainty of the DGP.
        """
        super().__init__()

        if aleatoric_uncertainty is None:
            aleatoric_uncertainty = AleatoricUncertainty()

        self.aleatoric_uncertainty = aleatoric_uncertainty

    @abstractmethod
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
        raise NotImplementedError

    def _aleatoric_uncertainty_terms(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Gets error terms representing aleatoric uncertainty for given observations' features x and targets y.

        Parameters
        ----------
        x : np.ndarray
            The observations' features x before considering feature uncertainty. Has shape (num_obs, ...).

        Returns
        -------
        uncertainty : Tuple[ndarray, ndarray]
            A tuple containing the observations' features x uncertainty terms and the observations' targets (y)
            uncertainty terms.
        """
        # Features uncertainty (for errors-in-variables cases, measurement errors)
        if self.aleatoric_uncertainty.feature_uncertainty is None:
            x_unc = zeros(len(x))   # len(x) is the number of observations

        elif isinstance(self.aleatoric_uncertainty.feature_uncertainty, UncertaintyDistribution):
            x_unc = self.aleatoric_uncertainty.feature_uncertainty.sample(x)

        else:
            raise Exception("Not a valid measure uncertainty distribution.")

        # Target uncertainty
        if self.aleatoric_uncertainty.target_uncertainty is None:
            y_unc = zeros(len(x))   # len(x) is the number of observations

        elif isinstance(self.aleatoric_uncertainty.target_uncertainty, UncertaintyDistribution):
            y_unc = self.aleatoric_uncertainty.target_uncertainty.sample(x)

        else:
            raise Exception("Not a valid aleatoric target uncertainty type")

        return x_unc, y_unc

    def sample_dataset(self, x: ndarray) -> List[SyntheticData]:
        """
        Samples a dataset from the DGP. Takes as input the observations' features x of the dataset to be sampled rather
        than simply the amount of observations to sample so that domain is inherently defined. Also, in practice, a
        dataset's domain can often be chosen (e.g. choosing to include more observations of some class in a training set
        or test set). For a random domain, one can simply input a sampled ndarray of observations features.

        Note that error terms are added to the features x only after the deterministic parts of the targets y have been
        calculated. This is more truthful to an error-in-variable/measurement error situation; the target is associated
        to the true but unobserved regressor. See https://en.wikipedia.org/wiki/Errors-in-variables_model.

        Parameters
        ----------
        x : ndarray
            The observations' features x. Has shape (num_obs, ...).

        Returns
        -------
        synthetic_data : List[SyntheticData]
            A list of SyntheticData named tuples. Each item of the list is a different observation.
        """
        y = self._deterministic_function(x)

        # Error terms are added to x after deterministic y(x) has been calculated. See error-in-variables models.
        x_unc, y_unc = self._aleatoric_uncertainty_terms(x)

        synthetic_data = []
        for i in range(len(x)):     # len(x) is the number of observations.
            synthetic_data.append(
                SyntheticData(
                    x=atleast_1d(x[i] + x_unc[i]),
                    y=atleast_1d(y[i] + y_unc[i])
                )
            )

        return synthetic_data
