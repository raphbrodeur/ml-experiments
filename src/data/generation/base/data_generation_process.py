"""
    @file:              data_generation_process.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 04/2025

    @Description:       This file contains the abstract class DataGenerationProcess that serves as a base class for all
                        data generation processes.
"""

from abc import ABC, abstractmethod
from typing import (
    List,
    NamedTuple,
    Optional,
    Tuple
)

from numpy import (
    atleast_1d,
    ndarray,
    zeros
)

from src.data.generation.base.aleatoric_uncertainty import AleatoricUncertainty


class SyntheticData(NamedTuple):
    """
    Stores a single example of synthetic data.

    Elements
    --------
    x : ndarray
        The example's features.
    y : ndarray
        The example's labels.
    """
    x: ndarray
    y: ndarray


class DataGenerationProcess(ABC):
    """
    This class serves as a base class for all data generation processes (DGPs).

    The DGP is decomposed into a deterministic function and stochastic terms representing aleatoric uncertainty.
    """

    def __init__(self, aleatoric_uncertainty: Optional[AleatoricUncertainty] = None):
        """
        Initializes the base DGP.

        Parameters
        ----------
        aleatoric_uncertainty : Optional[AleatoricUncertainty]
            The aleatoric uncertainty of the DGP. Defaults to no uncertainty.
        """
        super().__init__()

        if aleatoric_uncertainty is None:
            aleatoric_uncertainty = AleatoricUncertainty()

        self._aleatoric_uncertainty: AleatoricUncertainty = aleatoric_uncertainty

    @abstractmethod
    def deterministic_function(self, x: ndarray) -> ndarray:
        """
        Gets the deterministic component of each label for given examples' features according to the DGP's
        underlying deterministic function.

        Parameters
        ----------
        x : np.ndarray
            The examples' features. Has shape (num_examples, ...).

        Returns
        -------
        y : np.ndarray
            The examples' labels. Has shape (num_examples, ...).
        """
        raise NotImplementedError

    @property
    def aleatoric_uncertainty(self) -> AleatoricUncertainty:
        """
        The aleatoric uncertainty of the DGP.

        Returns
        -------
        aleatoric_uncertainty : AleatoricUncertainty
             The aleatoric uncertainty of the DGP.
        """
        return self._aleatoric_uncertainty

    def _aleatoric_uncertainty_terms(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Gets uncertainty terms representing aleatoric uncertainty of given examples' features and labels.

        Parameters
        ----------
        x : np.ndarray
            The examples' features before considering feature uncertainty. Has shape (num_examples, ...).

        Returns
        -------
        uncertainty : Tuple[ndarray, ndarray]
            A tuple containing the examples' features uncertainty terms and the examples' labels uncertainty terms.
        """
        # Features uncertainty (for errors-in-variables cases, measurement errors)
        if self._aleatoric_uncertainty.feature_uncertainty is None:
            x_unc = zeros(len(x))   # len(x) is the number of examples
        else:
            x_unc = self._aleatoric_uncertainty.feature_uncertainty.sample_noise(x)

        # Label uncertainty
        if self._aleatoric_uncertainty.label_uncertainty is None:
            y_unc = zeros(len(x))   # len(x) is the number of examples
        else:
            y_unc = self._aleatoric_uncertainty.label_uncertainty.sample_noise(x)

        return x_unc, y_unc

    def sample_data(self, x: ndarray) -> List[SyntheticData]:
        """
        Samples a dataset from the DGP.

        Takes as input the features (or approximate features if uncertainty is applied to features) of the examples to
        sample rather than simply the amount of examples to sample so that domain is inherently defined. Moreover, this
        reflects reality better as in practice a dataset's domain can often be chosen (e.g. choosing to include more
        examples of some class in a training set or test set). To sample truly random examples, one can simply input a
        random ndarray of examples' features.

        Notes
        -----
        Note that uncertainty terms are added to the features only after the deterministic parts of the labels have been
        calculated. This is more truthful to an error-in-variable/measurement-error situation; the label is tied to the
        true but unobserved features. See https://en.wikipedia.org/wiki/Errors-in-variables_model.

        Parameters
        ----------
        x : ndarray
            The examples' features. Has shape (num_examples, ...).

        Returns
        -------
        synthetic_data : List[SyntheticData]
            A list of SyntheticData named tuples. Has length num_examples.
        """
        y = self.deterministic_function(x)

        # Error terms are added to x after deterministic y=f(x) has been calculated. See error-in-variables models.
        x_unc, y_unc = self._aleatoric_uncertainty_terms(x)

        synthetic_data = []
        for i in range(len(x)):     # len(x) is the number of examples.
            synthetic_data.append(
                SyntheticData(
                    x=atleast_1d(x[i] + x_unc[i]),
                    y=atleast_1d(y[i] + y_unc[i])
                )
            )

        return synthetic_data
