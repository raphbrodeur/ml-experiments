"""
    @file:              visualization.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 02/2025

    @Description:       This file contains utility functions for visualization.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.generation import DataGenerationProcess, NormalUncertainty, SyntheticData


def plot_dgp(
        dgp: DataGenerationProcess,
        sampled_data: Optional[List[SyntheticData]] = None,
        domain: np.ndarray = np.linspace(-10, 10, 1000)
):
    """
    Plots the data generation process (DGP) in a general way. Currently, only implemented for R^1 -> R^1 DGP's with
    optional aleatoric normal uncertainty centered at 0. Treats such aleatoric uncertainty by filling the ±2 standard
    deviations' region.

    Parameters
    ----------
    dgp : DataGenerationProcess
        The data generation process to plot.
    sampled_data : Optional[List[SyntheticData]]
        The generated data to plot. If None, only the DGP function is plotted. Defaults to None.
    domain : ndarray
        The domain over which to plot the DGP. Defaults to np.linspace(-10, 10, 1000).
    """
    if dgp.aleatoric_uncertainty.feature_uncertainty is not None:
        raise Exception("Visualization not implemented for feature uncertainty (error-in-variables).")

    fig, ax = plt.subplots()

    # Plot the DGP's deterministic (or mean function)
    image = dgp.deterministic_function(domain)
    ax.plot(domain, image, label="DGP", zorder=2)

    # Plot the aleatoric uncertainty
    if isinstance(dgp.aleatoric_uncertainty.target_uncertainty, NormalUncertainty):
        ax.fill_between(
            domain,
            image - 2 * dgp.aleatoric_uncertainty.target_uncertainty.std,
            image + 2 * dgp.aleatoric_uncertainty.target_uncertainty.std,
            color="gray",
            zorder=0
        )

    # Plot the sampled data
    if sampled_data is not None:
        for data in sampled_data:
            ax.scatter(data.x, data.y, color="black", zorder=1)

    # Axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Set x-axis limits
    ax.set_xlim(np.min(domain), np.max(domain))

    plt.show()


def plot_trained_model(
        model,
        training_data,
        domain: np.ndarray = np.linspace(-3, 3, 1000)
):
    """
    Description.
    """
    fig, ax = plt.subplots()  # Create plot
    domain_torch = torch.tensor([[domain[i]] for i in range(len(domain))], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(domain_torch)
        y_pred = y_pred.cpu().numpy()

    ax.plot(domain, y_pred, label='trained model', color='red', zorder=3)  # Plot trained model

    if training_data is not None:
        for data in training_data:
            ax.scatter(data.x, data.y, color="black", zorder=1)

    plt.show()
