"""
    @file:              visualization.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains utility functions for visualization.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data.generation import DataGenerationProcess, SyntheticData


def plot_dgp(
        dgp: DataGenerationProcess,
        domain: Optional[np.ndarray] = None,
        data: Optional[List[SyntheticData]] = None,
        num_samples: int = 3000
):
    """
    This function is a generic way to plot any R^1 -> R^1 data generation process (DGP) and its sampled data. Plots the
    mean image over multiple datasets sampled from a given domain and colors the ± 2 standard deviations' interval. Can
    plot data.

    Parameters
    ----------
    dgp : DataGenerationProcess
        The data generation process to plot.
    domain : ndarray
        The domain over which to plot the DGP. Defaults to numpy.linspace(-10, 10, 1000).
    data : Optional[List[SyntheticData]]
        The data to plot. Optional. Defaults to None.
    num_samples : int
        The number of datasets to sample from the DGP. Defaults to 3000.

    Raises
    ------
    Exception
        If feature uncertainty is not None
    """
    if dgp.aleatoric_uncertainty.feature_uncertainty is not None:
        raise Exception("Not implemented for feature uncertainty (error-in-variables).")

    fig, ax = plt.subplots()

    # Sample data from the DGP
    sampled_images = []
    for _ in range(num_samples):
        sampled_data = dgp.sample_data(domain)
        sampled_images.append(
            np.array([sampled_data[i].y[0] for i in range(len(domain))])
        )

    image_mean = np.mean(sampled_images, axis=0)    # The mean image of the sampled images
    image_std = np.std(sampled_images, axis=0)      # The standard deviation of the sampled images

    # Plot the DGP
    ax.plot(domain, image_mean, zorder=1)

    # Plot the target aleatoric uncertainty
    if dgp.aleatoric_uncertainty.target_uncertainty is not None:
        ax.fill_between(
            domain,
            image_mean - 2 * image_std,
            image_mean + 2 * image_std,
            color="gray",
            zorder=0
        )

    # Overlay data
    if data is not None:
        for data_point in data:
            ax.scatter(data_point.x, data_point.y, color="black", zorder=2)

    # Axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Set x-axis limits
    ax.set_xlim(np.min(domain), np.max(domain))

    ax.set_xlim(-1.2, 2)
    ax.set_ylim(-2, 2)

    plt.show()


def numpy_input_to_torch_input(numpy_batch: np.ndarray) -> torch.Tensor:
    """
    Converts a numpy batch with shape (N, ...) to a torch batch with shape (N, C, ...).

    Parameters
    ----------
    numpy_batch : np.ndarray
        The numpy input to convert. Has shape (N, ...)

    Returns
    -------
    torch_batch : torch.Tensor
        The torch input. Has shape (N, C, ...)
    """
    torch_batch = torch.tensor(
        [[numpy_batch[i]] for i in range(len(numpy_batch))],
        dtype=torch.float32
    ).unsqueeze(1)

    return torch_batch


def plot_trained_model(
        model: nn.Module,
        domain: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        training_data: Optional[List[SyntheticData]] = None
):
    """
    This function is a generic way to any R^1 -> R^1 trained model and its training data on a given domain.

    Parameters
    ----------
    model : nn.Module
        The trained Torch model.
    domain : Optional[np.ndarray]
        The domain over which to plot the trained model. Defaults to numpy.linspace(-10, 10, 1000).
    device : Optional[torch.device]
        The device on which to evaluate the model. Defaults to GPU with index 0 if cuda is available, otherwise defaults
        to CPU.
    training_data : Optional[List[SyntheticData]]
        The data to plot. Optional. Defaults to None.
    """
    if domain is None:
        domain = np.linspace(-10, 10, 1000)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)    # Set module device

    fig, ax = plt.subplots()

    # Create torch tensor on appropriate device for domain
    domain_torch = numpy_input_to_torch_input(domain).to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(domain_torch)    # Model inference on domain

        # Remove channel dim, convert to ndarray and move to CPU for plotting
        y_pred = y_pred.squeeze(1).cpu().numpy()

    ax.plot(domain, y_pred, label='trained model', color='red', zorder=3)  # Plot trained model

    if training_data is not None:
        for data in training_data:
            ax.scatter(data.x, data.y, color="black", zorder=1)

    plt.show()
