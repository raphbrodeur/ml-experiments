"""
    @file:              activation_variance.py
    @Author:            Raphael Brodeur

    @Creation Date:     09/2025
    @Last modification: 09/2025

    @Description:       This file contains experiments regarding how the variance of the initialized parameters impacts
                        the variance of the output activations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.datasets import SyntheticDataset
from src.data.generation import (
    AleatoricUncertainty,
    NormalUncertainty,
    SimpleWigglyRegression
)
from src.models import enable_dropout, MLP
from src.utils import numpy_input_to_torch_input, set_determinism


if __name__ == "__main__":
    # Set random seed for reproducibility
    set_determinism(1010710)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #############################################
    # VARIANCE FOR 1 LINEAR LAYER -- Var[b] = 0 #
    #############################################

    # For 1 linear layer with Var[b] = 0 :
    # Var[y] = (0.5 * in_features * Var[W_ij]) * Var[x] ?
    print(f"Theorical var : {0.5 * 16 * 1 * 1}")

    y_list = []
    for _ in range(10000):
        # input
        x = torch.randn(16).to(device)

        # model
        lin = nn.Linear(16, 16)
        nn.init.normal_(lin.weight, mean=0., std=1)
        nn.init.constant_(lin.bias, 0.)

        act = nn.ReLU()

        # model = nn.Sequential(lin, act).to(device)
        model = nn.Sequential(act, lin).to(device)

        # forward pass
        y = model(x)
        y_list.append(y.cpu().detach().numpy())

    print("y_list", np.var(y_list, axis=0))
























