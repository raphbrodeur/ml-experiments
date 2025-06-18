"""
    @file:              utils.py
    @Author:            Raphael Brodeur

    @Creation Date:     03/2025
    @Last modification: 03/2025

    @Description:       This file contains utility functions.
"""

import random

from numpy.random import seed as numpy_seed
from torch import manual_seed


def set_determinism(seed: int):
    """
    This function sets a random seed for Python's random, NumPy and Torch.

    Parameters
    ----------
    seed : int
        The seed to set.
    """
    random.seed(seed)   # Random
    numpy_seed(seed)    # NumPy
    manual_seed(seed)   # Torch
