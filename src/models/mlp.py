"""
    @file:              mlp.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2025
    @Last modification: 03/2025

    @Description:       This file contains a simple MLP model.
"""

from typing import Optional, Sequence

from torch import Tensor, device
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.datasets import SyntheticDataset
from src.models.base import check_if_built, MLPBlock, Model


class MLP(Model):
    """
    This class is an MLP model.
    """

    def __init__(
            self,
            hidden_channels_width: Sequence[int],
            activation: str,
            dropout: Optional[float],
            normalization: Optional[str]
    ):
        """
        Initializes the MLP model.
        """
        super().__init__()

        self._hidden_channels_width: Sequence[int] = hidden_channels_width
        self._activation: str = activation
        self._dropout: float = dropout
        self._normalization = normalization

        self._mlp: Optional[Module] = None

    def build(self):
        """
        Builds the model and initializes weights.
        """
        super().build()

        self._mlp = MLPBlock(
            hidden_channels_width=self._hidden_channels_width,
            activation=self._activation,
            dropout=self._dropout,
            normalization=self._normalization
        )

    @check_if_built
    def forward(self, x: Tensor) -> Tensor:
        """
        Gets an output tensor for an input tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        y : Tensor
            The output tensor.
        """
        return self._mlp(x)

    @check_if_built
    def train_mse(
            self,
            ds: SyntheticDataset,
            lr: float,
            num_epoch: int,
            dev: device
    ):
        """
        Temporary method! An actual trainer will be implemented.
        """
        loader = DataLoader(
            dataset=ds,
            batch_size=16,
            shuffle=True,
            pin_memory=True
        )

        opt = Adam(
            params=self.parameters(),
            lr=lr
        )

        for epoch in range(num_epoch):
            # Training
            self.train()

            for batch in loader:
                x = batch.x.to(dev)
                y = batch.y.to(dev)

                opt.zero_grad()

                # Forward pass
                y_pred = self.forward(x)

                # Calculate cost
                mse = (y - y_pred) ** 2
                cost = mse.mean()
                print(f"At epoch {epoch}, batch cost MSE = {cost}")

                # Backward pass
                cost.backward()
                opt.step()

            print("Training done.")
