"""
    @file:              train_and_test_model.py
    @Author:            Raphael Brodeur

    @Creation Date:     06/2025
    @Last modification: 06/2025

    @Description:       This file contains an example experiment of training and testing a model as well as Monte-Carlo
                        Dropout.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
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

    # Define data generation process
    data_generator = SimpleWigglyRegression(
        aleatoric_uncertainty=AleatoricUncertainty(         # Aleatoric uncertainty can be set here if needed
            y_uncertainty=NormalUncertainty(
                mean=0.0,
                standard_deviation=0.075
            )
        ),
        a=0.4,                                              # Parameters for the wiggly function.
        b=-2.0,
        c=1.0,
        d=3.0,
        i=1,
        j=1,
        k=2
    )

    # Generate training data
    training_data = data_generator.sample_data(
        x=np.concatenate([                     # Define the learning domain, can be randomized if needed
            np.linspace(-0.8, 0.8, 250),
            np.linspace(1.2, 1.4, 65)
        ])
    )

    # Create training dataset
    training_ds = SyntheticDataset(data=training_data)

    # Set an MLP model with 4 hidden layers
    model = MLP(
        input_features=1,
        hidden_channels_width=[1024, 1024, 1024, 1024],
        output_features=1,
        activation="relu",
        dropout=0.2
    )

    # Other model configurations can be set here, for example:
    # model = MLP(
    #     input_features=1,
    #     hidden_channels_width=[1024, 1024, 1024, 1024],
    #     output_features=1,
    #     activation="leakyrelu",
    #     dropout=("spatialdropout", {"p": 0.2}),
    #     normalization=("batch", {"num_features": 1}),
    #     ordering="NAD"
    # )

    # No use in sending to model to device yet, as it is not built !

    # Define loss function
    def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean squared error of a batch.
        """
        return ((y_true - y_pred) ** 2).mean()

    mse_loss = mse

    # Initially, no training algorithm are defined
    print("Initially, no training algorithm are defined :", model.learning_algorithms)

    # Define a learning procedure for the model and register it to the Model class under the name given in the decorator
    @model.register_learning_algorithm("basic_mse_adam_training")
    def your_custom_training_procedure(
            dataset: SyntheticDataset,
            learning_rate: float,
            num_epochs: int,
            training_device: torch.device,
    ):
        """
        Defines a basic MSE loss training with Adam.
        """
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=True,
            pin_memory=True
        )

        opt = Adam(
            params=model.parameters(),
            lr=learning_rate
        )

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                x = batch.x.to(training_device)
                y = batch.y.to(training_device)

                # Reset grad
                opt.zero_grad()

                # Forward pass
                y_pred = model(x)

                # Compute loss
                loss = mse_loss(y_true=y, y_pred=y_pred)

                # Backward pass and optimization
                loss.backward()
                opt.step()

        print("Training complete.")

    # Now a learning algorithm is defined and registered to the model, it can be called when the model is built
    print("Now a learning algorithm is defined and registered to the model :", model.learning_algorithms)

    # Build the model. This actually initializes the model and its weights. See note in the Model base class for more...
    # NOTE !! Model has to be built before being sent to a device !
    model.build()

    # Now that the model is built, it can be sent to the device
    model.to(device)

    # Train the model
    model.learn["basic_mse_adam_training"](
        dataset=training_ds,
        learning_rate=5e-5,
        num_epochs=500,
        training_device=device
    )

    # Test the trained model
    test_domain = np.linspace(-1.2, 2.25, 1000)
    model.eval()
    with torch.no_grad():
        y_pred = model(
            numpy_input_to_torch_input(test_domain).to(device)
        )
        y_pred = y_pred.squeeze(1).cpu().numpy()    # From torch.Size([1000, 1, 1]) to (1000, 1)

    # Show the results
    fig, ax = plt.subplots()
    for data_point in training_data:
        ax.scatter(data_point.x, data_point.y, color="black", zorder=2, s=10, marker="o")
    ax.plot(test_domain, y_pred, color='#4F609C', zorder=3)  # Plot trained model
    ax.set_title("Trained Model Predictions And Training Data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    # Monte-Carlo Dropout inference on test domain
    num_mcd_samples = 1000
    mcd_samples = []
    for _ in range(num_mcd_samples):
        model.eval()                    # Puts all layers to train() mode False
        enable_dropout(model)           # Puts all Dropout layers to train() mode True
        with torch.no_grad():
            y_pred = model(
                numpy_input_to_torch_input(test_domain).to(device)
            )
            mcd_samples.append(y_pred.squeeze().cpu().numpy())

    # Get statistics across Monte-Carlo Dropout samples
    y_mean = np.mean(mcd_samples, axis=0)
    y_std = np.std(mcd_samples, axis=0)

    # Show the results
    fig, ax = plt.subplots()
    for data_point in training_data:
        ax.scatter(data_point.x, data_point.y, color="black", zorder=2, s=10, marker="o")
    ax.plot(test_domain, y_mean, color='#4F609C', zorder=3)     # Plot mean y curve across all mcd samples
    ax.fill_between(                                            # Fill +-2 y standard deviation across all mcd samples
        test_domain,
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        color="#C0DEF0",
        zorder=0
    )
    ax.set_title("Trained Model Predictions And Training Data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
