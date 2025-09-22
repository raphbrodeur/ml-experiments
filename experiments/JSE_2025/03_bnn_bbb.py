import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchbayesian.bnn as bnn

from src.data.datasets import SyntheticDataset
from src.data.generation import (
    AleatoricUncertainty,
    NormalUncertainty,
    SimpleWigglyRegression
)
from src.models import MLP
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

    # Build model
    model.build()

    # Transform into BNN
    model = bnn.BayesianModule(model)

    # Send model to device
    model.to(device)

    # Define loss function
    def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean squared error of a batch.
        """
        return ((y_true - y_pred) ** 2).mean()

    mse_loss = mse

    ######################
    # TRAINING HYPERPARAMETERS
    dataset = training_ds
    learning_rate = 1e-3    # 1e-3
    num_epochs = 3500  # 3500
    training_device = device

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
            loss = mse_loss(y_true=y, y_pred=y_pred) + 1e-4 * model.kl_divergence(reduction="mean")
            print(loss)

            # Backward pass and optimization
            loss.backward()
            opt.step()

    print("Training complete.")
    #################


    # Visualize the trained model
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
    ax.set_title("SINGLE MODEL")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.2, 2.25)
    ax.set_ylim(-2.0, 2.0)
    plt.show()


    # Visualize BNN inference and UQ
    fig, ax = plt.subplots()
    for data_point in training_data:
        ax.scatter(data_point.x, data_point.y, color="black", zorder=2, s=10, marker="o")

    num_bnn_samples = 1000
    bnn_samples = []
    for _ in range(num_bnn_samples):
        model.eval()                    # Puts all layers to train() mode False
        with torch.no_grad():
            y_pred = model(
                numpy_input_to_torch_input(test_domain).to(device)
            )

            y_pred = y_pred.squeeze().cpu().numpy()

            bnn_samples.append(y_pred)

            # Plot individual models of ensemble as a distribution
            # ax.plot(test_domain, y_pred, zorder=3, color="black", alpha=0.01)

    # Get statistics across BNN samples
    y_mean = np.mean(bnn_samples, axis=0)
    y_std = np.std(bnn_samples, axis=0)
    ax.plot(test_domain, y_mean, color='#4F609C', zorder=3)     # Plot mean y curve across all mcd samples
    ax.fill_between(                                            # Fill +-2 y standard deviation across all mcd samples
        test_domain,
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        color="#C0DEF0",
        zorder=0
    )

    ax.set_title("BBB")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.2, 2.25)
    ax.set_ylim(-2.0, 2.0)
    plt.show()
