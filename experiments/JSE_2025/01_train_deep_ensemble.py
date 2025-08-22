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
from src.models import MLP
from src.utils import set_determinism


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

    # Define loss function
    def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean squared error of a batch.
        """
        return ((y_true - y_pred) ** 2).mean()

    mse_loss = mse

    # Define a learning procedure for the model and register it to the Model class under the name given in the decorator
    @model.register_learning_algorithm("basic_mse_adam_training")
    def your_custom_training_procedure(
            dataset: SyntheticDataset,
            learning_rate: float,
            num_epochs: int,
            training_device: torch.device,
            param_name: int
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

        # torch.save(model.state_dict(), f"./ensemble_params/trained_params_{param_name}.pt")

        print("Training complete.")

    for i in range(1000):
        # Build model
        model.build()
        model.to(device)

        # Training i-th model of deep ensemble
        print(f"Training model {i}")
        model.learn["basic_mse_adam_training"](
            dataset=training_ds,
            learning_rate=5e-5,
            num_epochs=500,
            training_device=device,
            param_name=i
        )
