import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import SyntheticDataset
from src.data.generation import AleatoricUncertainty, NormalUncertainty, SimpleWigglyRegression
from src.models import enable_dropout, MLP
from src.utils import set_determinism
from src.utils import numpy_input_to_torch_input



if __name__ == "__main__":
    # Set random seed for reproducibility
    set_determinism(1010710)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate same training data
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
    training_data = data_generator.sample_data(
        x=np.concatenate([                     # Define the learning domain, can be randomized if needed
            np.linspace(-0.8, 0.8, 250),
            np.linspace(1.2, 1.4, 65)
        ])
    )

    # Set model
    model = MLP(
        input_features=1,
        hidden_channels_width=[1024, 1024, 1024, 1024],
        output_features=1,
        activation="relu",
        dropout=0.2
    )

    # Build model and send to device
    model.build()
    model.to(device)

    # Visualize training data
    fig, ax = plt.subplots()
    for data_point in training_data:
        ax.scatter(data_point.x, data_point.y, color="black", zorder=2, s=10, marker="o")

    # Ensemble predictions
    test_domain = np.linspace(-1.2, 2.25, 1000)
    ensemble_predictions = []
    for i in range(1000):
        # Load parameters
        model.load_state_dict(torch.load(f"./ensemble_params/trained_params_{i}.pt"))

        model.eval()
        with torch.no_grad():
            y_pred = model(
                numpy_input_to_torch_input(test_domain).to(device)
            )

            y_pred = y_pred.squeeze().cpu().numpy()

            ensemble_predictions.append(y_pred)

            # Plot individual models of ensemble as a distribution
            ax.plot(test_domain, y_pred, zorder=3, color="black", alpha=0.01)


    # Plot a confidence region
    # y_mean = np.mean(ensemble_predictions, axis=0)
    # y_std = np.std(ensemble_predictions, axis=0)
    # ax.plot(test_domain, y_mean, color='#4F609C', zorder=3)     # Plot mean y curve across all mcd samples
    # ax.fill_between(                                            # Fill +-2 y standard deviation across all mcd samples
    #     test_domain,
    #     y_mean - 2 * y_std,
    #     y_mean + 2 * y_std,
    #     color="#C0DEF0",
    #     zorder=0
    # )

    ax.set_title("DEEP ENSEMBLE")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.2, 2.25)
    ax.set_ylim(-2.0, 2.0)
    plt.show()











