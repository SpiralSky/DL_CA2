import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.models.util.ModelHistory import ModelHistory


def plot_reconstruction_history(
    history: ModelHistory,
    ax: Axes | None = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    epochs = history.values("epoch")

    ax.plot(
        epochs,
        history.values("train_reconstruction"),
        label="Train",
    )

    ax.plot(
        epochs,
        history.values("val_reconstruction"),
        label="Validation",
    )

    ax.set_title("Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.grid(alpha=0.3)
    ax.legend()


def plot_kl_history(
    history: ModelHistory,
    ax: Axes | None = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    epochs = history.values("epoch")

    ax.plot(
        epochs,
        history.values("train_kl_divergence"),
        label="Train",
    )

    ax.plot(
        epochs,
        history.values("val_kl_divergence"),
        label="Validation",
    )

    ax.set_title("KL Divergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL")

    ax.set_yscale(
        "symlog",
        linthresh=10,
    )

    ax.grid(alpha=0.3)
    ax.legend()


def plot_kl_weight(
    history: ModelHistory,
    ax: Axes | None = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2))

    weights = history.values("kl_weight")

    if weights:
        ax.plot(
            history.values("epoch"),
            weights,
            label="KL Weight",
        )

        ax.legend()

    ax.set_title("KL Warmup")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")

    ax.grid(alpha=0.3)