from matplotlib.figure import Figure, SubFigure
import matplotlib.pyplot as plt
from typing import Sequence, Any

from typing import Sequence, Any

from matplotlib.figure import Figure, SubFigure


def plot_gradient_history(
    history: Sequence[dict[str, Any]],
    fig: Figure | SubFigure | None = None,
) -> Figure | SubFigure:
    epochs = []
    grad_norms = []
    grad_means = []

    for entry in history:
        if "grad_norm" not in entry:
            continue

        epochs.append(entry["epoch"])
        grad_norms.append(entry["grad_norm"])

        if "grad_mean" in entry:
            grad_means.append(entry["grad_mean"])

    if fig is None:
        fig = plt.figure(figsize=(10, 4))

    ax = fig.subplots()

    if grad_norms:
        ax.plot(
            epochs,
            grad_norms,
            label="grad_norm",
        )

    if grad_means:
        ax.plot(
            epochs,
            grad_means,
            label="grad_mean",
        )

    ax.set_title("Gradient Statistics Over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Magnitude")
    ax.set_yscale("log")

    if grad_norms or grad_means:
        ax.legend()

    return fig