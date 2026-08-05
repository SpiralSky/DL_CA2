import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure


def plot_gradient_flow(
    model: nn.Module,
    fig: Figure | SubFigure | None = None,
) -> Figure | SubFigure:
    layer_names = []
    mean_gradients = []
    max_gradients = []

    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        layer_names.append(parameter_name)
        mean_gradients.append(parameter.grad.abs().mean().item())
        max_gradients.append(parameter.grad.abs().max().item())

    if fig is None:
        fig = plt.figure(figsize=(12, 6))

    ax = fig.subplots()

    x_positions = range(len(layer_names))

    ax.bar(x_positions, mean_gradients, color="steelblue", label="Mean")
    ax.bar(x_positions, max_gradients, color="coral", alpha=0.6, label="Max")

    for index, mean_gradient in enumerate(mean_gradients):
        if mean_gradient == 0:
            ax.text(
                index,
                ax.get_ylim()[0] * 2,
                "∅",
                ha="center",
                fontsize=7,
                color="red",
            )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(layer_names, rotation=90, fontsize=6)
    ax.set_ylabel("Gradient Magnitude (log scale)")
    ax.set_title("Gradient Flow Per Layer")
    ax.set_yscale("log")
    ax.legend()

    return fig