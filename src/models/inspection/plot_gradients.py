from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure

from src.models.util.ModelHistory import ModelHistory


def plot_gradient_summary(
    history: ModelHistory,
    fig: Figure | SubFigure | None = None,
) -> Figure | SubFigure:

    module_gradients = defaultdict(list)

    for entry in history:
        gradients = entry.get("gradients")
        if gradients is None:
            continue

        epoch_modules = defaultdict(list)

        for name, stats in gradients.items():
            if name.endswith(".bias"):
                continue

            module = name.removesuffix(".weight")

            # Aggregate all layers inside a Sequential block
            parts = module.split(".")
            if parts[-1].isdigit():
                module = ".".join(parts[:-1])

            epoch_modules[module].append(stats["norm"])

        for module, values in epoch_modules.items():
            epoch_value = np.mean(values)
            module_gradients[module].append(epoch_value)

    if fig is None:
        fig = plt.figure(figsize=(12, 6))

    ax = fig.subplots()

    if not module_gradients:
        ax.set_title("Gradient Summary")
        ax.text(
            0.5,
            0.5,
            "No gradient history available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    modules = list(module_gradients.keys())

    medians = []
    lower_errors = []
    upper_errors = []

    for module in modules:
        values = np.asarray(module_gradients[module])

        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)

        medians.append(median)
        lower_errors.append(median - q1)
        upper_errors.append(q3 - median)

    x = np.arange(len(modules))

    bars = ax.bar(
        x,
        medians,
        yerr=[lower_errors, upper_errors],
        capsize=4,
    )

    for bar, median, upper in zip(bars, medians, upper_errors):
        ax.annotate(
            f"{median:.2e}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                median + upper,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        modules,
        rotation=45,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel("Median Gradient Norm")
    ax.set_title("Gradient Magnitude per Module")
    ax.set_yscale("log")

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    return fig