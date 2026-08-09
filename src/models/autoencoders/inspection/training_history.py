from typing import Any, TypedDict

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.models.util.TrainingHistory import TrainingHistory


class MetricPlotSpec(TypedDict, total=False):
    """
    Config for one subplot in plot_metrics().

    title: plot title (also used as the y-axis label).
    metric: base metric name, e.g. "reconstruction" or "kl_divergence".
        For a normal metric this is looked up as "train_{metric}" /
        "val_{metric}" in the "train" / "val" splits. For an extra metric
        (extra_metric=True) it's looked up as-is in the "extra" split.
    extra_metric: if True, plot `metric` directly from the "extra" split
        instead of deriving train/val series from it.
    scale: y-axis scale to apply, e.g. "log", or ("symlog", {"linthresh": 10})
        to pass extra kwargs to ax.set_yscale(). None (default) leaves the
        default linear scale.
    """

    title: str
    metric: str
    extra_metric: bool
    scale: str | tuple[str, dict[str, Any]] | None


def _plot_one(history: TrainingHistory, spec: MetricPlotSpec, ax: Axes) -> None:
    title = spec["title"]
    metric = spec["metric"]
    is_extra = spec.get("extra_metric", False)
    scale = spec.get("scale")

    plotted = False

    if is_extra:
        values = history.values(metric, "extra")
        if values:
            ax.plot(history.epochs(metric, "extra"), values, label=title)
            plotted = True
    else:
        train_values = history.values(metric, "train")
        val_values = history.values(metric, "val")

        if train_values:
            ax.plot(
                history.epochs(metric, "train"),
                train_values,
                label="Train",
            )
            plotted = True

        if val_values:
            ax.plot(
                history.epochs(metric, "val"),
                val_values,
                label="Validation",
            )
            plotted = True

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)

    if scale is not None:
        if isinstance(scale, tuple):
            scale_name, scale_kwargs = scale
            ax.set_yscale(scale_name, **scale_kwargs)
        else:
            ax.set_yscale(scale)

    ax.grid(alpha=0.3)
    if plotted:
        ax.legend()


def plot_metrics(
    history: TrainingHistory,
    specs: list[MetricPlotSpec],
) -> tuple[Figure, list[Axes]]:
    """
    Plot a composite figure with one stacked row per spec in `specs`.
    Layout is deliberately simple: fixed width, fixed height per row.
    """
    fig, axes_grid = plt.subplots(
        nrows=len(specs),
        ncols=1,
        figsize=(8, 3 * len(specs)),
        squeeze=False,
    )
    axes = [row[0] for row in axes_grid]

    for ax, spec in zip(axes, specs):
        _plot_one(history, spec, ax)

    fig.tight_layout()
    return fig, axes