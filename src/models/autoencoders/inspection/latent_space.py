from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from models.autoencoders.AbstractVAE import AbstractVAE


def analyze_latent_space(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    n_samples: int | None = 5000,
    perplexity: float = 30.0,
    random_state: int = 42,
    class_names: list[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Uses TSNE to display n_samples of points in latent space on a 2d surface.
    :param model: Model to generate samples from.
    :param data_loader: DataLoader where samples will be picked from.
    :param n_samples: Number of samples to be picked. Defaults to 5000. If None is specified, uses all samples from DataLoader.
    :param perplexity: TSNE Perplexity.
    :param random_state: Random State to be used in TSNE.
    :param class_names: List of class names to act as legend in plot.
    :param ax: axes to plot on.
    :return: Plot axes.
    """

    device = next(model.parameters()).device
    model.eval()

    latent_means = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device)

            mu, _ = model.encoder(images)

            latent_means.append(mu.cpu())
            labels.append(batch_labels.cpu())

    latent_means = torch.cat(latent_means).numpy()
    labels = torch.cat(labels).numpy()

    if n_samples is not None and len(latent_means) > n_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(
            len(latent_means),
            size=n_samples,
            replace=False,
        )

        latent_means = latent_means[indices]
        labels = labels[indices]

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
    ).fit_transform(latent_means)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.5,
    )

    if class_names is None:
        fig.colorbar(scatter, ax=ax, label="Class")
    else:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=plt.cm.tab10(i / max(len(class_names) - 1, 1)), # type: ignore[attr-defined]
                color="w",
                markersize=8,
            )
            for i in range(len(class_names))
        ]

        ax.legend(
            handles,
            class_names,
            title="Class",
            fontsize=8,
        )

    ax.set_title(f"t-SNE (n={len(latent_means)}, perplexity={perplexity})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    return ax

def plot_latent_utilization(
    model: AbstractVAE,
    dataloader: DataLoader,
    *,
    ax: Axes | None = None,
) -> Axes:
    """
    Plots latent space utilization of each dimension in a line graph.
    :param model: Model for latent space analysis.
    :param dataloader: DataLoader to load images from.
    :param ax: Axes to plot on.
    :return: Returns the provided axes.
    """

    model.eval()

    device = next(model.parameters()).device

    kl_sum = None
    num_samples = 0

    with torch.no_grad():
        for images, *_ in dataloader:
            images = images.to(device)

            mu, logvar = model.encoder(images)

            kl = model.kl_divergence(mu, logvar)

            kl = kl.sum(dim=0)

            kl_sum = kl if kl_sum is None else kl_sum + kl
            num_samples += images.size(0)

    avg_kl = (kl_sum / num_samples).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    dimensions = np.arange(avg_kl.size)

    ax.plot(
        dimensions,
        avg_kl,
        linewidth=2,
    )

    ax.fill_between(
        dimensions,
        avg_kl,
        alpha=0.2,
    )

    ax.set_title("KL Contribution per Latent Dimension")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Average KL")

    ax.grid(alpha=0.3)

    return ax