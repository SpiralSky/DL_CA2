from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


def analyze_latent_space(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    n_samples: int | None = 5000,
    perplexity: float = 30.0,
    random_state: int = 42,
    save_file: Path | None = None,
    class_names: list[str] | None = None,
    ax: Axes | None = None,
) -> Figure:

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
                markerfacecolor=plt.cm.tab10(i / max(len(class_names) - 1, 1)),
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

    if save_file is not None:
        fig.savefig(
            save_file,
            dpi=300,
            bbox_inches="tight",
        )

    return fig

def plot_latent_utilization(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    ax: Axes | None = None,
) -> Figure:
    """
    Plot average KL contribution per latent dimension.

    The device is inferred automatically from the model parameters.
    """

    model.eval()

    device = next(model.parameters()).device

    kl_sum = None
    num_samples = 0

    with torch.no_grad():
        for images, *_ in dataloader:
            images = images.to(device)

            mu, logvar = model.encoder(images)

            kl = -0.5 * (
                1
                + logvar
                - mu.pow(2)
                - logvar.exp()
            )

            kl = kl.sum(dim=0)

            kl_sum = kl if kl_sum is None else kl_sum + kl
            num_samples += images.size(0)

    avg_kl = (kl_sum / num_samples).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

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

    return fig