import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.models.autoencoders.VAE import VAE


def analyze_latent_space(
    model: VAE,
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