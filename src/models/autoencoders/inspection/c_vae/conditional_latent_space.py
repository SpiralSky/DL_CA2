import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.models.autoencoders.AbstractVAE import AbstractVAE


def analyze_conditional_latent_space(
    model: AbstractVAE,
    data_loader: DataLoader,
    *,
    n_samples: int | None = 5000,
    perplexity: float = 30.0,
    random_state: int = 42,
    class_names: list[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Uses TSNE to display conditional VAE latent space.
    """

    device = next(model.parameters()).device
    model.eval()

    latent_means = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)

            embeddings = model.label_embeddings(
                batch_labels.long()
            )

            mu, _ = model.encoder(
                images,
                embeddings,
            )

            latent_means.append(mu.cpu())
            labels.append(batch_labels.cpu())

    latent_means = torch.cat(latent_means).numpy()
    labels = torch.cat(labels).numpy()

    if n_samples is not None and len(latent_means) > n_samples:
        rng = np.random.default_rng(random_state)

        indices = rng.choice(
            len(latent_means),
            n_samples,
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

    ax.set_title(
        f"Conditional t-SNE "
        f"(n={len(latent_means)}, perplexity={perplexity})"
    )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    return ax