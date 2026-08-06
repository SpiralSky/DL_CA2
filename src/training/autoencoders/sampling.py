import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
import torch
from torch.utils.data import DataLoader

from src.models.autoencoders.AbstractVAE import AbstractVAE


def prepare_image(image: torch.Tensor) -> np.ndarray:
    """Convert a tensor image into a matplotlib-compatible numpy array."""

    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)

        if image.shape[-1] == 1:
            image = image.squeeze(-1)

    return image.detach().cpu().numpy()


def plot_class_samples(
    model: AbstractVAE,
    dataloader: DataLoader,
    *,
    n_images: int = 2,
    cmap: str = "gray",
    class_names: list[str] | None = None,
    axes=None,
) -> Figure:
    """
    Plot reconstructed samples grouped by class.

    Samples are collected from the dataloader and grouped using labels.
    """

    model.eval()

    device = next(model.parameters()).device

    samples: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for images, labels, *_ in dataloader:
            images = images.to(device)

            reconstructed, _, _ = model(images)

            reconstructed = reconstructed.cpu()
            labels = labels.cpu()

            for image, label in zip(reconstructed, labels):
                class_idx = int(label)

                if class_idx not in samples:
                    samples[class_idx] = []

                if len(samples[class_idx]) < n_images:
                    samples[class_idx].append(image)

    class_indices = sorted(samples)
    n_classes = len(class_indices)

    if axes is None:
        fig, axes = plt.subplots(n_images, n_classes, figsize=(n_classes * 1.8, n_images * 2.0), squeeze=False)
    else:
        fig = axes[0, 0].figure

    for column, class_idx in enumerate(class_indices):
        title = class_names[class_idx] if class_names else f"Class {class_idx}"
        axes[0, column].set_title(title, fontsize=10)

        for row, image in enumerate(samples[class_idx]):
            axis = axes[row, column]
            axis.axis("off")
            axis.imshow(prepare_image(image), cmap=cmap)

        for row in range(len(samples[class_idx]), n_images):
            axes[row, column].axis("off")

    return fig


def plot_latent_utilization(
    model: AbstractVAE,
    dataloader: DataLoader,
    *,
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """
    Plot average KL contribution per latent dimension.

    The model device is inferred automatically.
    """

    model.eval()

    device = next(model.parameters()).device

    kl_total: torch.Tensor | None = None
    sample_count = 0

    with torch.no_grad():
        for images, *_ in dataloader:
            images = images.to(device)

            mu, logvar = model.encoder(images)

            kl = model.kl_divergence(mu, logvar)
            kl = kl.sum(dim=0)

            kl_total = kl if kl_total is None else kl_total + kl
            sample_count += images.size(0)

    if kl_total is None:
        raise RuntimeError("DataLoader produced no samples")

    avg_kl = (kl_total / sample_count).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    dimensions = np.arange(avg_kl.size)

    ax.plot(dimensions, avg_kl, linewidth=2)
    ax.fill_between(dimensions, avg_kl, alpha=0.2)

    ax.set_title("KL Contribution per Latent Dimension")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Average KL")
    ax.grid(alpha=0.3)

    return fig