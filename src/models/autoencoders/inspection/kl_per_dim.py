import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch.utils.data import DataLoader

from src.models.autoencoders.VAE import VAE


def plot_kl_per_dim(
    model: VAE,
    dataloader: DataLoader,
    *,
    ax: Axes | None = None,
) -> Axes:
    """
    Plots latent space utilisation of each dimension in a line graph.
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

            # Gets kl per dim. As kl_divergence() returns a single summed value, manual calculations are needed.
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

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
