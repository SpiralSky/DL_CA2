import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch.utils.data import DataLoader

from src.models.autoencoders.VAE import VAE


def plot_latent_utilization(
    model: VAE,
    dataloader: DataLoader,
    *,
    ax: Axes | None = None,
) -> Axes:

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
        _, ax = plt.subplots(figsize=(8, 3))

    dimensions = np.arange(avg_kl.size)

    ax.plot(dimensions, avg_kl, linewidth=2)
    ax.fill_between(dimensions, avg_kl, alpha=0.2)

    ax.set_title("KL Contribution per Latent Dimension")
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Average KL")
    ax.grid(alpha=0.3)

    return ax
