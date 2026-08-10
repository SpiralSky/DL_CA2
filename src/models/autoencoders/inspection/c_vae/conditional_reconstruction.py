import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.models.autoencoders.BetaConditionalVAE import BetaConditionalVAE


@torch.no_grad()
def plot_conditional_reconstructions(
    model: BetaConditionalVAE,
    data_loader: DataLoader,
    *,
    device=None,
    num_images: int = 8,
    axes: np.ndarray | None = None,
) -> np.ndarray:

    device = device or next(model.parameters()).device
    model.eval()

    images, labels = next(iter(data_loader))

    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)

    reconstructions, _, _ = model(
        images,
        labels,
    )

    images = (
        images.cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    reconstructions = (
        reconstructions.cpu()
        .permute(0, 2, 3, 1)
        .numpy()
        .clip(0, 1)
    )

    if axes is None:
        _, axes = plt.subplots(
            2,
            num_images,
            figsize=(num_images * 1.5, 3.5),
        )

    for i in range(num_images):
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructions[i])
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")

    return axes