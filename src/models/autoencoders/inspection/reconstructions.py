import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def plot_reconstructions(
    model,
    data_loader,
    *,
    device=None,
    num_images: int = 8,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Plots original images and their reconstructions.
    """

    device = device or next(model.parameters()).device
    model.eval()

    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)

    reconstructions, _, _ = model(images)

    images = images.cpu().permute(0, 2, 3, 1).numpy()
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