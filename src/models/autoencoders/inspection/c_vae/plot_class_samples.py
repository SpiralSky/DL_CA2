import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.models.autoencoders.BetaConditionalVAE import BetaConditionalVAE
from src.training.autoencoders.sampling import prepare_image


def plot_conditional_class_samples(
    model: BetaConditionalVAE,
    dataloader: DataLoader,
    *,
    n_images: int = 2,
    cmap: str = "gray",
    class_names: list[str] | None = None,
    axes: np.ndarray | None = None,
) -> np.ndarray:

    model.eval()

    device = next(model.parameters()).device

    samples: dict[int, list[torch.Tensor]] = {}

    with torch.no_grad():
        for images, labels, *_ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            reconstructed, _, _ = model(images, labels)

            reconstructed = reconstructed.cpu()
            labels = labels.cpu()

            for image, label in zip(reconstructed, labels):
                class_idx = int(label)

                if class_idx not in samples:
                    samples[class_idx] = []

                if len(samples[class_idx]) < n_images:
                    samples[class_idx].append(image)

            # Stop once every class has enough samples
            if all(len(v) >= n_images for v in samples.values()):
                break

    class_indices = sorted(samples)
    n_classes = len(class_indices)

    if axes is None:
        _, axes = plt.subplots(
            n_images,
            n_classes,
            figsize=(n_classes * 1.8, n_images * 2.0),
            squeeze=False,
        )

    for column, class_idx in enumerate(class_indices):
        title = (
            class_names[class_idx]
            if class_names
            else f"Class {class_idx}"
        )

        axes[0, column].set_title(title, fontsize=10)

        for row, image in enumerate(samples[class_idx]):
            axis = axes[row, column]
            axis.axis("off")
            axis.imshow(
                prepare_image(image),
                cmap=cmap,
            )

        for row in range(len(samples[class_idx]), n_images):
            axes[row, column].axis("off")

    return axes