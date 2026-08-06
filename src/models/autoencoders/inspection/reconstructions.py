import torch
from matplotlib import pyplot as plt


from matplotlib.figure import Figure, SubFigure

import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@torch.no_grad()
def plot_reconstructions(
    model,
    data_loader,
    *,
    device=None,
    num_images: int = 8,
    axes=None,
) -> Figure:

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
        fig, axes = plt.subplots(
            2,
            num_images,
            figsize=(num_images * 1.5, 3.5),
        )
    else:
        fig = axes[0, 0].figure

    for i in range(num_images):
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructions[i])
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")

    return fig