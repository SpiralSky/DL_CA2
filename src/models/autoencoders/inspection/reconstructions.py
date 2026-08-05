import torch
from matplotlib import pyplot as plt


from matplotlib.figure import Figure, SubFigure


@torch.no_grad()
def plot_reconstructions(
    model,
    data_loader,
    device=None,
    num_images: int = 8,
    fig: Figure | SubFigure | None = None,
) -> Figure | SubFigure:
    device = device or next(model.parameters()).device
    model.eval()

    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    recon, _, _ = model(images)

    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    recon_np = recon.cpu().permute(0, 2, 3, 1).numpy()

    if fig is None:
        fig = plt.figure(figsize=(num_images * 1.5, 3.5))

    axes = fig.subplots(2, num_images)

    for i in range(num_images):
        axes[0, i].imshow(images_np[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_np[i].clip(0, 1))
        axes[1, i].axis("off")

    fig.text(0.02, 0.75, "Original", rotation=90, va="center", fontsize=10)
    fig.text(0.02, 0.25, "Reconstructed", rotation=90, va="center", fontsize=10)

    return fig