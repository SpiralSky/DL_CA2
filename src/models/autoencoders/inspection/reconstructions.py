import torch
from matplotlib import pyplot as plt


@torch.no_grad()
def plot_reconstructions(model, data_loader, device=None, num_images=8):
    """
    Draws one batch from data_loader, runs it through the model, and plots
    original vs. reconstructed images side by side (originals on top row,
    reconstructions on bottom row). This catches issues loss curves alone
    can hide, e.g. whether the model has collapsed to near-identical blurry
    outputs regardless of input.
    """
    device = device or next(model.parameters()).device
    model.eval()

    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    recon, _, _ = model(images)

    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    recon_np = recon.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3.5))
    for i in range(num_images):
        axes[0, i].imshow(images_np[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_np[i].clip(0, 1))
        axes[1, i].axis("off")

    fig.text(0.02, 0.75, "original", rotation=90, va="center", fontsize=10)
    fig.text(0.02, 0.25, "reconstructed", rotation=90, va="center", fontsize=10)
    fig.tight_layout(rect=[0.04, 0, 1, 1])
    plt.show()