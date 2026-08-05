import math
import torch
import torch.nn as nn
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from src.models.autoencoders.AbstractVAE import AbstractVAE

class VAE(AbstractVAE):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        super().__init__(latent_dim)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def view_class_samples(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_images: int,
        device: torch.device = None
    ) -> dict[int, torch.Tensor]:
        device = device or next(self.parameters()).device
        self.to(device)
        self.eval()

        class_latents = defaultdict(list)

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            mu, logvar = self.encoder(images)
            z = self.reparameterize(mu, logvar)

            for i in range(z.size(0)):
                class_idx = int(labels[i].item())
                class_latents[class_idx].append(z[i])

        class_centroids = {}
        for class_idx, latents in class_latents.items():
            stacked = torch.stack(latents)
            centroid = stacked.mean(dim=0)
            class_centroids[class_idx] = centroid

        generated_images = {}

        for class_idx, centroid in class_centroids.items():
            stacked = torch.stack(class_latents[class_idx])
            class_std = stacked.std(dim=0).mean().item()
            if class_std == 0 or torch.isnan(torch.tensor(class_std)):
                class_std = 1.0

            noise = torch.randn(n_images, centroid.size(0), device=device) * class_std
            z_samples = centroid.unsqueeze(0).expand(n_images, -1) + noise

            decoded = self.decoder(z_samples)
            generated_images[class_idx] = decoded.cpu()

        return generated_images

    def plot_class_samples(
            self,
            dataloader: torch.utils.data.DataLoader,
            n_images: int = 10,
            n_cols: int = 5,
            figsize: tuple[int, int] | None = None,
            cmap: str = "gray",
            show: bool = False,
            device: torch.device = None,
    ) -> Figure | None:
        generated = self.view_class_samples(dataloader, n_images, device)
        n_classes = len(generated)
        n_rows_per_class = math.ceil(n_images / n_cols)

        if figsize is None:
            figsize = (n_cols * 2, n_classes * n_rows_per_class * 2)

        fig, axes = plt.subplots(
            n_classes,
            n_rows_per_class,
            figsize=figsize,
            squeeze=False,
        )

        for class_idx, ax_row in zip(sorted(generated.keys()), axes):
            images = generated[class_idx]

            for i in range(n_rows_per_class * n_cols):
                ax = ax_row[i] if i < len(ax_row) else None
                if ax is None:
                    continue

                ax.axis("off")
                if i < n_images:
                    img = images[i]
                    # Handle different channel configurations
                    if img.ndim == 3:
                        if img.shape[0] in (1, 3):  # CHW
                            img = img.permute(1, 2, 0)
                        if img.shape[-1] == 1:  # Single channel
                            img = img.squeeze(-1)
                    img_np = img.cpu().numpy()
                    ax.imshow(img_np, cmap=cmap if img_np.ndim == 2 else None)
                    if i == 0:
                        ax.set_title(f"Class {class_idx}", fontsize=10)

        plt.tight_layout()

        if show:
            plt.show()
            return None

        return fig