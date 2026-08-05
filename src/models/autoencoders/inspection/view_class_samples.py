import torch
from collections import defaultdict

from src.models.autoencoders.VAE import VAE


def view_class_samples(
        vae_model: VAE,
        dataloader: torch.utils.data.DataLoader,
        n_images: int,
        device: torch.device = torch.device("cpu")
) -> dict[int, torch.Tensor]:
    vae_model.to(device)
    vae_model.eval()

    class_latents = defaultdict(list)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            mu, logvar = vae_model.encoder(images)

            z = vae_model.reparameterize(mu, logvar)

            for i in range(z.size(0)):
                class_idx = int(labels[i].item())
                class_latents[class_idx].append(z[i])

    class_centroids = {}
    for class_idx, latents in class_latents.items():
        stacked = torch.stack(latents)
        centroid = stacked.mean(dim=0)
        class_centroids[class_idx] = centroid

    generated_images = {}

    with torch.no_grad():
        for class_idx, centroid in class_centroids.items():
            stacked = torch.stack(class_latents[class_idx])
            class_std = stacked.std(dim=0).mean().item()
            if class_std == 0 or torch.isnan(torch.tensor(class_std)):
                class_std = 1.0

            noise = torch.randn(n_images, centroid.size(0), device=device) * class_std
            z_samples = centroid.unsqueeze(0).expand(n_images, -1) + noise

            decoded = vae_model.decoder(z_samples)

            generated_images[class_idx] = decoded.cpu()

    return generated_images