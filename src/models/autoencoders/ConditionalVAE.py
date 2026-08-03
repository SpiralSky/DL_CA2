import torch
from torch import nn

from models.autoencoders.VAE import VAE

class ConditionalVAE(VAE):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int, num_classes: int, label_embed_dim: int = 16):
        super().__init__(encoder, decoder, latent_dim)
        self.num_classes = num_classes
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)


    def forward(self, input_features, labels):
        y = self.label_embed(labels)
        mu, logvar = self.encoder(input_features, y)
        z = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(z, y)
        return reconstructed_image, mu, logvar


    def run_epoch(self, loader, device, optimizer, beta, config, train):
        self.train() if train else self.eval()

        totals = {"total": 0.0, "reconstruction": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        with torch.enable_grad() if train else torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                if train:
                    optimizer.zero_grad()

                reconstructed_image, mu, logvar = self(images, labels)
                losses = self.get_loss(reconstructed_image, images, mu, logvar, beta=beta, recon_loss_type=config["recon_loss_type"], free_bits=config["free_bits"])

                if train:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=config["grad_clip_norm"])
                    optimizer.step()

                for k in totals:
                    totals[k] += losses[k].item()

                num_batches += 1
        return {k: v / num_batches for k, v in totals.items()}

    @torch.no_grad()
    def sample(self, num_samples: int, labels: torch.Tensor, device=None):

        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        y = self.label_embed(labels.to(device))
        return self.decoder(z, y)