import torch
import torch.nn as nn

from src.models.autoencoders.AbstractVAE import AbstractVAE

class ConditionalVAE(AbstractVAE):
    def __init__(self, encoder, decoder, latent_dim, num_classes, label_embed_dim=16):
        super().__init__(latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.label_embeddings = nn.Embedding(num_classes, label_embed_dim)

    def forward(self, images, labels):
        labels = labels.squeeze().long()
        embeddings = self.label_embeddings(labels)
        mu, logvar = self.encoder(images, embeddings)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, embeddings), mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, labels: torch.Tensor, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        y = self.label_embeddings(labels.to(device))
        return self.decoder(z, y)

    def run_epoch(self, loader, device, optimizer, config, train):
        self.train() if train else self.eval()

        totals = {"total": 0.0, "reconstruction": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        with torch.enable_grad() if train else torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)

                if train:
                    optimizer.zero_grad()

                recon, mu, logvar = self(images, labels)
                losses = self.get_loss(
                    recon,
                    images,
                    mu,
                    logvar,
                    recon_loss_type=config["recon_loss_type"],
                    free_bits=config["free_bits"],
                )

                if train:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=config["grad_clip_norm"]
                    )
                    optimizer.step()

                for k in totals:
                    totals[k] += losses[k].item()

                num_batches += 1

        return {k: v / num_batches for k, v in totals.items()}