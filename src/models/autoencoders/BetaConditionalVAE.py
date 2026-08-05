import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.autoencoders.AbstractVAE import AbstractVAE


class BetaConditionalVAE(AbstractVAE):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim: int,
        num_classes: int,
        label_embed_dim: int = 16,
        beta: float = 1.0,
    ):
        super().__init__(latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.label_embeddings = nn.Embedding(num_classes, label_embed_dim)
        self.beta = beta

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

    def get_loss(self, recon, images, mu, logvar, *, beta=None, **kwargs):
        losses = super().get_loss(recon, images, mu, logvar, **kwargs)
        beta = self.beta if beta is None else beta
        if beta != 1.0:
            losses["total"] = losses["reconstruction"] + beta * losses["kl_divergence"]
        return losses

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
                    beta=config.get("beta", self.beta),
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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        max_epochs: int,
        lr: float,
        grad_clip_norm: float,
        recon_loss_type: str = "mse",
        free_bits: float = 0.0,
        beta: float = 1.0,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        early_stopping=None,
    ) -> list[dict]:
        self.beta = beta
        return super().fit(
            train_loader,
            val_loader,
            device,
            max_epochs,
            lr,
            grad_clip_norm,
            recon_loss_type,
            free_bits,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
        )