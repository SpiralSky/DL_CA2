from typing import Literal, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

class VAETrainConfig(TypedDict):
    recon_loss_type: Literal["mse", "bce"]
    grad_clip_norm: float
    free_bits: int

class VAE(nn.Module):
    """
    Basic VAE Implementation
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        """
        Creates a new VAE model instance.
        :param encoder: Encoder instance.
        :param decoder: Decoder instance.
        :param latent_dim: Latent dimension of the VAE (equal to latent vector length).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_features):
        mu, logvar = self.encoder(input_features)
        z = self.reparameterize(mu, logvar)
        reconstructed_image = self.decoder(z)
        return reconstructed_image, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)

    def run_epoch(
            self,
            loader: DataLoader,
            device: torch.device,
            optimizer: torch.optim.Optimizer,
            beta: float,
            config: VAETrainConfig,
            train: bool,
    ) -> dict[str, float]:
        self.train() if train else self.eval()

        totals = {"total": 0.0, "reconstruction": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        with torch.enable_grad() if train else torch.no_grad():
            for images, _ in loader:
                images = images.to(device)

                if train:
                    optimizer.zero_grad()

                reconstructed_image, mu, logvar = self(images)

                losses = self.get_loss(
                    reconstructed_image, images, mu, logvar,
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

    def get_loss(
            self,
            reconstructed_image: torch.Tensor,
            image: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            *,
            recon_loss_type: Literal["mse", "bce"] = "bce",
            free_bits: float = 0.0,
    ):
        recon_loss = self.get_reconstruction_loss(reconstructed_image, image, recon_loss_type)
        kl_divergence = self.get_kl_divergence(mu, logvar, free_bits)

        return {
            "total": recon_loss + kl_divergence,
            "reconstruction": recon_loss,
            "kl_divergence": kl_divergence,
        }

    @staticmethod
    def get_reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor, loss_type: Literal["mse", "bce"], ) -> torch.Tensor:
        """
        Supports both BCE and MSE losses.
        :param recon_x:
        :param x:
        :param loss_type:
        :return:
        """
        if loss_type == "mse":
            return functional.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        if loss_type == "bce":
            return functional.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)

        raise ValueError(
            f"Unknown recon_loss_type '{loss_type}', expected 'mse' or 'bce'"
        )

    @staticmethod
    def get_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

        return kl_per_dim.sum() / mu.size(0)