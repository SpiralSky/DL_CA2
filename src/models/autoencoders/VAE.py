import torch
import torch.nn as nn
from torch import Tensor

from src.models.autoencoders.AbstractVAE import AbstractVAE


class VAE(AbstractVAE):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        super().__init__(latent_dim)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Runs a single forward pass.
        :param x: Input tensor. Shape (batch_size, latent_dim).
        :return: Tuple of: Reconstructed image, mu and logvar constructed by the encoder.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device=None):
        """
        Samples images from the latent space.
        :param num_samples: Number of samples to generate.
        :param device: Device to generate tensors on.
        :return: Image samples.
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)