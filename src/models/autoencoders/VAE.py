import torch
import torch.nn as nn
from torch import Tensor


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        """
        Creates a new AbstractVAE.

        :param encoder: Encoder module.
        :param decoder: Decoder module.
        :param latent_dim: Latent dimension size.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim


    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Runs a single forward pass.

        :param x: Input tensor.
        :return: Tuple of reconstructed image, mu and logvar.
        """

        mu, logvar = self.encoder(x)

        z = self.reparameterize(
            mu,
            logvar,
        )

        return self.decoder(z), mu, logvar


    @torch.no_grad()
    def sample(self, num_samples: int, device=None) -> Tensor:
        """
        Samples images from the latent space.

        :param num_samples: Number of samples to generate.
        :param device: Device to generate tensors on.
        :return: Image samples.
        """

        device = device or next(self.parameters()).device

        z = torch.randn(
            num_samples,
            self.latent_dim,
            device=device,
        )

        return self.decoder(z)


    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick.

        :param mu: Mean tensor.
        :param logvar: Log variance tensor.
        :return: Sampled latent tensor.
        """

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std