import torch
import torch.nn as nn
from torch import Tensor

from src.models.autoencoders.VAE import VAE


class BetaConditionalVAE(VAE):
    """
    Conditional beta-VAE.

    Uses label embeddings to condition both encoder and decoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        num_classes: int,
        label_embed_dim: int = 16,
    ):
        super().__init__(encoder, decoder, latent_dim)

        self.label_embeddings = nn.Embedding(num_classes, label_embed_dim)

    def forward(
        self,
        images: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Runs conditional VAE forward pass.

        :param images: Input images.
        :param labels: Class labels.
        :return: Reconstruction, mean, and log variance.
        """

        labels = labels.squeeze().long()

        embeddings = self.label_embeddings(labels)

        mu, logvar = self.encoder(images, embeddings)

        z = self.reparameterize(mu, logvar)

        return self.decoder(z, embeddings), mu, logvar

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        labels: Tensor,
        device=None,
    ) -> Tensor:
        """
        Generates conditional samples.

        :param num_samples: Number of samples.
        :param labels: Labels controlling generation.
        :param device: Device to generate samples on.
        :return: Generated images.
        """

        device = device or next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_dim, device=device)

        embeddings = self.label_embeddings(labels.to(device))

        return self.decoder(z, embeddings)