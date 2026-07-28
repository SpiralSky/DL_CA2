import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Generic VAE shell: takes an encoder and decoder as injected nn.Module
    instances rather than hardcoding architecture. Swapping in a different
    encoder/decoder (deeper convs, ResNet blocks, conditional variants that
    also accept a label) requires no changes here.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device=None):
        """
        Draws num_samples latent vectors from the prior N(0, I) and decodes
        them, for inspecting what the model has learned to generate.
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)
