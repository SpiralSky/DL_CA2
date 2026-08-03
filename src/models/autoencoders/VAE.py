# vae.py
import torch
import torch.nn as nn

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