import torch
from torch import nn

from src.models.autoencoders.encoders.basic_encoder import BasicEncoder


class ConditionalEncoder(BasicEncoder):
    def __init__(
        self,
        latent_dim: int = 128,
        label_embed_dim: int = 16,
    ):
        super().__init__(latent_dim=latent_dim)

        self.mean = nn.Linear(4096 + label_embed_dim, latent_dim)
        self.log_variance = nn.Linear(4096 + label_embed_dim, latent_dim)

    # TODO (Method overriding, Unknown resolution)
    # noinspection method-overriding
    def forward(
        self,
        inputs: torch.Tensor,
        embeddings: torch.Tensor,
    ):
        feature_maps = self.features(inputs)

        combined = torch.cat([feature_maps, embeddings], dim=1)

        mean = self.mean(combined)
        log_variance = self.log_variance(combined)

        return mean, log_variance