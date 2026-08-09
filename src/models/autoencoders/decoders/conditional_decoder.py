import torch
from torch import nn

from src.models.autoencoders.decoders.basic_decoder import BasicDecoder


class ConditionalDecoder(BasicDecoder):
    def __init__(
        self,
        latent_dim: int = 128,
        label_embed_dim: int = 16
    ):
        super().__init__(latent_dim=latent_dim)

        self.decoder[0] = nn.Linear(
            latent_dim + label_embed_dim,
            4096,
        )

    # noinspection method-overriding
    def forward(
        self,
        z: torch.Tensor,
        embeddings: torch.Tensor
    ):
        combined = torch.cat([z, embeddings], dim=1)

        return self.decoder(combined)