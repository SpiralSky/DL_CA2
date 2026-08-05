import torch
from torch import nn

from src.models.autoencoders.decoders.basic_decoder import BasicDecoder


class ConditionalDecoder(BasicDecoder):
    def __init__(self, out_channels=3, base_channels=32, latent_dim=128, label_embed_dim=16):
        super().__init__(out_channels, base_channels, latent_dim)
        self.decoder[0] = nn.Linear(latent_dim + label_embed_dim, base_channels * 4 * self.init_spatial * self.init_spatial)

    def forward(self, z, embeddings):
        combined = torch.cat([z, embeddings], dim=1)
        return self.decoder(combined)