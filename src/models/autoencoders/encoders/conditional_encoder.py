import torch
from torch import nn

from src.models.autoencoders.encoders.basic_encoder import BasicEncoder


class ConditionalEncoder(BasicEncoder):
    def __init__(self, input_channels=3, output_channels=32, latent_dim=128, label_embed_dim=16):
        super().__init__(input_channels, output_channels, latent_dim)

        self.mean = nn.Linear(8192 + label_embed_dim, latent_dim)
        self.log_variance = nn.Linear(8192 + label_embed_dim, latent_dim)

    def forward(self, inputs, embeddings):
        feature_maps = self.features(inputs)
        combined = torch.cat([feature_maps, embeddings], dim=1)
        return self.mean(combined), self.log_variance(combined)