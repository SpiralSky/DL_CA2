import torch.nn as nn

from src.models.autoencoders.encoders.basic_encoder import BasicEncoder


class ImprovedEncoder(BasicEncoder):
    def __init__(self, latent_dim: int = 256):
        super().__init__(latent_dim=latent_dim)

        self.features = nn.Sequential(
            self.down_block(3, 32),
            self.refine_block(32, 48),
            self.down_block(48, 48),
            self.refine_block(48, 64),
            self.down_block(64, 256),

            nn.Flatten(),
        )

    @staticmethod
    def refine_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )