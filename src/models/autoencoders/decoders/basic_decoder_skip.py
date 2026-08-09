from torch import nn

from src.models.autoencoders.decoders.basic_decoder import BasicDecoder


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))

# TODO Docstrings
class SkipDecoder(BasicDecoder):
    """
    Improved Decoder with additional convolutional capacity.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__(latent_dim=latent_dim)

        self.init_spatial = 4

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8192),
            nn.Unflatten(dim=1, unflattened_size=(512, self.init_spatial, self.init_spatial)),
            ResidualBlock(512),
            self.make_conv_block(in_channels=512, out_channels=256),
            ResidualBlock(256),
            self.make_conv_block(in_channels=256, out_channels=128),
            ResidualBlock(128),
            self.make_conv_block(in_channels=128, out_channels=64, upsample=False),
            self.make_conv_block(in_channels=64, out_channels=3),

            nn.Sigmoid()
        )
