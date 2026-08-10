from torch import nn

from src.models.autoencoders.encoders.basic_encoder import BasicEncoder
from src.models.autoencoders.util.ResidualBlock import ResidualBlock


class ResEncoder(BasicEncoder):
    """
    Improved Encoder with residual blocks.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__(latent_dim=latent_dim)

        self.features = nn.Sequential(
            self.down_block(3, 64),
            ResidualBlock(64),

            self.down_block(64, 128),
            ResidualBlock(128),

            self.down_block(128, 256),
            ResidualBlock(256),

            nn.Flatten(),
        )

        self.mean = nn.Linear(4096, latent_dim)
        self.log_variance = nn.Linear(4096, latent_dim)