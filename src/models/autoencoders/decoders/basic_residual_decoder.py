from torch import nn

from src.models.autoencoders.decoders.basic_decoder import BasicDecoder
from src.models.autoencoders.util.ResidualBlock import ResidualBlock


class ResDecoder(BasicDecoder):
    """
    Improved Decoder with residual blocks.
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__(latent_dim=latent_dim)
        self.init_spatial = 4
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            # Block 1: 4×4, refine features
            # 4 -> 8
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            # Block 2: 8×8, refine features
            # 8 -> 16
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            # Block 3: 16×16, refine features
            # 16 -> 32
            ResidualBlock(64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, padding=1),
            # Final: 3 channels
            # 32 -> 32
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )