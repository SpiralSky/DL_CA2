from torch import nn

from src.models.autoencoders.decoders.basic_decoder import BasicDecoder


class ResidualBlock(nn.Module):
    """
    Simple Residual Block.

    Learns residual mapping f(x) with two 3x3 convolutions with BatchNormalization
    and LeakyReLU activation.

    Input is added to output of convolutional block, allowing gradients to flow through the skip path.

    Input and output channels match.
    """
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
        """
        Runs a single forward pass through the residual block.
        :param x: Input features.
        :return: Output features.
        """
        return self.activation(x + self.block(x))

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
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),

            # Block 2: 8×8, refine features
            # 8 -> 16
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),

            # Block 3: 16×16, refine features
            # 16 -> 32
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),

            # Final: 3 channels
            # 32 -> 32
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
