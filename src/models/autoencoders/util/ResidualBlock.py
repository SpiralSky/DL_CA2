from torch import nn


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
