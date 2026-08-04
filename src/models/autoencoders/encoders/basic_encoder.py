import torch.nn as nn
from IPython.core.magic import output_can_be_silenced


class BasicEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        """
        Basic Encoder
        Input: Image with 3 channels
        Output: Latent Vector of mean and log variance of length lat
        """
        super().__init__()

        # NOTE: stride=2, kernel_size=3, and padding=1 makes Image dimensions halve each 2D convolution.
        # It is used in place of MaxPooling to preserve image.
        # Image Dimensions: 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(
            BasicEncoder.down_block(3, 64),
            BasicEncoder.down_block(64, 128),
            BasicEncoder.down_block(128, 256),

            nn.Flatten()
        )

        self.mean = nn.Linear(4096, latent_dim)
        self.log_variance = nn.Linear(4096, latent_dim)

    @staticmethod
    def down_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs):
        feature_maps = self.features(inputs)
        mean = self.mean(feature_maps)
        log_variance = self.log_variance(feature_maps)
        return mean, log_variance