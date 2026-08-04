import torch.nn as nn
from IPython.core.magic import output_can_be_silenced


class BasicEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        """
        Basic Encoder
        Input: Image with 3 channels
        Output: Latent Vector of mean and log variance of length lat
        """
        super().__init__()

        # NOTE: stride=2, kernel_size=3, and padding=1 makes Image dimensions halve each 2D convolution.
        # It is used in place of MaxPooling to preserve image.
        # Image Dimensions: 32 -> 16 -> 8 -> 4 -> 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()
        )

        self.mean = nn.Linear(256, latent_dim)
        self.log_variance = nn.Linear(256, latent_dim)

    def forward(self, inputs):
        feature_maps = self.features(inputs)
        mean = self.mean(feature_maps)
        log_variance = self.log_variance(feature_maps)
        return mean, log_variance