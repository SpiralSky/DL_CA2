import torch.nn as nn


class BasicEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        """
        Basic Encoder class.
        :param latent_dim: Length of latent dimension vector.
        """
        super().__init__()

        # NOTE: stride=2, kernel_size=3, and padding=1 makes Image dimensions halve each 2D convolution.
        # It is used in place of MaxPooling to preserve image.
        # Image Dimensions: 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(
            self.down_block(3, 64),
            self.down_block(64, 128),
            self.down_block(128, 256),

            nn.Flatten()
        )

        self.mean = nn.Linear(4096, latent_dim)
        self.log_variance = nn.Linear(4096, latent_dim)

    @staticmethod
    def down_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Standard downsampling block. Kernel_size=3, stride=2 and padding=1 causes spatial dimensions (W/H) to halve.
        BatchNormalization is used on raw logits to prevent gradient shrinking.
        :param in_channels: Input channels.
        :param out_channels: Output channels. Also corresponds to number of convolutional filters used.
        :return:
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs):
        """
        Standard forward pass. Uses inherited mean and log_variance linear layers to output mean and log variance.
        :param inputs: Feature maps.
        :return: Tuple of mean and log variance.
        """
        feature_maps = self.features(inputs)
        mean = self.mean(feature_maps)
        log_variance = self.log_variance(feature_maps)
        return mean, log_variance