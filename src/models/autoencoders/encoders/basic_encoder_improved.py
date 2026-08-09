import torch.nn as nn

from src.models.autoencoders.encoders.basic_encoder import BasicEncoder


# TODO Update Docstring
class ImprovedEncoder(BasicEncoder):
    """
    TODO: UPDATE ME
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__(latent_dim=latent_dim)

        self.features = nn.Sequential(
            self.down_block(3, 64),
            self.down_block(64, 128),
            self.down_block(128, 256),
            self.channel_block(256, 512),
            nn.Flatten()
        )

        self.mean = nn.Linear(8192, latent_dim)
        self.log_variance = nn.Linear(8192, latent_dim)

    @staticmethod
    def channel_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Block that increases channels with convolutional filters to learn features while keeping image size.
        BatchNormalization is used on raw logits to prevent gradient shrinking.
        :param in_channels: Input channels.
        :param out_channels: Output channels. Also corresponds to number of convolutional filters used.
        :return:
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )