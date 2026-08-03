import torch.nn as nn


class BasicEncoder(nn.Module):
    def __init__(self, input_channels=3, output_channels=32, latent_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        self.mean = nn.Linear(8192, latent_dim)
        self.log_variance = nn.Linear(8192, latent_dim)

    def forward(self, inputs):
        feature_maps = self.features(inputs)
        mean = self.mean(feature_maps)
        log_variance = self.log_variance(feature_maps)
        return mean, log_variance