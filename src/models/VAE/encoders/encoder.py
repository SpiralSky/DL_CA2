import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for 32x32 RGB images (e.g. CIFAR-10).
    Maps an image to the parameters (mu, logvar) of a diagonal Gaussian
    over the latent space. Downsamples 32 -> 16 -> 8 -> 4 via stride-2 convs.
    """

    def __init__(self, in_channels=3, base_channels=32, latent_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten_dim = base_channels * 4 * 4 * 4  # channels * H * W at 4x4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
