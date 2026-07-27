import torch.nn as nn


class BasicEncoder(nn.Module):
    """
    Convolutional encoder for 32x32 RGB images (e.g. CIFAR-10).
    Maps an image to the parameters (mu, logvar) of a diagonal Gaussian
    over the latent space. Downsamples 32 -> 16 -> 8 -> 4 via stride-2 convs,
    with a stride-1 refinement conv at each resolution so the network has
    capacity to learn shape/structure features before compressing further,
    rather than immediately squeezing spatial detail into the bottleneck.
    """

    def __init__(self, in_channels=3, base_channels=32, latent_dim=128):
        super().__init__()

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            down_block(in_channels, base_channels),
            down_block(base_channels, base_channels * 2),
            down_block(base_channels * 2, base_channels * 4),
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