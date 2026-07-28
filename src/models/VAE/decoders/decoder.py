import torch.nn as nn


import torch.nn as nn

class BasicDecoder(nn.Module):
    """
    Convolutional decoder mirroring ConvEncoder. Maps a latent vector back
    to a 32x32 RGB image via upsampling transposed convs: 4 -> 8 -> 16 -> 32,
    with a stride-1 refinement conv at each resolution (mirroring the
    encoder) so shape/structure can be reconstructed with more capacity than
    a single upsampling conv provides. Output is passed through sigmoid to
    match [0, 1]-scaled ToTensor() inputs.
    """

    def __init__(self, out_channels=3, base_channels=32, latent_dim=128):
        super().__init__()
        self.base_channels = base_channels
        self.init_spatial = 4
        self.fc = nn.Linear(latent_dim, base_channels * 4 * self.init_spatial * self.init_spatial)

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.deconv = nn.Sequential(
            up_block(base_channels * 4, base_channels * 2),
            up_block(base_channels * 2, base_channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 4, self.init_spatial, self.init_spatial)
        return self.deconv(h)
