import torch.nn as nn

class BasicDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        # W/H Dimension of the first image (after Unflatten).
        self.init_spatial = 4

        # Image dimensions (W/H):
        # 4 -> 8 -> 16 -> 32 -> 32
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.Unflatten(dim=1, unflattened_size=(256, self.init_spatial, self.init_spatial)),

            BasicDecoder.make_conv_block(in_channels=256, out_channels=64),
            BasicDecoder.make_conv_block(in_channels=64, out_channels=32),
            BasicDecoder.make_conv_block(in_channels=32, out_channels=3),

            nn.Sigmoid()
        )

    @staticmethod
    def make_conv_block(in_channels: int, out_channels: int, upsample=True) -> nn.Sequential:
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if upsample:
            conv_block.insert(
                0,
                nn.Upsample(scale_factor=2, mode="nearest")
            )

        return conv_block


    def forward(self, input_features):
        return self.decoder(input_features)
