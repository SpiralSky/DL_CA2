from torch import nn

from src.models.autoencoders.decoders.basic_decoder import BasicDecoder

class ImprovedDecoder(BasicDecoder):
    def __init__(self, latent_dim: int = 256):
        super().__init__(latent_dim=latent_dim)

        self.decoder =  nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.Unflatten(dim=1, unflattened_size=(256, self.init_spatial, self.init_spatial)),

            BasicDecoder.make_conv_block(in_channels=256, out_channels=64),
            BasicDecoder.make_conv_block(in_channels=64, out_channels=48, upsample=False),
            BasicDecoder.make_conv_block(in_channels=48, out_channels=48),
            BasicDecoder.make_conv_block(in_channels=48, out_channels=32, upsample=False),
            BasicDecoder.make_conv_block(in_channels=32, out_channels=3),

            nn.Sigmoid()
        )