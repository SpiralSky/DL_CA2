from models.autoencoders.decoders.decoder import BasicDecoder
from models.autoencoders.encoders.encoder import BasicEncoder
from models.autoencoders.models.VAE import VAE


def basic_autoencoder(in_channels=3, base_channels=32, latent_dim=128):
    """
    Assembles the baseline autoencoders from its component modules. Switching to a
    different encoder/decoder implementation later only means changing what
    gets constructed here (or adding an entry to MODEL_REGISTRY below) --
    everything downstream (training loop, loss function) is unaffected.
    """
    encoder = BasicEncoder(input_channels=in_channels, output_channels=base_channels, latent_dim=latent_dim)
    decoder = BasicDecoder(out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim)
    return VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)