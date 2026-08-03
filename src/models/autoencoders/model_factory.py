from src.models.autoencoders.ConditionalVAE import ConditionalVAE
from src.models.autoencoders.VAE import VAE
from src.models.autoencoders.decoders.basic_decoder import BasicDecoder
from src.models.autoencoders.decoders.conditional_decoder import ConditionalDecoder
from src.models.autoencoders.encoders.basic_encoder import BasicEncoder
from src.models.autoencoders.encoders.conditional_encoder import ConditionalEncoder


def basic_vae(in_channels=3, base_channels=32, latent_dim=128):
    """
    Assembles the baseline autoencoders from its component modules. Switching to a
    different encoder/decoder implementation later only means changing what
    gets constructed here (or adding an entry to MODEL_REGISTRY below) --
    everything downstream (training loop, loss function) is unaffected.
    """
    encoder = BasicEncoder(input_channels=in_channels, output_channels=base_channels, latent_dim=latent_dim)
    decoder = BasicDecoder(out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim)
    return VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

def conditional_vae(num_classes, in_channels=3, base_channels=32, latent_dim=128):
    encoder = ConditionalEncoder(input_channels=in_channels, output_channels=base_channels, latent_dim=latent_dim)
    decoder = ConditionalDecoder(out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim)

    return ConditionalVAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim, num_classes=num_classes)