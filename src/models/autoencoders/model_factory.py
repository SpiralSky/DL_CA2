from src.models.autoencoders.BetaConditionalVAE import BetaConditionalVAE
from src.models.autoencoders.VAE import VAE
from src.models.autoencoders.decoders.basic_decoder import BasicDecoder
from src.models.autoencoders.decoders.basic_decoder_improved import ImprovedDecoder
from src.models.autoencoders.decoders.basic_decoder_skip import SkipDecoder
from src.models.autoencoders.decoders.conditional_decoder import ConditionalDecoder
from src.models.autoencoders.encoders.basic_encoder import BasicEncoder
from src.models.autoencoders.encoders.basic_encoder_improved import ImprovedEncoder
from src.models.autoencoders.encoders.conditional_encoder import ConditionalEncoder


def basic_vae(latent_dim=128) -> VAE:
    """
    Creates a basic VAE model
    :param latent_dim: Size of the latent dimension.
    :return:
    """
    encoder = BasicEncoder(latent_dim=latent_dim)
    decoder = BasicDecoder(latent_dim=latent_dim)
    return VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

def improved_basic_vae(latent_dim=128):
    encoder = ImprovedEncoder(latent_dim=latent_dim)
    decoder = ImprovedDecoder(latent_dim=latent_dim)
    return VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

def skip_vae(latent_dim=128):
    encoder = ImprovedEncoder(latent_dim=latent_dim)
    decoder = SkipDecoder(latent_dim=latent_dim)
    return VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

def beta_conditional_vae(num_classes, in_channels=3, base_channels=32, latent_dim=128):
    encoder = ConditionalEncoder(input_channels=in_channels, output_channels=base_channels, latent_dim=latent_dim)
    decoder = ConditionalDecoder(out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim)

    return BetaConditionalVAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim, num_classes=num_classes)