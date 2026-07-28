from models.VAE.encoders.encoder import BasicEncoder
from models.VAE.decoders.decoder import BasicDecoder
from .model import VAE


def build_baseline_vae(in_channels=3, base_channels=32, latent_dim=128):
    """
    Assembles the baseline VAE from its component modules. Switching to a
    different encoder/decoder implementation later only means changing what
    gets constructed here (or adding an entry to MODEL_REGISTRY below) --
    everything downstream (training loop, loss function) is unaffected.
    """
    encoder = BasicEncoder(in_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim)
    decoder = BasicDecoder(out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim)
    return VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)


MODEL_REGISTRY = {
    "baseline_vae": build_baseline_vae,
}


def build_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)
