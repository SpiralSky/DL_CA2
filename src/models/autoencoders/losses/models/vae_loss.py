from typing import Literal

import torch

from src.models.autoencoders.losses.kl_divergence import kl_divergence
from src.models.autoencoders.losses.reconstruction import reconstruction_loss


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    recon_loss_type: Literal["mse", "bce"] = "bce",
    free_bits: float = 0.0,
    kl_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    Calculates standard VAE loss.

    :param recon: Reconstructed images.
    :param target: Original images.
    :param mu: Latent mean.
    :param logvar: Latent log variance.
    :param recon_loss_type: Reconstruction loss type.
    :param free_bits: Free bits value.
    :param kl_weight: Weight applied to KL divergence.
    :return: Dictionary containing loss values.
    """

    reconstruction = reconstruction_loss(
        recon,
        target,
        recon_loss_type,
    )

    kl = kl_divergence(
        mu,
        logvar,
        free_bits,
    )

    return {
        "loss": reconstruction + kl_weight * kl,
        "reconstruction": reconstruction,
        "kl_divergence": kl,
    }