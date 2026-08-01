import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=1.0, recon_loss_type="mse", free_bits=0.0):
    """
    Standard autoencoders loss: reconstruction term plus a beta-weighted KL divergence
    between the approximate posterior N(mu, sigma^2) and the standard normal
    prior N(0, I).

    recon_loss_type:
        "mse"  - appropriate for continuous natural-image pixels (default,
                 recommended for CIFAR-10-like photographic data)
        "bce"  - appropriate for near-binary pixel data (e.g. MNIST); assumes
                 decoder output is sigmoid-bounded to [0, 1]

    free_bits:
        Minimum nats each latent dimension is allowed before its KL term is
        penalized (per-dimension clamp, applied before summing). 0 disables
        this (original behavior). Use e.g. 0.5 to counter posterior collapse,
        where many dimensions drive KL to ~0 and stop encoding information.

    Returns individual terms too, since watching them separately during
    training reveals issues (e.g. posterior collapse) that the summed loss
    alone would hide.
    """
    if recon_loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.shape[0]
    elif recon_loss_type == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.shape[0]
    else:
        raise ValueError(f"Unknown recon_loss_type '{recon_loss_type}', expected 'mse' or 'bce'")

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_div = kl_per_dim.sum() / x.shape[0]

    total_loss = recon_loss + beta * kl_div
    return {
        "total": total_loss,
        "reconstruction": recon_loss,
        "kl_divergence": kl_div,
    }