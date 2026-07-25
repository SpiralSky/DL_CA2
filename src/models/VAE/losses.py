import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Standard VAE loss: reconstruction term (binary cross-entropy, since
    decoder output is sigmoid-bounded to [0, 1]) plus a beta-weighted KL
    divergence between the approximate posterior N(mu, sigma^2) and the
    standard normal prior N(0, I).

    Returns individual terms too, since watching them separately during
    training reveals issues (e.g. posterior collapse) that the summed loss
    alone would hide.
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.shape[0]
    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.shape[0]
    total_loss = recon_loss + beta * kl_div
    return {
        "total": total_loss,
        "reconstruction": recon_loss,
        "kl_divergence": kl_div,
    }
