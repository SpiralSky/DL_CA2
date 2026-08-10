import torch


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """
    Function to calculate KL divergence.

    Free bits prevents small KL values from being penalised.

    :param mu: Tensor of posterior means.
    :param logvar: Tensor of posterior log variances.
    :param free_bits: Minimum KL contribution per latent dimension.
    :return: KL divergence tensor.
    """

    kl_per_dim = -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    if free_bits > 0:
        kl_per_dim = torch.clamp(
            kl_per_dim - free_bits,
            min=0,
        )

    return kl_per_dim.sum() / mu.size(0)

