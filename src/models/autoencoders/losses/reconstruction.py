from typing import Literal

import torch
from torch.nn import functional


def reconstruction_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    loss_type: Literal["mse", "bce"],
) -> torch.Tensor:
    """
    Function used to get reconstruction loss.
    Uses torch.nn.functional's loss functions.

    :param recon_x: Tensor of image reconstructions.
    :param x: Tensor of image samples.
    :param loss_type: Type of loss. Either MSE (mean squared error) or BCE (binary cross-entropy).
    :return: Reconstruction loss tensor.
    """

    if loss_type == "mse":
        return functional.mse_loss(recon_x, x, reduction="sum") / x.size(0)

    if loss_type == "bce":
        return functional.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)

    raise ValueError(
        f"Unknown recon_loss_type '{loss_type}', expected 'mse' or 'bce'"
    )