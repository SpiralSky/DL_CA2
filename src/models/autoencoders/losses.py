from typing import Literal, TypedDict
import torch
import torch.nn.functional as F


class VAELossOutput(TypedDict):
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl_divergence: torch.Tensor


def _reconstruction_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    loss_type: Literal["mse", "bce"],
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    if loss_type == "bce":
        return F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    raise ValueError(
        f"Unknown recon_loss_type '{loss_type}', expected 'mse' or 'bce'"
    )


def _kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float,
) -> torch.Tensor:
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum() / mu.size(0)


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    recon_loss_type: Literal["mse", "bce"] = "mse",
    free_bits: float = 0.0,
) -> VAELossOutput:
    recon = _reconstruction_loss(recon_x, x, recon_loss_type)
    kl = _kl_divergence(mu, logvar, free_bits)
    return {
        "total": recon + beta * kl,
        "reconstruction": recon,
        "kl_divergence": kl,
    }
