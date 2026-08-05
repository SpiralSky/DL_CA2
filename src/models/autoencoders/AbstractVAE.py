import time
from typing import Literal, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping


class VAETrainConfig(TypedDict):
    recon_loss_type: Literal["mse", "bce"]
    grad_clip_norm: float
    free_bits: float

class AbstractVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor, loss_type: Literal["mse", "bce"]) -> torch.Tensor:
        if loss_type == "mse":
            return F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        if loss_type == "bce":
            return F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
        raise ValueError(f"Unknown recon_loss_type '{loss_type}', expected 'mse' or 'bce'")

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum() / mu.size(0)

    def get_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        recon_loss_type: Literal["mse", "bce"] = "bce",
        free_bits: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        recon_loss = self.reconstruction_loss(recon, target, recon_loss_type)
        kl = self.kl_divergence(mu, logvar, free_bits)
        return {
            "total": recon_loss + kl,
            "reconstruction": recon_loss,
            "kl_divergence": kl,
        }

    def run_epoch(self, loader, device, optimizer, config, train):
        self.train() if train else self.eval()

        totals = {"total": 0.0, "reconstruction": 0.0, "kl_divergence": 0.0}
        num_batches = 0

        with torch.enable_grad() if train else torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)

                if train:
                    optimizer.zero_grad()

                recon, mu, logvar = self(images)
                losses = self.get_loss(
                    recon,
                    images,
                    mu,
                    logvar,
                    recon_loss_type=config["recon_loss_type"],
                    free_bits=config["free_bits"],
                )

                if train:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=config["grad_clip_norm"]
                    )
                    optimizer.step()

                for k in totals:
                    totals[k] += losses[k].item()

                num_batches += 1

        return {k: v / num_batches for k, v in totals.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        max_epochs: int,
        lr: float,
        grad_clip_norm: float,
        recon_loss_type: str = "mse",
        free_bits: float = 0.0,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        early_stopping=None
    ) -> list[dict]:
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5
            )

        if early_stopping is None:
            early_stopping = EarlyStopping(monitor="val_loss")

        config: VAETrainConfig = {
            "recon_loss_type": recon_loss_type,  # type: ignore
            "free_bits": free_bits,
            "grad_clip_norm": grad_clip_norm,
        }

        history = []

        for epoch in range(1, max_epochs + 1):
            start = time.time()

            train_metrics = self.run_epoch(
                train_loader, device, optimizer, config, train=True
            )
            val_metrics = self.run_epoch(
                val_loader, device, optimizer, config, train=False
            )

            scheduler.step(val_metrics["total"])

            elapsed = time.time() - start

            logs = {
                "epoch": epoch,
                "max_epochs": max_epochs,
                "lr": optimizer.param_groups[0]["lr"],
                "time": elapsed,
                "loss": train_metrics["total"],
                "recon": train_metrics["reconstruction"],
                "kl": train_metrics["kl_divergence"],
                "val_loss": val_metrics["total"],
            }

            history.append(logs)

            print(
                f"epoch {epoch}/{max_epochs}  "
                f"loss={logs['loss']:.2f}  recon={logs['recon']:.2f}  kl={logs['kl']:.2f}  "
                f"lr={logs['lr']:.2e}  time={elapsed:.1f}s  "
                f"val_loss={logs['val_loss']:.2f}"
            )

            if early_stopping.on_epoch_end(epoch, logs, self):
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement > {early_stopping.min_delta} "
                    f"for {early_stopping.patience} epochs)"
                )
                break

        early_stopping.load_checkpoint(self)

        if early_stopping.best_value is not None:
            print(f"Restored best model weights (val_loss={early_stopping.best_value:.2f})")

        return history