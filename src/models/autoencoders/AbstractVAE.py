import time
from typing import Literal, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from src.models.util.TrainingHistory import TrainingHistory
from src.training.callbacks import EarlyStopping


# TODO Remove Train Config / Refactor
class VAETrainConfig(TypedDict):
    recon_loss_type: Literal["mse", "bce"]
    grad_clip_norm: float
    free_bits: float
    kl_weight: float


class AbstractVAE(nn.Module):
    def __init__(self, latent_dim: int):
        """
        Creates a new AbstractVAE.
        :param latent_dim: Latent dimension size.
        """
        super().__init__()
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        """
        Reparameterization trick: Samples from N(mu, sigma^2).

        :param mu: Tensor of posterior means, shape (batch_size, latent_dim).
        :param logvar: Tensor of log-variances, shape (batch_size, latent_dim).
        :param eps: Tensor of random values for sampling, shape (batch_size, latent_dim).
        :return:
        """
        std = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor, loss_type: Literal["mse", "bce"]) -> torch.Tensor:
        """
        Function used to get reconstruction loss.
        Uses torch.nn.functional's loss functions.
        :param recon_x: Tensor of image reconstructions.
        :param x: Tensor of image samples.
        :param loss_type: Type of loss. Either MSE (mean squared error) or BCE (binary cross-entropy) loss.
        :return:
        """
        if loss_type == "mse":
            return functional.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        if loss_type == "bce":
            return functional.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
        raise ValueError(f"Unknown recon_loss_type '{loss_type}', expected 'mse' or 'bce'")

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
        """
        Function to get Kl divergence given mu and logvar values,
        summed over all dimensions (in latent vector).

        Free bits ensures that KL divergence does not penalise for
        values that are smaller than free_bits.
        :param mu: Tensor of posterior means, shape (batch_size, latent_dim).
        :param logvar: Tensor of logvar values, shape (batch_size, latent_dim)
        :param free_bits: Free bits float value.
        :return: Tensor.
        """
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0)
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
            kl_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Function to get loss of VAE.
        :param recon: Tensor of reconstructed images. Shape: (batch_size, C, H, W).
        :param target: Real images in batch size. Shape: (batch_size, C, H, W).
        :param mu: mu value. Shape: (batch_size, latent_dim).
        :param logvar: logvar value. Shape: (batch_size, latent_dim).
        :param recon_loss_type: Reconstruction loss type. Supports bce or mse.
        :param free_bits: Free bits for reconstruction loss calculation.
        :param kl_weight: Kl divergence weight.
        :return:
        """

        recon_loss = self.reconstruction_loss(recon, target, recon_loss_type)
        kl = self.kl_divergence(mu, logvar, free_bits)

        return {
            "loss": recon_loss + kl_weight * kl,
            "reconstruction": recon_loss,
            "kl_divergence": kl,
        }

    def run_epoch(
            self,
            loader: DataLoader,
            device: torch.device,
            optimizer: torch.optim.Optimizer,
            config: VAETrainConfig,
            train: bool,
    ):
        """
        Trains the model on one epoch.
        Returns loss types over batches.
        :param loader: DataLoader to load images from.
        :param device: Device (cpu/gpu).
        :param optimizer: Optimizer type to use for training.
        :param config: Config for training.
        :param train: If False, training is disabled (no gradient updates).
        :return: Values of each loss type.
        """
        self.train() if train else self.eval()

        loss_types = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "kl_divergence": 0.0,
        }

        num_batches = 0

        with torch.enable_grad() if train else torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)

                if train:
                    optimizer.zero_grad()

                # Standard forward pass
                recon, mu, logvar = self(images)

                losses = self.get_loss(
                    recon,
                    images,
                    mu,
                    logvar,
                    recon_loss_type=config["recon_loss_type"],
                    free_bits=config["free_bits"],
                    kl_weight=config["kl_weight"]
                )

                if train:
                    losses["loss"].backward()

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=config["grad_clip_norm"])

                    optimizer.step()

                for key in loss_types:
                    loss_types[key] += losses[key].item()

                num_batches += 1

        return {
            key: value / num_batches
            for key, value in loss_types.items()
        }

    # TODO: Possible refactor/shift to dedicated VAETrainer.
    # TODO: Refactor callbacks system (possibly in VAETrainer).
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
            kl_warmup_epochs: int = 0,
            *,
            optimizer: torch.optim.Optimizer | None = None,
            scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
            early_stopping: EarlyStopping | None = None,
    ) -> TrainingHistory:
        """
        Function for training abstract VAE.
        Uses a standard training loop and assigns default optimizers and schedulers.

        :param train_loader: DataLoader to take image tensors for training.
        :param val_loader: DataLoader to take image tensors for validation.
        :param device: PyTorch device (cpu/gpu).
        :param max_epochs: Epoch count to stop at.
        :param lr: Default learning rate.
        :param grad_clip_norm: Gradient clipping norm (to stop exploding gradients).
        :param recon_loss_type: Reconstruction Loss type. Either bce or mse.
        :param free_bits: Free bits for reconstruction loss calculation.
        :param kl_warmup_epochs: Number of epochs for KL Warmup (reduces kl divergence penalties before that epoch).
        :param optimizer: PyTorch Optimizer to use for training.
        :param scheduler: PyTorch learning rate scheduler to use for training.
        :param early_stopping: EarlyStopping callback to use for training.
        :return:
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        if early_stopping is None:
            early_stopping = EarlyStopping(
                monitor=("val", "loss"),
                start_epoch=30,
            )

        history = TrainingHistory(max_epochs)
        gradient_history: dict[int, dict[str, dict[str, float]]] = {}

        for epoch in range(1, max_epochs + 1):
            start = time.time()

            kl_weight = self.get_kl_weight(
                epoch,
                kl_warmup_epochs,
            )

            config: VAETrainConfig = {
                "recon_loss_type": recon_loss_type,  # type: ignore
                "free_bits": free_bits,
                "grad_clip_norm": grad_clip_norm,
                "kl_weight": kl_weight,
            }

            train_metrics = self.run_epoch(train_loader, device, optimizer, config, train=True)
            val_metrics = self.run_epoch(val_loader, device, optimizer, config, train=False)

            gradient_history[epoch] = self.get_gradient_stats()
            scheduler.step(val_metrics["loss"])

            message = history.update(
                epoch=epoch,
                train_metrics={"loss" if k == "total" else k: v for k, v in train_metrics.items()},
                val_metrics={"loss" if k == "total" else k: v for k, v in val_metrics.items()},
                extra_metrics={
                    "kl_weight": kl_weight,
                    "lr": optimizer.param_groups[0]["lr"],
                    "time": time.time() - start,
                },
            )

            print(message)

            if stop_message := early_stopping.on_epoch_end(
                epoch,
                history,
                self,
            ):
                print(stop_message)
                break

        return history

    # TODO Possible(?) refactor to trainer/plotting function/callback hooks
    def get_gradient_stats(self) -> dict[str, dict[str, float]]:
        """
        Get gradient statistics over model parameter types for plotting.
        :return: Dictionary of model parameter types to gradient statistics.
        """
        gradients = {}

        for name, param in self.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach().abs()

            gradients[name] = {
                "mean": grad.mean().item(),
                "max": grad.max().item(),
                "norm": grad.norm().item(),
            }

        return gradients

    @staticmethod
    def get_kl_weight(
            epoch: int,
            kl_warmup_epochs: int,
    ) -> float:
        """
        Gets kl weight based on epoch for KL annealing.
        KL Weight scales linearly to 1.0 (reduces kl score when multiplied together)
        to reduce KL loss and prioritise reconstruction loss during kl warmup.
        :param epoch: Current epoch.
        :param kl_warmup_epochs: Number of epochs to warm up for.
        :return:
        """
        if kl_warmup_epochs <= 0:
            return 1.0

        return min(epoch / kl_warmup_epochs, 1.0)
