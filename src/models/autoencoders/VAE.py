import time
from typing import Literal, TypedDict

from torch.utils.data import DataLoader

from src.models.autoencoders.losses.models.vae_loss import vae_loss

class VAETrainConfig(TypedDict):
    recon_loss_type: Literal["mse", "bce"]
    grad_clip_norm: float
    free_bits: float
    kl_weight: float


import torch
import torch.nn as nn
from torch import Tensor

from src.models.util.TrainingHistory import TrainingHistory
from src.training.callbacks.EarlyStopping import EarlyStopping


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        """
        Creates a new AbstractVAE.

        :param encoder: Encoder module.
        :param decoder: Decoder module.
        :param latent_dim: Latent dimension size.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim


    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Runs a single forward pass.

        :param x: Input tensor.
        :return: Tuple of reconstructed image, mu and logvar.
        """

        mu, logvar = self.encoder(x)

        z = self.reparameterize(
            mu,
            logvar,
        )

        return self.decoder(z), mu, logvar


    @torch.no_grad()
    def sample(self, num_samples: int, device=None) -> Tensor:
        """
        Samples images from the latent space.

        :param num_samples: Number of samples to generate.
        :param device: Device to generate tensors on.
        :return: Image samples.
        """

        device = device or next(self.parameters()).device

        z = torch.randn(
            num_samples,
            self.latent_dim,
            device=device,
        )

        return self.decoder(z)


    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick.

        :param mu: Mean tensor.
        :param logvar: Log variance tensor.
        :return: Sampled latent tensor.
        """

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

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

        Wrapper around external VAE loss implementation.

        :param recon: Tensor of reconstructed images.
        :param target: Original image tensor.
        :param mu: Latent mean tensor.
        :param logvar: Latent log variance tensor.
        :param recon_loss_type: Reconstruction loss type.
        :param free_bits: Free bits value.
        :param kl_weight: KL divergence weight.
        :return: Dictionary of losses.
        """

        return vae_loss(
            recon,
            target,
            mu,
            logvar,
            recon_loss_type=recon_loss_type,
            free_bits=free_bits,
            kl_weight=kl_weight,
        )


    def run_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        config: VAETrainConfig,
        train: bool
    ):
        """
        Trains the model on one epoch.

        :param loader: DataLoader to load images from.
        :param device: Device to train on.
        :param optimizer: Optimizer used for training.
        :param config: Training configuration.
        :param train: Whether gradients should be updated.
        :return: Average losses.
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

                recon, mu, logvar = self(images)

                losses = self.get_loss(
                    recon,
                    images,
                    mu,
                    logvar,
                    recon_loss_type=config["recon_loss_type"],
                    free_bits=config["free_bits"],
                    kl_weight=config["kl_weight"],
                )

                if train:
                    losses["loss"].backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        max_norm=config["grad_clip_norm"],
                    )

                    optimizer.step()

                for key in loss_types:
                    loss_types[key] += losses[key].item()

                num_batches += 1

        return {
            key: value / num_batches
            for key, value in loss_types.items()
        }


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

        Uses standard training loop.

        :param train_loader: DataLoader for training.
        :param val_loader: DataLoader for validation.
        :param device: Training device.
        :param max_epochs: Maximum epoch count.
        :param lr: Learning rate.
        :param grad_clip_norm: Gradient clipping norm.
        :param recon_loss_type: Reconstruction loss type.
        :param free_bits: Free bits value.
        :param kl_warmup_epochs: Number of KL warmup epochs.
        :param optimizer: Optional optimizer.
        :param scheduler: Optional scheduler.
        :param early_stopping: Optional early stopping callback.
        :return: Training history.
        """

        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
            )

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=5,
                factor=0.5,
            )

        if early_stopping is None:
            early_stopping = EarlyStopping(
                monitor=("val", "loss"),
                start_epoch=30,
            )

        history = TrainingHistory(max_epochs)

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

            train_metrics = self.run_epoch(
                train_loader,
                device,
                optimizer,
                config,
                train=True,
            )

            val_metrics = self.run_epoch(
                val_loader,
                device,
                optimizer,
                config,
                train=False,
            )

            scheduler.step(
                val_metrics["loss"]
            )

            message = history.update(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
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
                early_stopping.restore_best_state(self)
                break

        return history


    @staticmethod
    def get_kl_weight(epoch: int, kl_warmup_epochs: int) -> float:
        """
        Gets KL weight based on epoch for KL annealing.

        KL weight increases linearly from zero to one.

        :param epoch: Current epoch.
        :param kl_warmup_epochs: Warmup duration.
        :return: KL weight.
        """

        if kl_warmup_epochs <= 0:
            return 1.0

        return min(
            epoch / kl_warmup_epochs,
            1.0,
        )