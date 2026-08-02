import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader

from src.models.autoencoders.losses import vae_loss
from src.models.training.callbacks import EarlyStopping


@dataclass(frozen=True, slots=True)
class TrainConfig:
    lr: float = 1e-3
    max_epochs: int = 100
    warmup_epochs: int = 10
    beta_target: float = 1.0
    recon_loss_type: str = "mse"
    free_bits: float = 0.0
    grad_clip_norm: float = 1.0
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

@runtime_checkable
class VAEModel(Protocol):
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def train(self, mode: bool = True) -> "VAEModel":
        ...

    def eval(self) -> "VAEModel":
        ...

    def parameters(self) -> torch.nn.Parameter:
        ...


class LossFn(Protocol):
    def __call__(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        beta: float,
        recon_loss_type: str,
        free_bits: float,
    ) -> dict[str, torch.Tensor]:
        ...


class BetaSchedule(Protocol):
    def __call__(self, epoch: int, warmup_epochs: int, beta_target: float) -> float:
        ...


class EpochFn(Protocol):
    def __call__(
        self,
        model: VAEModel,
        loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFn,
        config: TrainConfig,
        beta: float,
        train: bool,
    ) -> dict[str, float]:
        ...



def _default_beta_schedule(epoch: int, warmup_epochs: int, beta_target: float) -> float:
    if warmup_epochs <= 0:
        return beta_target
    return min(beta_target, beta_target * epoch / warmup_epochs)


def _default_loss_fn(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float,
    recon_loss_type: str,
    free_bits: float,
) -> dict[str, torch.Tensor]:
    return vae_loss(
        recon_x, x, mu, logvar,
        beta=beta,
        recon_loss_type=recon_loss_type,
        free_bits=free_bits,
    )


def _default_epoch_fn(
    model: VAEModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    config: TrainConfig,
    beta: float,
    train: bool,
) -> dict[str, float]:
    model.train() if train else model.eval()
    totals = {"total": 0.0, "reconstruction": 0.0, "kl_divergence": 0.0}
    num_batches = 0

    with torch.enable_grad() if train else torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            if train:
                optimizer.zero_grad()

            recon, mu, logvar = model(images)
            losses = loss_fn(
                recon, images, mu, logvar,
                beta=beta,
                recon_loss_type=config.recon_loss_type,
                free_bits=config.free_bits,
            )

            if train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.grad_clip_norm
                )
                optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            num_batches += 1

    return {k: v / num_batches for k, v in totals.items()}


def _format_logs(logs: dict) -> str:
    return (
        f"epoch {logs['epoch']}/{logs['max_epochs']}  "
        f"loss={logs['loss']:.2f}  "
        f"recon={logs['recon']:.2f}  "
        f"kl={logs['kl']:.2f}  "
        f"beta={logs['beta']:.3f}  "
        f"lr={logs['lr']:.2e}  "
        f"time={logs['time']:.1f}s  "
        f"val_loss={logs['val_loss']:.2f}"
    )

class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        device: torch.device,
        *,
        loss_fn: LossFn = _default_loss_fn,
        epoch_fn: EpochFn = _default_epoch_fn,
        beta_schedule: BetaSchedule = _default_beta_schedule,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        early_stopping: EarlyStopping | None = None,
    ):
        self.config = config
        self.device = device
        self.loss_fn = loss_fn
        self.epoch_fn = epoch_fn
        self.beta_schedule = beta_schedule
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._early_stopping = early_stopping
        self._history: list[dict] = []

    def _build_optimizer(self, model: VAEModel) -> torch.optim.Optimizer:
        if self._optimizer is not None:
            return self._optimizer
        return torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        if self._scheduler is not None:
            return self._scheduler
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.config.scheduler_patience,
            factor=self.config.scheduler_factor,
        )

    def _build_early_stopping(self) -> EarlyStopping:
        if self._early_stopping is not None:
            return self._early_stopping
        return EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )

    def fit(
        self,
        model: VAEModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> list[dict]:
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        early_stopping = self._build_early_stopping()
        history = []

        for epoch in range(1, self.config.max_epochs + 1):
            beta = self.beta_schedule(epoch, self.config.warmup_epochs, self.config.beta_target)
            start = time.time()

            train_metrics = self.epoch_fn(
                model, train_loader, self.device, optimizer,
                self.loss_fn, self.config, beta, train=True,
            )
            val_metrics = self.epoch_fn(
                model, val_loader, self.device, optimizer,
                self.loss_fn, self.config, beta, train=False,
            )
            scheduler.step(val_metrics["total"])
            elapsed = time.time() - start

            logs = {
                "epoch": epoch,
                "max_epochs": self.config.max_epochs,
                "beta": beta,
                "lr": optimizer.param_groups[0]["lr"],
                "time": elapsed,
                "loss": train_metrics["total"],
                "recon": train_metrics["reconstruction"],
                "kl": train_metrics["kl_divergence"],
                "val_loss": val_metrics["total"],
            }
            history.append(logs)
            print(_format_logs(logs))

            if beta >= self.config.beta_target:
                if early_stopping.on_epoch_end(epoch, logs, model):
                    print(
                        f"\nearly stopping at epoch {epoch} "
                        f"(no improvement > {self.config.early_stopping_min_delta} "
                        f"for {self.config.early_stopping_patience} epochs)"
                    )
                    break

        early_stopping.restore(model)
        if early_stopping.best_state is not None:
            print(f"restored best model weights (val_loss={early_stopping.best:.2f})")

        self._history.extend(history)
        return history
