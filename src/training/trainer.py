from typing import Callable, Optional
import time
import torch
from torch.utils.data import DataLoader
from src.training.callbacks import EarlyStopping

def beta_schedule(epoch: int, warmup_epochs: int, beta_target: float) -> float:
    if warmup_epochs <= 0:
        return beta_target
    return min(beta_target, beta_target * epoch / warmup_epochs)

def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    lr: float,
    grad_clip_norm: float,
    warmup_epochs: int = 0,
    beta_target: float = 1.0,
    recon_loss_type: str = "mse",
    free_bits: float = 0.0,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    *,
    optimizer: torch.optim.Optimizer | None= None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    early_stopping: EarlyStopping | None = None,
    beta_schedule_fn: Callable[[int, int, float], float] = beta_schedule,
) -> list[dict]:
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=scheduler_patience, factor=scheduler_factor
        )

    if early_stopping is None:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )

    config = {
        "recon_loss_type": recon_loss_type,
        "free_bits": free_bits,
        "grad_clip_norm": grad_clip_norm,
    }

    history = []

    for epoch in range(1, max_epochs + 1):
        beta = beta_schedule_fn(epoch, warmup_epochs, beta_target)
        start = time.time()

        train_metrics = model.run_epoch(
            train_loader, device, optimizer, beta, config, train=True
        )
        val_metrics = model.run_epoch(
            val_loader, device, optimizer, beta, config, train=False
        )
        scheduler.step(val_metrics["total"])
        elapsed = time.time() - start

        logs = {
            "epoch": epoch,
            "max_epochs": max_epochs,
            "beta": beta,
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
            f"beta={beta:.3f}  lr={logs['lr']:.2e}  time={elapsed:.1f}s  "
            f"val_loss={logs['val_loss']:.2f}"
        )

        if beta >= beta_target:
            if early_stopping.on_epoch_end(epoch, logs, model):
                print(
                    f"\nearly stopping at epoch {epoch} "
                    f"(no improvement > {early_stopping_min_delta} "
                    f"for {early_stopping_patience} epochs)"
                )
                break

    early_stopping.restore(model)
    if early_stopping.best_value is not None:
        print(f"restored best model weights (val_loss={early_stopping.restore_best_weights:.2f})")

    return history