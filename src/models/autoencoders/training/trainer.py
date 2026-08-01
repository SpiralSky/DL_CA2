from typing import TypedDict, Optional
import time
import torch
from torch.utils.data import DataLoader

from models.autoencoders.losses import vae_loss
from models.autoencoders.training.callbacks import EarlyStopping


class TrainConfig(TypedDict):
    lr: float
    max_epochs: int
    warmup_epochs: int
    beta_target: float
    recon_loss_type: str
    free_bits: float
    grad_clip_norm: float
    scheduler_patience: int
    scheduler_factor: float
    early_stopping_patience: int
    early_stopping_min_delta: float


def beta_schedule(epoch: int, warmup_epochs: int, beta_target: float) -> float:
    if warmup_epochs <= 0:
        return beta_target
    return min(beta_target, beta_target * epoch / warmup_epochs)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
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
            losses = vae_loss(
                recon,
                images,
                mu,
                logvar,
                beta=beta,
                recon_loss_type=config["recon_loss_type"],
                free_bits=config["free_bits"],
            )

            if train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config["grad_clip_norm"]
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


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    early_stopping: EarlyStopping | None = None,
    run_epoch_fn = run_epoch,
    beta_schedule_fn = beta_schedule,
) -> list[dict]:
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config["scheduler_patience"],
            factor=config["scheduler_factor"],
        )

    if early_stopping is None:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            min_delta=config["early_stopping_min_delta"],
        )

    history = []
    for epoch in range(1, config["max_epochs"] + 1):
        beta = beta_schedule_fn(epoch, config["warmup_epochs"], config["beta_target"])
        start = time.time()

        train_metrics = run_epoch_fn(
            model, train_loader, device, optimizer, config, beta, train=True
        )
        val_metrics = run_epoch_fn(
            model, val_loader, device, optimizer, config, beta, train=False
        )
        scheduler.step(val_metrics["total"])
        elapsed = time.time() - start

        logs = {
            "epoch": epoch,
            "max_epochs": config["max_epochs"],
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

        if beta >= config["beta_target"]:
            if early_stopping.on_epoch_end(epoch, logs, model):
                print(
                    f"\nearly stopping at epoch {epoch} "
                    f"(no improvement > {config['early_stopping_min_delta']} "
                    f"for {config['early_stopping_patience']} epochs)"
                )
                break

    early_stopping.restore(model)
    if early_stopping.best_state is not None:
        print(f"restored best model weights (val_loss={early_stopping.best:.2f})")

    return history
