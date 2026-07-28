import time

import torch

from .callbacks import EarlyStopping
from ..losses import vae_loss

def beta_schedule(epoch, warmup_epochs, beta_target):
    if warmup_epochs <= 0:
        return beta_target
    return min(beta_target, beta_target * epoch / warmup_epochs)


def run_epoch(model, loader, device, optimizer, config, beta, train):
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
                recon, images, mu, logvar,
                beta=beta,
                recon_loss_type=config.recon_loss_type,
                free_bits=config.free_bits,
            )

            if train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            num_batches += 1

    return {k: v / num_batches for k, v in totals.items()}


def fit(model, train_loader, val_loader, config, device):
    """
    Keras-style fit(): takes a TrainConfig instead of loose kwargs, and delegates
    stopping/checkpointing to a callback instead of inlining it in the loop.
    Returns the per-epoch history (list of dicts) for later plotting/inspection.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.scheduler_patience, factor=config.scheduler_factor
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    history = []
    for epoch in range(1, config.max_epochs + 1):
        beta = beta_schedule(epoch, config.warmup_epochs, config.beta_target)
        start = time.time()

        train_metrics = run_epoch(model, train_loader, device, optimizer, config, beta, train=True)
        val_metrics = run_epoch(model, val_loader, device, optimizer, config, beta, train=False)
        scheduler.step(val_metrics["total"])
        elapsed = time.time() - start

        logs = {
            "epoch": epoch,
            "beta": beta,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
            "loss": train_metrics["total"],
            "recon": train_metrics["reconstruction"],
            "kl": train_metrics["kl_divergence"],
            "val_loss": val_metrics["total"],
        }
        history.append(logs)
        print(f"epoch {epoch}/{config.max_epochs}  loss={logs['loss']:.2f}  recon={logs['recon']:.2f}  "
              f"kl={logs['kl']:.2f}  beta={beta:.3f}  lr={logs['lr']:.2e}  "
              f"time={elapsed:.1f}s  val_loss={logs['val_loss']:.2f}")

        # only start counting patience once beta has fully warmed up -- val_loss
        # is expected to shift during warmup regardless of model quality
        if beta >= config.beta_target:
            if early_stopping.on_epoch_end(epoch, logs, model):
                print(f"\nearly stopping at epoch {epoch} "
                      f"(no improvement > {config.early_stopping_min_delta} for {config.early_stopping_patience} epochs)")
                break

    early_stopping.restore(model)
    if early_stopping.best_state is not None:
        print(f"restored best model weights (val_loss={early_stopping.best:.2f})")

    return history