from pathlib import Path

import torch
from src.training.trainer import fit
from src.datasets.cifar10 import get_dataloaders
from src.models.autoencoders.model_factory import basic_autoencoder

config = {
    "recon_loss_type": "mse",
    "free_bits": 0.0,
    "grad_clip_norm": 1.0,
}

def train_base_vae(data_path: Path):
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = basic_autoencoder(in_channels=3, base_channels=32, latent_dim=128).to(device)

    history = fit(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=100,
        lr=1e-3,
        grad_clip_norm=1.0,
        warmup_epochs=1,
        beta_target=1.0,
        recon_loss_type="mse",
        free_bits=0.0,
        scheduler_patience=5,
        scheduler_factor=0.5,
        early_stopping_patience=10,
        early_stopping_min_delta=0.0,
    )
    print(f"\nDone. Trained {len(history)} epochs.")
