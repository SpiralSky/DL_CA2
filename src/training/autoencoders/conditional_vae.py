from pathlib import Path

import torch

from src.datasets.cifar10 import get_dataset
from src.datasets.cifar10 import get_dataloaders
from src.models.autoencoders.model_factory import conditional_vae


def train_conditional_vae(data_path: Path):
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path)
    dataset = get_dataset(data_path=data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    model = conditional_vae(len(dataset.classes), in_channels=3, base_channels=32, latent_dim=128).to(device)

    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=100,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4,
        scheduler_patience=5,
        scheduler_factor=0.5,
        early_stopping_patience=10,
        early_stopping_min_delta=0.0,
    )

    print(f"\nDone. Trained {len(history)} epochs.")
