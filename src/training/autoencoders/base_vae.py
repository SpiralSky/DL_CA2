from pathlib import Path

import torch
from src.datasets.cifar10 import get_dataloaders
from src.models.autoencoders.model_factory import basic_vae
from src.models.autoencoders.inspection.reconstructions import plot_reconstructions

config = {
    "recon_loss_type": "mse",
    "free_bits": 0.0,
    "grad_clip_norm": 1.0,
}

def train_base_vae(data_path: Path):
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = basic_vae(latent_dim=128).to(device)


    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=100,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4
    )

    plot_reconstructions(model, test_dataloader)

    print(f"\nDone. Trained {len(history)} epochs.")
