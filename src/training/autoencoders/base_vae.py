from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.autoencoders.VAE import VAE
from src.models.inspection.plot_gradients import plot_gradient_flow
from src.models.autoencoders.inspection.latent_space import analyze_latent_space
from src.datasets.cifar10 import get_dataloaders, get_dataset
from src.models.autoencoders.model_factory import basic_vae
from src.models.autoencoders.inspection.reconstructions import plot_reconstructions

config = {
    "recon_loss_type": "mse",
    "free_bits": 0.0,
    "grad_clip_norm": 1.0,
}

def train_base_vae(data_path: Path) -> tuple[VAE, DataLoader, list]:
    labels = get_dataset(data_path).classes
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path, batch_size=256)

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

    print(f"\nDone. Trained {len(history)} epochs.")

    return model, test_dataloader, labels

def view_model_results(model: VAE, test_dataloader: DataLoader, class_names: list) -> None:
    plot_reconstructions(model, test_dataloader)
    analyze_latent_space(model, test_dataloader, class_names=class_names)
    plot_gradient_flow(model)
    model.plot_class_samples(test_dataloader)

def train_and_analysis(data_path: Path) -> None:
    model, test_dataloader, class_names = train_base_vae(data_path)
    view_model_results(model, test_dataloader, class_names)



