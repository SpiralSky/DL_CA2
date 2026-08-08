from pathlib import Path

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from src.models.util.TrainingHistory import TrainingHistory
from src.datasets.cifar10 import get_dataloaders, get_dataset
from src.models.autoencoders.VAE import VAE
from src.models.autoencoders.inspection.latent_space import analyze_latent_space, plot_kl_per_dim
from src.models.autoencoders.inspection.reconstructions import plot_reconstructions
from src.models.autoencoders.model_factory import basic_vae, improved_basic_vae
from src.training.autoencoders.sampling import plot_class_samples

config = {
    "recon_loss_type": "mse",
    "free_bits": 0.0,
    "grad_clip_norm": 1.0,
}

def train_base_vae(data_path: Path) -> tuple[VAE, DataLoader, list, TrainingHistory]:
    """
    Function to train base VAE.
    :param data_path: Path to folder containing data
    :return: Tuple of: Model, DataLoader, Class Labels, Training History
    """
    labels = get_dataset(data_path).classes
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = basic_vae(latent_dim=128).to(device)

    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=300,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=30
    )

    print(f"\nDone. Trained {len(history)} epochs.")

    return model, test_dataloader, labels, history

def train_base_vae_reduced(data_path: Path) -> tuple[VAE, DataLoader, list, TrainingHistory]:
    """
    Function to train base VAE (reduced latent dim size).
    :param data_path: Path to folder containing data
    :return: Tuple of: Model, DataLoader, Class Labels, Training History
    """
    labels = get_dataset(data_path).classes
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = basic_vae(latent_dim=64).to(device)

    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=300,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=30
    )

    print(f"\nDone. Trained {len(history)} epochs.")

    return model, test_dataloader, labels, history

def train_improved_base_vae(data_path: Path):
    """
    Function to train improved base VAE
    :param data_path:
    :return:
    """
    labels = get_dataset(data_path).classes
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = improved_basic_vae(latent_dim=128).to(device)

    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=300,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=30
    )

    print(f"\nDone. Trained {len(history)} epochs.")

    return model, test_dataloader, labels, history

