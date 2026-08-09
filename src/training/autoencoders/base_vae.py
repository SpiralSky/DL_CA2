from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.cifar10 import get_dataloaders, get_dataset
from src.models.autoencoders.VAE import VAE
from src.models.autoencoders.model_factory import (
    basic_vae,
    improved_basic_vae,
    skip_vae,
)
from src.models.util.TrainingHistory import TrainingHistory
from src.training.save_state import load_checkpoint, save_checkpoint

# TODO Add docstring
def print_history(history: TrainingHistory, *, loaded: bool = False) -> None:
    print(history)
    status = "Loaded checkpoint" if loaded else "Ended"
    print(f"\n[Training] {status} at {len(history)} epochs")

def train_base_vae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False,
) -> tuple[VAE, DataLoader, list, TrainingHistory]:
    """
    Function to train base VAE
    :param data_path: Path to folder containing data
    :param checkpoint_path: Optional checkpoint path.
    :param override: If True, ignores existing checkpoint and overwrites after training.
    :return: Tuple of: Model, DataLoader, Class Labels, Training History
    """
    labels = get_dataset(data_path).classes
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        data_path,
        batch_size=256,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = basic_vae(latent_dim=128).to(device)

    if checkpoint_path and checkpoint_path.exists() and not override:
        history = load_checkpoint(model, checkpoint_path)
        print_history(history, loaded=True)
        return model, test_dataloader, labels, history

    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=300,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=30,
    )

    if checkpoint_path:
        save_checkpoint(model, history, checkpoint_path)

    print_history(history)

    return model, test_dataloader, labels, history


def train_improved_base_vae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False,
) -> tuple[VAE, DataLoader, list, TrainingHistory]:
    """
    Function to train improved base VAE
    :param data_path: Path to folder containing data
    :param checkpoint_path: Optional checkpoint path.
    :param override: If True, ignores existing checkpoint and overwrites after training.
    :return: Tuple of: Model, DataLoader, Class Labels, Training History
    """
    labels = get_dataset(data_path).classes
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        data_path,
        batch_size=256,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = improved_basic_vae(latent_dim=128).to(device)

    if checkpoint_path and checkpoint_path.exists() and not override:
        history = load_checkpoint(model, checkpoint_path)
        print_history(history, loaded=True)
        return model, test_dataloader, labels, history

    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=300,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=30,
    )

    if checkpoint_path:
        save_checkpoint(model, history, checkpoint_path)

    print_history(history)

    return model, test_dataloader, labels, history


def train_skip_vae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False,
) -> tuple[VAE, DataLoader, list, TrainingHistory]:
    """
    Function to train skip VAE
    :param data_path: Path to folder containing data
    :param checkpoint_path: Optional checkpoint path.
    :param override: If True, ignores existing checkpoint and overwrites after training.
    :return: Tuple of: Model, DataLoader, Class Labels, Training History
    """
    labels = get_dataset(data_path).classes
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        data_path,
        batch_size=256,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = skip_vae(latent_dim=128).to(device)

    if checkpoint_path and checkpoint_path.exists() and not override:
        history = load_checkpoint(model, checkpoint_path)
        print_history(history, loaded=True)
        return model, test_dataloader, labels, history

    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=300,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=30,
    )

    if checkpoint_path:
        save_checkpoint(model, history, checkpoint_path)

    print_history(history)

    return model, test_dataloader, labels, history