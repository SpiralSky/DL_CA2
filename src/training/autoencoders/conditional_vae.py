from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.autoencoders.VAE import VAE
from src.models.util.TrainingHistory import TrainingHistory
from src.datasets.cifar10 import get_dataset, get_dataloaders
from src.models.autoencoders.model_factory import beta_conditional_vae
from src.training.autoencoders.base_vae import print_history
from src.training.save_state import load_checkpoint, save_checkpoint
from training.callbacks import EarlyStopping


def train_bcvae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False,
) -> tuple[VAE, DataLoader, list, TrainingHistory]:
    """
    Function to train beta conditional VAE.
    :param data_path: Path to folder containing data.
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

    model = beta_conditional_vae(
        num_classes=len(labels),
        latent_dim=128,
    ).to(device)

    if checkpoint_path and checkpoint_path.exists() and not override:
        history = load_checkpoint(model, checkpoint_path)
        print_history(history, loaded=True)
        return model, test_dataloader, labels, history

    early_stopping = EarlyStopping(start_epoch=50)

    history = model.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=300,
        lr=3e-3,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=50,
        beta=4,
        early_stopping=early_stopping
    )

    if checkpoint_path:
        save_checkpoint(model, history, checkpoint_path)

    print_history(history)

    return model, test_dataloader, labels, history
