from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.cifar10 import get_dataset, get_dataloaders
from src.models.autoencoders.VAE import VAE
from src.models.autoencoders.model_factory import improved_basic_vae, basic_vae, residual_vae
from src.models.util.TrainingHistory import TrainingHistory
from src.training.callbacks.EarlyStopping import EarlyStopping
from src.training.save_state import load_checkpoint, save_checkpoint
from src.training.trainers.VAETrainer import VAETrainer


def print_history(
    history: TrainingHistory,
    *,
    loaded: bool = False,
) -> None:
    print(history)

    status = "Loaded checkpoint" if loaded else "Ended"

    print(
        f"\n[Training] {status} at {len(history)} epochs"
    )

def train_vae(
    model: VAE,
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False,
    transform=None,
) -> tuple[VAE, DataLoader, list, TrainingHistory]:
    """
    Generic VAE training function.

    :param model: VAE model to train.
    :param data_path: Path to dataset.
    :param checkpoint_path: Optional checkpoint path.
    :param override: Whether to ignore an existing checkpoint.
    :param transform: Optional transform applied to training images.
    :return: Model, test DataLoader, class labels and training history.
    """

    labels = get_dataset(data_path).classes

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        data_path,
        batch_size=256,
        transform=transform,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = model.to(device)

    if checkpoint_path and checkpoint_path.exists() and not override:
        history = load_checkpoint(model, checkpoint_path)
        print_history(history, loaded=True)
        return model, test_dataloader, labels, history

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    trainer = VAETrainer(
        model,
        optimizer,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=30,
        callbacks=[
            EarlyStopping(
                patience=30,
                start_epoch=30,
                min_delta=1e-4
            )
        ],
    )

    history = trainer.fit(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        max_epochs=300,
    )

    if checkpoint_path:
        save_checkpoint(model, history, checkpoint_path)


    return model, test_dataloader, labels, history


def train_base_vae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False
) -> tuple[VAE, DataLoader, list, TrainingHistory]:

    return train_vae(
        basic_vae(latent_dim=128),
        data_path,
        checkpoint_path,
        override=override
    )

def train_improved_base_vae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False
) -> tuple[VAE, DataLoader, list, TrainingHistory]:

    return train_vae(
        improved_basic_vae(latent_dim=128),
        data_path,
        checkpoint_path,
        override=override
    )

def train_res_vae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False
) -> tuple[VAE, DataLoader, list, TrainingHistory]:

    return train_vae(
        residual_vae(latent_dim=128),
        data_path,
        checkpoint_path,
        override=override
    )

def train_augmented_base_vae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False,
) -> tuple[VAE, DataLoader, list, TrainingHistory]:
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    return train_vae(
        basic_vae(latent_dim=128),
        data_path,
        checkpoint_path,
        override=override,
        transform=augmentation,
    )