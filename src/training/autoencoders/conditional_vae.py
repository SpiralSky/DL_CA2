from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.cifar10 import get_dataset, get_dataloaders
from src.models.autoencoders.BetaConditionalVAE import BetaConditionalVAE
from src.models.autoencoders.model_factory import beta_conditional_vae
from src.models.util.TrainingHistory import TrainingHistory
from src.training.autoencoders.base_vae import print_history
from src.training.callbacks.EarlyStopping import EarlyStopping
from src.training.save_state import load_checkpoint, save_checkpoint
from src.training.trainers.BetaVAETrainer import BetaVAETrainer


def train_bcvae(
    data_path: Path,
    checkpoint_path: Path | None = None,
    *,
    override: bool = False,
) -> tuple[BetaConditionalVAE, DataLoader, list, TrainingHistory]:
    """
    Function to train beta conditional VAE.

    :param data_path: Path to folder containing data.
    :param checkpoint_path: Optional checkpoint path.
    :param override: If True, ignores existing checkpoint and overwrites after training.
    :return: Tuple of model, test dataloader, class labels, and training history.
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-3,
    )

    trainer = BetaVAETrainer(
        model,
        optimizer,
        beta=4,
        grad_clip_norm=1.0,
        free_bits=0.4,
        kl_warmup_epochs=50,
        callbacks=[
            EarlyStopping(start_epoch=50),
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

    print_history(history)

    return model, test_dataloader, labels, history