from pathlib import Path

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from src.models.autoencoders.inspection.training_history import plot_reconstruction_history, plot_kl_history, plot_kl_weight
from src.models.util.ModelHistory import ModelHistory
from src.datasets.cifar10 import get_dataloaders, get_dataset
from src.models.autoencoders.VAE import VAE
from src.models.autoencoders.inspection.latent_space import analyze_latent_space, plot_latent_utilization
from src.models.autoencoders.inspection.reconstructions import plot_reconstructions
from src.models.autoencoders.model_factory import basic_vae, improved_basic_vae
from src.training.autoencoders.sampling import plot_class_samples

config = {
    "recon_loss_type": "mse",
    "free_bits": 0.0,
    "grad_clip_norm": 1.0,
}

def train_base_vae(data_path: Path) -> tuple[VAE, DataLoader, list, ModelHistory]:
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

def train_improved_base_vae(data_path: Path):
    labels = get_dataset(data_path).classes
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_path, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = improved_basic_vae(latent_dim=64).to(device)

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

def view_model_results(
    model: VAE,
    test_dataloader: DataLoader,
    class_names: list[str],
    history: ModelHistory,
) -> Figure:

    dashboard = plt.figure(
        figsize=(18, 20),
        constrained_layout=True,
    )

    gs = dashboard.add_gridspec(
        5,
        1,
        height_ratios=[
            2.0,  # Reconstructions
            2.5,  # t-SNE
            3.2,  # Random samples
            2.0,  # Latent utilization
            2.2,  # Training statistics
        ],
    )

    recon_gs = gs[0].subgridspec(2, 8, hspace=0.05, wspace=0.05)
    recon_axes = recon_gs.subplots()

    plot_reconstructions(
        model,
        test_dataloader,
        axes=recon_axes,
    )

    latent_ax = dashboard.add_subplot(gs[1])

    analyze_latent_space(
        model,
        test_dataloader,
        class_names=class_names,
        ax=latent_ax,
    )

    sample_axes = gs[2].subgridspec(
        2,
        len(class_names),
        wspace=0.05,
        hspace=0.05,
    ).subplots(squeeze=False)

    plot_class_samples(
        model,
        test_dataloader,
        class_names=class_names,
        axes=sample_axes,
    )

    utilization_ax = dashboard.add_subplot(gs[3])

    plot_latent_utilization(
        model=model,
        dataloader=test_dataloader,
        ax=utilization_ax,
    )

    history_gs = gs[4].subgridspec(
        1,
        3,
        wspace=0.35,
    )

    reconstruction_ax = dashboard.add_subplot(history_gs[0])
    kl_ax = dashboard.add_subplot(history_gs[1])
    weight_ax = dashboard.add_subplot(history_gs[2])

    plot_reconstruction_history(
        history=history,
        ax=reconstruction_ax,
    )

    plot_kl_history(
        history=history,
        ax=kl_ax,
    )

    plot_kl_weight(
        history=history,
        ax=weight_ax,
    )

    return dashboard

def train_and_analysis(data_path: Path) -> None:
    model, test_dataloader, class_names, history = train_base_vae(data_path)
    view_model_results(model, test_dataloader, class_names, history)



