from pathlib import Path

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from src.models.util.ModelHistory import ModelHistory
from src.datasets.cifar10 import get_dataloaders, get_dataset
from src.models.autoencoders.VAE import VAE
from src.models.autoencoders.inspection.latent_space import analyze_latent_space
from src.models.autoencoders.inspection.reconstructions import plot_reconstructions
from src.models.autoencoders.model_factory import basic_vae
from src.models.inspection.plot_gradients import plot_gradient_summary

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

def view_model_results(
    model: VAE,
    test_dataloader: DataLoader,
    class_names: list,
    history: ModelHistory
) -> Figure:
    dashboard = plt.figure(
        figsize=(18, 18),
        constrained_layout=True,
    )

    gs = dashboard.add_gridspec(
        4,
        1,
        height_ratios=[
            1.0,  # Reconstructions
            3.0,  # t-SNE
            3.2,  # Class samples
            1.6,  # Gradient flow
        ],
    )

    recon_subfig = dashboard.add_subfigure(gs[0])
    latent_subfig = dashboard.add_subfigure(gs[1])
    classes_subfig = dashboard.add_subfigure(gs[2])
    gradient_subfig = dashboard.add_subfigure(gs[3])

    plot_reconstructions(
        model,
        test_dataloader,
        fig=recon_subfig,
    )

    analyze_latent_space(
        model,
        test_dataloader,
        class_names=class_names,
        fig=latent_subfig,
    )

    model.plot_class_samples(
        test_dataloader,
        fig=classes_subfig,
        class_names=class_names
    )

    plot_gradient_summary(
        history=history,
        fig=gradient_subfig,
    )

    dashboard.canvas.draw()

    return dashboard

def train_and_analysis(data_path: Path) -> None:
    model, test_dataloader, class_names, history = train_base_vae(data_path)
    view_model_results(model, test_dataloader, class_names, history)



