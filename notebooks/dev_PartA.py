# %%
import os
# %load_ext magics.magics

# %% [markdown]
# # Part A
#

# %% [markdown]
# ### Imports

# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision
from matplotlib import pyplot as plt, ticker as mticker
from skimage.filters import laplace
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# %% [markdown]
# ### Configuration
# - PROJECT_ROOT: Root of project. Note that the data folder is expected to be `PROJECT_ROOT/data/`

# %%
PROJECT_ROOT = Path.cwd().parent

# %% [markdown]
# ## 0. Loading Data

# %%
# %%load_clean
from src.datasets.cifar10 import get_dataset #noqa

def get_dataset(data_path: Path, train: bool = True, transform=None) -> Dataset:
    """
    Loads the CIFAR-10 dataset.
    :param data_path: Path object to the data folder containing the dataset.
    :param train: If true, loads the training set, else loads the test set.
    :param transform: A function that takes in a PIL image and returns a transformed verison.
    :return:
    """
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.CIFAR10(
        root=data_path, train=train, download=True, transform=transform
    )



# %%
cifar_10_dataset = get_dataset(PROJECT_ROOT / "data")

# %% [markdown]
# ## 1. Exploratory Data Analysis

# %%
eda_dataloader = DataLoader(cifar_10_dataset, batch_size=256, shuffle=False)

# %% [markdown]
# ### 1.1. Displaying Images
# Images are displayed in an 32x32 Grid.
#
# Here, we can see that images are polychromatic, with contrast of bright and dark images. They also seem to be quite saturated.

# %%
# %%load_clean
import src.analysis.cifar10.display_images #noqa

def display_images(dataloader: DataLoader, dim: tuple[int, int]) -> None:
    """
    Displays images in a grid.
    :param dataloader: Pytorch DataLoader object.
    :param dim: Dimensions of the grid, in the format: (rows, columns)
    :return:
    """

    dataiter = iter(dataloader)
    images = []

    while len(images) < (dim[0] * dim[1]):
        batch_images, _ = next(dataiter)
        for img in batch_images:
            images.append(img)

    fig, ax = plt.subplots(dim[0], dim[1], figsize=(dim[1] * 2, dim[0] * 2))
    axes = ax.flatten()

    for i in range(dim[0] * dim[1]):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



# %%
display_images(eda_dataloader, (10, 5))

# %% [markdown]
# ### EDA

# %%
# %%load_clean
import src.analysis.analyse_images #noqa

"""
Per-class image statistics for datasets served via a PyTorch DataLoader.

Works for any (image, label) DataLoader where images are equal-sized tensors
(e.g. CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST). No dataset-specific code:
only `num_classes` and `class_names` need to change per dataset.
"""


def accumulate_pixel_statistics(dataloader, num_classes):
    """
    Single pass over the DataLoader, accumulating running sums needed for
    per-class mean/std, without holding the full dataset in memory.

    Returns a dict of tensors, each indexed by class:
        count:      (num_classes,)          number of images per class
        sum_:       (num_classes, C)        running per-channel sum
        sumsq:      (num_classes, C)        running per-channel sum of squares
        brightness_sum:    (num_classes,)   running sum of per-image mean brightness
        brightness_sumsq:  (num_classes,)   running sum of squared per-image brightness
    """
    images, _ = next(iter(dataloader))
    num_channels = images.shape[1]
    pixels_per_image = images.shape[2] * images.shape[3]

    count = torch.zeros(num_classes, dtype=torch.float64)
    sum_ = torch.zeros(num_classes, num_channels, dtype=torch.float64)
    sumsq = torch.zeros(num_classes, num_channels, dtype=torch.float64)
    brightness_sum = torch.zeros(num_classes, dtype=torch.float64)
    brightness_sumsq = torch.zeros(num_classes, dtype=torch.float64)

    for images, labels in dataloader:
        images = images.double()
        for c in labels.unique():
            mask = labels == c
            imgs_c = images[mask]
            count[c] += imgs_c.shape[0]
            sum_[c] += imgs_c.sum(dim=[0, 2, 3])
            sumsq[c] += (imgs_c ** 2).sum(dim=[0, 2, 3])
            per_image_brightness = imgs_c.mean(dim=[1, 2, 3])
            brightness_sum[c] += per_image_brightness.sum()
            brightness_sumsq[c] += (per_image_brightness ** 2).sum()

    return {
        "count": count,
        "pixels_per_image": pixels_per_image,
        "sum_": sum_,
        "sumsq": sumsq,
        "brightness_sum": brightness_sum,
        "brightness_sumsq": brightness_sumsq,
    }


def compute_channel_mean_std(accum):
    """
    Convert accumulated sums into per-class, per-channel mean and std.
    Returns two (num_classes, C) tensors.
    """
    n_pixels = accum["count"].unsqueeze(1) * accum["pixels_per_image"]
    mean = accum["sum_"] / n_pixels
    var = accum["sumsq"] / n_pixels - mean ** 2
    std = var.clamp(min=0).sqrt()
    return mean, std


def compute_brightness_mean_std(accum):
    """
    Convert accumulated brightness sums into per-class mean and std.
    Returns two (num_classes,) tensors.
    """
    n = accum["count"]
    mean = accum["brightness_sum"] / n
    var = accum["brightness_sumsq"] / n - mean ** 2
    std = var.clamp(min=0).sqrt()
    return mean, std


def compute_texture_scores(dataloader, num_classes, samples_per_class=200):
    """
    Estimate per-class texture/edge density using Laplacian variance on a
    grayscale-averaged version of each image. Subsamples for speed since this
    runs on CPU via scikit-image rather than as a batched tensor op.
    Returns a (num_classes,) tensor of mean edge variance per class.
    """
    collected = {c: [] for c in range(num_classes)}
    remaining = {c: samples_per_class for c in range(num_classes)}

    for images, labels in dataloader:
        if all(v == 0 for v in remaining.values()):
            break
        for img, label in zip(images, labels):
            c = label.item()
            if remaining[c] > 0:
                gray = img.mean(dim=0).numpy()
                collected[c].append(laplace(gray).var())
                remaining[c] -= 1

    scores = torch.tensor([np.mean(collected[c]) for c in range(num_classes)])
    return scores


def build_stats_dataframe(class_names, count, channel_mean, brightness_mean,
                           brightness_std, texture_scores=None):
    """
    Assembles per-class statistics into a pandas DataFrame, one row per class.
    Kept separate from display logic so the raw table can also be inspected,
    filtered, or exported (e.g. df.to_csv()) without touching styling code.
    """
    data = {
        "n": count.numpy().astype(int),
        "mean_R": channel_mean[:, 0].numpy(),
        "mean_G": channel_mean[:, 1].numpy(),
        "mean_B": channel_mean[:, 2].numpy(),
        "brightness_mean": brightness_mean.numpy(),
        "brightness_std": brightness_std.numpy(),
    }
    if texture_scores is not None:
        data["texture"] = texture_scores.numpy()

    return pd.DataFrame(data, index=pd.Index(class_names, name="class"))


def is_notebook_environment():
    """
    Detects whether code is running inside a Jupyter kernel, so display
    logic can choose between rich HTML output and a plain-text fallback.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def style_stats_dataframe(df):
    """
    Applies conditional formatting: color gradients on the RGB/brightness
    columns so intensity differences are visible at a glance, and an inline
    bar chart on the texture column so relative magnitude reads instantly.
    """
    mean_columns = [c for c in df.columns if c.startswith("mean_")]
    styler = (
        df.style
        .format(precision=3)
        .background_gradient(subset=mean_columns, cmap="coolwarm")
        .background_gradient(subset=["brightness_mean"], cmap="binary")
    )
    if "texture" in df.columns:
        styler = styler.bar(subset=["texture"], color="#5fba7d")
    return styler


def display_class_statistics(df):
    """
    Renders the stats table as a styled HTML table in Jupyter, or a plain
    aligned text table otherwise (e.g. running as a plain .py script).
    """
    if is_notebook_environment():
        display(style_stats_dataframe(df))
    else:
        print(df.to_string(float_format=lambda x: f"{x:.3f}"))


def plot_class_balance(class_names, count, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(class_names, count.numpy())
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Images per class")

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_channel_means(class_names, channel_mean, channel_names=("R", "G", "B"), ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(class_names))
    for i, ch in enumerate(channel_names):
        ax.plot(x, channel_mean[:, i].numpy(), marker="o", label=ch, color=ch.lower())
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.set_title("Mean channel intensity per class")

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_texture_scores(class_names, texture_scores, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    order = torch.argsort(texture_scores, descending=True)
    ax.bar([class_names[i] for i in order], texture_scores[order].numpy())
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Texture (edge variance) per class")

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_class_eda_dashboard(class_names, count, channel_mean, texture_scores=None):
    """
    Docks the individual plots into a single composite figure. Texture scores
    (which vary meaningfully per class) take the tall left spot; channel means
    and class balance (constant across classes, so demoted) are stacked on
    the right. Each plot function is unchanged and just handed an ax to draw
    on instead of creating its own figure.
    """
    if texture_scores is None:
        fig, (ax_means, ax_balance) = plt.subplots(1, 2, figsize=(12, 5))
        plot_channel_means(class_names, channel_mean, ax=ax_means)
        plot_class_balance(class_names, count, ax=ax_balance)
        for ax in (ax_means, ax_balance):
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        fig.tight_layout()
        plt.show()
        return

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 2)

    ax_texture = fig.add_subplot(gs[:, 0])
    plot_texture_scores(class_names, texture_scores, ax=ax_texture)

    ax_means = fig.add_subplot(gs[0, 1])
    plot_channel_means(class_names, channel_mean, ax=ax_means)

    ax_balance = fig.add_subplot(gs[1, 1])
    plot_class_balance(class_names, count, ax=ax_balance)

    for ax in (ax_means, ax_balance):
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))

    fig.tight_layout()
    plt.show()


def run_class_eda(dataloader, class_names, compute_texture=True, samples_per_class=200):
    """
    Orchestrator: runs the full per-class EDA pipeline, prints statistics,
    and displays plots docked into a single dashboard figure.
    """
    num_classes = len(class_names)

    accum = accumulate_pixel_statistics(dataloader, num_classes)
    channel_mean, channel_std = compute_channel_mean_std(accum)
    brightness_mean, brightness_std = compute_brightness_mean_std(accum)

    texture_scores = None
    if compute_texture:
        texture_scores = compute_texture_scores(dataloader, num_classes, samples_per_class)

    df = build_stats_dataframe(class_names, accum["count"], channel_mean,
                                brightness_mean, brightness_std, texture_scores)
    display_class_statistics(df)

    plot_class_eda_dashboard(class_names, accum["count"], channel_mean, texture_scores)

    return {
        "stats_df": df,
        "count": accum["count"],
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "texture_scores": texture_scores,
    }



# %%
_ = run_class_eda(eda_dataloader, cifar_10_dataset.classes)

# %% [markdown]
# ### 1. Train/Test Split

# %%
train_data, validation_data = torch.utils.data.random_split(cifar_10_dataset, [0.8, 0.2])
num_workers = min(2, os.cpu_count())

train_data_loader = DataLoader(
    train_data, batch_size=256, shuffle=True,
    num_workers=num_workers, pin_memory=True, persistent_workers=True,
)
val_data_loader = DataLoader(
    validation_data, batch_size=256, shuffle=False,
    num_workers=num_workers, pin_memory=True, persistent_workers=True,
)

# %%
# %%load_clean
import src.models.autoencoders.decoders.decoder #noqa

class BasicDecoder(nn.Module):
    """
    Convolutional decoder mirroring ConvEncoder. Maps a latent vector back
    to a 32x32 RGB image via upsampling transposed convs: 4 -> 8 -> 16 -> 32,
    with a stride-1 refinement conv at each resolution (mirroring the
    encoder) so shape/structure can be reconstructed with more capacity than
    a single upsampling conv provides. Output is passed through sigmoid to
    match [0, 1]-scaled ToTensor() inputs.
    """

    def __init__(self, out_channels=3, base_channels=32, latent_dim=128):
        super().__init__()
        self.base_channels = base_channels
        self.init_spatial = 4
        self.fc = nn.Linear(latent_dim, base_channels * 4 * self.init_spatial * self.init_spatial)

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.deconv = nn.Sequential(
            up_block(base_channels * 4, base_channels * 2),
            up_block(base_channels * 2, base_channels),
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 4, self.init_spatial, self.init_spatial)
        return self.deconv(h)



# %%
# %%load_clean
import src.models.autoencoders.encoders.encoder #noqa

class BasicEncoder(nn.Module):
    """
    Convolutional encoder for 32x32 RGB images (e.g. CIFAR-10).
    Maps an image to the parameters (mu, logvar) of a diagonal Gaussian
    over the latent space. Downsamples 32 -> 16 -> 8 -> 4 via stride-2 convs,
    with a stride-1 refinement conv at each resolution so the network has
    capacity to learn shape/structure features before compressing further,
    rather than immediately squeezing spatial detail into the bottleneck.
    """

    def __init__(self, in_channels=3, base_channels=32, latent_dim=128):
        super().__init__()

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            down_block(in_channels, base_channels),
            down_block(base_channels, base_channels * 2),
            down_block(base_channels * 2, base_channels * 4),
        )
        self.flatten_dim = base_channels * 4 * 4 * 4  # channels * H * W at 4x4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar



# %%
# %%load_clean
import src.models.autoencoders.factory #noqa

def build_baseline_vae(in_channels=3, base_channels=32, latent_dim=128):
    """
    Assembles the baseline autoencoders from its component modules. Switching to a
    different encoder/decoder implementation later only means changing what
    gets constructed here (or adding an entry to MODEL_REGISTRY below) --
    everything downstream (training loop, loss function) is unaffected.
    """
    encoder = BasicEncoder(in_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim)
    decoder = BasicDecoder(out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim)
    return VAE(encoder=encoder, decoder=decoder, latent_dim=latent_dim)


MODEL_REGISTRY = {
    "baseline_vae": build_baseline_vae,
}


def build_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)



# %%
# %%load_clean
import src.models.autoencoders.losses

def vae_loss(recon_x, x, mu, logvar, beta=1.0, recon_loss_type="mse"):
    """
    Standard autoencoders loss: reconstruction term plus a beta-weighted KL divergence
    between the approximate posterior N(mu, sigma^2) and the standard normal
    prior N(0, I).

    recon_loss_type:
        "mse"  - appropriate for continuous natural-image pixels (default,
                 recommended for CIFAR-10-like photographic data)
        "bce"  - appropriate for near-binary pixel data (e.g. MNIST); assumes
                 decoder output is sigmoid-bounded to [0, 1]

    Returns individual terms too, since watching them separately during
    training reveals issues (e.g. posterior collapse) that the summed loss
    alone would hide.
    """
    if recon_loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.shape[0]
    elif recon_loss_type == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.shape[0]
    else:
        raise ValueError(f"Unknown recon_loss_type '{recon_loss_type}', expected 'mse' or 'bce'")

    kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.shape[0]
    total_loss = recon_loss + beta * kl_div
    return {
        "total": total_loss,
        "reconstruction": recon_loss,
        "kl_divergence": kl_div,
    }



# %%
# %%load_clean
import src.models.autoencoders.model

class VAE(nn.Module):
    """
    Generic autoencoders shell: takes an encoder and decoder as injected nn.Module
    instances rather than hardcoding architecture. Swapping in a different
    encoder/decoder (deeper convs, ResNet blocks, conditional variants that
    also accept a label) requires no changes here.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, num_samples: int, device=None):
        """
        Draws num_samples latent vectors from the prior N(0, I) and decodes
        them, for inspecting what the model has learned to generate.
        """
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)



# %%
@torch.no_grad()
def plot_reconstructions(model, data_loader, device=None, num_images=8):
    """
    Draws one batch from data_loader, runs it through the model, and plots
    original vs. reconstructed images side by side (originals on top row,
    reconstructions on bottom row). This catches issues loss curves alone
    can hide, e.g. whether the model has collapsed to near-identical blurry
    outputs regardless of input.
    """
    device = device or next(model.parameters()).device
    model.eval()

    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    recon, _, _ = model(images)

    images_np = images.cpu().permute(0, 2, 3, 1).numpy()
    recon_np = recon.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3.5))
    for i in range(num_images):
        axes[0, i].imshow(images_np[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_np[i].clip(0, 1))
        axes[1, i].axis("off")

    fig.text(0.02, 0.75, "original", rotation=90, va="center", fontsize=10)
    fig.text(0.02, 0.25, "reconstructed", rotation=90, va="center", fontsize=10)
    fig.tight_layout(rect=[0.04, 0, 1, 1])
    plt.show()


# %%
import copy
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# --- tunable knobs for this run ---
latent_dim = 256
base_channels = 64
max_epochs = 300          # upper cap; early stopping will likely end training sooner
warmup_epochs = 20
lr = 3e-3                 # raised from 1e-3 -- batch_size=256 supports a higher LR
recon_loss_type = "mse"

early_stopping_patience = 15     # epochs to wait for improvement before stopping
early_stopping_min_delta = 0.01  # minimum change in val_loss to count as improvement

model = build_baseline_vae(latent_dim=latent_dim, base_channels=base_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

best_val_loss = float("inf")
best_model_state = None
epochs_without_improvement = 0
epoch_times = []

for epoch in range(1, max_epochs + 1):
    beta = min(1.0, epoch / warmup_epochs) if warmup_epochs > 0 else 1.0
    start = time.time()

    model.train()
    total_loss, total_recon, total_kl, num_batches = 0.0, 0.0, 0.0, 0
    for images, _ in train_data_loader:
        images = images.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(images)
        losses = vae_loss(recon, images, mu, logvar, beta=beta, recon_loss_type=recon_loss_type)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += losses["total"].item()
        total_recon += losses["reconstruction"].item()
        total_kl += losses["kl_divergence"].item()
        num_batches += 1

    train_loss = total_loss / num_batches
    train_recon = total_recon / num_batches
    train_kl = total_kl / num_batches
    elapsed = time.time() - start
    epoch_times.append(elapsed)

    model.eval()
    val_total, val_batches = 0.0, 0
    with torch.no_grad():
        for images, _ in val_data_loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            losses = vae_loss(recon, images, mu, logvar, beta=beta, recon_loss_type=recon_loss_type)
            val_total += losses["total"].item()
            val_batches += 1
    val_loss = val_total / val_batches
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch}/{max_epochs}  loss={train_loss:.2f}  recon={train_recon:.2f}  "
          f"kl={train_kl:.2f}  beta={beta:.3f}  lr={current_lr:.2e}  "
          f"time={elapsed:.1f}s  val_loss={val_loss:.2f}")

    # Only start counting early-stopping patience once beta has fully warmed up --
    # val_loss is expected to rise/shift during warmup regardless of model quality,
    # so comparing against "best" before beta stabilizes would trigger prematurely.
    if beta >= 1.0:
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nearly stopping at epoch {epoch} "
                  f"(no improvement > {early_stopping_min_delta} for {early_stopping_patience} epochs)")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"restored best model weights (val_loss={best_val_loss:.2f})")

avg_epoch_time = sum(epoch_times) / len(epoch_times)
print(f"\naverage epoch time: {avg_epoch_time:.1f}s")
print(f"total epochs run: {len(epoch_times)}")

plot_reconstructions(model, val_data_loader)
