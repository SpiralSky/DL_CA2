# %%
# %load_ext magics.magics

# %% [markdown]
# # Part A
#

# %% [markdown]
# ### Imports

import os
# %%
# <$IMPORTS>
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# %% [markdown]
# ### Configuration
# - PROJECT_ROOT: Root of project. Note that the data folder is expected to be `PROJECT_ROOT/data/`

# %%
PROJECT_ROOT = Path.cwd().parent

# %% [markdown]
# ## 0. Loading Data

# %%
# %%load_clean
from src.datasets.cifar10 import get_dataset # noqa: F401

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
from src.analysis.cifar10.display_images import display_images # noqa: F401

# %%
display_images(eda_dataloader, (10, 5))

# %% [markdown]
# ### EDA

# %%
# %%load_clean
from src.analysis.analyse_images import run_class_eda # noqa: F401

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
from src.models.autoencoders.decoders.decoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_f68f = [
    BasicDecoder,
]

# %%
# %%load_clean
from src.models.autoencoders.encoders.encoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_ec6f = [
    BasicEncoder,
]

# %%
# %%load_clean
from src.models.autoencoders.models.VAE import VAE # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.model_factory import basic_autoencoder # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.losses import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_6c08 = [
    VAELossOutput,
    kl_divergence,
    reconstruction_loss,
    vae_loss,
]

# %%
# %%load_clean
from src.models.autoencoders.inspection.reconstructions import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_7b4b = [
    plot_reconstructions,
]

# %%
# %%load_clean
from src.models.training.callbacks import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_0936 = [
    Callback,
    EarlyStopping,
]

# %%
# %%load_clean
from src.models.training.trainer import Trainer, TrainConfig # noqa: F401

# %%
# ─── MINIMAL TRAINER TEST ───

import torch

# You already have loaded:
#   TrainConfig, Trainer, basic_autoencoder, vae_loss, cifar_10_dataset
#   train_data_loader, val_data_loader (from your split above)

# 1. Config (short run for quick test)
config = TrainConfig(
    lr=1e-3,
    max_epochs=100,           # small for speed
    warmup_epochs=1,
    beta_target=1.0,
    recon_loss_type="mse",
    free_bits=0.0,
    grad_clip_norm=1.0,
    scheduler_patience=5,   # won't trigger with 3 epochs
    scheduler_factor=0.5,
    early_stopping_patience=10,
    early_stopping_min_delta=0.0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = basic_autoencoder(in_channels=3, base_channels=32, latent_dim=128).to(device)

trainer = Trainer(config=config, device=device)
history = trainer.fit(model, train_data_loader, val_data_loader)

print(f"\n✅ Done. Trained {len(history)} epochs.")
