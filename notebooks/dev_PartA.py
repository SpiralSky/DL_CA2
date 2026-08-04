# %%
# %load_ext magics.magics

# %% [markdown]
# # Part A
#

# %% [markdown]
# ### Imports
# Pytorch and torchvision is used for this module
#
# There are also some miscellanous imports for typing, etc.

# %%
# <$IMPORTS>
from pathlib import Path
from torch.utils.data import DataLoader

# %% [markdown]
# ### Configuration
# - PROJECT_ROOT: Root of project. Note that the data folder is expected to be `PROJECT_ROOT/data/`

# %%
PROJECT_ROOT = Path.cwd().parent

# %% [markdown]
# ## 0. Loading Data
# `get_dataset` is a simple

# %%
# %%load_clean
from src.datasets.cifar10 import get_dataset  # noqa: F401

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
from src.analysis.cifar10.display_images import display_images  # noqa: F401

# %%
display_images(eda_dataloader, (10, 5))

# %% [markdown]
# ### 1.2. Pixel-based image analysis

# %%
# %%load_clean
from src.analysis.analyse_images import run_class_eda # noqa: F401

# %%
_ = run_class_eda(eda_dataloader, cifar_10_dataset.classes)

# %% [markdown]
# ## 2. Models

# %% [markdown]
# ### 2.1. Splitting Data
# Pytorch's `DataLoader` class is used to split data into **80/15/15** distributions amongst train, test and split.
#
# Test set is withheld during experiments.

# %%
# %%load_clean
from src.datasets.cifar10 import get_dataloaders  # noqa: F401

# %% [markdown]
# ### 2.2. Basic Encoder and Decoder
# We first used a basic Encoder and Decoder setup.
#
# For the Encoder:
# - A Convolutional Layers are used

# %%
# %%load_clean
from src.models.autoencoders.encoders.basic_encoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_2d36 = [
    BasicEncoder,
]

# %%
# %%load_clean
from src.models.autoencoders.decoders.basic_decoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_958e = [
    BasicDecoder,
]

# %%
# %%load_clean
from src.models.autoencoders.model_factory import basic_vae # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.AbstractVAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_8162 = [
    AbstractVAE,
    VAETrainConfig,
]

# %%
# %%load_clean
from src.models.autoencoders.VAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_6760 = [
    VAE,
]

# %%
# %%load_clean
from src.models.autoencoders.inspection.reconstructions import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_7b4b = [
    plot_reconstructions,
]

# %%
# %%load_clean
from src.training import *  # noqa: F401

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_base_vae  # noqa: F401

# %%
train_base_vae(PROJECT_ROOT.joinpath("data"))

# %%
# %%load_clean
from src.training.autoencoders.conditional_vae import train_conditional_vae  # noqa: F401

# %%
train_conditional_vae(PROJECT_ROOT.joinpath("data"))
