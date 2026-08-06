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
# Pytorch's `random_split` class is used to split data into **80/15/15** distributions amongst train, test and split.
# The classes are split into `DataLoaders`

# %%
# %%load_clean
from src.datasets.cifar10 import get_dataloaders  # noqa: F401

# %% [markdown]
# ### 2.2. Basic Encoder and Decoder
# We first used a basic Encoder and Decoder setup.

# %% [markdown]
# #### 2.2.1. Encoder Architecture
# For the Encoder Base:
# - The encoder takes in 32x32 image with 3 channels.
# - It features 3 down blocks which is a simple convolutional layer with batch normalization.
# - For each convolutional layer, `kernel_size=3` and `stride=2` are used which halves the image size.
# - Convolutional filter is turned to 64 filters then doubles in the next 2 layers to learn **deep features of the images**
# - `LeakyRelu` is used to prevent dead neurons
# - Flatten is used after the 3 down blocks. As Image dimensions halve per down block, reaching **4x4x256**, total tensor size is 4096, thus output vector length is 4096. A Linear layer then learns information and also reduces the size of the output to `latent_dim` which in this project is 256.
# - The encoder outputs `mean` and `log_variance` (Standard VAE), which is then used with reparameterization trick for the decoder input.

# %%
# %%load_clean
from src.models.autoencoders.encoders.basic_encoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_2d36 = [
    BasicEncoder,
]

# %% [markdown]
# #### 2.2.2. Decoder Architecture
# BasicDecoder takes in latent_vector of length `latent_dim` as an input.
#
# It has 3 convolution blocks:
# - Unless `upsample=False` (defaults to `True`), an `Upsample` layer is added which uses nearest neighbour (simple algorithm, duplicates pixels).
# - Upsample is used with `Conv2d` layers to double image spatial (width/height) dimensions per convolution block, while convolution learns features from it. `kernel_size=3` and `padding=1` ensures the convolution block keeps the image width/height of the same size while learning how to reduce channels while increasing spatial dimensions.
# - Channels decrease from 256 -> 64 -> 32 -> 3, halving each time, mirroring the encoder (except for 32 -> 3)

# %%
# %%load_clean
from src.models.autoencoders.decoders.basic_decoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_958e = [
    BasicDecoder,
]

# %% [markdown]
# ### ModelHistory class

# %%
# %%load_clean
from src.models.util.ModelHistory import ModelHistory  # noqa: F401

# %% [markdown]
# ### Callbacks

# %%
# %%load_clean
from src.training.callbacks import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_a022 = [
    Callback,
    EarlyStopping,
]

# %% [markdown]
# ### 2.3. AbstractVAE
# AbstractAVE is a basic pytorch module that provides methods such as:
# - `reparameterize`: Reparameterization trick (formula)
# - `reconstruction_loss`: bce/mse loss function from torch
# - `kl_divergence`: KL Divergence loss
# - `run_epoch` and `fit`: Basic Training Loop functionality

# %%
# %%load_clean
from src.models.autoencoders.AbstractVAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_8162 = [
    AbstractVAE,
    VAETrainConfig,
]

# %% [markdown]
# ### 2.4. VAE
# VAE simply extends AbstractVAE with simple encoder and decoder along with a basic `sample` method

# %%
# %%load_clean
from src.models.autoencoders.VAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_6760 = [
    VAE,
]

# %% [markdown]
# ### 2.4. Basic VAE
# This function returns a simple VAE using the encoders and decoders mentioned above.

# %%
# %%load_clean
from src.models.autoencoders.model_factory import basic_vae # noqa: F401

# %% [markdown]
# ## 3. Base VAE: Train & Analysis

# %% [markdown]
# ### 3.1. Training Base VAE
# `train_base_vae` is a basic training function to train and evaluate the VAE by running model.fit
#
# By default, batch size is 256 and lr is 3e-3 (can be changed for smaller gpu with less vram).
#
# It returns the model, the test dataloader and labels for analysis.

# %%
# %%load_clean
from src.training.autoencoders.base_vae import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_5c3c = [
    config,
    train_and_analysis,
    train_base_vae,
    view_model_results,
]

# %%
model, test_dataloader, labels, history = train_base_vae(PROJECT_ROOT.joinpath("data"))

# %% [markdown]
# ### 3.2. Analysis of Base VAE results

# %%
# %%load_clean
from src.models.autoencoders.inspection.latent_space import analyze_latent_space  # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.inspection.reconstructions import plot_reconstructions

# %%
# %%load_clean
from src.models.inspection.plot_gradients import plot_gradient_heatmap  # noqa: F401

# %%
# %%load_clea
from src.training.autoencoders.base_vae import view_model_results

# %%
_ = view_model_results(model, test_dataloader, labels, history)

# %%
# train_conditional_vae(PROJECT_ROOT.joinpath("data"))

# %%
# # %%load_clean
# from src.models.autoencoders.inspection.reconstructions import *  # noqa: F401
#
# _LOAD_CLEAN_IMPORTS_7b4b = [
#     plot_reconstructions,
# ]

# %%
# # %%load_clean
# from src.training.autoencoders.conditional_vae import train_conditional_vae  # noqa: F401
