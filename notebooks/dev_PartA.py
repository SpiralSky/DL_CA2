# %%
# %load_ext magics.magics

# %% [markdown]
# # PART A: GAN/VAE (45 marks)
# Apply some suitable GAN/VAE architectures to the problem of image generation. You can use either GAN or VAE, or both. Use the given dataset for CIFAR10 to create 1000 small images. There should be 10 classes of images for you to generate. You must submit your generated images (otherwise, there would be marks deduction).

# %% [markdown]
# ### Imports
# Pytorch and torchvision is used for this module.
#
# There are also some miscellanous imports for typing, etc.

# %%
# <$IMPORTS>
from pathlib import Path

from torch.utils.data import DataLoader

# %% [markdown]
# ### Configuration
# Some **configuration settings** for this project:
# - DATA_DIR = Directory where the python data folder resides. Set to `None` to fallback to default os data directory.

# %%
DATA_DIR = Path.cwd().parent.joinpath("data")

# %% [markdown]
# ## 0. Loading Data
# **CIFAR-10** is used for this assignment.
# `torchvision.datasets.CIFAR10` wraps and loads the CIFAR10 dataset at the specified data path.
#
# The function `get_dataset` applies a basic `ToTensor()` transform to the CIFAR10 dataset for general use.
# This means the output `Dataset` has **images normalized from a range of 0-1**.

# %%
# %%load_clean
from src.datasets.cifar10 import get_dataset  # noqa: F401

# %%
cifar_10_dataset = get_dataset(DATA_DIR)

# %% [markdown]
# ## 1. Exploratory Data Analysis
# We approached EDA via analyzing class distributions and image statistics

# %%
eda_dataloader = DataLoader(cifar_10_dataset, batch_size=256, shuffle=False)

# %% [markdown]
# ### 1.1. Displaying Images
# We selected **5 images** from each class to compare between-class and within-class differences.
#
# Insights:
# - **High Intra-Class Variability**: Instances differ in colour, texture, and background (E.g. horses have white, black and brown coats while standing in different backgrounds). This means the latent space must **capture each of these disentangled factors** as separate dimensions within the latent space, meaning that the latent space needs to be **sufficiently large**.
# - Object types and background colours vary within classes. This implies genreated μ-values may not be truly grouped by classes, and instead grouped by images within subclasses (e.g. Separate cluster for stealth bomber vs commercial plane).
# - Object scale and rotation varies even within classes. This means that more convolutional layers are needed to learn spatial information like positioning.
#
# Images also looked pixelated and some images are not even recognisable.
#
# *Note: This project standardises using Axes as an input for easier composite plots*

# %%
# %%load_clean
from src.analysis.cifar10.display_images import display_class_images  # noqa: F401

# %%
# Capturing output so no axes are returned
_ = display_class_images(eda_dataloader, cifar_10_dataset.classes, 5)

# %% [markdown]
# ### 1.2. General Image statistics.
# We analysed per-class image statistics such as mean RGB values and brightness mean and std.
# - Mean pixel values
# - Mean brightness and standard deviation of brightness
#
# Insights:
# For brightness, Latent dimensions associated with general brightness will have airplane, bird, cat and frog closer to each other while the rest are further apart.
#
# - Cat and Dog exhibit nearly identical global RGB profiles (ΔR=0.005, ΔG=0.009, ΔB=0.001, brightness Δ=0.004). With no strong chromatic or luminance separation, the encoder cannot rely on coarse pixel statistics and must discover fine texture/shape cues to avoid mapping both classes to the same latent region.
# - Action: Use random crops, horizontal flips, and small translations to force shape/texture invariance rather than background memorisation.
#
# - Airplane and Ship share a distinct B > G > R ordering (blue-dominant cast) with Airplane uniformly brighter across all channels. This suggests the encoder may learn a colour shortcut—treating blue backgrounds as a class cue rather than learning object structure.
# - Action: Apply moderate colour jitter (hue ±0.1, saturation ±0.2) to break the blue-sky/ocean association, and occasional grayscale transforms to ensure the encoder learns shape-independent colour.

# %%
# %%load_clean
from src.analysis.analyse_images import get_class_statistics # noqa: F401

# %%
get_class_statistics(eda_dataloader, cifar_10_dataset.classes)

# %% [markdown]
# ## 2. Models

# %% [markdown]
# ### 2.1. Splitting Data
# Pytorch's `random_split` class is used to split data into **80/15/15** distributions amongst train, test and split.
# The classes are split into `DataLoaders`, while the Train dataset has `Shuffle=True` to stop models from learning fake patterns.

# %%
# %%load_clean
from src.datasets.cifar10 import get_dataloaders  # noqa: F401

# %% [markdown]
# ### 2.2. Basic Encoder and Decoder
# Basic Encoder and Decoder Setup.
# We approached these with standard **convolutional layers** for encoder and decoder.
# Multiple **convolutional blocks** with 3x3 layers learn to **compress spatial information** from the image into the latent vector while the decoder learns to take the compressed spatial information and **reconstruct it** back into an image.
#
#
# Pytorch's `nn.Module` was subclassed for modularity.

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
    BasicEncoder
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
    BasicDecoder
]

# %% [markdown]
# ### 2.3. TrainingHistory class
# TrainingHistory is a utility class used to store history when training in **train, val and extra** metrics within dataframes.
# It also is used to format training history to log as outputs within fit() loops.

# %%
# %%load_clean
from src.models.util.TrainingHistory import TrainingHistory  # noqa: F401

# %% [markdown]
# ### 2.4. Callbacks
# **Callbacks** are defined here with EarlyStopping.
# Instead of training with a *fixed number of epochs*, we used EarlyStopping with a high epoch count so that the model keeps training until training improvements < `min_delta`, signalling that the model has stopped improving significantly.

# %%
# %%load_clean
from src.training.callbacks import Callback  # noqa: F401

# %%
# %%load_clean
from src.training.callbacks import EarlyStopping  # noqa: F401

# %% [markdown]
# ### 2.3. AbstractVAE
# AbstractVAE is a basic pytorch module for VAE.
#
# Fields:
# - `latent_dim`: Latent vector length. forward() on superclasses should use this parameter.
# ---
# AbstractVAE also contains methods, such as:
#
# `reparameterize`: Provides the *reparameterization trick*. As in standard VAE architecture, the encoder outputs a *logvar* value, which is then converted into σ (standard deviation) using the formula below:
#
# $$
# \sigma = \sqrt{\exp(\text{logvar})} \;=\; \exp\!\left(\frac{\text{logvar}}{2}\right)
# $$
#
# where
#
# $$
# \text{logvar} = \log(\sigma^2)
# $$
#
# and
#
# $$
# \exp(x) = e^x
# $$
#
# `reconstruction_loss`: Used to calculate *reconstruction loss* using `mse` or `bce`
#
# `kl_divergence`: Used to calculate *kl_divergence*. The default (AbstractVAE) implementation includes free bits, which does not penalize KL divergence per dimension below `free_bits` to prevent posterior collapse.
#
# `run_epoch`: Runs a single epoch over AbstractVAE and calculates gradients.
#
# `fit`: Runs a full training session with callbacks.

# %%
# %%load_clean
from src.models.autoencoders.AbstractVAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_8162 = [
    AbstractVAE,
    VAETrainConfig
]

# %% [markdown]
# ### 2.4. VAE
# `VAE` is a simple implementation of `AbstractVAE` with the `forward` method.
#
# The `forward` method is run when an instance of `VAE` is called and returns the reconstructed image along with *mu* and *logvar* values.

# %%
# %%load_clean
from src.models.autoencoders.VAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_6760 = [
    VAE
]

# %% [markdown]
# ### 2.4. Creating a Basic VAE
# Using the [Encoder and Decoder](#22-basic-encoder-and-decoder) defined previously, we create a basic convolutional VAE.

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
from src.training.autoencoders.base_vae import train_base_vae  # noqa: F401

# %%
model, test_dataloader, labels, history = train_base_vae(DATA_DIR)

# %% [markdown]
# #### 3.1.2. Training Curves
#
# **Training** and **validation curves** along with KL warmup are plotted to ensure that the model is not overfitting and **KL Warmup** is working.
#
# ---
#
# Results:
#
# Training and Validation curves **match properly**, proving that the model is not overfitting. KL Warmup works correctly, ramping up to 1.0 at 30 epochs.
# This is also reflected in **reconstruction loss** going down and then rising back up to 30 when KL penalty increases.
#
# <details>
# <summary>Saved Output</summary>
#
# ![image.png](attachment:74298dd1-8303-4d76-a8dc-bd95890f55e1.png)![description](path/to/image.png)
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.training_history import plot_metrics, MetricPlotSpec  # noqa: F401

# %%
specs: list[MetricPlotSpec] = [
    {
        "title": "Reconstruction Loss",
        "metric": "reconstruction",
    },
    {
        "title": "KL Divergence",
        "metric": "kl_divergence",
        "scale": ("symlog", {"linthresh": 10}),
    },
    {
        "title": "KL Warmup",
        "metric": "kl_weight",
        "extra_metric": True,
    },
]

_ = plot_metrics(history, specs)

# %% [markdown]
# ### 3.2. Latent Space Analysis
# **TSNE** is used to determine if the encoder clusters points within clusters separately.
#
# As the points are **densely scattered with no clear separation**, it may suggest that due to there being too many unused dimensions, the random variations mean points end up close to each other in 128D space (latent dim).

# %%
# %%load_clean
from src.models.autoencoders.inspection.latent_space import analyze_latent_space  # noqa: F401

# %%
fig, ax = plt.subplots()
analyze_latent_space(model, test_dataloader, class_names=labels, ax=ax)
plt.show()

# %% [markdown]
# ### 3.3. Plotting Reconstructions
# Plotting reconstructions is important to see how good the decoder can recreate encoded images.
#
# Based on the current results, we can see the reconstructions correctly encode general texture and shape of images, but are blurry.

# %%
# %%load_clean
from src.models.autoencoders.inspection.reconstructions import plot_reconstructions # noqa: F401

# %%
_ = plot_reconstructions(model, test_dataloader)

# %% [markdown]
# ### 3.4. Plotting Latent Space Utilization
# KL Divergence score per dimension is plotted to show latent space utilisation per dimension.
#
# **KL Divergence per dimension** directly corresponds to **latent space utilisation** as the lower the KL Divergence of each dimension, the closer it approaches N(0, 1), suggesting it is **not used for encoding information**.
#
# Results:
#
# **Many dimensions not having information encoded within** them may suggest that the latent space is too large and can be downsized for the current VAE architecture.
#

# %%
# %%load_clean
from src.models.autoencoders.inspection.latent_space import plot_kl_per_dim  # noqa: F401

# %%
fig, ax = plt.subplots()
plot_kl_per_dim(model, test_dataloader, ax=ax)
plt.show()

# %% [markdown]
# ### 3.5. Plotting Class Samples
# Random class samples are plotted.
# Based on randomly generated class samples, some generated images roughly resemble the original class. However, most samples do not have any resemblance to the original class.
#
# ---
# Results:
#
# Based on these results, Bird and Truck barely resemble their real-life counterparts.
#
# <details>
# <summary>Saved Output (Click to show)</summary>
#
# ![image.png](attachment:869b4aaa-f12e-4763-b4ec-2f514f9085f8.png)
#
# </details>

# %%
# %%load_clean
from src.training.autoencoders.sampling import prepare_image, plot_class_samples  # noqa: F401

# %%
_ = plot_class_samples(model, test_dataloader, class_names=labels)

# %% [markdown]
# ## 4. Experiments in improving results with Base VAE

# %% [markdown]
# ### 4.1. Reducing Dimensionality to 64
# `train_based_vae_reduced` is an identical training script with dimensionality reduced to 64.
#
# This is to test if 64 dimensions are sufficient for the current VAE architecture.

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_base_vae_reduced # noqa: F401

# %%
model, test_dataloader, labels, history = train_base_vae_reduced(DATA_DIR)

# %% [markdown]
# #### 4.1.1. Inspecting Training Curves
# Training Curves are inspected again, to ensure model training is stable.

# %%
specs: list[MetricPlotSpec] = [
    {
        "title": "Reconstruction Loss",
        "metric": "reconstruction",
    },
    {
        "title": "KL Divergence",
        "metric": "kl_divergence",
        "scale": ("symlog", {"linthresh": 10}),
    },
    {
        "title": "KL Warmup",
        "metric": "kl_weight",
        "extra_metric": True,
    },
]

_ = plot_metrics(history, specs)

# %% [markdown]
# ### 4.2. Redoing Analysis with `latent_dim=64`
# All analyses were rerun on the VAE with the **latent dimension reduced to 64**.
#
# However, **reconstructions clearly degraded**. They failed to capture **core image statistics like texture and shape**, ending up noticeably worse than the 128D model.
#
# This suggests that 64 dimensions was too restrictive for the current shallow encoder as it could not encode spatial information properly.
#
# The t-SNE plot also remained a dense, overlapping mess with **no visible class separation**. Thus, it can be concluded that reducing latent space does not help class seperation. 

# %%
_ = plot_reconstructions(model, test_dataloader)

fig, ax = plt.subplots()
plot_kl_per_dim(model, test_dataloader, ax=ax)
plt.show()

fig, ax = plt.subplots()
analyze_latent_space(model, test_dataloader, class_names=labels, ax=ax)
plt.show()

_ = plot_class_samples(model, test_dataloader, class_names=labels)

# %% [markdown]
# ### 4.3. Improving Encoder/Decoder architecture
# Instead of **reducing the latent dim**, another approach was tried to **improve encoder and decoder architectures**.
#
# This time, convolutional layers are increased with a deeper encoder and decoder layers.

# %% [markdown]
# ### 4.4. Improved Encoder/Decoders
# Improved Encoder and Decoders have 2 additional convolutional layers, while 2 layers do not have upsampling/downsampling to preserve spatial size increases (so that the output vector sizes are still the same).
#
# The increased convolutional layers help the Encoder/Decoders learn more detailed features from images to reproduce higher quality images and also encode more features within the latent dimension.

# %%
# %%load_clean
from src.models.autoencoders.encoders.basic_encoder_improved import ImprovedEncoder  # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.decoders.basic_decoder_improved import ImprovedDecoder  # noqa: F401

# %% [markdown]
# ### 4.5. Improved Basic VAE
# A new VAE is created with this function using the improved encoders/decoders

# %%
# %%load_clean
from src.models.autoencoders.model_factory import improved_basic_vae  # noqa: F401

# %% [markdown]
# ### 4.6. Training the Improved VAE
# The improved VAE is next trained.

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_improved_base_vae  # noqa: F401

# %%
model, test_dataloader, labels, history = train_improved_base_vae(DATA_DIR)

# %% [markdown]
# #### 4.6.1. Inspecting Training Curves

# %%
specs: list[MetricPlotSpec] = [
    {
        "title": "Reconstruction Loss",
        "metric": "reconstruction",
    },
    {
        "title": "KL Divergence",
        "metric": "kl_divergence",
        "scale": ("symlog", {"linthresh": 10}),
    },
    {
        "title": "KL Warmup",
        "metric": "kl_weight",
        "extra_metric": True,
    },
]

_ = plot_metrics(history, specs)

# %% [markdown]
# ### 4.7. Analysing the Improved VAE
#
# Results:
# Based on the KL Contribution per dim plot, this improved architecture has higher KL contribution in some dimensions, meaning that the VAE learns to utilise more of the latent space.
#
# However, reconstructions are still blurry and the T-SNE plot is still tightly clustered together.

# %%
_ = plot_reconstructions(model, test_dataloader)

fig, ax = plt.subplots()
plot_kl_per_dim(model, test_dataloader, ax=ax)
plt.show()

fig, ax = plt.subplots()
analyze_latent_space(model, test_dataloader, class_names=labels, ax=ax)
plt.show()

_ = plot_class_samples(model, test_dataloader, class_names=labels)
