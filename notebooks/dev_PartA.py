# %%
import gc
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
# - DATA_DIR: Directory where the python data folder resides. Set to `None` to fall back to default os data directory.
# - WEIGHTS_DIR: Directory where weights are. If weights are present, loads them

# %%
DATA_DIR = Path.cwd().parent.joinpath("data")
WEIGHTS_DIR = Path.cwd().parent.joinpath("weights")

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
# ## 2. Models and Required Dependencies

# %% [markdown]
# ### 2.1. Splitting Data
# Pytorch's `random_split` class is used to split data into **80/15/15** distributions amongst train, test and split.
# The classes are split into `DataLoaders`, while the Train dataset has `Shuffle=True` to stop models from learning fake patterns.

# %%
# %%load_clean
from src.datasets.cifar10 import get_dataloaders  # noqa: F401

# %% [markdown]
# ### 2.2. Callbacks
#

# %% [markdown]
# #### 2.2.1. Callback Class
# The `Callback` class comes with default methods and returns a `CallbackSignal` for flexibility within training.

# %%
# %%load_clean
from src.training.callbacks.Callback import * # noqa: F401

_LOAD_CLEAN_IMPORTS_5f06 = [
    Callback,
    CallbackSignal
]

# %% [markdown]
# #### 2.2.2. EarlyStopping
# EarlyStopping is a simple callback that stops the training session when there is no model improvement since `patience` epochs. By default, that is 15.
#
# It also restores best weights for the model at the end of the session.

# %%
# %%load_clean
from src.training.callbacks.EarlyStopping import EarlyStopping  # noqa: F401

# %% [markdown]
# ### 2.2. Losses

# %% [markdown]
# #### 2.2.1. Reconstruction Loss
# This function simply lets you choose whether to use **mse** (Mean Squared Error) or **bce** (Binary Cross-Entropy) loss.

# %%
# %%load_clean
from src.models.autoencoders.losses.reconstruction import reconstruction_loss  # noqa: F401

# %% [markdown]
# #### 2.2.1. KL Divergence
# KL divergence per dim is obtained and its sum is averaged over the number of batches (for batch tensors).
#
# This implementation also uses free bits as free bits are used for standard VAEs to not penalise KL divergence when the divergence is below a certain value, pretending overaggressive KL penalties affecting reconstructions.

# %%
# %%load_clean
from src.models.autoencoders.losses.kl_divergence import kl_divergence  # noqa: F401

# %% [markdown]
# #### 2.2.3. Vae Loss
# As we know, standard VAEs use a combination of KL Divergence and reconstruction loss for their loss functions.
#
# Vae Loss returns a dictionary, allowing model to track standard **vae loss** (reconstruction + kl, but in this case `kl` is multiplied by `kl_weight` for **KL warmup**) along with the separate loss functions for analysis.

# %%
# %%load_clean
from src.models.autoencoders.losses.models.vae_loss import vae_loss  # noqa: F401

# %% [markdown]
# ### 2.3. Trainers

# %% [markdown]
# #### 2.2.1. TrainingHistory
# TrainingHistory is a utility class used to store history when training in **train, val and extra** metrics within dataframes.
# It also is used to format training history to log as outputs within fit() loops.

# %%
# %%load_clean
from src.models.util.TrainingHistory import TrainingHistory  # noqa: F401

# %% [markdown]
# #### 2.2.2. Trainer Class
# The `Trainer` class provides generalised code for training.
#
# The method `fit` runs epochs up to max epochs and also runs provided callbacks during epoch start and end.
#
# Every epoch, it calls it's `run_epoch` method (unimplemented for base `Trainer`) with train and validation loaders and updates loss accordingly.
#
# Then, it updates `TrainingHistory`. On epoch start and end, callback hooks are called.
#
# Trainer also applies gradient clipping by default to prevent gradient explosions.

# %%
# %%load_clean
from src.training.trainers.Trainer import Trainer  # noqa: F401

# %% [markdown]
# #### 2.2.3. VAE Trainer
#
# `VAETrainer` extends the base `Trainer` class with Variational Autoencoder-specific training logic.
#
# The trainer handles:
# - Forward passes through the VAE model.
# - Computing reconstruction and KL divergence losses.
# - KL warmup scheduling.
# - Gradient updates and clipping.
# - Aggregating metrics across batches.
#
# The training objective is the standard VAE loss:
#
# $$
# \mathcal{L} = \mathcal{L}_{reconstruction} + \beta_{KL}\mathcal{L}_{KL}
# $$
# where:
# - $\mathcal{L}_{reconstruction}$ (_Reconstruction Loss_) measures how well the decoder reconstructs the input.
# - $\mathcal{L}_{KL}$ (_KL Divergence_) regularises the latent distribution towards the prior distribution.
# - $\beta_{KL}$ (_KL weight_) is gradually increased during KL warmup to prevent the latent space from collapsing too early.
#
# During training, the model outputs:
#
# $$
# (\hat{x}, \mu, \log\sigma^2)
# $$
#
# where $\hat{x}$ is the reconstructed input and $(\mu, \log\sigma^2)$ define the latent Gaussian distribution used for sampling.
#
# The trainer supports:
# - Configurable reconstruction losses (`MSE` or `BCE`).
# - Free bits regularization to reduce excessive KL penalties on small latent dimensions.
# - Gradient clipping for training stability.
# - Optional learning rate schedulers and callbacks through the base `Trainer`.
#
# KL warmup linearly increases the KL contribution:
#
# $$
# \beta_{KL} = \min\left(\frac{epoch}{warmup\_epochs}, 1\right)
# $$
#
# allowing the model to prioritize reconstruction early in training before enforcing stronger latent regularization.

# %%
# %%load_clean
from src.training.trainers.VAETrainer import VAETrainer  # noqa: F401

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
#
# The Basic Encoder is responsible for extracting image features and mapping them into the latent distribution parameters required by the VAE.
#
# The encoder:
# - Takes a **32x32 RGB image** as input, resulting in an input shape of `3x32x32`.
# - Uses three convolutional **downsampling blocks** to progressively reduce spatial dimensions while increasing feature channels.
# - Each down block consists of:
#   - A convolutional layer.
#   - Batch normalization.
#   - LeakyReLU activation.
#
# The convolutional layers use:
#
# $$
# kernel\_size=3,\quad stride=2,\quad padding=1
# $$
#
# which reduces the spatial dimensions by half after each block:
#
# $$
# 32 \rightarrow 16 \rightarrow 8 \rightarrow 4
# $$
#
# The number of convolutional filters increases throughout the network:
#
# $$
# 3 \rightarrow 64 \rightarrow 128 \rightarrow 256
# $$
#
# Increasing the number of filters allows the encoder to learn increasingly complex and abstract feature representations from the input images.
#
# Unlike max pooling, strided convolutions are used for downsampling. This allows the network to learn the downsampling operation while preserving important spatial information.
#
# After the convolutional feature extraction layers, the feature maps have dimensions:
#
# $$
# 4 \times 4 \times 256 = 4096
# $$
#
# The feature maps are flattened into a vector of size `4096`, which is passed through two separate fully connected layers:
#
# - `mean`: Produces the latent mean vector $\mu$.
# - `log_variance`: Produces the latent log variance vector $\log(\sigma^2)$.
#
# For a latent dimension of 256, the encoder outputs:
#
# $$
# \mu \in \mathbb{R}^{256}
# $$
#
# $$
# \log(\sigma^2) \in \mathbb{R}^{256}
# $$
#
# These values define the latent Gaussian distribution:
#
# $$
# z \sim \mathcal{N}(\mu, \sigma^2)
# $$
#
# which is sampled using the reparameterization trick before being passed to the decoder.
#
# The use of separate mean and log variance outputs allows the VAE to learn a continuous latent space instead of directly encoding deterministic latent vectors.

# %%
# %%load_clean
from src.models.autoencoders.encoders.basic_encoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_2d36 = [
    BasicEncoder
]

# %% [markdown]
# #### 2.2.2. Decoder Architecture
#
# The Basic Decoder reconstructs an image from the latent representation produced by the encoder.
#
# The decoder:
# - Takes a latent vector of length `latent_dim` as input.
# - Uses a fully connected layer to expand the latent representation back into a feature map.
# - Applies three convolutional upsampling blocks to progressively increase spatial dimensions while reducing feature channels.
# - Outputs a reconstructed RGB image with dimensions `3x32x32`.
#
# The latent vector is first expanded using a linear layer:
#
# $$
# latent\_dim \rightarrow 4096
# $$
#
# The resulting vector is reshaped into the initial feature map:
#
# $$
# 4096 = 256 \times 4 \times 4
# $$
#
# giving an initial spatial representation of:
#
# $$
# 256 \times 4 \times 4
# $$
#
# The decoder then increases the spatial dimensions through three convolutional blocks:
#
# $$
# 4 \rightarrow 8 \rightarrow 16 \rightarrow 32
# $$
#
# Each convolutional block consists of:
# - An optional nearest-neighbour `Upsample` layer, which doubles the spatial dimensions by duplicating neighbouring pixels.
# - A convolutional layer.
# - Batch normalization.
# - LeakyReLU activation.
#
# Nearest-neighbour upsampling is used instead of transposed convolutions to avoid checkerboard patterns found in TransposedConvolutions.
#
# The convolution layers use:
#
# $$
# kernel\_size=3,\quad stride=1,\quad padding=1
# $$
#
# which preserves the spatial dimensions after convolution, allowing the upsampling operation to control image size changes.
#
# The number of feature channels decreases throughout the decoder:
#
# $$
# 256 \rightarrow 64 \rightarrow 32 \rightarrow 3
# $$
#
# This mirrors the encoder structure, where the encoder increases feature channels while reducing spatial dimensions. The decoder performs the reverse operation by reducing feature channels while reconstructing spatial information.
#
# Finally, a `Sigmoid` activation is applied to constrain the reconstructed image values between 0 and 1:
#
# $$
# 0 \leq \hat{x} \leq 1
# $$
#
# allowing the output to represent normalized image pixel values.
#
# Overall, the decoder learns to transform the compact latent representation back into an image by gradually restoring spatial resolution and converting high-level latent features into pixel-level information.

# %%
# %%load_clean
from src.models.autoencoders.decoders.basic_decoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_958e = [
    BasicDecoder
]

# %% [markdown]
# ### 2.4. VAE
# VAE is a simple Variational AutoEncoder implementation created by extending pytorch `nn.Module`.
#
# `forward` provides code for forward pass, running its stored `encoder` and `decoder`
#
# Fields:
# - `latent_dim`: Latent vector length. forward() on superclasses should use this parameter.
# ---
# VAE also contains methods, such as:
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
# `VAE` uses loss functions provided earlier to get its loss, by calling `vae_loss` in its `get_loss` method.
#
# `run_epoch`: Runs a single epoch over AbstractVAE and calculates gradients.
#
# `fit`: Runs a full training session with callbacks.

# %%
# %%load_clean
from src.models.autoencoders.VAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_6760 = [
    VAE
]

# %%
# %%load_clean
from src.models.autoencoders.VAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_6760 = [
    VAE
]

# %% [markdown]
# ### 2.6. Creating a Basic VAE
# Using the [Encoder and Decoder](#22-basic-encoder-and-decoder) defined previously, a basic convolutional VAE is created.

# %%
# %%load_clean
from src.models.autoencoders.model_factory import basic_vae # noqa: F401

# %% [markdown]
# ## 3. Base VAE: Train & Analysis

# %% [markdown]
# ### 3.1. Training Utils

# %% [markdown]
# #### 3.1.1. Load and Save Checkpoints
# `load_checkpoint` is used to load a model's **state dict and history** from a file with `torch.load`
#
# `save_checkpoint` is used to save a model's **state dict and history** from a file with `torch.save`
#
# **NOTE**: Both **just the weights** and **weights with history** will be submitted for this reason (these functions require the weights and history).
#

# %%
# %%load_clean
from src.training.save_state import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_0530 = [
    load_checkpoint,
    save_checkpoint
]

# %% [markdown]
# #### 3.1.2. Printing History
# This small function is used in training scripts to log information when the history is loaded (if loaded from a checkpoint).

# %%
# %%load_clean
from src.training.autoencoders.base_vae import print_history  # noqa: F401

# %% [markdown]
# #### 3.1.3. Train VAE Script
#
# ## `train_vae` is a simple function that handles the complete training workflow for a VAE model.
#
# The function performs the following steps:
#
# 1. **Dataset preparation**
#    - Loads the CIFAR-10 dataset using the provided `data_path`.
#    - Extracts class labels for later evaluation and visualization.
#    - Creates training, validation, and testing dataloaders with a batch size of 256.
#
# 2. **Device configuration**
#    - Automatically selects CUDA if a compatible GPU is available.
#    - Otherwise, training falls back to CPU.
#    - The model is moved to the selected device before training.
#
# 3. **Checkpoint handling**
#    - If a checkpoint exists and `override=False`, the saved model state and training history are loaded.
#    - This allows previously trained models to be reused without retraining.
#
# 4. **Training setup**
#    - Creates an Adam optimiser with a learning rate of `3e-3`. This is because batch_size of `256` is used for dataloaders
#    - Initialises a `VAETrainer`, which manages the training loop, validation, optimisation, gradient clipping, KL warmup, and callbacks.
#
# 5. **Model training**
#    - The VAE is trained for 300 epochs.
#    - Free bits regularisation is enabled with a value of `0.4` to prevent latent dimensions from becoming inactive.
#    - KL warmup is applied over the first 30 epochs, gradually increasing the KL divergence contribution to stabilise training.
#
# 6. **Saving results**
#    - After training, the model weights and training history are saved if a checkpoint path is provided.
#    - The final training summary is printed.
#
# The function returns:
# - The trained VAE model.
# - The test dataloader for evaluation.
# - Dataset class labels.
# - The training history containing loss and metric information.

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_vae  # noqa: F401

# %% [markdown]
# ### 3.2. Training Base VAE
# `train_base_vae` simply wraps around `train_vae` and to train the simple VAE model we created earlier.
#
# **NOTE**: Saved weights, to be loaded require history included within them.
#
# **They will be included within submission, however, weights stripped of history are also included due to submission requirements**

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_base_vae  # noqa: F401

# %%
model, test_dataloader, labels, history = train_base_vae(DATA_DIR, WEIGHTS_DIR / "basic_vae.pt")

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
# ![image.png](attachments/base_vae/training_curves.png)
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.training_history import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_b756 = [
    MetricPlotSpec,
    plot_metrics,
    plot_one
]

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
# ### 3.2. Model Analysis for Improvement

# %% [markdown]
# #### 3.2.1. Latent Space Analysis
# **TSNE** is used to determine if the encoder clusters points within clusters separately.
#
# As the points are **densely scattered with no clear separation**, it may suggest that due to there being too many unused dimensions, the random variations mean points end up close to each other in 128D space (latent dim).
#
# ---
# Results:
#
# <details>
# <summary>Saved Output</summary>
#
# ![image.png](attachments/base_vae/t-sne.png)
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.standard_vae.latent_space import analyze_latent_space  # noqa: F401

# %%
fig, ax = plt.subplots()
analyze_latent_space(model, test_dataloader, class_names=labels, ax=ax)
plt.show()

# %% [markdown]
# ![image.png](attachment:6fad270d-bd6a-42e0-89d8-89d46b58b195.png)#### 3.2.2 Plotting Reconstructions
# Plotting reconstructions is important to see how good the decoder can recreate encoded images.
#
# Based on the current results, we can see the reconstructions correctly encode general texture and shape of images, but are blurry.
#
# ---
# Results:
#
#
# <details>
# <summary>Saved Output</summary>
#
# ![image.png](attachments/base_vae/reconstructions.png)
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.standard_vae.reconstructions import plot_reconstructions # noqa: F401

# %%
_ = plot_reconstructions(model, test_dataloader)

# %% [markdown]
# #### 3.2.3. Plotting Latent Space Utilization
# The plot illustrates the average KL divergence contribution across all 128 latent dimensions.
#
# **KL Divergence per dimension** directly corresponds to **latent space utilisation** as the lower the KL Divergence of each dimension, the closer it approaches N(0, 1), suggesting it is **not used for encoding information**.
#
# ---
# Results:
#
# Sharp spikes are observed in only a select few dimensions (e.g., near dimensions 1, 30, 41, 59, and 126), while the vast majority of dimensions remain near the baseline (~0.35).
#
# This indicates that information is sparsely distributed across the latent space, with only a small fraction of dimensions actively encoding meaningful features. This suggests that the current latent dimension size (128) is sufficient for the model's current capacity.
#
# <details>
# <summary>Saved Output</summary>
#
# ![image.png](attachments/base_vae/kl_per_dim.png)
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.kl_per_dim import plot_kl_per_dim  # noqa: F401

# %%
fig, ax = plt.subplots()
plot_kl_per_dim(model, test_dataloader, ax=ax)
plt.show()

# %% [markdown]
# ### 3.3. Image Generation Analysis
# We approach Image Generation Analysis via visual and automatic inspection:
# - Manually plotting samples per class and calculating statistics
# - Calculating Frenchet-Inception Distance

# %% [markdown]
# #### 3.3.1 Plotting Class Samples
# Random class samples are plotted.
# Based on randomly generated class samples, some generated images roughly resemble the original class. However, most samples do not have any resemblance to the original class.
#
# ---
# Results:
#
# Based on these results, Bird and Truck barely resemble their real-life counterparts.
#
# <details>
# <summary>Saved Output</summary>
#
# ![image.png](attachments/base_vae/class_samples.png)
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.standard_vae.sampling import prepare_image, plot_class_samples  # noqa: F401

# %%
_ = plot_class_samples(model, test_dataloader, class_names=labels)

# %% [markdown]
# #### 3.3.2. Frenchet-Inception Distance
# The table below presents the **Fréchet Inception Distance (FID)** scores computed across individual classes in the dataset.
#
# **Frenchet-Inception Distance** compares the quality of images generated by VAEs by comparing them to real images.
#
# Real and Fake images are passed through a **pre-trained classifier**, _Inception-v3_, and feature vectors are extracted from an intermediate level.
#
# Features of all real images are then summarised into a **multidimensional Gaussian distribution** and the distance between the two distributions are calculated using the _Frechet distance formula_.
#
# ---
# Results:
#
# The calculated FID scores across all classes **range between 156.92 and 201.33**.
#
# These relatively **high FID values** (>150) indicate that while the VAE produces basic structural outputs, the generated images lack **fine detail, sharpness, and high-frequency textures**.
#
# This is further supported by the **dense, overlapping clusters** in the t-SNE plot, which suggest that the encoder struggles to map distinct class features smoothly across the latent space, leading to **significant overlap in learned representations**.
#
# <details>
# <summary>Saved Output</summary>
#
# | Index | Class | FID | Samples |
# | :--- | :--- | :--- | :--- |
# | 0 | all | 115.695000 | 10000 |
# | 1 | airplane | 161.667023 | 1000 |
# | 2 | automobile | 195.564560 | 1000 |
# | 3 | bird | 158.928101 | 1000 |
# | 4 | cat | 151.696259 | 1000 |
# | 5 | deer | 171.366440 | 1000 |
# | 6 | dog | 157.990524 | 1000 |
# | 7 | frog | 178.288055 | 1000 |
# | 8 | horse | 197.150528 | 1000 |
# | 9 | ship | 167.699295 | 1000 |
# | 10 | truck | 200.041092 | 1000 |
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.standard_vae.class_fid import calculate_class_fid  # noqa: F401

# %%
calculate_class_fid(model, test_dataloader, labels)

# %% [markdown]
# ## 4. Improving Encoder/Decoder architecture
# To improve latent space utilisation and make the encoder learn to separate classes cleanly in higher-dimensional space, Encoder and Decoder architecture are improved.

# %% [markdown]
# ### 4.1. Improved Encoder/Decoders
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
# ### 4.2. Improved Basic VAE
# A new VAE is created with this function using the improved encoders/decoders

# %%
# %%load_clean
from src.models.autoencoders.model_factory import improved_basic_vae  # noqa: F401

# %% [markdown]
# ### 4.3. Training the Improved VAE
# The improved VAE is next trained.

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_improved_base_vae  # noqa: F401

# %%
model, test_dataloader, labels, history = train_improved_base_vae(DATA_DIR, WEIGHTS_DIR / "improved_vae.pt")

# %% [markdown]
# #### 4.3.1. Inspecting Training Curves
# Training Curves are similarly inspected.
#
# ---
# Results:
#
# <details>
# <summary>Saved Output</summary>
#
# ![image.png](attachments/improved_vae/training_curves.png)
#
# </details>

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
# ### 4.4. Analysing the Improved VAE
#
# Results:
# Based on the KL Contribution per dim plot, this improved architecture has higher KL contribution in some dimensions, meaning that the VAE learns to utilise more of the latent space.
#
# However, reconstructions are still blurry and the T-SNE plot is still tightly clustered together.
#
# <details>
# <summary>Saved Output (Reconstructions)</summary>
#
# ![image.png](attachments/improved_vae/reconstructions.png)
#
# </details>
# <details>
# <summary>Saved Output (KL per dim)</summary>
#
# ![image.png](attachments/improved_vae/kl_per_dim.png)
#
# </details>
# <details>
# <summary>Saved Output (T-SNE)</summary>
#
# ![image.png](attachments/improved_vae/t-sne.png)
#
# </details>

# %%
_ = plot_reconstructions(model, test_dataloader)

fig, ax = plt.subplots()
plot_kl_per_dim(model, test_dataloader, ax=ax)
plt.show()

fig, ax = plt.subplots()
analyze_latent_space(model, test_dataloader, class_names=labels, ax=ax)
plt.show()

# %% [markdown]
# ### 4.5. Evaluating Generational Capabilities of Improved VAE
# Class Samples are once again plotted along with FID.
#
# ---
# Results:
#
# Based on visual inspection of generated images, classes like automobile, horse and truck and airplane are barely recognisable.
# However, most of the classes cannot be discerned. Thus, increasing encoder and decoder layer depths do not seem to help.
#
# <details>
# <summary>Saved Output (Class Samples)</summary>
#
# ![image.png](attachments/improved_vae/class_samples.png)
#
# </details>
#
# <details>
# <summary>Saved Output (FID Scores)</summary>
#
# | Index | Class | FID | Samples |
# | :--- | :--- | :--- | :--- |
# | 0 | all | 112.638329 | 10000 |
# | 1 | airplane | 155.213272 | 1000 |
# | 2 | automobile | 189.595535 | 1000 |
# | 3 | bird | 154.895233 | 1000 |
# | 4 | cat | 148.365662 | 1000 |
# | 5 | deer | 169.937744 | 1000 |
# | 6 | dog | 154.648071 | 1000 |
# | 7 | frog | 179.888794 | 1000 |
# | 8 | horse | 188.818771 | 1000 |
# | 9 | ship | 163.521957 | 1000 |
# | 10 | truck | 197.247482 | 1000 |
# </details>
#
#
#

# %%
_ = plot_class_samples(model, test_dataloader, class_names=labels)

# %%
calculate_class_fid(model, test_dataloader, labels)

# %% [markdown]
# ## 5. VAE with Residual Layers
# A simple VAE with a residual block is used for the Decoder while the encoder remains the same

# %% [markdown]
# ### 5.1. Residual Block
# ResidualBlock is a simple subclass of `nn.Module` that adds the input feature map to the output of 2 convolutional blocks within itself.
#
# The **skip connection** allows gradients to flow directly through during backpropagation, preventing vanishing gradient and is also easier to optimise.

# %%
# %%load_clean
from src.models.autoencoders.util.ResidualBlock import ResidualBlock  # noqa: F401

# %% [markdown]
# ### 5.2. Residual Encoder & Residual Decoder
# The `ResDecoder` extends the standard decoder by inserting **ResidualBlock**s before each upsampling convolution block. This adds depth to the upsampling pathway without vanishing gradients, allowing the decoder to refine spatial features at each resolution before expanding them.
#
# Sigmoid activation is used at the final layer to normalise outputs to [0, 1], matching the input normalisation.
#
# ---
#
# Similarly, the `ResEncoder` class extends the standard encoder with `ResidualBlock`s to allow input feature maps to pass through

# %%
# %%load_clean
from src.models.autoencoders.encoders.basic_residual_encoder import ResEncoder  # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.decoders.basic_residual_decoder import ResDecoder  # noqa: F401

# %% [markdown]
# ### 5.2. Residual VAE
# Simple VAE is created using the residual blocks.

# %%
# %%load_clean
from src.models.autoencoders.model_factory import residual_vae  # noqa: F401

# %% [markdown]
# ### 5.3. Training Res VAE
# Residual Encoder and Decoder are trained with the standard training script.

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_res_vae  # noqa: F401

# %%
model, test_dataloader, labels, history = train_res_vae(DATA_DIR, checkpoint_path=WEIGHTS_DIR / "res_vae.pt")

# %% [markdown]
# #### 5.3.1. Checking Training Curves
# History of Residual Encoder/Decoder is plotted.
#
# ---
# Results:
#
# Loss curves are all standard, validation and train losses are the same.
# KL Warmup works correctly
#
# <details>
# <summary>Saved Output (Generated Images)</summary>
#
# ![image.png](./attachments/res_vae/training_curves.png)
#
# </details>

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
# attachment:670245c4-e0df-4aff-a981-21fa1cd2800d.pngattachment:670245c4-e0df-4aff-a981-21fa1cd2800d.png### 5.4. Analysing VAE with Residual Layers
#
# Results:
#
# Output Images are still blurry and the VAE can still generate general outlines of shapes like ship and the car.
#
# <details>
# <summary>Saved Output (Generated Images)</summary>
#
# ![image.png](./attachments/res_vae/reconstructions.png)
# </details>
#
# ---
# KL Contribution per dimension has decreased somewhat, however, many dimensions do not seem to contribute.
#
# <details>
# <summary>Saved Output (Generated Images)</summary>
#
# ![image.png](./attachments/res_vae/kl_per_dim.png)
# </details>
#
# ---
# TSNE seems to be somewhat separating, as colour clusters can be distinguished separately (e.g. red cluster). However, there is not much improvement
#
# <details>
# <summary>Saved Output (Generated Images)</summary>
#
# ![image.png](./attachments/res_vae/t-sne.png)
# </details>

# %%
_ = plot_reconstructions(model, test_dataloader)

fig, ax = plt.subplots()
plot_kl_per_dim(model, test_dataloader, ax=ax)
plt.show()

fig, ax = plt.subplots()
analyze_latent_space(model, test_dataloader, class_names=labels, ax=ax)
plt.show()

# %% [markdown]
# ### 5.5. Evaluating Generational Capabilities of ResVAE
# The generational capabilities of `ResVAE` are inspected using the FID score and generally plotting class samples.
#
# ---
# Results:
#
# While a subset of generated samples exhibits identifiable class structures, most outputs remain visually indistinct.
#
# This indicates that integrating a ResNet architecture does not substantially improve visual synthesis quality.
#
# Instead, performance appears constrained by the VAE's loss dynamics. KL divergence regularisation term penalises latent complexity, preventing further optimisation of the reconstruction loss.
#
# This is further supported by FID scores of the range 165-200, which is roughly the same as the previous architectures' generation scores.
#
# <details>
# <summary>Saved Output (Generated Image of classes)</summary>
#
# ![image.png](./attachments/res_vae/class_samples.png)
# </details>
#
# <details>
# <summary>Saved Output (FIDs)</summary>
#
# | Class       | FID       | Samples |
# |-------------|-----------|---------|
# | all         | 125.731445 | 10000 |
# | airplane    | 165.890259 | 1000 |
# | automobile  | 196.610382 | 1000 |
# | bird        | 164.755859 | 1000 |
# | cat         | 158.209320 | 1000 |
# | deer        | 184.687515 | 1000 |
# | dog         | 167.784058 | 1000 |
# | frog        | 194.918518 | 1000 |
# | horse       | 206.613083 | 1000 |
# | ship        | 178.539490 | 1000 |
# | truck       | 209.251633 | 1000 |
#
# </details>

# %%
_ = plot_class_samples(model, test_dataloader, class_names=labels)

# %%
calculate_class_fid(model, test_dataloader, labels)

# %% [markdown]
# ## 6. Better Architecture - Conditional and Beta VAEs.
#
# **Conditional VAE**:
# In Conditional VAE setups, class labels are fed to the encoder **alongside the image and concatenated to z before the decoder**. This removes the burden of encoding class identity into z , freeing the latent dimensions to capture intra-class variation such as pose, colour, and background.
#
# ---
# **β-VAE**:
# β-VAEs add a weight to the KL divergence term with the scalar β > 1, applying stronger pressure on the latent dimensions to match the prior. This encourages, independent latent representations as each dimension controls a single generative factor.

# %% [markdown]
# ### 6.1. BetaConditionalVAE
# `BetaConditionalVAE` extends `VAE` with a beta parameter in training which is delegates the rest to its super's training loop.
#
# There are a few notable changes:
# In its `forward` pass, labels are encoded into embeddings and then passed into the encoder and decoder respectively.
#
# `sample` takes in a  tensor of class labels for sampling from specific classes.
#
# For loss calculation (`get_loss`), beta value is used as a multiplier with kl divergence to control how much kl divergence affects loss.
#
# This affects the distribution of image mu, logvar values across the encoder's latent space.
#
# By supporting kl warmup where the input `beta` value to `self.get_loss` is:
# $$
# kl_weight * β
# $$
# Kl warmup is supported and helps the VAE learn reconstructions in early epochs without kl divergence dominating.

# %%
# %%load_clean
from src.models.autoencoders.BetaConditionalVAE import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_ae7a = [
    BetaConditionalVAE
]

# %% [markdown]
# ### 6.2. Conditional Encoder & Decoders

# %% [markdown]
# #### 6.2.1. Conditional Encoder
# The conditional encoder mirrors the latent encoder, except that it takes in **embeddings** along with the standard input image tensor, concatenating them so that the Encoder can directly condition latent distribution on class labels during feature extraction.

# %%
# %%load_clean
from src.models.autoencoders.encoders.conditional_encoder import ConditionalEncoder  # noqa: F401

# %% [markdown]
# #### 6.2.2. Conditional Decoder
# The conditional Decoder takes similarly takes in a Tensor of embeddings that contains class information, reducing the burden on the decoder for class inference
#
# This allows more controlled and targeted class generation so it can focus on generating more fine details.

# %%
# %%load_clean
from src.models.autoencoders.decoders.conditional_decoder import ConditionalDecoder  # noqa: F401

# %% [markdown]
# ### 6.2. Creating Beta Conditional VAE
# A simple Beta Conditional VAE is created using `ConditionalEncoder` and `ConditionalDecoder`

# %%
# %%load_clean
from src.models.autoencoders.model_factory import beta_conditional_vae  # noqa: F401

# %% [markdown]
# ### 6.3. Training

# %% [markdown]
# #### 6.3.1. BetaVAETrainer

# %%
# %%load_clean
from src.training.trainers.BetaVAETrainer import BetaVAETrainer # noqa: F401

# %% [markdown]
# #### 6.3.2. Training BCVAE

# %%
# %%load_clean
from src.training.autoencoders.conditional_vae import train_bcvae  # noqa: F401

# %%
model, test_dataloader, labels, history = train_bcvae(DATA_DIR, checkpoint_path=WEIGHTS_DIR / "bc_vae.pt")

# %% [markdown]
# #### 6.3.3. Training Analysis
# Training curves are plotted again. Train and validation loss follow the same line, thus, this suggests that the model is learning properly.
#
# KL warmup works as intended.
#
# ---
# Based on KL divergence output, KL divergence seems to keep decreasing up till the 300th epoch, whilst  reconstruction loss stays the same. This might suggest that the beta value is too high.
#
# <details>
# <summary>Saved Output (Generated Images)</summary>
#
# ![image.png](./attachments/bcvae/training_curves.png)
# </details>

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
# ### 6.4. Analysing Model

# %% [markdown]
# #### 6.4.1. Analysis Functions
# Model analysis functions are altered slightly to use class-based sampling for C-VAE.

# %%
# %%load_clean
from src.models.autoencoders.inspection.c_vae.conditional_fid import calculate_conditional_class_fid  # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.inspection.c_vae.conditional_latent_space import analyze_conditional_latent_space  # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.inspection.c_vae.conditional_reconstruction import plot_conditional_reconstructions #noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.inspection.c_vae.conditional_kl_per_dim import plot_conditional_kl_per_dim  # noqa: F401

# %%
# %%load_clean
from src.models.autoencoders.inspection.c_vae.plot_class_samples import plot_conditional_class_samples  # noqa: F401

# %% [markdown]
# ### 6.4.2. Plotting Analysis on BC-VAE results
#
# The slightly different functions are now plotted.
#
# T-SNE still seems very dense. However, this is to be expected for C-VAEs as classes are just a few dimensions on latent space and now should be roughly around N(0,1) instead of seperating into clusters.
#
# Reconstructions seem more blurry than before. This may be because of prioritising KL divergence (and thus higher reconstruction loss)
#
#
# ---
# Results:
#
# Reconstructions are noticeably more blurry, a common feature amongst b-vaes as they prioritise KL divergence.
# <details>
# <summary>Figure 1: Reconstructions</summary>
#
#
# ![image.png](./attachments/bcvae/reconstructions.png)
#
# </details>
#
# KL divergence seems to be constant throughout with only one dimension fluctuating.
# <details>
# <summary>Figure 2: KL Divergence Per Dimension</summary>
#
# ![image.png](./attachments/bcvae/kl_per_dim.png)
#
# </details>
#
#
# T-SNE surprisingly separates much better.  Furthermore, points within classes are spread out.
# <details>
# <summary>Figure 3: T-SNE Plot</summary>
#
# ![image.png](./attachments/bcvae/t-sne.png)
#
# </details>

# %%
_ = plot_conditional_reconstructions(model, test_dataloader)

fig, ax = plt.subplots()
plot_conditional_kl_per_dim(model, test_dataloader, ax=ax)
plt.show()

fig, ax = plt.subplots()
analyze_conditional_latent_space(model, test_dataloader, class_names=labels, ax=ax)
plt.show()

# %% [markdown]
# ### 6.5. Generative Capabilities of BC-VAE
# The **Generative Capabilities** of BC-VAE are also tested
#
# <details>
# <summary>Figure 1: Class Samples</summary>
#
#
# ![image.png](./attachments/bcvae/class_samples.png)
#
# </details>
#
# FID score instead shows deprovement. This is because B-VAE's beta parameter is too strong, making KL dominate reconstruction.
# <details>
# <summary>Figure 2: FID</summary>
#
#
# | Index | Class | FID | Samples |
# | :--- | :--- | :--- | :--- |
# | 0 | airplane | 196.586319 | 5000 |
# | 1 | automobile | 214.678940 | 5000 |
# | 2 | bird | 171.696518 | 5000 |
# | 3 | cat | 163.691528 | 5000 |
# | 4 | deer | 169.706238 | 5000 |
# | 5 | dog | 172.439697 | 5000 |
# | 6 | frog | 167.861145 | 5000 |
# | 7 | horse | 205.373535 | 5000 |
# | 8 | ship | 194.187103 | 5000 |
# | 9 | truck | 216.143845 | 5000 |
#
# </details>
#

# %%
_ = plot_conditional_class_samples(model, test_dataloader, class_names=labels)

# %%
gc.collect()
torch.cuda.empty_cache()
calculate_conditional_class_fid(model, test_dataloader, labels)
