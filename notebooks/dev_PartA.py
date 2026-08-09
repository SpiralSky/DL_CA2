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
# ### 3.2. Training Base VAE
# `train_base_vae` is a basic training function to train and evaluate the VAE by running model.fit
#
# By default, batch size is 256 and lr is 3e-3 (can be changed for smaller gpu with less vram).
#
# It returns the model, the test dataloader and labels for analysis.

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_base_vae  # noqa: F401

# %%
model, test_dataloader, labels, history = train_base_vae(DATA_DIR, WEIGHTS_DIR / "base_vae.pt")

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

# %%
# %%load_clean
from src.models.autoencoders.inspection.latent_space import analyze_latent_space  # noqa: F401

# %%
fig, ax = plt.subplots()
analyze_latent_space(model, test_dataloader, class_names=labels, ax=ax)
plt.show()

# %% [markdown]
# #### 3.2.2 Plotting Reconstructions
# Plotting reconstructions is important to see how good the decoder can recreate encoded images.
#
# Based on the current results, we can see the reconstructions correctly encode general texture and shape of images, but are blurry.

# %%
# %%load_clean
from src.models.autoencoders.inspection.reconstructions import plot_reconstructions # noqa: F401

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
# <summary>Saved Output (Click to show)</summary>
#
# ![image.png](attachment:f8926bc9-d6f9-4ac2-ada8-e6b14d7f69de.png)
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.latent_space import plot_kl_per_dim  # noqa: F401

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
# <summary>Saved Output (Click to show)</summary>
#
# ![image.png](attachment:3a53fd37-ce42-442c-87c2-7ba4d9e284ff.png)
#
# </details>

# %%
# %%load_clean
from src.training.autoencoders.sampling import prepare_image, plot_class_samples  # noqa: F401

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
# | Class ID | Class Name | FID Score | Samples |
# | :---: | :--- | :---: | :---: |
# | 0 | airplane | 171.25 | 770 |
# | 1 | automobile | 194.65 | 790 |
# | 2 | bird | 170.22 | 732 |
# | 3 | cat | 156.92 | 772 |
# | 4 | deer | 175.26 | 771 |
# | 5 | dog | 166.62 | 749 |
# | 6 | frog | 193.65 | 723 |
# | 7 | horse | 201.33 | 720 |
# | 8 | ship | 163.24 | 721 |
# | 9 | truck | 197.32 | 752 |
#
# </details>

# %%
# %%load_clean
from src.models.autoencoders.inspection.inspect_generation_quality import calculate_class_fid  # noqa: F401

# %%
calculate_class_fid(model, test_dataloader, labels)

# %% [markdown]
# ### 4. Improving Encoder/Decoder architecture
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
# Based on visual inspection of generated images, classes like automobile, horse andtruck are barely recognisable.
# However, most of the classes cannot be discerned. Thus, increasing encoder and decoder layer depths do not seem to help.
#
# <details>
# <summary>Saved Output (Generated Images)</summary>
#
# ![image.png](attachment:83584516-29dc-48d2-a611-95012b9995db.png)
#
# </details>
#
# <details>
# <summary>Saved Output (FID Scores)</summary>
#
# | Class ID | Class | FID | Samples |
# | :---: | :--- | :---: | :---: |
# | 0 | all | 118.304970 | 7500 |
# | 1 | airplane | 171.665024 | 733 |
# | 2 | automobile | 201.175278 | 764 |
# | 3 | bird | 167.266205 | 753 |
# | 4 | cat | 161.766663 | 765 |
# | 5 | deer | 177.994354 | 762 |
# | 6 | dog | 168.486664 | 729 |
# | 7 | frog | 195.047302 | 778 |
# | 8 | horse | 208.284286 | 733 |
# | 9 | ship | 165.884674 | 746 |
# | 10 | truck | 205.983459 | 737 |
#
# </details>
#
#
#

# %%
_ = plot_class_samples(model, test_dataloader, class_names=labels)

# %%
calculate_class_fid(model, test_dataloader, labels)

# %% [markdown]
# ### 5. VAE with Residual Layers
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
# ### 5.3. Residual Encoder & Residual Decoder
# The **ResDecoder** extends the standard decoder by inserting ResidualBlocks before each upsampling convolution block. This adds depth to the **upsampling pathway without vanishing gradients**, allowing the decoder to refine spatial features at each resolution before expanding them.
# Sigmoid activation is used at the final layer to **normalise outputs to [0, 1]**, matching the input normalisation.
#
# Similarly, the ResEncoder extends the standard encoder by placing ResidualBlocks after each downsampling block. The s**kip connections in each residual block** allow gradients to flow directly through the encoder during backpropagation, enabling **stable training** of a deeper feature extraction stack before the latent projection.

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_res_vae  # noqa: F401

# %%
model, test_dataloader, labels, history = train_res_vae(DATA_DIR, checkpoint_path=WEIGHTS_DIR / "res_vae.pt")

# %% [markdown]
# ### 6. Better Architecture - Conditional and Beta VAEs.
#
# **Conditional VAE**:
# Within encoder input features and the decoder's input feature map, labels for each class are encoded. This encoded pattern, given a pattern with images between classes allows the VAE to differentiate between classes and use the encoded features as a crutch to easily encode and decode features.
#
# **β-VAE**:
# Beta Warmup
