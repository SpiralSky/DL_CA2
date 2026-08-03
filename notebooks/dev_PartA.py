# %%
# %load_ext magics.magics

# %% [markdown]
# # Part A
#

# %% [markdown]
# ### Imports

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
# ### EDA

# %%
# %%load_clean
from src.analysis.analyse_images import run_class_eda # noqa: F401

# %%
_ = run_class_eda(eda_dataloader, cifar_10_dataset.classes)

# %% [markdown]
# ### 1. Train/Test Split

# %%

# %%
# %%load_clean
<<<<<<< HEAD
from src.models.autoencoders.decoders.basic_decoder import *  # noqa: F401
=======
import src.models.VAE.decoders.decoder #noqa

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
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.deconv = nn.Sequential(
            up_block(base_channels * 4, base_channels * 2),
            up_block(base_channels * 2, base_channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_channels * 4, self.init_spatial, self.init_spatial)
        return self.deconv(h)

>>>>>>> 43b96fe (Updated PartA)

_LOAD_CLEAN_IMPORTS_f68f = [
    BasicDecoder,
]

# %%
# %%load_clean
from src.models.autoencoders.encoders.basic_encoder import *  # noqa: F401

_LOAD_CLEAN_IMPORTS_ec6f = [
    BasicEncoder,
]

# %%
# %%load_clean
from src.models.autoencoders.VAE import VAE # noqa: F401

# %%
# %%load_clean
<<<<<<< HEAD
from src.models.autoencoders.model_factory import basic_autoencoder # noqa: F401
=======
import src.models.VAE.losses

def vae_loss(recon_x, x, mu, logvar, beta=1.0, recon_loss_type="mse", free_bits=0.0):
    """
    Standard VAE loss: reconstruction term plus a beta-weighted KL divergence
    between the approximate posterior N(mu, sigma^2) and the standard normal
    prior N(0, I).

    recon_loss_type:
        "mse"  - appropriate for continuous natural-image pixels (default,
                 recommended for CIFAR-10-like photographic data)
        "bce"  - appropriate for near-binary pixel data (e.g. MNIST); assumes
                 decoder output is sigmoid-bounded to [0, 1]

    free_bits:
        Minimum nats each latent dimension is allowed before its KL term is
        penalized (per-dimension clamp, applied before summing). 0 disables
        this (original behavior). Use e.g. 0.5 to counter posterior collapse,
        where many dimensions drive KL to ~0 and stop encoding information.

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

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_div = kl_per_dim.sum() / x.shape[0]

    total_loss = recon_loss + beta * kl_div
    return {
        "total": total_loss,
        "reconstruction": recon_loss,
        "kl_divergence": kl_div,
    }


>>>>>>> 43b96fe (Updated PartA)

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
from src.training import *  # noqa: F401

# %%
# %%load_clean
from src.training.trainer import Trainer, TrainConfig # noqa: F401

# %%
# %%load_clean
from src.training.autoencoders.base_vae import train_base_vae  # noqa: F401

# %%
train_base_vae(PROJECT_ROOT.joinpath("data"))
