# %%
# %load_ext magics.magics

# %% [markdown]
# # Part A
#

# %% [markdown]
# ### Imports

# %%
from pathlib import Path

import numpy as np
import torchvision
from scipy.spatial.distance import correlation
from skimage.color import rgb2hsv
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

# %% [markdown]
# ### Configuration
# - PROJECT_ROOT: Root of project. Note that the data folder is expected to be `PROJECT_ROOT/data/`

# %%
PROJECT_ROOT = Path.cwd().parent

# %% [markdown]
# ## 0. Loading Data

# %%
# %%load_clean
from datasets.cifar10 import get_cifar10

def get_cifar10(data_path: Path) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(download=True, root=data_path, train=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return train_loader


# %%
cifar_10 = get_cifar10(PROJECT_ROOT / "data")

# %% [markdown]
# ## 1. Exploratory Data Analysis

# %% [markdown]
# ### 1.1. Displaying Images
# Images are displayed in an 32x32 Grid.
#
# Here, we can see that images are polychromatic, with contrast of bright and dark images. They also seem to be quite saturated.

# %%
# %%load_clean
import src.analysis.cifar10.display_images

def display_images(dataloader: DataLoader, dim: tuple[int, int]) -> None:
    dataiter = iter(dataloader)
    images = []
    while len(images) < dim[0] * dim[1]:
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
display_images(cifar_10, (10, 5))

# %% [markdown]
# ### 1.1. TSNE Distribution of Images
# Images are displayed in TSNE.

# %%


def compare_structural_similarity(image_a: np.ndarray, image_b: np.ndarray) -> float:
    data_range = float(image_a.max() - image_a.min())
    if len(image_a.shape) == 3:
        return float(ssim(image_a, image_b, channel_axis=2, data_range=data_range))
    return float(ssim(image_a, image_b, data_range=data_range))

def compare_color_correlation(image_a: np.ndarray, image_b: np.ndarray) -> float:
    hsv_a = rgb2hsv(image_a)
    hsv_b = rgb2hsv(image_b)

    bins = (8, 8, 8)
    hsv_range = ((0, 1), (0, 1), (0, 1))

    hist_a, _ = np.histogramdd(hsv_a.reshape(-1, 3), bins=bins, range=hsv_range)
    hist_b, _ = np.histogramdd(hsv_b.reshape(-1, 3), bins=bins, range=hsv_range)

    return 1.0 - float(correlation(hist_a.flatten(), hist_b.flatten()))


# %%
import numpy as np
from torch.utils.data import DataLoader

def get_image_samples(dataloader: DataLoader, n_samples: int = 100) -> np.ndarray:
    data_iterator = iter(dataloader)
    samples = []

    while len(samples) < n_samples:
        image_batch, _ = next(data_iterator)
        for image in image_batch:
            # Convert PyTorch tensor [C, H, W] to NumPy [H, W, C]
            img_np = image.permute(1, 2, 0).numpy()
            # Normalize to [0, 1] range if not already
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            samples.append(img_np)
            if len(samples) == n_samples:
                break

    return np.array(samples)

def compute_similarity_matrices(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_images = len(images)
    ssim_matrix = np.zeros((n_images, n_images))
    color_matrix = np.zeros((n_images, n_images))

    for i in range(n_images):
        for j in range(i, n_images):
            if i == j:
                ssim_matrix[i, j] = 1.0
                color_matrix[i, j] = 1.0
            else:
                s_val = compare_structural_similarity(images[i], images[j])
                c_val = compare_color_correlation(images[i], images[j])

                ssim_matrix[i, j] = ssim_matrix[j, i] = s_val
                color_matrix[i, j] = color_matrix[j, i] = c_val

    return ssim_matrix, color_matrix

# Execution
images = get_image_samples(cifar_10, n_samples=100)
ssim_results, color_results = compute_similarity_matrices(images)

# %%
import numpy as np
from matplotlib import pyplot as plt

def plot_similarity_heatmaps(ssim_matrix: np.ndarray, color_matrix: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    im_ssim = axes[0].imshow(ssim_matrix, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Structural Similarity Index (SSIM) Matrix")
    fig.colorbar(im_ssim, ax=axes[0])

    im_color = axes[1].imshow(color_matrix, cmap="plasma", vmin=0, vmax=1)
    axes[1].set_title("Color Correlation (HSV) Matrix")
    fig.colorbar(im_color, ax=axes[1])

    plt.tight_layout()
    plt.show()

plot_similarity_heatmaps(ssim_results, color_results)
