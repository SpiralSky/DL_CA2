"""
Per-class image statistics for datasets served via a PyTorch DataLoader.

Works for any (image, label) DataLoader where images are equal-sized tensors
(e.g. CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST). No dataset-specific code:
only `num_classes` and `class_names` need to change per dataset.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from skimage.filters import laplace
from IPython.display import display


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