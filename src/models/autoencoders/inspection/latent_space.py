import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torch.nn.functional as functional

def analyze_latent_space(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    n_samples: int | None = 5000,
    perplexity: float = 30.0,
    random_state: int = 42,
    save_file: Path | None = None,
    class_names: list[str] | None = None,
) -> plt.Figure:
    if not hasattr(model, "state_dict"):
        raise ValueError("model must have a state_dict method")
    if not hasattr(data_loader, "__iter__"):
        raise TypeError("data_loader must be iterable")

    device = next(model.parameters()).device
    model.eval()

    all_mu: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
                raise ValueError("data_loader must yield (images, labels) tuples")

            images, labels = batch
            images = images.to(device)

            if hasattr(model, "encoder"):
                mu, _ = model.encoder(images)
            else:
                _, mu, _ = model(images)

            all_mu.append(mu.cpu())
            all_labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    mu = torch.cat(all_mu).numpy()
    labels = torch.cat(all_labels).numpy()

    if n_samples is not None and (n := len(mu)) > n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=n_samples, replace=False)
        mu, labels = mu[idx], labels[idx]

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1,
    ).fit_transform(mu)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.5,
        s=10,
    )

    if class_names is not None:
        n_classes = len(class_names)
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab10(i / 9), markersize=8) for i in range(n_classes)]
        ax.legend(handles, class_names, title="class", loc="best", fontsize=8)
    else:
        plt.colorbar(scatter, ax=ax, label="class")

    ax.set_title(f"t-SNE (n={len(mu)}, perplexity={perplexity})")
    plt.tight_layout()

    if save_file is not None:
        fig.savefig(save_file, dpi=300, bbox_inches="tight")

    try:
        __IPYTHON__  # type: ignore[name-defined]
    except NameError:
        plt.show()

    return fig