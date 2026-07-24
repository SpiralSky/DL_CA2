import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


def plot_tsne(
    dataloader: DataLoader,
    n_samples: int = 2000,
    class_names: list[str] | None = None
) -> None:
    data_iterator = iter(dataloader)
    flattened_samples = []
    class_labels = []

    while len(flattened_samples) < n_samples:
        sample_batch, label_batch = next(data_iterator)
        for sample, label in zip(sample_batch, label_batch):
            flattened_samples.append(sample.flatten().numpy())
            class_labels.append(label.item())
            if len(flattened_samples) == n_samples:
                break

    feature_matrix = np.array(flattened_samples)
    label_vector = np.array(class_labels)

    tsne_model = TSNE(n_components=2, random_state=42, n_jobs=-1)
    low_dim_projections = tsne_model.fit_transform(feature_matrix)

    unique_classes = np.unique(label_vector)
    if class_names is None:
        class_names = [f"Class {int(c)}" for c in unique_classes]

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, class_val in enumerate(unique_classes):
        class_mask = (label_vector == class_val)
        ax.scatter(
            low_dim_projections[class_mask, 0],
            low_dim_projections[class_mask, 1],
            label=class_names[idx] if idx < len(class_names) else f"Class {int(class_val)}",
            alpha=0.6,
            s=15
        )

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.axis('off')
    plt.tight_layout()
    plt.show()