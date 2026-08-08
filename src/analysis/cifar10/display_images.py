import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch.utils.data import DataLoader


def display_class_images(
    dataloader: DataLoader,
    classes: list[str],
    num_images: int,
    ax: Axes | None = None,
) -> Axes:
    """
    Displays a grid of images, one column per class, with each column labelled.

    :param dataloader: PyTorch DataLoader object.
    :param num_images: Number of images to display per class (rows).
    :param classes: List of class names, indexed by class id.
    :param ax: Optional pre-existing array of Axes with shape
        (num_images, len(classes)). If None, a new figure/axes is created
        and shown via plt.show().
    :return: The Axes array used for plotting.
    """

    # Gather all images/labels from the dataloader once.
    all_images = []
    all_labels = []
    for batch_images, batch_labels in dataloader:
        all_images.extend(batch_images)
        all_labels.extend(batch_labels)

    show_when_done = ax is None
    if ax is None:
        _, ax = plt.subplots(
            num_images,
            len(classes),
            figsize=(len(classes), num_images),
        )

    # Normalize ax to a 2D array of shape (num_images, len(classes)).
    axes = np.atleast_2d(ax)
    if axes.shape != (num_images, len(classes)):
        axes = axes.reshape(num_images, len(classes))

    for class_id, class_name in enumerate(classes):
        class_images = [
            img for img, label in zip(all_images, all_labels)
            if label.item() == class_id
        ]

        if len(class_images) < num_images:
            raise ValueError(
                f"Not enough images available for class {class_id} "
                f"({class_name}). Requested {num_images}, "
                f"found {len(class_images)}."
            )

        indices = np.random.choice(
            len(class_images), size=num_images, replace=False
        )

        for row, idx in enumerate(indices):
            col_ax = axes[row, class_id]
            col_ax.imshow(class_images[idx].permute(1, 2, 0))
            col_ax.set_xticks([])
            col_ax.set_yticks([])

        axes[0, class_id].set_title(class_name)

    plt.tight_layout()

    if show_when_done:
        plt.show()

    return axes