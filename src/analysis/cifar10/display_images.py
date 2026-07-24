from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def display_images(dataloader: DataLoader, dim: tuple[int, int]) -> None:
    dataiter = iter(dataloader)
    images = []

    while len(images) < (dim[0] * dim[1]):
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