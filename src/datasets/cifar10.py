from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10(data_path: Path) -> DataLoader:
    """
    Loads and returns the CIFAR10 dataset.
    :return: DataLoader of CIFAR10 dataset.
    """

    # test
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(download=True, root=data_path, train=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    return train_loader