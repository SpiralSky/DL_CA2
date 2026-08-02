from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_dataset(data_path: Path, train: bool = True, transform=None) -> Dataset:
    """
    Loads the CIFAR-10 dataset.
    :param data_path: Path object to the data folder containing the dataset.
    :param train: If true, loads the training set, else loads the test set.
    :param transform: A function that takes in a PIL image and returns a transformed verison.
    :return:
    """
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.CIFAR10(
        root=data_path, train=train, download=True, transform=transform
    )