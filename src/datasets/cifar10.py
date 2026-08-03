import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def get_dataset(data_path: Path, train: bool = True, transform=None) -> Dataset:
    """
    Loads the CIFAR-10 dataset.
    NOTE: transforms.ToTensor() transforms provided
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

def get_dataloaders(data_path: Path, core_count=2) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data, validation_data, test_data = torch.utils.data.random_split(get_dataset(data_path=data_path),[0.7, 0.15, 0.15])

    # Saves at least 1 cpu core to prevent overloading the cpu.
    num_workers = min(core_count, os.cpu_count() - 1)

    train_data_loader = DataLoader(
        train_data, batch_size=256, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_data_loader = DataLoader(
        validation_data, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    test_data_loader = DataLoader(
        validation_data, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    return train_data_loader, val_data_loader, test_data_loader