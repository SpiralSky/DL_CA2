import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader


def get_dataset(data_path: Path, train: bool = True, transform: nn.Module | None = None) -> torchvision.datasets.CIFAR10:
    """
    Loads the CIFAR-10 dataset.
    NOTE: transforms.ToTensor() transforms provided by default
    :param data_path: Path object to the data folder containing the dataset.
    :param train: If true, loads the training set, else loads the test set.
    :param transform: Defaults with a standard ToTensor() transform. Replace with custom transform for data augmentation. (Must have ToTensor() as first layer for normalization.)
    :return:
    """
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.CIFAR10(
        root=data_path, train=train, download=True, transform=transform
    )

def get_dataloaders(data_path: Path, core_count=2, batch_size=256) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits data into train, test and validation sets with ratio 70/15/15, and returns them as torch DataLoaders.
    :param data_path: Path to dataset.
    :param core_count: Number of core/workers to use, caps at processor's max core count - 1.
    :return: Train DataLoader, Validation DataLoader, Test DataLoader.
    """
    train_data, validation_data, test_data = torch.utils.data.random_split(get_dataset(data_path=data_path),[0.7, 0.15, 0.15])

    # Saves at least 1 cpu core to prevent overloading the cpu.
    num_workers = min(core_count, os.cpu_count() - 1)

    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    val_data_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=False
    )

    return train_data_loader, val_data_loader, test_data_loader