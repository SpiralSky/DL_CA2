import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader


def get_dataset(data_path: Path, train: bool = True, transform: nn.Module | None = None) -> torchvision.datasets.CIFAR10:
    """
    Loads CIFAR-10 dataset.

    :param data_path: Dataset directory.
    :param train: Loads training or test split.
    :param transform: Dataset transform.
    :return: CIFAR-10 dataset.
    """
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    return torchvision.datasets.CIFAR10(
        root=data_path,
        train=train,
        download=True,
        transform=transform,
    )

def get_dataloaders(
    data_path: Path,
    core_count=2,
    batch_size=256,
    transform: nn.Module | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation and test dataloaders.

    Training data receives the provided transform.
    Validation and test data always use the default transform.

    :param data_path: Dataset directory.
    :param core_count: Number of dataloader workers.
    :param batch_size: Batch size.
    :param transform: Training data augmentation transform.
    :return: Train, validation and test dataloaders.
    """

    train_dataset = get_dataset(data_path, train=True, transform=transform)

    eval_dataset = get_dataset(data_path, train=True)

    test_dataset = get_dataset(data_path, train=False)

    train_data, validation_data = torch.utils.data.random_split(range(len(train_dataset)), [0.7, 0.3])

    train_data = torch.utils.data.Subset(train_dataset, train_data.indices)

    validation_data = torch.utils.data.Subset(eval_dataset, validation_data.indices)

    num_workers = min(core_count, os.cpu_count() - 1)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False
    )

    return train_loader, val_loader, test_loader