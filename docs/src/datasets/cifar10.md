Module src.datasets.cifar10
===========================

Functions
---------

`get_dataset(data_path: pathlib.Path, train: bool = True, transform=None) ‑> torch.utils.data.dataset.Dataset`
:   Loads the CIFAR-10 dataset.
    :param data_path: Path object to the data folder containing the dataset.
    :param train: If true, loads the training set, else loads the test set.
    :param transform: A function that takes in a PIL image and returns a transformed verison.
    :return: