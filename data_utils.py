import torch
from torchvision.datasets import CIFAR10, CIFAR100, Imagenette, SVHN


def dataset_x(dataset_name, transform=None, download=True):
    datasets = {
        "cifar10": (
            CIFAR10(root='./data', download=download, train=True),
            CIFAR10(root='./data', download=download, train=False),
            10,  # classes
            (32,32),  # imgresol
            (0.4914, 0.4822, 0.4465),  # mean
            (0.2023, 0.1994, 0.2010),  # std
        ),
        "cifar100": (
            CIFAR100(root='./data', download=download, train=True),
            CIFAR100(root='./data', download=download, train=False),
            100,
            (32,32),
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
        "imagenette": (
            Imagenette(root='./data', download=download, split="train", size="160px"),
            Imagenette(root='./data', download=download, split="val", size="160px"),
            10,
            (160, 160),
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
        "svhn": (
            SVHN(root='./data', download=download, split="train", size="160px"),
            SVHN(root='./data', download=download, split="val", size="160px"),
            10,
            (160, 160),
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    }
    assert dataset_name in datasets.keys(), "Invalid dataset name"
    return datasets[dataset_name]

