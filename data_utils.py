import torch
from torchvision.datasets import CIFAR10, CIFAR100, STL10, Imagenette, CelebA, OxfordIIITPet, Flowers102


def dataset_x(dataset_name, transform=None, download=True):
    if dataset_name == 'cifar10':
        return  (
            CIFAR10(root='./data', download=download, train=True),
            CIFAR10(root='./data', download=download, train=False),
            10,  # classes
            (32,32),  # imgresol
            (0.4914, 0.4822, 0.4465),  # mean
            (0.2023, 0.1994, 0.2010),  # std
        )
    elif dataset_name == 'cifar100':
        return (
            CIFAR100(root='./data', download=download, train=True),
            CIFAR100(root='./data', download=download, train=False),
            100,
            (32,32),
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    elif dataset_name == 'stl10_unlabeled':
        return (
            STL10(root='./data', download=download, split='train+unlabeled'),
            STL10(root='./data', download=download, split='test'),
            10,
            (32, 32),
            [0.4467, 0.4398, 0.4066],
            [0.2242, 0.2215, 0.2239]
        )
    elif dataset_name == 'stl10_labeled':
        return (
            STL10(root='./data', download=download, split='train'),
            STL10(root='./data', download=download, split='test'),
            10,
            (96, 96),
            [0.4467, 0.4398, 0.4066],
            [0.2242, 0.2215, 0.2239]
        )
    elif dataset_name == 'imagenette':
        return (  # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
            Imagenette(root='./data', download=False, split="train", size="160px"),
            Imagenette(root='./data', download=False, split="val", size="160px"),
            10,
            (160, 160),
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    elif dataset_name == 'oxfordIIItpet':
        return ( # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            OxfordIIITPet(root='./data', download=download, split="trainval", target_types="category"),
            OxfordIIITPet(root='./data', download=download, split="test", target_types="category"),
            37,
            (160, 160),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    elif dataset_name == 'flowers102':
        return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            Flowers102(root='./data', download=download, split="train"),
            Flowers102(root='./data', download=download, split="test"),
            37,
            (160, 160),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
        # "svhn": (
        #     SVHN(root='./data', download=download, split="train", size="160px"),
        #     SVHN(root='./data', download=download, split="val", size="160px"),
        #     10,
        #     (160, 160),
        #     (0.5071, 0.4867, 0.4408),
        #     (0.2675, 0.2565, 0.2761)
        # )
        # "celeba": ( # https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/datasets/kitti_dataset.py
        #     CelebA(root='./data', download=download, split='train', target_type="identity"),
        #     CelebA(root='./data', download=download, split='test', target_type="identity"),
        #     10,
        #     (96, 96),
        #     [0.4467, 0.4398, 0.4066],
        #     [0.2242, 0.2215, 0.2239]
        # ),

    # assert dataset_name in datasets.keys(), "Invalid dataset name"
    # return datasets[dataset_name]

# def dataset_x(dataset_name, transform=None, download=True):
#     datasets = {
#         "cifar10": (
#             CIFAR10(root='./data', download=download, train=True),
#             CIFAR10(root='./data', download=download, train=False),
#             10,  # classes
#             (32,32),  # imgresol
#             (0.4914, 0.4822, 0.4465),  # mean
#             (0.2023, 0.1994, 0.2010),  # std
#         ),
#         "cifar100": (
#             CIFAR100(root='./data', download=download, train=True),
#             CIFAR100(root='./data', download=download, train=False),
#             100,
#             (32,32),
#             (0.5071, 0.4867, 0.4408),
#             (0.2675, 0.2565, 0.2761)
#         ),
#         "stl10_unlabeled": (
#             STL10(root='./data', download=download, split='train+unlabeled'),
#             STL10(root='./data', download=download, split='test'),
#             10,
#             (32, 32),
#             [0.4467, 0.4398, 0.4066],
#             [0.2242, 0.2215, 0.2239]
#         ),
#         "stl10_labeled": (
#             STL10(root='./data', download=download, split='train'),
#             STL10(root='./data', download=download, split='test'),
#             10,
#             (96, 96),
#             [0.4467, 0.4398, 0.4066],
#             [0.2242, 0.2215, 0.2239]
#         ),
#         "imagenette": (  # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
#             Imagenette(root='./data', download=download, split="train", size="160px"),
#             Imagenette(root='./data', download=download, split="val", size="160px"),
#             10,
#             (160, 160),
#             (0.5071, 0.4867, 0.4408),
#             (0.2675, 0.2565, 0.2761)
#         ),
#         "oxfordIIItpet": ( # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
#             OxfordIIITPet(root='./data', download=download, split="trainval", target_types="category"),
#             OxfordIIITPet(root='./data', download=download, split="test", target_types="category"),
#             37,
#             (160, 160),
#             (0.485, 0.456, 0.406),
#             (0.229, 0.224, 0.225)
#         ),
#         # "svhn": (
#         #     SVHN(root='./data', download=download, split="train", size="160px"),
#         #     SVHN(root='./data', download=download, split="val", size="160px"),
#         #     10,
#         #     (160, 160),
#         #     (0.5071, 0.4867, 0.4408),
#         #     (0.2675, 0.2565, 0.2761)
#         # )
#         # "celeba": ( # https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/datasets/kitti_dataset.py
#         #     CelebA(root='./data', download=download, split='train', target_type="identity"),
#         #     CelebA(root='./data', download=download, split='test', target_type="identity"),
#         #     10,
#         #     (96, 96),
#         #     [0.4467, 0.4398, 0.4066],
#         #     [0.2242, 0.2215, 0.2239]
#         # ),
#
#
#     }
#     assert dataset_name in datasets.keys(), "Invalid dataset name"
#     return datasets[dataset_name]

