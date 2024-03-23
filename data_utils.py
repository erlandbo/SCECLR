import torch
from torchvision.datasets import CIFAR10, CIFAR100, STL10, Imagenette, ImageNet, CelebA, OxfordIIITPet, Flowers102, Food101
from data import CIFARAugmentations, ImageNetAugmentations, STL10Augmentations
from data_ffcv_ssl import CIFARFFCVAugmentations, STL10FFCVAugmentations, ImageNetFFCVAugmentations


def dataset_x(dataset_name, download=False, transform_mode="contrastive_pretrain"):
    if dataset_name == 'cifar10':
        CIFAR10_IMGSIZE = (32, 32)
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD = (0.2023, 0.1994, 0.2010)
        return (
            CIFAR10(root='./data', download=download, train=True),
            CIFAR10(root='./data', download=download, train=False),
            10,  # classes
            CIFARAugmentations(imgsize=CIFAR10_IMGSIZE, mean=CIFAR10_MEAN, std=CIFAR10_STD, mode=transform_mode),
            CIFARFFCVAugmentations(imgsize=CIFAR10_IMGSIZE, mean=CIFAR10_MEAN, std=CIFAR10_STD, mode=transform_mode)
        )
    elif dataset_name == 'cifar100':
        CIFAR100_IMGSIZE = (32, 32)
        CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        CIFAR100_STD = (0.2675, 0.2565, 0.2761)
        return (
            CIFAR100(root='./data', download=download, train=True),
            CIFAR100(root='./data', download=download, train=False),
            100,
            CIFARAugmentations(imgsize=CIFAR100_IMGSIZE, mean=CIFAR100_MEAN, std=CIFAR100_STD, mode=transform_mode),
            CIFARFFCVAugmentations(imgsize=CIFAR100_IMGSIZE, mean=CIFAR100_MEAN, std=CIFAR100_STD, mode=transform_mode)
        )
    elif dataset_name == 'stl10_unlabeled':
        STL10_IMGSIZE = (96, 96),
        STL10_MEAN = [0.4467, 0.4398, 0.4066],
        STL10_STD = [0.2242, 0.2215, 0.2239],
        return (
            STL10(root='./data', download=download, split='train+unlabeled'),
            STL10(root='./data', download=download, split='test'),
            10,
            STL10Augmentations(imgsize=STL10_IMGSIZE, mean=STL10_MEAN, std=STL10_STD, mode=transform_mode),
            STL10FFCVAugmentations(imgsize=STL10_IMGSIZE, mean=STL10_MEAN, std=STL10_STD, mode=transform_mode)
        )
    elif dataset_name == 'stl10_labeled':
        STL10_IMGSIZE = (96, 96),
        STL10_MEAN = [0.4467, 0.4398, 0.4066],
        STL10_STD = [0.2242, 0.2215, 0.2239],
        return (
            STL10(root='./data', download=download, split='train'),
            STL10(root='./data', download=download, split='test'),
            10,
            STL10Augmentations(imgsize=STL10_IMGSIZE, mean=STL10_MEAN, std=STL10_STD, mode=transform_mode),
            STL10FFCVAugmentations(imgsize=STL10_IMGSIZE, mean=STL10_MEAN, std=STL10_STD, mode=transform_mode)
        )
    elif dataset_name == 'imagenette':  # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
        IMAGENETTE_IMGSIZE = (160, 160),
        IMAGENETTE_MEAN = [0.485, 0.456, 0.406],
        IMAGENETTE_STD = [0.229, 0.224, 0.225],
        return (
            Imagenette(root='./data', download=download, split="train", size="160px"),
            Imagenette(root='./data', download=download, split="val", size="160px"),
            10,
            ImageNetAugmentations(imgsize=IMAGENETTE_IMGSIZE, mean=IMAGENETTE_MEAN, std=IMAGENETTE_STD, mode=transform_mode),
            ImageNetFFCVAugmentations(imgsize=IMAGENETTE_IMGSIZE, mean=IMAGENETTE_MEAN, std=IMAGENETTE_STD, mode=transform_mode)
        )
    elif dataset_name == 'imagenet': # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
        IMAGENET_IMGSIZE = (224, 224),
        IMAGENET_MEAN = [0.485, 0.456, 0.406],
        IMAGENET_STD = [0.229, 0.224, 0.225],
        return (
            ImageNet(root='./data', download=download, split="train"),
            ImageNet(root='./data', download=download, split="val"),
            1000,
            ImageNetAugmentations(imgsize=IMAGENET_IMGSIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD, mode=transform_mode),
            ImageNetFFCVAugmentations(imgsize=IMAGENET_IMGSIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD, mode=transform_mode)
        )
    elif dataset_name == 'oxfordIIItpet':  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
        OXFORDPET_IMGSIZE = (160, 160),
        OXFORDPET_MEAN = (0.485, 0.456, 0.406),
        OXFORDPET_STD = (0.229, 0.224, 0.225),
        return (
            OxfordIIITPet(root='./data', download=download, split="trainval", target_types="category"),
            OxfordIIITPet(root='./data', download=download, split="test", target_types="category"),
            37,
            ImageNetAugmentations(imgsize=OXFORDPET_IMGSIZE, mean=OXFORDPET_MEAN, std=OXFORDPET_STD, mode=transform_mode),
            ImageNetFFCVAugmentations(imgsize=OXFORDPET_IMGSIZE, mean=OXFORDPET_MEAN, std=OXFORDPET_STD, mode=transform_mode)
        )
    elif dataset_name == 'flowers102':
        FLOWER102_IMGSIZE = (160, 160),
        FLOWER102_MEAN = [0.485, 0.456, 0.406],
        FLOWER102_STD = [0.229, 0.224, 0.225],
        return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            Flowers102(root='./data', download=download, split="train"),
            Flowers102(root='./data', download=download, split="test"),
            102,
            ImageNetAugmentations(imgsize=FLOWER102_IMGSIZE, mean=FLOWER102_MEAN, std=FLOWER102_STD, mode=transform_mode),
            ImageNetFFCVAugmentations(imgsize=FLOWER102_IMGSIZE, mean=FLOWER102_MEAN, std=FLOWER102_STD, mode=transform_mode)
        )
    elif dataset_name == 'food101':
        FOOD101_IMGSIZE = (224, 224),
        FOOD101_MEAN = [0.485, 0.456, 0.406],
        FOOD101_STD = [0.229, 0.224, 0.225],
        return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            Food101(root='./data', download=download, split="train"),
            Food101(root='./data', download=download, split="test"),
            101,
            ImageNetAugmentations(imgsize=FOOD101_IMGSIZE, mean=FOOD101_MEAN, std=FOOD101_STD, mode=transform_mode),
            ImageNetFFCVAugmentations(imgsize=FOOD101_IMGSIZE, mean=FOOD101_MEAN, std=FOOD101_STD,mode=transform_mode)
        )
    elif dataset_name == "celeba":  #( # https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/datasets/kitti_dataset.py
        CELEBA_IMGSIZE = (128, 128),
        CELEBA_MEAN = [0.4467, 0.4398, 0.4066],
        CELEBA_STD = [0.2242, 0.2215, 0.2239],
        return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            CelebA(root='./data', download=download, split='train', target_type="identity"),
            CelebA(root='./data', download=download, split='test', target_type="identity"),
            10_177,
            ImageNetAugmentations(imgsize=CELEBA_IMGSIZE, mean=CELEBA_MEAN, std=CELEBA_STD, mode=transform_mode),
            ImageNetFFCVAugmentations(imgsize=CELEBA_IMGSIZE, mean=CELEBA_MEAN, std=CELEBA_STD, mode=transform_mode)
        )
        # "svhn": (
        #     SVHN(root='./data', download=download, split="train", size="160px"),
        #     SVHN(root='./data', download=download, split="val", size="160px"),
        #     10,
        #     (160, 160),
        #     (0.5071, 0.4867, 0.4408),
        #     (0.2675, 0.2565, 0.2761)
        # )



# def dataset_x(dataset_name, transform=None, download=False):
#     if dataset_name == 'cifar10':
#         return  (
#             CIFAR10(root='./data', download=download, train=True),
#             CIFAR10(root='./data', download=download, train=False),
#             10,  # classes
#             (32,32),  # imgresol
#             (0.4914, 0.4822, 0.4465),  # mean
#             (0.2023, 0.1994, 0.2010),  # std
#         )
#     elif dataset_name == 'cifar100':
#         return (
#             CIFAR100(root='./data', download=download, train=True),
#             CIFAR100(root='./data', download=download, train=False),
#             100,
#             (32,32),
#             (0.5071, 0.4867, 0.4408),
#             (0.2675, 0.2565, 0.2761)
#         )
#     elif dataset_name == 'stl10_unlabeled':
#         return (
#             STL10(root='./data', download=download, split='train+unlabeled'),
#             STL10(root='./data', download=download, split='test'),
#             10,
#             (32, 32),
#             [0.4467, 0.4398, 0.4066],
#             [0.2242, 0.2215, 0.2239]
#         )
#     elif dataset_name == 'stl10_labeled':
#         return (
#             STL10(root='./data', download=download, split='train'),
#             STL10(root='./data', download=download, split='test'),
#             10,
#             (96, 96),
#             [0.4467, 0.4398, 0.4066],
#             [0.2242, 0.2215, 0.2239]
#         )
#     elif dataset_name == 'imagenette':
#         return (  # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
#             Imagenette(root='./data', download=download, split="train", size="160px"),
#             Imagenette(root='./data', download=download, split="val", size="160px"),
#             10,
#             (160, 160),
#             [0.485, 0.456, 0.406],
#             [0.229, 0.224, 0.225]
#         )
#     elif dataset_name == 'imagenet':
#         return (  # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
#             ImageNet(root='./data', download=download, split="train"),
#             ImageNet(root='./data', download=download, split="val"),
#             1000,
#             (224, 224),
#             [0.485, 0.456, 0.406],
#             [0.229, 0.224, 0.225]
#         )
#     elif dataset_name == 'oxfordIIItpet':
#         return ( # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
#             OxfordIIITPet(root='./data', download=download, split="trainval", target_types="category"),
#             OxfordIIITPet(root='./data', download=download, split="test", target_types="category"),
#             37,
#             (160, 160),
#             (0.485, 0.456, 0.406),
#             (0.229, 0.224, 0.225)
#         ),
#     elif dataset_name == 'flowers102':
#         return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
#             Flowers102(root='./data', download=download, split="train"),
#             Flowers102(root='./data', download=download, split="test"),
#             102,
#             (160, 160),
#             [0.485, 0.456, 0.406],
#             [0.229, 0.224, 0.225]
#         ),
#     elif dataset_name == 'food101':
#         return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
#             Food101(root='./data', download=download, split="train"),
#             Food101(root='./data', download=download, split="test"),
#             101,
#             (224, 224),
#             [0.485, 0.456, 0.406],
#             [0.229, 0.224, 0.225]
#         ),
#     elif dataset_name == "celeba":  #( # https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/datasets/kitti_dataset.py
#         return (
#             CelebA(root='./data', download=download, split='train', target_type="identity"),
#             CelebA(root='./data', download=download, split='test', target_type="identity"),
#             10_177,
#             (128, 128),
#             [0.4467, 0.4398, 0.4066],
#             [0.2242, 0.2215, 0.2239]
#         ),
#         # "svhn": (
#         #     SVHN(root='./data', download=download, split="train", size="160px"),
#         #     SVHN(root='./data', download=download, split="val", size="160px"),
#         #     10,
#         #     (160, 160),
#         #     (0.5071, 0.4867, 0.4408),
#         #     (0.2675, 0.2565, 0.2761)
#         # )
#

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

