from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2
from PIL import Image, ImageFilter
import random
import torch
from torchvision.datasets import CIFAR10, CIFAR100, STL10, Imagenette, ImageNet, CelebA, OxfordIIITPet, Flowers102, Food101


class SSLImageDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        xa_i, xa_j = self.transform(x)
        return xa_i, y, idx, xa_j

    def __len__(self):
        return len(self.dataset)


class CIFARAugmentations(object):
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="contrastive_pretrain",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 ):
        if mode == "contrastive_pretrain":
            self.num_views = 2
            augmentations = [
                v2.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops)),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomApply([v2.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8 * jitter_strength,
                    hue=0.2 * jitter_strength
                )], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        elif mode == "train_classifier":
            self.num_views = 1
            augmentations = [
                v2.RandomResizedCrop(size=imgsize, scale=(0.2, 1.0)),
                v2.RandomHorizontalFlip(0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        elif mode == "test_classifier":
            self.num_views = 1
            augmentations = [
                v2.Resize(size=imgsize),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        else: raise ValueError(f"Unrecognized mode: {mode}")

        self.augmentations = v2.Compose(augmentations)

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views) ] if self.num_views > 1 else self.augmentations(x)


class ImageNetAugmentations(object):
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="contrastive_pretrain",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 ):
        if mode == "contrastive_pretrain":
            self.num_views = 2
            augmentations = [
                v2.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops), interpolation=Image.BICUBIC),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomApply([v2.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8 * jitter_strength,
                    hue=0.2 * jitter_strength
                )], p=0.8),
                v2.RandomGrayscale(p=0.2),
                GaussianBlur(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        elif mode == "train_classifier":
            self.num_views = 1
            augmentations = [
                v2.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops)),
                v2.RandomHorizontalFlip(0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        elif mode == "test_classifier":
            self.num_views = 1
            augmentations = [
                v2.Resize(size=imgsize + 32),  # 256
                v2.CenterCrop(size=imgsize),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        else: raise ValueError(f"Unrecognized mode: {mode}")

        self.augmentations = v2.Compose(augmentations)

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views) ] if self.num_views > 1 else self.augmentations(x)


class STL10Augmentations(object):
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="contrastive_pretrain",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 ):
        if mode == "contrastive_pretrain":
            self.num_views = 2
            augmentations = [
                v2.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops), interpolation=Image.BICUBIC),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomApply([v2.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8 * jitter_strength,
                    hue=0.2 * jitter_strength
                )], p=0.8),
                v2.RandomGrayscale(p=0.2),
                GaussianBlur(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        elif mode == "train_classifier":
            self.num_views = 1
            augmentations = [
                v2.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops)),
                v2.RandomHorizontalFlip(0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        elif mode == "test_classifier":
            self.num_views = 1
            augmentations = [
                v2.Resize(size=imgsize + 12),  # 256
                v2.CenterCrop(size=imgsize),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        else: raise ValueError(f"Unrecognized mode: {mode}")

        self.augmentations = v2.Compose(augmentations)

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views) ] if self.num_views > 1 else self.augmentations(x)


# https://github.com/facebookresearch/dino/blob/main/utils.py
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def builddataset_x(dataset_name, download=False, transform_mode="contrastive_pretrain"):
    if dataset_name == 'cifar10':
        CIFAR10_IMGSIZE = (32, 32)
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD = (0.2023, 0.1994, 0.2010)
        NUM_CLASSES = 10
        return (
            CIFAR10(root='./data', download=download, train=True),
            CIFAR10(root='./data', download=download, train=False),
            CIFARAugmentations(imgsize=CIFAR10_IMGSIZE, mean=CIFAR10_MEAN, std=CIFAR10_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'cifar100':
        CIFAR100_IMGSIZE = (32, 32)
        CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        CIFAR100_STD = (0.2675, 0.2565, 0.2761)
        NUM_CLASSES = 100
        return (
            CIFAR100(root='./data', download=download, train=True),
            CIFAR100(root='./data', download=download, train=False),
            CIFARAugmentations(imgsize=CIFAR100_IMGSIZE, mean=CIFAR100_MEAN, std=CIFAR100_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'stl10_unlabeled':
        STL10_IMGSIZE = (96, 96)
        STL10_MEAN = [0.4467, 0.4398, 0.4066]
        STL10_STD = [0.2242, 0.2215, 0.2239]
        NUM_CLASSES = 10
        return (
            STL10(root='./data', download=download, split='train+unlabeled'),
            STL10(root='./data', download=download, split='test'),
            STL10Augmentations(imgsize=STL10_IMGSIZE, mean=STL10_MEAN, std=STL10_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'stl10_labeled':
        STL10_IMGSIZE = (96, 96)
        STL10_MEAN = [0.4467, 0.4398, 0.4066]
        STL10_STD = [0.2242, 0.2215, 0.2239]
        NUM_CLASSES = 10
        return (
            STL10(root='./data', download=download, split='train'),
            STL10(root='./data', download=download, split='test'),
            STL10Augmentations(imgsize=STL10_IMGSIZE, mean=STL10_MEAN, std=STL10_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'imagenette':  # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
        IMAGENETTE_IMGSIZE = (160, 160)
        IMAGENETTE_MEAN = [0.485, 0.456, 0.406]
        IMAGENETTE_STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 10
        return (
            Imagenette(root='./data', download=download, split="train", size="160px"),
            Imagenette(root='./data', download=download, split="val", size="160px"),
            ImageNetAugmentations(imgsize=IMAGENETTE_IMGSIZE, mean=IMAGENETTE_MEAN, std=IMAGENETTE_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'imagenet':  # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
        IMAGENET_IMGSIZE = (224, 224)
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 1000
        return (
            ImageNet(root='./data', download=download, split="train"),
            ImageNet(root='./data', download=download, split="val"),
            ImageNetAugmentations(imgsize=IMAGENET_IMGSIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'oxfordIIItpet':  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
        OXFORDPET_IMGSIZE = (160, 160)
        OXFORDPET_MEAN = (0.485, 0.456, 0.406)
        OXFORDPET_STD = (0.229, 0.224, 0.225)
        NUM_CLASSES = 37
        return (
            OxfordIIITPet(root='./data', download=download, split="trainval", target_types="category"),
            OxfordIIITPet(root='./data', download=download, split="test", target_types="category"),
            ImageNetAugmentations(imgsize=OXFORDPET_IMGSIZE, mean=OXFORDPET_MEAN, std=OXFORDPET_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'flowers102':
        FLOWER102_IMGSIZE = (160, 160)
        FLOWER102_MEAN = [0.485, 0.456, 0.406]
        FLOWER102_STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 102
        return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            Flowers102(root='./data', download=download, split="train"),
            Flowers102(root='./data', download=download, split="test"),
            ImageNetAugmentations(imgsize=FLOWER102_IMGSIZE, mean=FLOWER102_MEAN, std=FLOWER102_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'food101':
        FOOD101_IMGSIZE = (224, 224)
        FOOD101_MEAN = [0.485, 0.456, 0.406]
        FOOD101_STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 101
        return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            Food101(root='./data', download=download, split="train"),
            Food101(root='./data', download=download, split="test"),
            ImageNetAugmentations(imgsize=FOOD101_IMGSIZE, mean=FOOD101_MEAN, std=FOOD101_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == "celeba":  #( # https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/datasets/kitti_dataset.py
        CELEBA_IMGSIZE = (128, 128)
        CELEBA_MEAN = [0.4467, 0.4398, 0.4066]
        CELEBA_STD = [0.2242, 0.2215, 0.2239]
        NUM_CLASSES = 10_177
        return (  # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            CelebA(root='./data', download=download, split='train', target_type="identity"),
            CelebA(root='./data', download=download, split='test', target_type="identity"),
            ImageNetAugmentations(imgsize=CELEBA_IMGSIZE, mean=CELEBA_MEAN, std=CELEBA_STD, mode=transform_mode),
            NUM_CLASSES
        )
        # "svhn": (
        #     SVHN(root='./data', download=download, split="train", size="160px"),
        #     SVHN(root='./data', download=download, split="val", size="160px"),
        #     10,
        #     (160, 160),
        #     (0.5071, 0.4867, 0.4408),
        #     (0.2675, 0.2565, 0.2761)
        # )

