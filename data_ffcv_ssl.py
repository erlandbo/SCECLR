import ffcv
import numpy as np

from ffcv.fields.basics import IntDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, ToTorchImage, ToDevice, Squeeze
from torchvision.datasets import CIFAR10, CIFAR100, STL10, Imagenette, ImageNet, CelebA, OxfordIIITPet, Flowers102, Food101

# from
# https://github.com/SerezD/ffcv_pytorch_lightning/
# blob/master/src/ffcv_pl/ffcv_utils/augmentations.py
from dataclasses import replace
from typing import Callable, Optional, Tuple

import torch
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
import torchvision


# From https://github.com/berenslab/t-simcne/blob/master/tsimcne/ffcv_augmentation.py
class DivideImageBy255(Operation):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        assert (
                dtype == torch.float16
                or dtype == torch.float32
                or dtype == torch.float64
        ), f"wrong dtype passed: {dtype}"
        self.dtype = dtype

    def generate_code(self) -> Callable:
        def divide(image, dst):
            dst = image.to(self.dtype) / 255.0
            return dst

        divide.is_parallel = True

        return divide

    def declare_state_and_memory(
            self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, dtype=self.dtype), None


class CIFARFFCVAugmentations(object):
    def __init__(self, imgsize, mean, std, mode="contrastive_pretrain"):
        if mode == "contrastive_pretrain":
            self.augmentations = [
                ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
                ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
                ffcv.transforms.colorjitter.RandomColorJitter(jitter_prob=0.8, brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),
                ffcv.transforms.grayscale.RandomGrayscale(gray_prob=0.2),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        elif mode == "train_classifier":
            self.augmentations = [
                ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
                ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        elif mode == "test_classifier":
            self.augmentations = [
                ffcv.fields.rgb_image.SimpleRGBImageDecoder(),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        else:
            raise ValueError(f"Unrecognized mode: {mode}")


class STL10FFCVAugmentations(object):

    def __init__(self, imgsize, mean, std, mode="contrastive_pretrain"):
        if mode == "contrastive_pretrain":
            self.augmentations = [
                ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
                ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
                ffcv.transforms.colorjitter.RandomColorJitter(jitter_prob=0.8, brightness=0.4, contrast=0.4,
                                                              saturation=0.4, hue=0.1),
                ffcv.transforms.grayscale.RandomGrayscale(gray_prob=0.2),
                ffcv.transforms.gaussian_blur.GaussianBlur(blur_prob=0.5),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        elif mode == "train_classifier":
            self.augmentations = [
                ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
                ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        elif mode == "test_classifier":
            crop_ration = imgsize[0] / (imgsize[1] + 12)
            #crop_ration = 224 / 256
            self.augmentations = [
                ffcv.fields.rgb_image.CenterCropRGBImageDecoder(output_size=imgsize, ratio=crop_ration),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        else:
            raise ValueError(f"Unrecognized mode: {mode}")


class ImageNetFFCVAugmentations(object):
    def __init__(self, imgsize, mean, std, mode="contrastive_pretrain"):
        if mode == "contrastive_pretrain":
            self.augmentations = [
                ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
                ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
                ffcv.transforms.colorjitter.RandomColorJitter(jitter_prob=0.8, brightness=0.4, contrast=0.4,
                                                              saturation=0.4, hue=0.1),
                ffcv.transforms.grayscale.RandomGrayscale(gray_prob=0.2),
                ffcv.transforms.gaussian_blur.GaussianBlur(blur_prob=0.5),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        elif mode == "train_classifier":
            self.augmentations = [
                ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
                ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        elif mode == "test_classifier":
            #crop_ration = imgsize[0] / (imgsize[1] + 32)
            crop_ration = 224 / 256
            self.augmentations = [
                ffcv.fields.rgb_image.CenterCropRGBImageDecoder(output_size=imgsize, ratio=crop_ration),
                ffcv.transforms.ops.ToTensor(),
                ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
                DivideImageBy255(torch.float32),
                torchvision.transforms.Normalize(mean, std)
            ]
        else:
            raise ValueError(f"Unrecognized mode: {mode}")


def builddataset_ffcv_x(dataset_name, download=False, transform_mode="contrastive_pretrain"):
    if dataset_name == 'cifar10':
        CIFAR10_IMGSIZE = (32, 32)
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD = (0.2023, 0.1994, 0.2010)
        NUM_CLASSES = 10
        return (
            CIFAR10(root='./data', download=download, train=True),
            CIFAR10(root='./data', download=download, train=False),
            CIFARFFCVAugmentations(imgsize=CIFAR10_IMGSIZE, mean=CIFAR10_MEAN, std=CIFAR10_STD, mode=transform_mode),
            CIFARFFCVAugmentations(imgsize=CIFAR10_IMGSIZE, mean=CIFAR10_MEAN, std=CIFAR10_STD, mode=transform_mode),
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
            CIFARFFCVAugmentations(imgsize=CIFAR100_IMGSIZE, mean=CIFAR100_MEAN, std=CIFAR100_STD, mode=transform_mode),
            CIFARFFCVAugmentations(imgsize=CIFAR100_IMGSIZE, mean=CIFAR100_MEAN, std=CIFAR100_STD, mode=transform_mode),
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
            STL10FFCVAugmentations(imgsize=STL10_IMGSIZE, mean=STL10_MEAN, std=STL10_STD, mode=transform_mode),
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
            STL10FFCVAugmentations(imgsize=STL10_IMGSIZE, mean=STL10_MEAN, std=STL10_STD, mode=transform_mode),
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
            ImageNetFFCVAugmentations(imgsize=IMAGENETTE_IMGSIZE, mean=IMAGENETTE_MEAN, std=IMAGENETTE_STD, mode=transform_mode),
            NUM_CLASSES
        )
    elif dataset_name == 'imagenet': # https://github.com/kumarkrishna/fastssl/blob/main/fastssl/data/imagenet_dataloaders.py
        IMAGENET_IMGSIZE = (224, 224)
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 1000
        return (
            ImageNet(root='./data', download=download, split="train"),
            ImageNet(root='./data', download=download, split="val"),
            ImageNetFFCVAugmentations(imgsize=IMAGENET_IMGSIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD, mode=transform_mode),
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
            ImageNetFFCVAugmentations(imgsize=OXFORDPET_IMGSIZE, mean=OXFORDPET_MEAN, std=OXFORDPET_STD, mode=transform_mode),
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
            ImageNetFFCVAugmentations(imgsize=FLOWER102_IMGSIZE, mean=FLOWER102_MEAN, std=FLOWER102_STD, mode=transform_mode),
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
            ImageNetFFCVAugmentations(imgsize=FOOD101_IMGSIZE, mean=FOOD101_MEAN, std=FOOD101_STD, mode=transform_mode),
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
            ImageNetFFCVAugmentations(imgsize=CELEBA_IMGSIZE, mean=CELEBA_MEAN, std=CELEBA_STD, mode=transform_mode),
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




# Download :
# https://github.com/facebookresearch/FFCV-SSL/blob/main/examples/test_ffcv_augmentations_ssl.py
# and find examples
# def build_ffcv_sslloader(write_path, imgsize, mean, std, batchsize, numworkers, shuffle=True, gaus_blur=False):
#     image_pipeline1_jitter_flip = [
#         ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
#         #ffcv.transforms.RandomResizedCrop(output_size=imgsize, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333)),
#         ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
#         ffcv.transforms.colorjitter.RandomColorJitter(jitter_prob=0.8, brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),
#         ffcv.transforms.grayscale.RandomGrayscale(gray_prob=0.2)
#     ]
#     image_pipeline1_gaus_blur = [ffcv.transforms.gaussian_blur.GaussianBlur(blur_prob=0.5, kernel_size=int(0.1 * imgsize[0]))]
#     image_pipeline1_totensor = [
#         ffcv.transforms.ops.ToTensor(),
#         #ffcv.transforms.ToDevice(torch.device('cuda:0'), non_blocking=True),
#         ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
#         #ffcv.transforms.NormalizeImage(np.array(mean)*255, np.array(std)*255, np.float32),
#         DivideImageBy255(torch.float32),
#         torchvision.transforms.Normalize(mean, std)
#     ]
#     #####################################3
#     image_pipeline2_jitter_flip = [
#         ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
#         # ffcv.transforms.RandomResizedCrop(output_size=imgsize, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333)),
#         ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
#         ffcv.transforms.colorjitter.RandomColorJitter(jitter_prob=0.8, brightness=0.4, contrast=0.4, saturation=0.4,
#                                                       hue=0.1),
#         ffcv.transforms.grayscale.RandomGrayscale(gray_prob=0.2)
#     ]
#     image_pipeline2_gaus_blur = [
#         ffcv.transforms.gaussian_blur.GaussianBlur(blur_prob=0.5, kernel_size=int(0.1 * imgsize[0]))]
#     image_pipeline2_totensor = [
#         ffcv.transforms.ops.ToTensor(),
#         # ffcv.transforms.ToDevice(torch.device('cuda:0'), non_blocking=True),
#         ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
#         # ffcv.transforms.NormalizeImage(np.array(mean)*255, np.array(std)*255, np.float32),
#         DivideImageBy255(torch.float32),
#         torchvision.transforms.Normalize(mean, std)
#     ]
#
#     image_pipeline1 = image_pipeline1_jitter_flip + image_pipeline1_gaus_blur if gaus_blur else [] + image_pipeline1_totensor
#     image_pipeline2 = image_pipeline2_jitter_flip + image_pipeline2_gaus_blur if gaus_blur else [] + image_pipeline2_totensor
#
#     label_pipeline = [
#         ffcv.fields.basics.IntDecoder(),
#         ffcv.transforms.ops.ToTensor(),
#         ffcv.transforms.common.Squeeze(1)
#     ]
#
#     idx_pipeline = [
#         ffcv.fields.basics.IntDecoder(),
#         ffcv.transforms.ops.ToTensor(),
#         ffcv.transforms.common.Squeeze(1)
#     ]
#
#     loader = ffcv.loader.Loader(
#         write_path,
#         num_workers=numworkers,
#         batch_size=batchsize,
#         pipelines={
#             "image": image_pipeline1,
#             "image2": image_pipeline2,
#             "label": label_pipeline,
#             "idx": idx_pipeline,
#         },
#         # We need this custom mapper to map the additional pipeline to
#         # the label used in the dataset (image in this case)
#         custom_field_mapper={"image2": "image"},
#         order=ffcv.loader.OrderOption.RANDOM if shuffle else ffcv.loader.OrderOption.SEQUENTIAL,
#         drop_last=False,
#         os_cache=True,
#         seed=42
#     )
#     return loader
#

# def build_ffcv_nonsslloader(write_path, imgsize, mean, std, batchsize, numworkers, augmode="train_linear", shuffle=True, crop_ratio=None):
#     if augmode == "train_linear":
#         image_pipeline1 = [
#             ffcv.fields.rgb_image.SimpleRGBImageDecoder() if crop_ratio is None else ffcv.fields.rgb_image.CenterCropRGBImageDecoder(output_size=imgsize, ratio=crop_ratio),
#             ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
#             ffcv.transforms.ops.ToTensor(),
#             #ToDevice(torch.device('cuda:0'), non_blocking=True),
#             ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
#             DivideImageBy255(torch.float32),
#             torchvision.transforms.Normalize(mean, std)
#             #ffcv.transforms.normalize.NormalizeImage(mean=np.array(mean)*255.0, std=np.array(std)*255.0, type=np.float32)
#         ]
#     else:
#         image_pipeline1 = [
#             ffcv.fields.rgb_image.SimpleRGBImageDecoder() if crop_ratio is None else ffcv.fields.rgb_image.CenterCropRGBImageDecoder(output_size=imgsize, ratio=crop_ratio),
#             ffcv.transforms.ops.ToTensor(),
#             #ToDevice(torch.device('cuda:0'), non_blocking=True),
#             ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
#             DivideImageBy255(torch.float32),
#             torchvision.transforms.Normalize(mean, std)
#             #ffcv.transforms.normalize.NormalizeImage(mean=np.array(mean)*255.0, std=np.array(std)*255.0, type=np.float32)
#         ]
#
#     label_pipeline = [
#         ffcv.fields.basics.IntDecoder(),
#         ffcv.transforms.ops.ToTensor(),
#         ffcv.transforms.common.Squeeze(1),
#         #ToDevice(torch.device('cuda:0'), non_blocking=True),  # not int on gpu
#     ]
#
#
#     loader = ffcv.loader.Loader(
#         write_path,
#         num_workers=numworkers,
#         batch_size=batchsize,
#         pipelines={
#             "image": image_pipeline1,
#             "label": label_pipeline,
#         },
#         order=OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL,
#         drop_last=False,
#         os_cache=True,
#         seed=42
#     )
#     return loader

