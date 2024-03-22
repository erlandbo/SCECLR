import ffcv
import numpy as np

from ffcv.fields.basics import IntDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, ToTorchImage, ToDevice, Squeeze

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

# Download :
# https://github.com/facebookresearch/FFCV-SSL/blob/main/examples/test_ffcv_augmentations_ssl.py
# and find examples
def build_ffcv_sslloader(write_path, imgsize, mean, std, batchsize, numworkers, mode="train"):
    image_pipeline1 = [
        ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
        #ffcv.transforms.RandomResizedCrop(output_size=imgsize, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333)),
        ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
        ffcv.transforms.colorjitter.RandomColorJitter(jitter_prob=0.8, brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),
        ffcv.transforms.grayscale.RandomGrayscale(gray_prob=0.2),
        ffcv.transforms.ops.ToTensor(),
        #ffcv.transforms.ToDevice(torch.device('cuda:0'), non_blocking=True),
        ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
        #ffcv.transforms.NormalizeImage(np.array(mean)*255, np.array(std)*255, np.float32),
        DivideImageBy255(torch.float32),
        torchvision.transforms.Normalize(mean, std)
    ]
    image_pipeline2 = [
        ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder(output_size=imgsize, scale=(0.2, 1.0)),
        # ffcv.transforms.RandomResizedCrop(output_size=imgsize, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333)),
        ffcv.transforms.flip.RandomHorizontalFlip(flip_prob=0.5),
        ffcv.transforms.colorjitter.RandomColorJitter(jitter_prob=0.8, brightness=0.4, contrast=0.4, saturation=0.4,hue=0.1),
        ffcv.transforms.grayscale.RandomGrayscale(gray_prob=0.2),
        ffcv.transforms.ops.ToTensor(),
        #ffcv.transforms.ToDevice(torch.device('cuda:0'), non_blocking=True),
        ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
        #ffcv.transforms.NormalizeImage(np.array(mean)*255, np.array(std)*255, np.float32),
        DivideImageBy255(torch.float32),
        torchvision.transforms.Normalize(mean, std)
    ]

    label_pipeline = [
        ffcv.fields.basics.IntDecoder(),
        ffcv.transforms.ops.ToTensor(),
        ffcv.transforms.common.Squeeze(1)
    ]

    idx_pipeline = [
        ffcv.fields.basics.IntDecoder(),
        ffcv.transforms.ops.ToTensor(),
        ffcv.transforms.common.Squeeze(1)
    ]

    loader = ffcv.loader.Loader(
        write_path,
        num_workers=numworkers,
        batch_size=batchsize,
        pipelines={
            "image": image_pipeline1,
            "image2": image_pipeline2,
            "label": label_pipeline,
            "idx": idx_pipeline,
        },
        # We need this custom mapper to map the additional pipeline to
        # the label used in the dataset (image in this case)
        custom_field_mapper={"image2": "image"},
        order=ffcv.loader.OrderOption.RANDOM if mode=="train" else ffcv.loader.OrderOption.SEQUENTIAL,
        drop_last=False,
        os_cache=True,
        seed=42
    )
    return loader


def build_ffcv_nonsslloader(write_path, imgsize, mean, std, batchsize, numworkers, mode="train"):
    image_pipeline1 = [
        ffcv.fields.rgb_image.SimpleRGBImageDecoder(),
        ffcv.transforms.ops.ToTensor(),
        #ToDevice(torch.device('cuda:0'), non_blocking=True),
        ffcv.transforms.ops.ToTorchImage(convert_back_int16=False),
        DivideImageBy255(torch.float32),
        torchvision.transforms.Normalize(mean, std)
        #ffcv.transforms.normalize.NormalizeImage(mean=np.array(mean)*255.0, std=np.array(std)*255.0, type=np.float32)
    ]

    label_pipeline = [
        ffcv.fields.basics.IntDecoder(),
        ffcv.transforms.ops.ToTensor(),
        ffcv.transforms.common.Squeeze(1),
        #ToDevice(torch.device('cuda:0'), non_blocking=True),  # not int on gpu
    ]

    loader = ffcv.loader.Loader(
        write_path,
        num_workers=numworkers,
        batch_size=batchsize,
        pipelines={
            "image": image_pipeline1,
            "label": label_pipeline,
        },
        order=OrderOption.RANDOM if mode == 'train' else OrderOption.SEQUENTIAL,
        drop_last=False,
        os_cache=True,
        seed=42
    )
    return loader

