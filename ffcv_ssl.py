import ffcv
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.fields.basics import IntDecoder
from ffcv import transforms as T
import torch
import torchvision.transforms as transforms
from data_utils import dataset_x
import numpy as np
from data import SSLImageDataset

from ffcv.pipeline.operation import Operation

# from
# https://github.com/SerezD/ffcv_pytorch_lightning/
# blob/master/src/ffcv_pl/ffcv_utils/augmentations.py
from dataclasses import replace
from typing import Callable, Optional, Tuple

import torch
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State

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
    MEAN = np.array(mean) * 255
    STD = np.array(std) * 255
    image_pipeline1 = [
        ffcv.transforms.RandomResizedCrop(imgsize, scale=(0.4, 1)),
        ffcv.transforms.RandomGrayscale(seed=0),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(torch.device('cuda:0'), non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        DivideImageBy255(torch.float32),
        ffcv.transforms.NormalizeImage(MEAN, STD, np.float32),
    ]
    image_pipeline2 = [
        ffcv.transforms.RandomResizedCrop(imgsize, scale=(0.4, 1)),
        ffcv.transforms.RandomGrayscale(seed=0),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(torch.device('cuda:0'), non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        DivideImageBy255(torch.float32),
        ffcv.transforms.NormalizeImage(MEAN, STD, np.float32),
    ]

    label_pipeline = [
        IntDecoder(),
        ffcv.transforms.ToTensor(),
        T.Squeeze()
    ]

    idx_pipeline = [
        IntDecoder(),
        ffcv.transforms.ToTensor(),
        T.Squeeze()
    ]

    loader = Loader(
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
        order=OrderOption.RANDOM if mode == 'train' else OrderOption.SEQUENTIAL,
    )
    return loader


def build_ffcv_nonsslloader(write_path, imgsize, mean, std, batchsize, numworkers, mode="train"):
    MEAN = np.array(mean) * 255
    STD = np.array(std) * 255
    image_pipeline1 = [
        ffcv.transforms.RandomResizedCrop(imgsize, scale=(0.2, 1)),
        ffcv.transforms.RandomGrayscale(seed=0),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(torch.device('cuda:0'), non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        DivideImageBy255(torch.float32),
        ffcv.transforms.NormalizeImage(MEAN, STD, np.float32),
    ]

    label_pipeline = [
        IntDecoder(),
        ffcv.transforms.ToTensor(),
        T.Squeeze()
    ]

    idx_pipeline = [
        IntDecoder(),
        ffcv.transforms.ToTensor(),
        T.Squeeze()
    ]

    loader = Loader(
        write_path,
        num_workers=numworkers,
        batch_size=batchsize,
        pipelines={
            "image": image_pipeline1,
            #"image2": image_pipeline2,
            "label": label_pipeline,
            "idx": idx_pipeline,
        },
        # We need this custom mapper to map the additional pipeline to
        # the label used in the dataset (image in this case)
        #custom_field_mapper={"image2": "image"},
        order=OrderOption.RANDOM if mode == 'train' else OrderOption.SEQUENTIAL,
    )
    return loader

