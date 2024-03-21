from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
import torch
from data_utils import dataset_x
import argparse
from torch.utils.data import Dataset
import os

# Use FFCV
parser = argparse.ArgumentParser(description='ffcv_writer')

parser.add_argument('--basedataset', default='cifar10', type=str, choices=["cifar10", "cifar100"])


def write_ssl_dataset(basedataset):
    train_basedataset, test_basedataset, _ , imgsize, mean, std = dataset_x(basedataset)

    train_basedataset = FFCVSSLImageDataset(train_basedataset)
    test_basedataset = FFCVSSLImageDataset(test_basedataset)

    if not os.path.exists(f'./output/{basedataset}'):
        os.makedirs(f'./output/{basedataset}')

    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    write_path_train = f'./output/{basedataset}/ssltrainds.beton'

    # Pass a type for each data field
    writer = DatasetWriter(write_path_train, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(max_resolution=max(imgsize)),
        'label': IntField(),
        'idx': IntField()
    })
    # Write dataset
    writer.from_indexed_dataset(train_basedataset)

    ####################################

    write_path_test = f'./output/{basedataset}/ssltestds.beton'
    # Pass a type for each data field
    writer = DatasetWriter(write_path_test, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(max_resolution=max(imgsize)),
        'label': IntField(),
        'idx': IntField()
    })
    # Write dataset
    writer.from_indexed_dataset(test_basedataset)


class FFCVSSLImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx

    def __len__(self):
        return len(self.dataset)


def write_nonssl_dataset(basedataset):
    train_basedataset, test_basedataset, _ , imgsize, mean, std = dataset_x(basedataset)

    if not os.path.exists(f'./output/{basedataset}'):
        os.makedirs(f'./output/{basedataset}')

    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    write_path_train = f'./output/{basedataset}/trainds.beton'

    # Pass a type for each data field
    writer = DatasetWriter(write_path_train, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(max_resolution=max(imgsize)),
        'label': IntField(),
    })
    # Write dataset
    writer.from_indexed_dataset(train_basedataset)

    ####################################

    write_path_test = f'./output/{basedataset}/testds.beton'
    # Pass a type for each data field
    writer = DatasetWriter(write_path_test, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(max_resolution=max(imgsize)),
        'label': IntField(),
    })
    # Write dataset
    writer.from_indexed_dataset(test_basedataset)


def main():
    args = parser.parse_args()

    write_ssl_dataset(args.basedataset)
    write_nonssl_dataset(args.basedataset)


if __name__ == "__main__":
    main()

