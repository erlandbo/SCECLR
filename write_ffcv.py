from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from data import builddataset_x
import argparse
from torch.utils.data import Dataset
import os

# Use FFCV
parser = argparse.ArgumentParser(description='ffcv_writer')

parser.add_argument('--basedataset', required=True, type=str, choices=["cifar10", "cifar100", "stl10_unlabeled", "stl10_labeled", "imagenette", "oxfordIIItpet", "food101"])

parser.add_argument('--aug_mode', default="contrastive", type=str, choices=["contrastive", "non_contrastive"])
parser.add_argument('--write_mode', default="raw", type=str, choices=["raw", "smart", "jpg", "proportion"])
parser.add_argument('--jpg_quality', default=90, type=float)

parser.add_argument('--max_resolution', default=None, type=int)
parser.add_argument('--compress_probability', default=0.5, type=float)

parser.add_argument('--num_workers', default=-1, type=int)
parser.add_argument('--chunksize', default=100, type=int)
parser.add_argument('--shuffle_indices', default=True, action=argparse.BooleanOptionalAction)


class FFCVSSLImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx

    def __len__(self):
        return len(self.dataset)


def write_ssl_dataset(args):
    train_basedataset, test_basedataset, _, _ = builddataset_x(args.basedataset)

    train_basedataset = FFCVSSLImageDataset(train_basedataset)
    test_basedataset = FFCVSSLImageDataset(test_basedataset)

    if not os.path.exists(f'./output/{args.basedataset}'):
        os.makedirs(f'./output/{args.basedataset}')

    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    write_path_train = f'./output/{args.basedataset}/ssltrainds.beton'

    # Pass a type for each data field
    writer = DatasetWriter(
        write_path_train,
    {
            # Tune options to optimize dataset size, throughput at train-time
            'image': RGBImageField(
                max_resolution=args.max_resolution,
                write_mode=args.write_mode,
                jpeg_quality=args.jpg_quality,
                compress_probability=args.compress_probability
            ),
            'label': IntField(),
            'idx': IntField()
        },
        num_workers=args.num_workers
    )
    writer.from_indexed_dataset(
        train_basedataset,
        shuffle_indices=args.shuffle_indices,
        chunksize=args.chunksize
    )


def write_nonssl_dataset(args):
    train_basedataset, test_basedataset, _, _ = builddataset_x(args.basedataset)

    if not os.path.exists(f'./output/{args.basedataset}'):
        os.makedirs(f'./output/{args.basedataset}')

    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    write_path_train = f'./output/{args.basedataset}/trainds.beton'

    # Pass a type for each data field
    writer = DatasetWriter(
        write_path_train,
        {
            # Tune options to optimize dataset size, throughput at train-time
            'image': RGBImageField(
                max_resolution=args.max_resolution,
                write_mode=args.write_mode,
                jpeg_quality=args.jpg_quality,
                compress_probability=args.compress_probability
            ),
            'label': IntField(),
        },
        num_workers=args.num_workers
    )
    writer.from_indexed_dataset(
        train_basedataset,
        shuffle_indices=args.shuffle_indices,
        chunksize=args.chunksize
    )

    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    write_path_test = f'./output/{args.basedataset}/testds.beton'

    # Pass a type for each data field
    writer = DatasetWriter(
        write_path_test,
        {
            # Tune options to optimize dataset size, throughput at train-time
            'image': RGBImageField(
                max_resolution=args.max_resolution,
                write_mode=args.write_mode,
                jpeg_quality=args.jpg_quality,
                compress_probability=args.compress_probability
            ),
            'label': IntField(),
        },
        num_workers=args.num_workers
    )
    writer.from_indexed_dataset(
        test_basedataset,
        shuffle_indices=False,
        chunksize=args.chunksize
    )


def main():
    args = parser.parse_args()

    print(args)

    if args.aug_mode == "contrastive":
        write_ssl_dataset(args)
    else:
        write_nonssl_dataset(args)


if __name__ == "__main__":
    main()

