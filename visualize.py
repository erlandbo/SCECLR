import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from data_utils import dataset_x
from data import Augmentation
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from torch.nn import functional as F
import argparse
from models import build_model_from_hparams
from logger_utils import read_hyperparameters
from eval_knn import encode_tofeatures


def visualize_feats(X_train, y_train, X_test, y_test, num_classes, filepath_plot):
    X = torch.cat([X_train, X_test], dim=0)
    y = torch.cat([y_train, y_test], dim=0)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(*X.T, c=y, cmap="jet")
    fig.savefig(filepath_plot + "/2dplotfeats.png")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize')

    torch.backends.cudnn.benchmark = True

    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--numworkers', default=10, type=int)
    parser.add_argument('--basedataset', default='cifar10', type=str, choices=["cifar10", "cifar100"])

    parser.add_argument('--hparams_path', required=True, type=str, help="path to hparams to re-construct model")
    parser.add_argument('--checkpoint_path', required=True, type=str, help="path to model weights")
    parser.add_argument('--filepath_plot', required=True, type=str, help="path to model weights")

    parser.add_argument('--use_ffcv', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_fp16', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    train_dataset, test_dataset, num_classes , imgsize, mean, std = dataset_x(args.basedataset)

    if not args.use_ffcv:

        test_augmentation = Augmentation(imgsize, mean, std, mode="test", num_views=1)
        train_dataset.transform = test_dataset.transform = test_augmentation
        trainloader = DataLoader(train_dataset, batch_size=args.batchsize, num_workers=args.numworkers,shuffle=True, pin_memory=True, drop_last=False)
        testloader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.numworkers, shuffle=False, pin_memory=True, drop_last=False)
    else:
        from ffcv_ssl import build_ffcv_nonsslloader

        trainloader = build_ffcv_nonsslloader(
            write_path=f"output/{args.basedataset}/trainds.beton",
            mean=mean,
            std=std,
            imgsize=imgsize,
            batchsize=args.batchsize,
            numworkers=args.numworkers,
            mode="train"
        )
        testloader = build_ffcv_nonsslloader(
            write_path=f"output/{args.basedataset}/testds.beton",
            mean=mean,
            std=std,
            imgsize=imgsize,
            batchsize=args.batchsize,
            numworkers=args.numworkers,
            mode="test"
        )

    model = build_model_from_hparams(read_hyperparameters(args.hparams_path))
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    train_features, train_outs, train_targets = encode_tofeatures(model, trainloader,use_fp16=args.use_fp16)
    test_features, test_outs, test_targets = encode_tofeatures(model, testloader, use_fp16=args.use_fp16)

    visualize_feats(train_outs, train_targets, test_outs, test_targets, num_classes, args.filepath_plot)

