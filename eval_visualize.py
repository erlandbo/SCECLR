import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from data import builddataset_x
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from torch.nn import functional as F
import argparse
from models import build_model_from_hparams
from logger_utils import read_hyperparameters
from eval_knn import encode_tofeatures
from models import change_model


def visualize_feats(X_train, y_train, X_test, y_test, num_classes, filepath_plot):
    X = torch.cat([X_train, X_test], dim=0).cpu().numpy()
    y = torch.cat([y_train, y_test], dim=0).cpu().numpy()
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


    if not args.use_ffcv:
        train_basedataset, test_basedataset, test_augmentation, NUM_CLASSES = builddataset_x(args.basedataset,
                                                                                             transform_mode="test_classifier")
        train_basedataset.transform = test_basedataset.transform = test_augmentation
        trainloader = DataLoader(train_basedataset, batch_size=args.batchsize, shuffle=True,
                                 num_workers=args.numworkers, pin_memory=True, drop_last=False)
        testloader = DataLoader(test_basedataset, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers,
                                pin_memory=True, drop_last=False)
    else:
        from data_ffcv_ssl import builddataset_ffcv_x
        import ffcv

        train_basedataset, test_basedataset, test_augmentation, NUM_CLASSES = builddataset_ffcv_x(args.basedataset,
                                                                                                  transform_mode="test_classifier")
        trainloader = ffcv.loader.Loader(
            f"output/{args.basedataset}/trainds.beton",
            num_workers=args.numworkers,
            batch_size=args.batchsize,
            pipelines={
                "image": test_augmentation.augmentations,
                "label": [ffcv.fields.basics.IntDecoder(), ffcv.transforms.ops.ToTensor(),
                          ffcv.transforms.common.Squeeze(1)],
            },
            order=ffcv.loader.OrderOption.RANDOM,
            drop_last=False,
            os_cache=True,
            seed=42
        )
        testloader = ffcv.loader.Loader(
            f"output/{args.basedataset}/testds.beton",
            num_workers=args.numworkers,
            batch_size=args.batchsize,
            pipelines={
                "image": test_augmentation.augmentations,
                "label": [ffcv.fields.basics.IntDecoder(), ffcv.transforms.ops.ToTensor(),
                          ffcv.transforms.common.Squeeze(1)],
            },
            order=ffcv.loader.OrderOption.SEQUENTIAL,
            drop_last=False,
            os_cache=True,
            seed=42
        )

    hparams = read_hyperparameters(args.hparams_path)
    model = build_model_from_hparams(hparams)
    model = change_model(model, projection_dim=2, device=torch.device("cuda:0"), change_layer="last")
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    train_features, train_outs, train_targets = encode_tofeatures(model, trainloader,use_fp16=args.use_fp16)
    test_features, test_outs, test_targets = encode_tofeatures(model, testloader, use_fp16=args.use_fp16)

    visualize_feats(train_outs, train_targets, test_outs, test_targets, NUM_CLASSES, args.filepath_plot)

