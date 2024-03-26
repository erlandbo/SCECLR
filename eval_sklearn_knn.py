import numpy as np
import torch
import math
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from torch.nn import functional as F
import argparse
from models import build_model_from_hparams, change_model
from logger_utils import read_hyperparameters
from models import ResProjModel
from data import builddataset_x
from eval_knn import encode_tofeatures


def knn_classifier(X_train, X_test, y_train, y_test, k=15):
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return acc


# def logreg_classifier(X_train, X_test, y_train, y_test):
#     model = LogisticRegression(solver="saga", n_jobs=-1)
#     model.fit(X_train, y_train)
#     acc = model.score(X_test, y_test)
#     return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval sklearn knn')

    torch.backends.cudnn.benchmark = True

    parser.add_argument('--nn_k', default=15, type=int, help='k-nearest neighbors')
    parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon for inverse euclidean weighting')
    parser.add_argument('--fx_distance', default="cosine", type=str,choices=["cosine", "euclidean"], help='Function for computing distance')
    parser.add_argument('--weights', default="distance", type=str,choices=["distance", "uniform"], help='Weights computing distance')
    parser.add_argument('--batchsize', default=256, type=int)

    parser.add_argument('--numworkers', default=0, type=int)

    parser.add_argument('--basedataset', default='cifar10', type=str, choices=["cifar10", "cifar100", "stl10_unlabeled", "stl10_labeled", "imagenette", "oxfordIIItpet"])

    parser.add_argument('--hparams_path', required=True, type=str, help="path to hparams to re-construct model")
    parser.add_argument('--checkpoint_path', required=True, type=str, help="path to model weights")

    parser.add_argument('--use_ffcv', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_fp16', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--use_2dfeats', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if not args.use_ffcv:
        train_basedataset, test_basedataset, test_augmentation, NUM_CLASSES = builddataset_x(args.basedataset, transform_mode="test_classifier")
        train_basedataset.transform = test_basedataset.transform = test_augmentation
        trainloader = DataLoader(train_basedataset, batch_size=args.batchsize, shuffle=True,num_workers=args.numworkers,pin_memory=True, drop_last=False)
        testloader = DataLoader(test_basedataset, batch_size=args.batchsize, shuffle=False,num_workers=args.numworkers,pin_memory=True, drop_last=False)
    else:
        from data_ffcv_ssl import builddataset_ffcv_x
        import ffcv

        train_basedataset, test_basedataset, test_augmentation1, test_augmentation2, NUM_CLASSES = builddataset_ffcv_x(args.basedataset, transform_mode="test_classifier")
        trainloader = ffcv.loader.Loader(
            f"output/{args.basedataset}/trainds.beton",
            num_workers=args.numworkers,
            batch_size=args.batchsize,
            pipelines={
                "image": test_augmentation1.augmentations,
                "label": [ffcv.fields.basics.IntDecoder(),ffcv.transforms.ops.ToTensor(),ffcv.transforms.common.Squeeze(1)],
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
                "image": test_augmentation2.augmentations,
                "label": [ffcv.fields.basics.IntDecoder(), ffcv.transforms.ops.ToTensor(),ffcv.transforms.common.Squeeze(1)],
            },
            order=ffcv.loader.OrderOption.SEQUENTIAL,
            drop_last=False,
            os_cache=True,
            seed=42
        )

    hparams = read_hyperparameters(args.hparams_path)
    model = build_model_from_hparams(hparams)

    #print(model)

    if args.use_2dfeats:
        model = change_model(model, projection_dim=2, device=torch.device("cuda:0"), change_layer="last")
        print("change to 2D feats")


    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cuda:0"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    train_features, train_outs, train_targets = encode_tofeatures(model, trainloader, use_fp16=args.use_fp16)
    test_features, test_outs, test_targets = encode_tofeatures(model, testloader, use_fp16=args.use_fp16)

    train_features, train_outs, train_targets = train_features.cpu().numpy(), train_outs.cpu().numpy(), train_targets.cpu().numpy()
    test_features, test_outs, test_targets = test_features.cpu().numpy(), test_outs.cpu().numpy(), test_targets.cpu().numpy()

    knn_acc_feats = knn_classifier(train_features, test_features, train_targets, test_targets, k=args.nn_k)
    knn_acc_outs = knn_classifier(train_outs, test_outs, train_targets, test_targets, k=args.nn_k)

    #logreg_acc_feats = logreg_classifier(train_features, test_features, train_targets, test_targets)
    #logreg_acc_outs = logreg_classifier(train_outs, test_outs, train_targets, test_targets)

    print("kNN Acc features", knn_acc_feats)
    print("kNN Acc output", knn_acc_outs)

    #print("Logisitic regression Acc features", logreg_acc_feats)
    #print("Logisitic regression Acc output", logreg_acc_outs)



