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

@torch.no_grad()
def encode_tofeatures(model, dataloader):
    features, outs, targets = [], [], []
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        x, target = batch
        z, h = model(x.cuda(non_blocking=True))
        features.append(h)
        outs.append(z)
        targets.append(target.cuda(non_blocking=True))
    features = torch.cat(features)
    outs = torch.cat(outs)
    targets = torch.cat(targets)
    return features, outs, targets


# Weighted kNN used in Dino
# https://github.com/facebookresearch/dino/blob/main/eval_knn.py
# https://github.com/vturrisi/solo-learn/blob/main/solo/utils/knn.py
# https://github.com/leftthomas/SimCLR/blob/master/main.py
@torch.no_grad()
def run_knn(X_train, y_train, X_test, y_test, num_classes, k=20, fx_distance="cosine", weights="distance",eps=1e-8, temp=0.5):
    if fx_distance == 'cosine':
        X_train = F.normalize(X_train, p=2, dim=1)
        X_test = F.normalize(X_test, p=2, dim=1)
    batchsize = len(X_test) // 100
    num_batches = (len(X_test) + batchsize - 1) // batchsize
    correct = 0.0
    for i in range(0, num_batches):
        X_test_batch = X_test[i*batchsize : min((i+1)*batchsize, len(X_test))]
        y_test_batch = y_test[i*batchsize : min((i+1)*batchsize, len(y_test))]

        B = X_test_batch.shape[0]  # effective batchsize

        if fx_distance == 'cosine':
            sim_mat = torch.matmul(X_test_batch, X_train.T)  # (B,N)
        elif fx_distance == 'euclidean':
            sim_mat = 1.0 / (torch.cdist(X_test_batch, X_train).pow(2) + eps)  # (B,N)
        else:
            raise ValueError("Invalid fx_distance", fx_distance)
        sim_candidates, idx_candidates = sim_mat.topk(k=k, dim=1)  # (B, k)

        if fx_distance == 'cosine':
            sim_candidates = torch.exp(sim_candidates / temp)

        # import pdb; pdb.set_trace()

        candidates_targets = torch.gather(y_train.expand(B, -1), dim=-1, index=idx_candidates)  # (N,) -> (B,N) -> (B,k)
        cand_class_count = torch.zeros(B * k, num_classes, device=X_test_batch.device)  # (B*k, C)
        cand_class_count = torch.scatter( cand_class_count, dim=1, index=candidates_targets.view(-1, 1), value=1.0).view(B, k, num_classes)  # (B,k,C)
        if weights == "distance":
            y_prob = torch.sum( cand_class_count * sim_candidates.view(B, k, -1) , dim=1)  # bcast sum( (B,k,C) * (B,k,1), dim=1) -> (B,C)
        elif weights == "uniform":
            y_prob = torch.sum( cand_class_count, dim=1)  # bcast sum( (B,k,C) , dim=1) -> (B,C)
        else:
            raise ValueError("Invalid weights", weights)
        y_pred = torch.argmax(y_prob, dim=1)  # (B,)
        correct += torch.sum(y_pred == y_test_batch).item()

    return correct / len(y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval knn')

    torch.backends.cudnn.benchmark = True

    parser.add_argument('--nn_ks', default=[20, 200], nargs='+',type=int, help='k-nearest neighbors')
    parser.add_argument('--temp', default=[0.07, 0.2, 0.5], nargs='+', type=float, help='Temperature cosine similarity exp(cossim/temp)')
    parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon for inverse euclidean weighting')
    parser.add_argument('--fx_distance', default="cosine", type=str,choices=["cosine", "euclidean"], help='Function for computing distance')
    parser.add_argument('--weights', default="distance", type=str,choices=["distance", "uniform"], help='Weights computing distance')
    parser.add_argument('--batchsize', default=256, type=int)

    parser.add_argument('--numworkers', default=10, type=int)

    parser.add_argument('--basedataset', default='cifar10', type=str, choices=["cifar10", "cifar100"])

    parser.add_argument('--hparams_path', default="", type=str, help="path to hparams to re-construct model")
    parser.add_argument('--checkpoint_path', default="", type=str, help="path to model weights")

    args = parser.parse_args()

    train_dataset, test_dataset, num_classes , imgsize, mean, std = dataset_x(args.basedataset)
    test_augmentation = Augmentation(imgsize, mean, std, mode="test", num_views=1)
    train_dataset.transform = test_dataset.transform = test_augmentation
    trainloader = DataLoader(train_dataset, batch_size=args.batchsize, num_workers=args.numworkers,shuffle=True, pin_memory=True, drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.numworkers, shuffle=False, pin_memory=True, drop_last=False)

    model = build_model_from_hparams(read_hyperparameters(args.hparams_path))
    print(model)
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=torch.device("cuda:0"))["model_state_dict"],
    )
    model.cuda()


    train_features, train_outs, train_targets = encode_tofeatures(model, trainloader)
    test_features, test_outs, test_targets = encode_tofeatures(model, testloader)

    nearest_neighbors = args.nn_ks
    temperatures = args.temp if args.fx_distance == "cosine" else [None]

    scores = {}
    for k in nearest_neighbors:
        for temp in temperatures:
            acc_features = run_knn(
                train_features,
                train_targets,
                test_features,
                test_targets,
                num_classes,
                k=k,
                fx_distance=args.fx_distance,
                weights=args.weights,
                temp=temp,
                eps=args.eps
            )

            scores[f"knn_feats_top1_k{k}_temp{temp}_fxdistance{args.fx_distance}_weights{args.weights}"] = acc_features

            acc_outs = run_knn(
                train_outs,
                train_targets,
                test_outs,
                test_targets,
                num_classes,
                k=k,
                fx_distance=args.fx_distance,
                weights=args.weights,
                temp=temp,
                eps=args.eps
            )

            scores[f"knn_outs_top1_k={k}_temp={temp}_fxdistance={args.fx_distance}_weights={args.weights}"] = acc_outs

    for name, val in scores.items():
        print(f'{name}: {val}')
