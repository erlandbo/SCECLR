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
from models import ResSCECLR
from data import builddataset_x


@torch.no_grad()
def encode_tofeatures(model, dataloader, use_fp16=False):
    features, outs, targets = [], [], []
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        x, target = batch
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # print(target, target.dtype)
        if not use_fp16:
            z, h = model(x)
        else:
            with torch.cuda.amp.autocast(use_fp16):
                z, h = model(x)
        features.append(h)
        outs.append(z)
        targets.append(target)

    features = torch.cat(features)
    outs = torch.cat(outs)
    targets = torch.cat(targets)

    #check_if_inf(features)
    #check_if_inf(outs)

    return features, outs, targets


def check_if_inf(data):
    mask = torch.isinf(data)
    if torch.all(data[mask == False] == float('-inf')):
        print("Infinity value in data")
        sys.exit(1)

# Weighted kNN used in Dino
# https://github.com/facebookresearch/dino/blob/main/eval_knn.py
# https://github.com/vturrisi/solo-learn/blob/main/solo/utils/knn.py
# https://github.com/leftthomas/SimCLR/blob/master/main.py
@torch.no_grad()
def run_knn(X_train, y_train, X_test, y_test, num_classes, k=20, fx_distance="cosine", weights="distance", eps=1e-8, temp=0.5, use_fp16=False):
    if fx_distance == 'cosine':
        if not use_fp16:
            X_train = F.normalize(X_train, p=2, dim=1)
            X_test = F.normalize(X_test, p=2, dim=1)
        else:
            with torch.cuda.amp.autocast(use_fp16):
                X_train = F.normalize(X_train, p=2, dim=1)
                X_test = F.normalize(X_test, p=2, dim=1)
        # check_if_inf(X_train)
        # check_if_inf(X_test)
    num_chunks = 100
    num_test_samples = X_test.shape[0]
    chunks_per_samples = num_test_samples // num_chunks
    correct = 0.0

    for i in range(0, num_test_samples, chunks_per_samples):
        X_test_batch = X_test[i : min(i + chunks_per_samples, num_test_samples)]
        y_test_batch = y_test[i : min(i + chunks_per_samples, num_test_samples)]

        B = X_test_batch.shape[0]  # effective batchsize

        if fx_distance == 'cosine':
            sim_mat = torch.mm(X_test_batch, X_train.T)  # (B,N)
            #sim_mat = torch.matmul(X_test_batch, X_train.T)  # (B,N)
            distances_neighbours, idx_neighbours = sim_mat.topk(k=k, dim=1)  # (B, k)
            sim_neighbours = torch.exp(distances_neighbours / temp)
        elif fx_distance == 'euclidean':
            sim_mat = 1.0 / (torch.cdist(X_test_batch, X_train).pow(2) + eps)  # (B,N)
            sim_neighbours, idx_neighbours = sim_mat.topk(k=k, dim=1)  # (B, k)
        else:
            raise ValueError("Invalid fx_distance", fx_distance)

        #import pdb; pdb.set_trace()
        #print(num_test_samples)

        neighbours_targets = torch.gather(y_train.expand(B, -1), dim=-1, index=idx_neighbours)  # (N,) -> (B,N) -> (B,k)
        neighbours_one_hot = torch.zeros(B * k, num_classes, device=X_test_batch.device)  # (B*k, C)
        class_count_neighbours = torch.scatter( neighbours_one_hot, dim=1, index=neighbours_targets.view(-1, 1), value=1.0).view(B, k, num_classes)  # (B,k,C)
        if weights == "distance":
            y_prob = torch.sum( class_count_neighbours * sim_neighbours.view(B, k, -1) , dim=1)  # bcast sum( (B,k,C) * (B,k,1), dim=1) -> (B,C)
        elif weights == "uniform":
            y_prob = torch.sum( class_count_neighbours, dim=1)  # bcast sum( (B,k,C) , dim=1) -> (B,C)
        else:
            raise ValueError("Invalid weights", weights)
        y_pred = torch.argmax(y_prob, dim=1)  # (B,)
        correct += torch.sum(y_pred == y_test_batch).item()
        #print(y_prob.max())
    return correct / num_test_samples

# DINO
@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=10, use_fp16=False):
    if not use_fp16:
        train_features = F.normalize(train_features, p=2, dim=1)
        test_features = F.normalize(test_features, p=2, dim=1)
    else:
        with torch.cuda.amp.autocast(use_fp16):
            train_features = F.normalize(train_features, p=2, dim=1)
            test_features = F.normalize(test_features, p=2, dim=1)
    top1, top5, total = 0.0, 0.0, 0
    #train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        #similarity = torch.mm(features, train_features)
        similarity = torch.mm(features, train_features.T)
        #import pdb; pdb.set_trace()
        check_if_inf(similarity)

        distances, indices = similarity.topk(k)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()

        check_if_inf(distances_transform)

        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)
        preds = probs.argmax(dim=1)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        #top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top1 += torch.sum(preds == targets).item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1_ = top1 / total
    top5_ = top5 / total

    #import pdb; pdb.set_trace()

    return top1_, top5_

def run_knn_classifier(net, train_loader, test_data_loader, num_classes, use_fp16=False):
    net.eval()
    scores = {}
    c = num_classes
    feature_bank, feature_labels = [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in train_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            if not use_fp16:
                out, feature = net(data)
                out, feature = F.normalize(out, p=2, dim=1), F.normalize(feature, p=2, dim=1)
            else:
                with torch.cuda.amp.autocast(use_fp16):
                    out, feature = net(data)
                    out, feature = F.normalize(out, p=2, dim=1), F.normalize(feature, p=2, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.dataset.targets, device=feature_bank.device)
        feature_labels = torch.cat(feature_labels).to(feature_bank.device)
        # loop test data to predict the label by weighted knn search
        check_if_inf(feature_bank)
        temps = [0.07, 0.5, 0.2]
        #temps = [0.5]
        nns = [20, 200]
        for k in nns:
            for temperature in temps:
                #k = 20
                #temperature = 0.5
                total_top1, total_top5, total_num = 0.0, 0.0, 0,

                for data, target in test_data_loader:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    if not use_fp16:
                        out, feature = net(data)
                        out, feature = F.normalize(out, p=2, dim=1), F.normalize(feature, p=2, dim=1)

                    else:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            out, feature = net(data)
                            out, feature = F.normalize(out, p=2, dim=1), F.normalize(feature, p=2, dim=1)
                    total_num += data.size(0)
                    # compute cos similarity between each feature vector and feature bank ---> [B, N]
                    sim_matrix = torch.mm(feature, feature_bank)
                    # [B, K]
                    sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                    # [B, K]
                    sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                    sim_weight = (sim_weight / temperature).exp()

                    # counts for each class
                    one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
                    # [B*K, C]
                    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                    # weighted score ---> [B, C]
                    pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

                    pred_labels = pred_scores.argsort(dim=-1, descending=True)
                    total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    #test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                    #                         .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
                    # import pdb; pdb.set_trace()
                    scores[f"top1_k{k}_temp{temperature}"] = total_top1 / total_num
                    #scores[f"top5_k{k}_temp{temperature}"] = total_top5 / total_num

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval knn')

    torch.backends.cudnn.benchmark = True

    parser.add_argument('--nn_ks', default=[20, 200], nargs='+',type=int, help='k-nearest neighbors')
    parser.add_argument('--temp', default=[0.07, 0.2, 0.5], nargs='+', type=float, help='Temperature cosine similarity exp(cossim/temp)')
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

        train_basedataset, test_basedataset, test_augmentation, NUM_CLASSES = builddataset_ffcv_x(args.basedataset, transform_mode="test_classifier")
        trainloader = ffcv.loader.Loader(
            f"output/{args.basedataset}/trainds.beton",
            num_workers=args.numworkers,
            batch_size=args.batchsize,
            pipelines={
                "image": test_augmentation.augmentations,
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
                "image": test_augmentation.augmentations,
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

    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cuda:0"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    scores = run_knn_classifier(model, trainloader, testloader, NUM_CLASSES, use_fp16=args.use_fp16)
    print(scores)

    train_features, train_outs, train_targets = encode_tofeatures(model, trainloader, use_fp16=args.use_fp16)
    test_features, test_outs, test_targets = encode_tofeatures(model, testloader, use_fp16=args.use_fp16)

    top1, top5 = knn_classifier(train_features, train_targets, test_features, test_targets, k=20, T=0.5, num_classes=NUM_CLASSES, use_fp16=args.use_fp16)
    print(top1)

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
                NUM_CLASSES,
                k=k,
                fx_distance=args.fx_distance,
                weights=args.weights,
                temp=temp,
                eps=args.eps,
                use_fp16=args.use_fp16
            )

            scores[f"knn_feats_top1_k{k}_temp{temp}_fxdistance{args.fx_distance}_weights{args.weights}"] = acc_features

            acc_outs = run_knn(
                train_outs,
                train_targets,
                test_outs,
                test_targets,
                NUM_CLASSES,
                k=k,
                fx_distance=args.fx_distance,
                weights=args.weights,
                temp=temp,
                eps=args.eps,
                use_fp16=args.use_fp16
            )

            scores[f"knn_outs_top1_k={k}_temp={temp}_fxdistance={args.fx_distance}_weights={args.weights}"] = acc_outs

    for name, val in scores.items():
        print(f'{name}: {val}')
