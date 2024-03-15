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


def data_features(model, dataloader, device):
    H, Z, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x, target = batch
            z, h = model(x.to(device))
            Z.append(z.cpu().numpy())
            H.append(h.cpu().numpy())
            targets.append(target.cpu().numpy())
    Z, H, targets = np.concatenate(Z, axis=0), np.concatenate(H, axis=0), np.concatenate(targets, axis=0)  # (N, Z), (N,H), (N,)
    return Z, H, targets


def evaluate(model, device, args):
    traindata, testdata, _, imgsize, mean, std = dataset_x(args.basedataset, download=False)
    eval_transform = Augmentation(imgsize, mean, std, mode="test", num_views=1)
    # eval_transform = Augmentation(imgsize, mean, std, mode="eval", num_views=1)
    traindata.transform = testdata.transform = eval_transform
    trainloader = DataLoader(traindata, batch_size=args.batchsize, shuffle=True, pin_memory=True)
    # print(traindata.classes)
    testloader = DataLoader(testdata, batch_size=args.batchsize, shuffle=False, pin_memory=True)

    Z_train, H_train, train_targets = data_features(model, trainloader, device)
    Z_test, H_test, test_targets = data_features(model, testloader, device)

    knn1 = KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2)
    knn2 = KNeighborsClassifier(n_neighbors=200, metric='minkowski', p=2)
    linear = LogisticRegression(solver='saga', n_jobs=-1)
    # mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam')

    scores = {}
    # knn1.fit(Z_train, train_targets)
    # scores["knn1_n20_score_Z"] = knn1.score(Z_test, test_targets)
    # knn1.fit(H_train, train_targets)
    # scores["knn1_n20_score_H"] = knn1.score(H_test, test_targets)
    #
    # knn2.fit(Z_train, train_targets)
    # scores["knn2_n200_score_Z"] = knn2.score(Z_test, test_targets)
    # knn2.fit(H_train, train_targets)
    # scores["knn2_n200_score_H"] = knn2.score(H_test, test_targets)

    # linear.fit(H_train, train_targets)
    # scores["linear_score_h"] = linear.score(H_test, test_targets)
    # mlp.fit(H_train, train_targets)
    # scores["mlp_score_h"] = mlp.score(H_test, test_targets)
    # temperature = 0.5

    #temps = [0.07, 0.2, 0.5]
    temps = [0.5]
    knns = [20]
    #knns = [20, 200]
    for nn in knns:
        for temp in temps:
            top1, top5 = test(model, trainloader, testloader, k=nn, temperature=temp)
            scores[f"top1_nn{nn}_temp{temp}"] = top1
        # top1, top5 = test(model, trainloader, testloader, k=200, temperature=0.5)
        # scores["top1_n200_t0.5"] = top1

    return scores


def visualize_feats(model, stage, epoch, device, args):
    traindata, testdata, _, imgsize, mean, std = dataset_x(args.basedataset, download=False)
    eval_transform = Augmentation(imgsize, mean, std, mode="test", num_views=1)
    traindata.transform = testdata.transform = eval_transform
    data = ConcatDataset([traindata, testdata])
    dataloader = DataLoader(data, batch_size=512, shuffle=True)
    Y, _, train_targets = data_features(model, dataloader, device)
    # plot the data
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(*Y.T, c=train_targets, cmap="jet")
    fig.savefig(args.exppath + "/2dfeats_stage_{}_epoch_{}.png".format(stage, epoch))
    plt.close(fig)


########################################################
# Copied from https://github.com/leftthomas/SimCLR/blob/master/main.py
# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, k=20, temperature=0.5):
    net.eval()
    # temperature = 0.5
    # k = 200
    c = 10
    total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            out, feature = net(data.cuda(non_blocking=True))
            out, feature = F.normalize(out, p=2, dim=-1), F.normalize(feature, p=2, dim=-1)
            feature_bank.append(feature)
            feature_labels.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.dataset.targets, device=feature_bank.device)
        feature_labels = torch.cat(feature_labels).to(feature_bank.device)
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out, feature = net(data)
            out, feature = F.normalize(out, p=2, dim=-1), F.normalize(feature, p=2, dim=-1)

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

    return total_top1 / total_num , total_top5 / total_num