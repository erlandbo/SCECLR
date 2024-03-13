import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from data_utils import dataset_x
from data import Augmentation
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt


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
    trainloader = DataLoader(traindata, batch_size=512, shuffle=True)
    testloader = DataLoader(testdata, batch_size=512, shuffle=False)

    Z_train, H_train, train_targets = data_features(model, trainloader, device)
    Z_test, H_test, test_targets = data_features(model, testloader, device)

    knn = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)
    linear = LogisticRegression(solver='saga', n_jobs=-1)
    # mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam')

    scores = {}
    knn.fit(Z_train, train_targets)
    scores["knn_score_Z"] = knn.score(Z_test, test_targets)
    knn.fit(H_train, train_targets)
    scores["knn_score_H"] = knn.score(H_test, test_targets)
    linear.fit(H_train, train_targets)
    scores["linear_score_h"] = linear.score(H_test, test_targets)
    # mlp.fit(H_train, train_targets)
    # scores["mlp_score_h"] = mlp.score(H_test, test_targets)
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


