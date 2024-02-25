import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from data_utils import dataset_x
from data import Augmentation
from torch.utils.data import DataLoader


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
    H, Z, targets = np.concatenate(H, axis=0), np.concatenate(Z, axis=0), np.concatenate(targets, axis=0)  # (N, H), (N,Z), (N,)
    return H, Z, targets


def evaluate(model, device, args):
    traindata, testdata, _, imgsize, mean, std = dataset_x(args.dataset)
    eval_transform = Augmentation(imgsize, mean, std, mode="eval", num_views=1)
    traindata.transform = testdata.transform = eval_transform
    trainloader = DataLoader(traindata, batch_size=512, shuffle=True)
    testloader = DataLoader(testdata, batch_size=512, shuffle=False)

    H_train, Z_train, train_targets = data_features(model, trainloader, device)
    H_test, Z_test, test_targets = data_features(model, testloader, device)

    knn = KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2)
    linear = LogisticRegression(solver='saga', n_jobs=-1)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam')

    scores = {}
    knn.fit(Z_train, Z_train)
    scores["knn_score_z"] = knn.score(Z_test, Z_test)
    knn.fit(H_train, H_train)
    scores["knn_score_h"] = knn.score(H_test, H_test)
    linear.fit(H_train, H_train)
    scores["linear_score_h"] = linear.score(H_test, H_test)
    mlp.fit(H_train, H_train)
    scores["mlp_score_h"] = mlp.score(H_test, H_test)

    return scores
