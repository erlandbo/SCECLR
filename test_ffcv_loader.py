from ffcv_ssl import build_ffcv_sslloader, build_ffcv_nonsslloader
from data_utils import dataset_x
import numpy as np


if __name__ == '__main__':
    train_basedataset, test_basedataset, num_classes, imgsize, mean, std = dataset_x("cifar10")

    def get_loader(name):
        trainloader_ssl = build_ffcv_sslloader(
            write_path=f"output/cifar10/ssltrainds.beton",
            mean=mean,
            std=std,
            imgsize=imgsize,
            batchsize=512,
            numworkers=20,
            mode="test"
        )

        trainloader_nossl = build_ffcv_nonsslloader(
            write_path=f"output/cifar10/trainds.beton",
            mean=mean,
            std=std,
            imgsize=imgsize,
            batchsize=512,
            numworkers=20,
            mode="train"
        )
        testloader_nossl = build_ffcv_nonsslloader(
            write_path=f"output/cifar10/testds.beton",
            mean=mean,
            std=std,
            imgsize=imgsize,
            batchsize=512,
            numworkers=20,
            mode="test"
        )

        if name == "train_ssl": return trainloader_ssl
        elif name == "train_nossl": return trainloader_nossl
        elif name == "test_nossl": return testloader_nossl
        else: pass

    loader = get_loader("test_nossl")

    for epoch in range(1):

        indices = []

        for batch_idx, batch in enumerate(loader):
            #x1, target, idx, x2 = batch
            x1, y = batch
            x1 = x1.cuda(non_blocking=True)
            #x2 = x2.cuda(non_blocking=True)
            #idx = idx.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            indices.extend(y.cpu().numpy().tolist())

        print(indices)
        #print(len(indices))
        print(all(np.array(indices) == np.array(test_basedataset.targets)) )
        print(indices[0:10])
        print(test_basedataset.targets[0:10])
        #print(test_basedataset.targets)
