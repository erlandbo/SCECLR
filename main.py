import numpy as np
import torch
from eval import evaluate
from data_utils import dataset_x, collate_fn_sce
from data import Augmentation, SCEImageDataset, SSLImageDataset
from Models import ResSCECLR, change_model
from Loss import SCELoss
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from torch.optim import SGD
import argparse
from tqdm import trange


parser = argparse.ArgumentParser(description='SCECLR')

parser.add_argument('--basedataset', default='cifar10', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--traindataset', default='sce', type=str)
parser.add_argument('--combinetraintest', default=False, action=argparse.BooleanOptionalAction, help='whether to combine train and test for visualization')
parser.add_argument('--imgsize', nargs=2, default=(32, 32), type=int)
parser.add_argument('--augmode', default='train', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--lr', nargs=3, default=None, type=int, help='learning rate for the 3 train stages. If None automatically set by batchsize')
parser.add_argument('--epochs', nargs=3, default=(1000, 450, 250), type=int, help='epochs for the 3 train stages')
parser.add_argument('--numworkers', default=10, type=int)

parser.add_argument('--loss_fn', default='sce', type=str, help='loss function to use')
# SCELoss
parser.add_argument('--rho', default=-1, type=int, help='constant rho parameter for sce-loss or disable -1 for automatically set by batchsize')
parser.add_argument('--alpha', default=0.5, type=int)
parser.add_argument('--s_init', default=2.0, type=int)
parser.add_argument('--metric', default="student-t", type=str, help='similarity metric to use')
parser.add_argument('--sce_triplet', default=False, action=argparse.BooleanOptionalAction, help='whether to form triplets in sce')

# ResNet
parser.add_argument('--backbone_depth', default=18, type=int, help='ResNet backbone depth 18, 34')
parser.add_argument('--in_channels', default=3, type=int, help='Initial in channels for the images')
parser.add_argument('--activation', default="ReLU", type=str)
parser.add_argument('--zero_init_residual', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--mlp_hidden_features', default=1024, type=int)
parser.add_argument('--outfeatures', default=2, type=int, help='Latent space features')
parser.add_argument('--norm_layer', default=True, action=argparse.BooleanOptionalAction, help='whether to use batch-normalization layers')
parser.add_argument('--hidden_mlp', default=True, action=argparse.BooleanOptionalAction, help='One or none MLP hidden layers')


def train_one_epoch(model, dataloader, loss_fn, optimizer, lr_schedule, device, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        x1, x2 = batch
        x = torch.cat([x1, x2], dim=0).to(device)
        z, h = model(x)
        optimizer.zero_grad()
        loss = loss_fn(z)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    lr_schedule.step()
    return running_loss / len(dataloader)


def train(model, dataloader, loss_fn, optimizer, lr_schedule, device, epochs, args):
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, lr_schedule, device, epoch)
        print(epoch, epoch_loss)

        if epoch % args.eval_epoch == 0:
            scores = evaluate(model, device, args)
            print(scores)


def main():
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.device=="cuda" else "cpu")

    traindata, testdata, _ , imgsize, mean, std = dataset_x(args.basedataset)
    if args.combinetraintest:
        dataset = ConcatDataset([traindata, testdata])
    else:
        dataset = traindata
    if args.traindataset == "sce":
        augmentation = Augmentation(imgsize, mean, std, mode="train", num_views=1)
        dataset = SCEImageDataset(dataset, augmentation, args.sce_triplet)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            sampler=RandomSampler(data_source=dataset, replacement=True),
            num_workers=args.numworkers,
            collate_fn=collate_fn_sce
        )
    else:  # standard ssl
        augmentation = Augmentation(imgsize, mean, std, mode="train", num_views=2)
        dataset = SSLImageDataset(dataset, augmentation)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.numworkers
        )

    if args.loss_fn == "sce":
        loss_fn = SCELoss(
            N=len(dataloader),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
            metric=args.metric
        ).to(device)
    else:
        loss_fn = None


    model = ResSCECLR(
        backbone_depth=args.backbone_depth,
        in_channels=args.in_channels,
        activation=args.activation,
        zero_init_residual=args.zero_init_residual,
        mlp_hidden_features=args.mlp_hidden_features,
        mlp_outfeatures=args.outfeatures,
        norm_layer=args.norm_layer,
        hidden_mlp=args.hidden_mlp
    ).to(device)

    optimizer_1 = SGD(model.parameters(), lr=args.lr[0] if args.lr else 1e-4, momentum=0.9, weight_decay=5e-4)
    lr_schedule_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=100)


    train(model, dataloader, loss_fn, optimizer_1, lr_schedule_1, device, args.epochs[0], args)


if __name__ == "__main__":
    main()

