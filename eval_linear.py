import numpy as np
import torch
import math
import sys
from data_utils import dataset_x
from data import Augmentation
from torch.utils.data import DataLoader, ConcatDataset
import argparse
from models import build_model_from_hparams, change_model
from logger_utils import read_hyperparameters
from torch import nn
from optimization import build_optimizer_epoch, auto_lr
import time


@torch.no_grad()
def linear_evaluation(backbone_model, linear_classifier, criterion, dataloader, use_fp16=False):
    linear_classifier.eval()
    backbone_model.eval()

    running_loss = 0.0
    num_correct = 0.0
    num_total = 0

    for batch_idx, batch in enumerate(dataloader):

        x, y = batch

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        if not use_fp16:
            with torch.no_grad():
                z, h = backbone_model(x)
            h = h.detach()
            logits = linear_classifier(h)
            loss = criterion(logits, y)
        else:
            with torch.cuda.amp.autocast(args.use_fp16):
                with torch.no_grad():
                    z, h = backbone_model(x)
                h = h.detach()
                logits = linear_classifier(h)
                loss = criterion(logits, y)

        running_loss += loss.item()
        y_pred = torch.argmax(logits, dim=1)  # (B,)
        num_correct += torch.sum(y_pred == y).item()
        num_total += y.shape[0].item()

        if not math.isfinite(loss.item()):
            print("Break testing infinity value in loss", force=True)
            sys.exit()

    avg_loss = running_loss / len(trainloader)
    acc = num_correct / num_total

    return avg_loss, acc


class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval linear')

    torch.backends.cudnn.benchmark = True

    # Hyperparameters and optimization parameters
    parser.add_argument('--batchsize', default=512, type=int)
    parser.add_argument('--base_lr', default=None, type=float,help='Automatically set from batchsize if None')

    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--lr_anneal', default="cosine_anneal", choices=["cosine_anneal", "linear_anneal"])
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmupepochs', default=10, type=int)
    parser.add_argument('--numworkers', default=0, type=int)

    parser.add_argument('--basedataset', default='cifar10', type=str, choices=["cifar10", "cifar100"])

    parser.add_argument('--backbone_hparams_path', required=True, type=str, help="path to hparams to re-construct backbone model")
    parser.add_argument('--backbone_checkpoint_path', required=True, type=str, help="path to model weights")

    parser.add_argument('--use_ffcv', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_fp16', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--use_2dfeats', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

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
    hparams = read_hyperparameters(args.hparams_path)
    backbone_model = build_model_from_hparams(hparams)

    if args.use_2dfeats:
        backbone_model = change_model(backbone_model, projection_dim=2, device=torch.device("cuda:0"), change_layer="last")

    checkpoint = torch.load(args.checkpoint_path)
    backbone_model.load_state_dict(checkpoint['model_state_dict'])
    backbone_model.cuda()

    for name, param in backbone_model.named_parameters():
        param.requires_grad = False
    backbone_model.eval()

    print(backbone_model)

    in_features = backbone_model.qprojector.mlp[-1].weight.shape[1]
    linear_classifier = LinearClassifier(in_features=in_features, out_features=num_classes)

    linear_classifier.train()

    print(linear_classifier)

    base_lr = auto_lr(args.batchsize) if args.base_lr is None else args.base_lr

    optimizer, lr_schedule = build_optimizer_epoch(linear_classifier, base_lr, args.warmupepochs, args.epochs,args.lr_anneal, args.momentum, args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    criterion = nn.CrossEntropyLoss()

    print("Starting training Time {}".format(time.strftime("%Y_%m_%d_%H_%M_%S")))
    start_time_training = time.time()

    for epoch in range(1, args.epochs + 1):
        linear_classifier.train()
        backbone_model.eval()

        start_time_epoch = time.time()

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch]

        running_loss = 0.0
        num_correct = 0.0
        num_total = 0

        for batch_idx, batch in enumerate(trainloader):

            optimizer.zero_grad()

            x, y = batch

            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            if scaler is None:
                with torch.no_grad():
                    z, h = backbone_model(x)
                h = h.detach()
                logits = linear_classifier(h)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            else:
                with torch.cuda.amp.autocast(args.use_fp16):
                    with torch.no_grad():
                        z, h = backbone_model(x)
                    h = h.detach()
                    logits = linear_classifier(h)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()

            y_pred = torch.argmax(logits, dim=1)  # (B,)
            num_correct += torch.sum(y_pred == y).item()
            num_total += y.shape[0].item()

            if not math.isfinite(loss.item()):
                print("Break training infinity value in loss", force=True)
                sys.exit()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = num_correct / num_total

        log_str = f'Time:{time.time() - start_time_epoch}seconds Epoch:{epoch} LR:{lr_schedule[epoch]} Loss:{epoch_loss} ACC:{epoch_acc}'
        #logger.info(log_str)
        print(log_str)

        test_loss, test_acc = linear_evaluation(backbone_model, linear_classifier, criterion, testloader, use_fp16=args.use_fp16)
        log_str = f'Validation: Loss:{test_loss} ACC:{test_acc}'


