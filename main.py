import torch
from torch.utils.data import DataLoader, RandomSampler
import argparse

from eval import evaluate, visualize_feats
from data_utils import dataset_x, collate_fn_sce
from data import Augmentation, SCEImageDataset, SSLImageDataset
from models import ResSCECLR, change_model
from criterions.scelosses import SCELoss
from criterions.sceclrlosses import SCECLRLoss
from criterions.tsimcnelosses import InfoNCELoss
from logger_utils import update_pbar, update_log, initialize_logger, write_model
from optimization import auto_lr, build_optimizer

parser = argparse.ArgumentParser(description='SCECLR')

# Model parameters
# ResNet
parser.add_argument('--backbone_depth', default=18, type=int, choices=[18, 34], help='backbone depth resnet 18, 34')
parser.add_argument('--in_channels', default=3, type=int, help='Images')
parser.add_argument('--activation', default="ReLU", type=str, choices=["ReLU", "GELU"])
parser.add_argument('--zero_init_residual', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--mlp_hidden_features', default=1024, type=int)
parser.add_argument('--outfeatures', default=2, type=int, help='Latent space features')
parser.add_argument('--norm_layer', default=True, action=argparse.BooleanOptionalAction, help='whether to use batch-normalization every layers')
parser.add_argument('--hidden_mlp', default=True, action=argparse.BooleanOptionalAction, help='One or none MLP hidden layers')
parser.add_argument('--device', default='cuda', type=str, choices=["cuda", "cpu"])

# Hyperparameters and optimization parameters
parser.add_argument('--batchsize', default=512, type=int)
parser.add_argument('--eval_epoch', default=5, type=int, help='interval for evaluation epoch')
parser.add_argument('--lr', nargs=3, default=(None, None, None), type=float, help='Automatically set from batchsize None')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--lr_anneal', default="cosine_anneal", choices=["cosine_anneal", "linear_anneal"])
parser.add_argument('--epochs', nargs=3, default=(1000, 450, 250), type=int)
parser.add_argument('--warmupepochs', nargs=3, default=(10, 10, 10), type=int)
parser.add_argument('--numworkers', default=0, type=int)

# Loss function
parser.add_argument('--criterion', default='sceclr', type=str, choices=["sce", "sceclr", "infonce"])
parser.add_argument('--metric', default="student-t", type=str, choices=["student-t", "gaussian", "cosine", "dotprod"])

# SCE and SCECLR
parser.add_argument('--rho', default=-1., type=float, help='Set constant rho, or automatically from batchsize -1')
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--s_init', default=2.0, type=float)

# Data
parser.add_argument('--basedataset', default='cifar10', type=str, choices=["cifar10", "cifar100"])
parser.add_argument('--imgsize', nargs=2, default=(32, 32), type=int)
parser.add_argument('--augmode', default='train', type=str, choices=["train", "eval", "test"], help=
    "augmentation train mode simclr, eval mode random resize-crop flip , test mode nothing")

# Logging
parser.add_argument('--mainpath', default="./", type=str, help="path to store logs from experiments")

# TODO remove?
parser.add_argument('--traindataset', default='ssl',choices=["ssl", "sce"], type=str, help="Generic SSL dataset or SCE")


def train_one_epoch(model, dataloader, criterion, optimizer, lr_schedule, device, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch * len(dataloader) + batch_idx]

        x1, x2 = batch
        x = torch.cat([x1, x2], dim=0).to(device)
        z, _ = model(x)
        optimizer.zero_grad()
        loss = criterion(z)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        update_pbar(batch_idx, num_batches=len(dataloader))

    return running_loss / len(dataloader)


def main():
    args = parser.parse_args()

    logger = initialize_logger(args)

    device = torch.device("cuda:0" if args.device=="cuda" else "cpu")

    traindata, testdata, _ , imgsize, mean, std = dataset_x(args.basedataset)
    # TODO remove?
    if args.traindataset == "sce":
        augmentation = Augmentation(imgsize, mean, std, mode="train", num_views=1)
        dataset = SCEImageDataset(traindata, augmentation, args.sce_triplet)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            sampler=RandomSampler(data_source=dataset, replacement=True),
            num_workers=args.numworkers,
            collate_fn=collate_fn_sce
        )
    else:  # standard ssl
        augmentation = Augmentation(imgsize, mean, std, mode="train", num_views=2)
        dataset = SSLImageDataset(traindata, augmentation)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=True,
            num_workers=args.numworkers
        )

    if args.criterion == "sce":
        criterion = SCELoss(
            metric=args.metric,
            N=len(dataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "sceclr":
        criterion = SCECLRLoss(
            metric=args.metric,
            N=len(dataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "infonce":
        criterion = InfoNCELoss(
            metric=args.metric
        ).to(device)

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

    for i in range(3):

        if i == 1:
            model = change_model(model, projection_dim=2, device=device, freeze_layer="keeplast", change_layer="last")
        elif i == 2:
            model = change_model(model, device=device, freeze_layer=None)

        base_lri = auto_lr(args.batchsize) if args.lr[i] is None else args.lr[i]
        optimizer_i, lr_schedule_i = build_optimizer(
            model=model,
            lr=base_lri,
            warmup_epochs=args.warmupepochs[i],
            max_epochs=args.epochs[i],
            num_batches=len(dataloader),
            cosine_anneal=args.lr_anneal,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        write_model(model, args)

        for epoch in range(0, args.epochs[i]):
            epoch_loss = train_one_epoch(model, dataloader, criterion, optimizer_i, lr_schedule_i, device, epoch)

            scores = None
            if epoch % args.eval_epoch == 0:
                scores = evaluate(model, device, args)
                if model.qprojector.mlp[-1].weight.shape[0] == 2:
                    visualize_feats(model, stage=i, epoch=epoch, device=device, args=args)

            update_log(
                logger,
                stage=i,
                epoch=epoch,
                epoch_loss=epoch_loss,
                lr=lr_schedule_i[(epoch+1) * len(dataloader)-1],
                scores=scores,
                buffer_vals=" ".join([f"{name}:{val.item()}" for (name, val) in criterion.named_buffers()])
            )


        torch.save(model.state_dict(), args.exppath + "/model_stage_{}.pth".format(i))
        torch.save(criterion.state_dict(), args.exppath + "/loss_stage_{}.pth".format(i))


if __name__ == "__main__":
    main()

