import torch
from torch.utils.data import DataLoader, RandomSampler
import argparse

from eval import evaluate, visualize_feats
from data_utils import dataset_x
from data import Augmentation, SSLImageDataset
from models import ResSCECLR, change_model
from criterions.scelosses import SCELoss
from criterions.scempairlosses import SCEMPairLoss
from criterions.sceclrlossesv1 import SCECLRV1Loss
from criterions.sceclrlossesv2 import SCECLRV2Loss
from criterions.tsimcnelosses import InfoNCELoss
from logger_utils import update_pbar, update_log, initialize_logger, write_model
from optimization import auto_lr, build_optimizer
from criterions.criterion_utils import change_criterion

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
parser.add_argument('--eval_epoch', default=10, type=int, help='interval for evaluation epoch')
parser.add_argument('--lr', nargs=3, default=(None, None, None), type=float, help='Automatically set from batchsize if None')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--lr_anneal', default="cosine_anneal", choices=["cosine_anneal", "linear_anneal"])
parser.add_argument('--epochs', nargs=3, default=(1000, 450, 250), type=int)
parser.add_argument('--warmupepochs', nargs=3, default=(10, 0, 10), type=int)
parser.add_argument('--numworkers', default=0, type=int)

# Loss function
parser.add_argument('--criterion', default='sce', type=str, choices=["sce", "sceclrv1", "scempair", "infonce"])
parser.add_argument('--metric', default="cauchy", type=str, choices=["cauchy", "heavy-tailed", "gaussian", "cosine", "dotprod"])

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

# Re-start training
parser.add_argument('--checkpoint_path', default="", type=str, help="path to model weights")
parser.add_argument('--start_stage', default=0, type=int, choices=[0, 1, 2], help="start stage of training")
# Change criterion for later stages. Can be combined with re-starting training from saved checkpoint
parser.add_argument('--change_metric', default=None, type=str, choices=["cauchy", "heavy-tailed", "gaussian", "cosine", "dotprod"])
parser.add_argument('--change_rho', default=None, type=float, help='Set constant rho, or automatically from batchsize -1')
parser.add_argument('--change_alpha', default=None, type=float)


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
    if args.criterion == "sceclrv1":
        criterion = SCECLRV1Loss(
            metric=args.metric,
            N=len(dataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "sceclrv2":
        criterion = SCECLRV2Loss(
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

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        criterion.load_state_dict(checkpoint["criterion_state_dict"])

    for stage in range(args.start_stage, 3):

        if stage == 1:
            # model = change_model(model, projection_dim=2, device=device, freeze_layer="keeplast", change_layer="last")
            model = change_model(model, projection_dim=2, device=device, freeze_layer="mixer", change_layer="mlp")

            if args.change_metric:
                criterion = change_criterion(criterion, device, args.change_metric, new_rho=args.change_rho, new_alpha=args.change_alpha)

        elif stage == 2:
            model = change_model(model, device=device, freeze_layer=None)

        base_lri = auto_lr(args.batchsize) if stage < 2 else auto_lr(args.batchsize) / 1000 if args.lr[stage] is None else args.lr[stage]
        optimizer_i, lr_schedule_i = build_optimizer(
            model=model,
            lr=base_lri,
            warmup_epochs=args.warmupepochs[stage],
            max_epochs=args.epochs[stage],
            num_batches=len(dataloader),
            lr_anneal=args.lr_anneal,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        write_model(model, args)

        for epoch in range(0, args.epochs[stage]):
            epoch_loss = train_one_epoch(model, dataloader, criterion, optimizer_i, lr_schedule_i, device, epoch)

            scores = None
            if epoch % args.eval_epoch == 0:
                scores = evaluate(model, device, args)
                if model.qprojector.mlp[-1].weight.shape[0] == 2:
                    visualize_feats(model, stage=stage, epoch=epoch, device=device, args=args)

            update_log(
                logger,
                stage=stage,
                epoch=epoch,
                epoch_loss=epoch_loss,
                lr=lr_schedule_i[(epoch+1) * len(dataloader)-1],
                scores=scores,
                buffer_vals=" ".join([f"{name}:{val.item()}" for (name, val) in criterion.named_buffers()])
            )


        torch.save({
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
        }, args.exppath + "/checkpoint_stage_{}.pth".format(stage))

        # torch.save(model.state_dict(), args.exppath + "/model_stage_{}.pth".format(stage))
        # torch.save(criterion.state_dict(), args.exppath + "/loss_stage_{}.pth".format(stage))


if __name__ == "__main__":
    main()

