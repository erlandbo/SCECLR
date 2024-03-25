import torch
from torch.utils.data import DataLoader
import argparse
import math
import sys
from data import SSLImageDataset, builddataset_x
from models import ResProjModel, change_model
from criterions.scelosses import SCELoss
from criterions.clsav1global import CLSAv1Loss
from criterions.clsav2 import CLSAv2Loss
from criterions.clsav2NoPos import CLSAv2NoPosLoss
from criterions.clsav2global import CLSAv2GlobalLoss
from criterions.clsav4 import CLSAv4Loss
from criterions.clsav4NoPos import CLSAv4NoPosLoss
from criterions.clbnglobal import CLSAv3GlobalLoss
from criterions.tsimcnelosses import InfoNCELoss
from criterions.sogclr import SogCLR
from logger_utils import initialize_logger, store_hyperparameters
from optimization import auto_lr, build_optimizer, build_optimizer_epoch
import time
import numpy as np


parser = argparse.ArgumentParser(description='CLSA')

# Model parameters
# ResNet
parser.add_argument('--backbone_depth', default=18, type=int, choices=[18, 34], help='backbone depth resnet 18, 34')
parser.add_argument('--in_channels', default=3, type=int, help='Images')
parser.add_argument('--activation', default="ReLU", type=str, choices=["ReLU", "GELU"])
parser.add_argument('--zero_init_residual', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--mlp_hidden_features', default=1024, type=int)
parser.add_argument('--mlp_outfeatures', default=2, type=int, help='Latent space features')
parser.add_argument('--norm_layer', default=True, action=argparse.BooleanOptionalAction, help='whether to use batch-normalization every layers')
parser.add_argument('--norm_mlp_layer', default=True, action=argparse.BooleanOptionalAction, help='whether to use batch-normalization last mlp layer')
parser.add_argument('--hidden_mlp', default=True, action=argparse.BooleanOptionalAction, help='One or none MLP hidden layers')
parser.add_argument('--device', default='cuda:0', type=str, choices=["cuda:0", "cpu"])

parser.add_argument('--first_conv', default=False, action=argparse.BooleanOptionalAction, help='whether to use batch-normalization last mlp layer')
parser.add_argument('--first_maxpool', default=False, action=argparse.BooleanOptionalAction, help='whether to use batch-normalization last mlp layer')

# Hyperparameters and optimization parameters
parser.add_argument('--batchsize', default=512, type=int)
parser.add_argument('--lr', nargs=3, default=(None, None, None), type=float, help='Automatically set from batchsize if None')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--lr_anneal', default="cosine_anneal", choices=["cosine_anneal", "linear_anneal"])
parser.add_argument('--epochs', nargs=3, default=(1000, 450, 250), type=int)
parser.add_argument('--warmupepochs', nargs=3, default=(10, 0, 10), type=int)
parser.add_argument('--numworkers', default=0, type=int)

parser.add_argument('--rseed', default=None, type=int)

# Loss function
parser.add_argument('--criterion', default='sce', type=str, choices=["clsav1", "clsav2","clsav2nopos", "clsav2global","clsav4", "clsav4NoPos", "sce", "infonce", "sogclr"])
parser.add_argument('--metric', default="cauchy", type=str, choices=["cauchy", "gaussian", "cosine"])

# SCE and SCECLR
parser.add_argument('--rho', default=-1., type=float, help='Set constant rho, or automatically from batchsize -1')
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--s_init', default=2.0, type=float)

# Data
parser.add_argument('--basedataset', default='cifar10', type=str, choices=["cifar10", "cifar100", "stl10_unlabeled", "stl10_labeled", "imagenette", "oxfordIIItpet"])


parser.add_argument('--checkpoint_interval', default=100, type=int, help='interval for saving checkpoint')

parser.add_argument('--use_ffcv', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--use_fp16', default=False, action=argparse.BooleanOptionalAction)

# Logging
parser.add_argument('--mainpath', default="./", type=str, help="path to store logs from experiments")

# Re-start training
parser.add_argument('--checkpoint_path', default="", type=str, help="path to model weights")
parser.add_argument('--start_stage', default=0, type=int, choices=[0, 1, 2], help="start stage of training")
# Change criterion for later stages. Can be combined with re-starting training from saved checkpoint
parser.add_argument('--change_metric', default=None, type=str, choices=["cauchy", "heavy-tailed", "gaussian", "cosine", "dotprod"])
parser.add_argument('--change_rho', default=None, type=float, help='Set constant rho, or automatically from batchsize -1')
parser.add_argument('--change_alpha', default=None, type=float)


def main():
    args = parser.parse_args()

    logger = initialize_logger(args)

    store_hyperparameters(args)

    device = torch.device("cuda:0")

    if not args.use_ffcv:
        train_basedataset, val_basedataset, contrastive_augmentation, NUM_CLASSES = builddataset_x(args.basedataset, transform_mode="contrastive_pretrain")
        train_dataset = SSLImageDataset(train_basedataset, contrastive_augmentation)
        trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,num_workers=args.numworkers,pin_memory=True, drop_last=False)
    else:
        from data_ffcv_ssl import builddataset_ffcv_x
        import ffcv
        train_basedataset, val_basedataset, contrastive_augmentation, NUM_CLASSES = builddataset_ffcv_x(args.basedataset, transform_mode="contrastive_pretrain")
        trainloader = ffcv.loader.Loader(
            f"output/{args.basedataset}/ssltrainds.beton",
            num_workers=args.numworkers,
            batch_size=args.batchsize,
            pipelines={
                "image": contrastive_augmentation.augmentations,
                "image2": contrastive_augmentation.augmentations,
                "label": [ffcv.fields.basics.IntDecoder(),ffcv.transforms.ops.ToTensor(),ffcv.transforms.common.Squeeze(1)],
                "idx": [ffcv.fields.basics.IntDecoder(),ffcv.transforms.ops.ToTensor(),ffcv.transforms.common.Squeeze(1)],
            },
                custom_field_mapper={"image2": "image"},
                order=ffcv.loader.OrderOption.RANDOM,
                drop_last=False,
                os_cache=True,
                seed=42
            )

    if args.criterion == "sce":
        criterion = SCELoss(
            metric=args.metric,
            N=len(train_basedataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "clsav1":
        criterion = CLSAv1Loss(
            metric=args.metric,
            N=len(train_basedataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "clsav2":
        criterion = CLSAv2Loss(
            metric=args.metric,
            N=len(train_basedataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "clsav2nopos":
        criterion = CLSAv2NoPosLoss(
            metric=args.metric,
            N=len(train_basedataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "clsav2global":
        criterion = CLSAv2GlobalLoss(
            metric=args.metric,
            N=len(train_basedataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "clsav4":
        criterion = CLSAv4Loss(
            metric=args.metric,
            N=len(train_basedataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "clsav4NoPos":
        criterion = CLSAv4NoPosLoss(
            metric=args.metric,
            N=len(train_basedataset),
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.criterion == "infonce":
        criterion = InfoNCELoss(
            metric=args.metric
        ).to(device)
    elif args.criterion == "sogclr":
        criterion = SogCLR(
            N=len(train_basedataset),
        ).to(device)

    model = ResProjModel(
        backbone_depth=args.backbone_depth,
        in_channels=args.in_channels,
        activation=args.activation,
        zero_init_residual=args.zero_init_residual,
        mlp_hidden_features=args.mlp_hidden_features,
        mlp_outfeatures=args.mlp_outfeatures,
        norm_mlp_layer=args.norm_mlp_layer,
        hidden_mlp=args.hidden_mlp,
        first_conv=args.first_conv,
        first_maxpool=args.first_maxpool,
    ).to(device)

    torch.backends.cudnn.benchmark = True

    if args.rseed is not None:
        torch.cuda.manual_seed_all(args.rseed)
        np.random.seed(args.rseed)
        torch.manual_seed(args.rseed)

    for stage in range(args.start_stage, 3):

        if stage == 1:
            model = change_model(model, projection_dim=2, device=device, freeze_layer="keeplast", change_layer="last")
            # model = change_model(model, projection_dim=2, device=device, freeze_layer="mixer", change_layer="mlp")

        elif stage == 2:
            model = change_model(model, device=device, freeze_layer=None)

        print(model)
        print("Starting training stage {} Time {}".format(stage, time.strftime("%Y_%m_%d_%H_%M_%S")))
        start_time_training = time.time()

        base_lr = auto_lr(args.batchsize) if stage < 2 else auto_lr(args.batchsize) / 1000 if args.lr[stage] is None else args.lr[stage]

        optimizer, lr_schedule = build_optimizer_epoch(model,base_lr,args.warmupepochs[stage],args.epochs[stage],args.lr_anneal,args.momentum,args.weight_decay)

        scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

        for epoch in range(1, args.epochs[stage] + 1):
            model.train()

            start_time_epoch = time.time()

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[epoch]

            running_loss = 0.0

            for batch_idx, batch in enumerate(trainloader):

                optimizer.zero_grad()

                x1, target, idx, x2 = batch

                x1 = x1.cuda(non_blocking=True)
                x2 = x2.cuda(non_blocking=True)
                idx = idx.cuda(non_blocking=True)

                x = torch.cat([x1, x2], dim=0)

                #print(x.shape)

                if scaler is None:
                    z, _ = model(x)
                    loss = criterion(z, idx)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.cuda.amp.autocast(args.use_fp16):
                        z, _ = model(x)
                        loss = criterion(z, idx)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item()

                if not math.isfinite(loss.item()):
                    print("Break training infinity value in loss", force=True)
                    sys.exit(1)

            epoch_loss = running_loss / len(trainloader)

            stats = {"loss": epoch_loss, "lr": lr_schedule[epoch]}
            log_str = f'Time:{time.time() - start_time_epoch}seconds Stage:{stage} Epoch:{epoch} LR:{lr_schedule[epoch]} Loss:{epoch_loss}'
            logger.info(log_str)
            print(log_str)

            if epoch % args.checkpoint_interval == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "criterion_state_dict": criterion.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "epoch": epoch
                }, args.exppath + "/checkpoint_stage_{}_epoch_{}.pth".format(stage, epoch))

        print("Completed training stage {} Time {} Training took {} seconds".format(stage, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), time.time()-start_time_training))

        torch.save({
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": args.epochs[stage]
        }, args.exppath + "/checkpoint_stage_{}.pth".format(stage))


if __name__ == "__main__":
    main()

