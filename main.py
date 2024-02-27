import torch
from torch.utils.data import DataLoader, RandomSampler
import argparse

from eval import evaluate, visualize_feats
from data_utils import dataset_x, collate_fn_sce
from data import Augmentation, SCEImageDataset, SSLImageDataset
from Models import ResSCECLR, change_model
from criterions.scelosses import SCELoss
from criterions.sceclrlosses import SCECLRLoss
from criterions.tsimcnelosses import InfoNCECauchy
from logger_utils import update_pbar, update_log, initialize_logger, write_model
from optimization import auto_lr, build_optimizer


parser = argparse.ArgumentParser(description='SCECLR')

parser.add_argument('--basedataset', default='cifar10', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--traindataset', default='ssl', type=str)
parser.add_argument('--imgsize', nargs=2, default=(32, 32), type=int)
parser.add_argument('--augmode', default='train', type=str)
parser.add_argument('--batchsize', default=512, type=int)
parser.add_argument('--eval_epoch', default=5, type=int, help='knn and mlp evaluation epochs')
parser.add_argument('--lr', nargs=3, default=(None, None, None), type=float, help='learning rate for the 3 train stages. If None automatically set by batchsize')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--cosine_anneal', default=True, action=argparse.BooleanOptionalAction, help='cosine or linear anneal lr')
parser.add_argument('--epochs', nargs=3, default=(1000, 450, 250), type=int, help='epochs for the 3 train stages')
parser.add_argument('--warmupepochs', nargs=3, default=(10, 10, 10), type=int, help='warmup epochs for the 3 train stages')
parser.add_argument('--numworkers', default=10, type=int)

parser.add_argument('--mainpath', default="./", type=str)

parser.add_argument('--loss_fn', default='sce', type=str, help='loss function to use')
# SCELoss
parser.add_argument('--rho', default=-1., type=float, help='constant rho parameter for sce-loss or disable -1 for automatically set by batchsize')
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--s_init', default=2.0, type=float)
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

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch * len(dataloader) + batch_idx]

        x1, x2 = batch
        x = torch.cat([x1, x2], dim=0).to(device)
        z, _ = model(x)
        optimizer.zero_grad()
        loss = loss_fn(z)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        update_pbar(batch_idx, len(dataloader))

    return running_loss / len(dataloader)


def main():
    args = parser.parse_args()

    logger = initialize_logger(args)

    device = torch.device("cuda:0" if args.device=="cuda" else "cpu")

    traindata, testdata, _ , imgsize, mean, std = dataset_x(args.basedataset)
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

    N_dataitems = len(dataset)
    num_batches = len(dataloader)

    if args.loss_fn == "sce":
        loss_fn = SCELoss(
            N=N_dataitems,
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.loss_fn == "sceclr":
        loss_fn = SCECLRLoss(
            metric=args.metric,
            N=N_dataitems,
            rho=args.rho,
            alpha=args.alpha,
            S_init=args.s_init,
        ).to(device)
    elif args.loss_fn == "infonce":
        loss_fn = InfoNCECauchy()


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
            model = change_model(model,device=device, freeze_layer=None)

        base_lri = auto_lr(args.batchsize) if args.lr[i] is None else args.lr[i]
        optimizer_i, lr_schedule_i = build_optimizer(
            model=model,
            lr=base_lri,
            warmup_epochs=args.warmupepochs[i],
            max_epochs=args.epochs[i],
            num_batches=num_batches,
            cosine_anneal=args.cosine_anneal,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        write_model(model, args)
        # train(model, dataloader, loss_fn, optimizer_i, lr_schedule_i, device, args.epochs[i], args, stage=i)

        for epoch in range(0, args.epochs[i]):
            epoch_loss = train_one_epoch(model, dataloader, loss_fn, optimizer_i, lr_schedule_i, device, epoch)

            scores = None
            if epoch % args.eval_epoch == 0:
                scores = evaluate(model, device, args)
                if model.qprojector.mlp[-1].weight.shape[0] == 2:
                    visualize_feats(model, stage=i, epoch=epoch, device=device, args=args)

            update_log(logger, i, epoch, epoch_loss, lr_schedule_i[(epoch+1) * N_dataitems-1], scores)

            # import pdb; pdb.set_trace()

            # print(loss_fn)

        torch.save(model.state_dict(), args.exppath + "/model_stage_{}.pth".format(i))
        torch.save(loss_fn.state_dict(), args.exppath + "/loss_stage_{}.pth".format(i))


if __name__ == "__main__":
    main()

