import torch

from criterions.scelosses import SCELoss
from criterions.sceclrlosses import SCECLRLoss
from criterions.tsimcnelosses import InfoNCELoss


def change_criterion(criterion, device, new_metric, new_rho=None, new_alpha=None):
    criterion_name = criterion.__class__.__name__.lower().split("loss")[0]
    if criterion_name == "infonce":
        new_criterion = InfoNCELoss(
            metric=new_metric
        ).to(device)
    else:
        if criterion_name == "sce":
            new_criterion = SCELoss(
                metric=new_metric,
                N=criterion.criterion.N.item(),
                rho=criterion.criterion.rho.item(),
                alpha=criterion.criterion.alpha.item(),
            ).to(device)
        elif criterion_name == "sceclr":
            new_criterion = SCECLRLoss(
                metric=new_metric,
                N=criterion.criterion.N.item(),
                rho=criterion.criterion.rho.item(),
                alpha=criterion.criterion.alpha.item(),
            ).to(device)
        old_criterion_buffers = dict(criterion.named_buffers())
        new_criterion_buffers = dict(new_criterion.named_buffers())
        for name, buff_value in old_criterion_buffers.items():
            new_criterion_buffers[name].copy_(buff_value)

        if new_rho: new_criterion_buffers["criterion.rho"].copy_(torch.zeros(1, device=device) + new_rho)
        if new_alpha: new_criterion_buffers["criterion.alpha"].copy_(torch.zeros(1,device=device) + new_alpha)

    return new_criterion.to(device)