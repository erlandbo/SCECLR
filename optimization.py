import torch
import numpy as np


def build_optimizer(model, lr, warmup_epochs, max_epochs, num_batches, lr_anneal="cosine_anneal", momentum=0.9, weight_decay=5e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_schedule_warmup = np.linspace(0.0, stop=lr, num=warmup_epochs * num_batches)
    T_max = num_batches*(max_epochs - warmup_epochs)
    if lr_anneal == "cosine_anneal":
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        anneal_steps = np.arange(0, T_max)
        lr_schedule_anneal = 0.5 * lr * (1 + np.cos(anneal_steps/T_max * np.pi))
    elif lr_anneal == "linear_anneal":
        lr_schedule_anneal = np.linspace(start=lr, stop=0.0, num=T_max)
    else: raise ValueError("Unknown learning rate anneal strategy {}".format(lr_anneal))
    # warmupsteps + annealsteps = lendata*max_epochs
    lr_schedule = np.concatenate([lr_schedule_warmup, lr_schedule_anneal])
    return optimizer, lr_schedule


def build_optimizer_epoch(model, lr, warmup_epochs, max_epochs, lr_anneal="cosine_anneal", momentum=0.9, weight_decay=5e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_schedule_warmup = np.linspace(0.0, stop=lr, num=warmup_epochs + 1)
    T_max = (max_epochs + 1 - warmup_epochs)
    if lr_anneal == "cosine_anneal":
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        anneal_steps = np.arange(0, T_max)
        lr_schedule_anneal = 0.5 * lr * (1 + np.cos(anneal_steps/T_max * np.pi))
    elif lr_anneal == "linear_anneal":
        lr_schedule_anneal = np.linspace(start=lr, stop=0.0, num=T_max)
    else: raise ValueError("Unknown learning rate anneal strategy {}".format(lr_anneal))
    lr_schedule = np.concatenate([lr_schedule_warmup, lr_schedule_anneal])
    return optimizer, lr_schedule


# Auto linear lr from [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour] https://arxiv.org/pdf/1706.02677.pdf
# Auto sqrt lr from SimCLR https://arxiv.org/pdf/2002.05709.pdf
def auto_lr(batchsize, scale="linear"):
    if scale == "linear":
        base_lr = 0.03 * batchsize / 256
    else:
        base_lr = 0.075 * batchsize**0.5
    return base_lr
