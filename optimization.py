import torch
import numpy as np


# TODO use torch instead of np?
# TODO add rmsprop, adam?
def build_optimizer(model, lr, warmup_epochs, max_epochs, num_batches, cosine_anneal=True, momentum=0.9, weight_decay=5e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_schedule_warmup = np.linspace(0.0, lr, warmup_epochs * num_batches)
    T_max = num_batches*(max_epochs - warmup_epochs)
    if cosine_anneal:
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
        anneal_steps = np.arange(0, T_max)
        lr_schedule_anneal = 0.5 * lr * (1 + np.cos(anneal_steps/T_max * np.pi))
    else:
        lr_schedule_anneal = np.linspace(lr, 0.0, T_max)
    # warmupsteps + annealsteps = lendata*max_epochs
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
