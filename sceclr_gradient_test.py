import torch
from criterions.sceclrlossesv1 import SCECLRV1Loss

torch.manual_seed(0)

def simclr(feats):
    B = feats.size(0) // 2

    q = 1.0 / ( torch.cdist(feats, feats, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)

    self_mask = torch.eye(2*B, device=feats.device, dtype=torch.bool)
    pos_mask = torch.roll(self_mask, shifts=B, dims=1)

    q.masked_fill(self_mask, 0.0)

    loss = -torch.log(q[pos_mask] / torch.sum(q, dim=1)).mean()
    return loss


def sceclr(feats):
    B = feats.size(0) // 2

    q = 1.0 / ( torch.cdist(feats, feats, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)

    self_mask = torch.eye(2*B, device=feats.device, dtype=torch.bool)
    pos_mask = torch.roll(self_mask, shifts=B, dims=1)

    q.masked_fill(self_mask, 0.0)

    pos = -torch.log(q[pos_mask])

    s = 1.0
    Z = torch.sum(q.detach(), dim=1,keepdim=True)
    Q = q / Z
    neg = torch.sum(Q, dim=1,keepdim=True) * s

    loss = pos.mean() + neg.mean()

    return loss

def sceclr_alt(feats):
    B = feats.size(0) // 2

    q = 1.0 / ( torch.cdist(feats, feats, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)

    self_mask = torch.eye(2*B, device=feats.device, dtype=torch.bool)
    pos_mask = torch.roll(self_mask, shifts=B, dims=1)

    q.masked_fill(self_mask, 0.0)

    pos = -torch.log(q[pos_mask])

    s = 1.0
    #Z = torch.sum(q.detach(), dim=1,keepdim=True)
    #Q = q / Z
    neg = torch.log( torch.sum(q, dim=1, keepdim=True)) * s

    loss = pos.mean() + neg.mean()

    return loss


#sceclr = SCECLRLoss(metric="cauchy")

x1 = torch.rand(8, 4, requires_grad=True)
x2 = x1.detach().clone().requires_grad_(True)

y1 = sceclr(x1)
y2 = simclr(x2)
# y3 = sceclr_alt(x1)

print(y1)
print(y2)

y1.backward()
y2.backward()

print(x1.grad)
print(x2.grad)

print(torch.abs(x1.grad - x2.grad))