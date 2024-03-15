import torch
from criterions.sceclrlossesv2 import SCECLRV2Loss
from criterions.sceclrlossesv2_real import SCECLRV2RealLoss

torch.manual_seed(42)


def simclr(feats):
    B = feats.size(0) // 2

    q = 1.0 / ( torch.cdist(feats, feats, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)

    self_mask = torch.eye(2*B, device=feats.device, dtype=torch.bool)
    pos_mask = torch.roll(self_mask, shifts=B, dims=1)

    q.masked_fill(self_mask, 0.0)

    loss = -torch.log(q[pos_mask] / torch.sum(q, dim=1)).mean()
    return loss


sceclrv2 = SCECLRV2Loss(metric="cauchy", N=1)
sceclrv2_alt = SCECLRV2RealLoss(metric="cauchy", N=1)

x1 = torch.rand(8, 4, requires_grad=True)
x2 = x1.detach().clone().requires_grad_(True)

idx = torch.tensor([0])

y1 = sceclrv2(x1, idx)
y2 = sceclrv2_alt(x2, idx)
# y3 = sceclr_alt(x1)

print(y1)
print(y2)

y1.backward()
y2.backward()

print(x1.grad)
print(x2.grad)

print(torch.abs(x1.grad - x2.grad))