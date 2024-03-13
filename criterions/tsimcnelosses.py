# InfoNCE loss from t-SimCNE https://arxiv.org/pdf/2210.09879.pdf
# Adapted and copied from https://github.com/berenslab/t-simcne/blob/iclr2023/tsimcne/losses/infonce.py

import torch
from torch import nn
from torch.nn import functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, metric, **kwargs):
        super().__init__()
        if metric == "cosine":
            self.criterion = InfoNCECosine(**kwargs)
        elif metric == "cauchy":  # actually Cauchy
            self.criterion = InfoNCECauchy(**kwargs)
        elif metric == "gaussian":
            self.cls = InfoNCEGaussian(**kwargs)
        else:
            raise ValueError(f"Unknown {metric = !r} for InfoNCE loss")

    def forward(self, x, idx):
        return self.criterion(x)


class InfoNCECosine(nn.Module):
    def __init__(
            self,
            temperature: float = 0.5,
            reg_coef: float = 0,
            reg_radius: float = 200,
    ):
        super().__init__()
        self.temperature = temperature
        self.reg_coef = reg_coef
        self.reg_radius = reg_radius

    def forward(self, features):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        # mean deviation from the sphere with radius `reg_radius`
        vecnorms = torch.linalg.vector_norm(features, dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * F.mse_loss(vecnorms, target)

        a = F.normalize(a)
        b = F.normalize(b)

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        cos_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return loss


class InfoNCECauchy(nn.Module):
    def __init__(self, temperature: float = 1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        tempered_alignment = torch.diagonal_copy(sim_ab).log_().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCEGaussian(InfoNCECauchy):
    def forward(self, features):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()

        tempered_alignment = sim_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss

