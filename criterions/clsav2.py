import torch
from torch import nn
from torch.nn import functional as F

# Contrastive Learning by Stochastic Approximation with Momentum
# CLSAM
class CLSAv2Loss(nn.Module):
    def __init__(self, metric, **kwargs):
        super().__init__()
        if metric == 'cauchy':
            self.criterion = CauchyLoss(**kwargs)
        elif metric == 'gaussian':
            self.criterion = GaussianLoss(**kwargs)
        elif metric == 'cosine':
            self.criterion = CosineLoss(**kwargs)
        else: raise ValueError('Unknown metric {}'.format(metric))

    def forward(self, x, idx):
        return self.criterion(x, idx)


class CLSABase(nn.Module):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=1.0):
        super().__init__()
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1, ))  # weighted sum q
        self.register_buffer("omega", torch.zeros(1, ))  # count q
        self.register_buffer("N", torch.zeros(1, ) + N)  # N samples in dataset
        self.register_buffer("s_inv", torch.zeros(N, ) + N**S_init)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.register_buffer("rho", torch.zeros(1, ) + rho)  # Automatically set rho or constant
        ########### Debug
        self.register_buffer("qii", torch.zeros(1, ) )
        self.register_buffer("qij", torch.zeros(1, ) )
        self.register_buffer("qcoeff", torch.zeros(1, ) )
        ##################

    @torch.no_grad()
    def update_s(self, qii, qij, qji, feats_idx):
        #####################
        self.qii = qii.mean()
        self.qij = (qij.mean() + qji.mean()) / 2
        self.qcoeff = torch.mean(self.s_inv / self.N.pow(1))

        #######################
        B = feats_idx.size(0)

        self.xi = torch.zeros(B, ).to(qii.device)
        self.omega = torch.zeros(B, ).to(qii.device)

        # Attraction
        self.xi = self.xi + torch.sum(self.alpha * qii.detach())
        self.omega = self.omega + self.alpha * B

        # Repulsion
        E_rep1 = torch.sum(qij.detach(), dim=1) / (2.0 * B - 1.0)  # drop self-mask
        E_rep2 = torch.sum(qji.detach(), dim=1) / (2.0 * B - 1.0)  # drop self-mask
        E_rep = (E_rep1 + E_rep2) / 2.0

        self.xi = self.xi + torch.sum( (1 - self.alpha) * E_rep )
        self.omega = self.omega + (1 - self.alpha) * B

        # Automatically set rho or constant
        momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho
        weighted_sum_count = self.xi / self.omega
        self.s_inv[feats_idx] = momentum * self.s_inv[feats_idx] + (1.0 - momentum) * self.N.pow(1) * weighted_sum_count


class CauchyLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=1.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)

    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2

        q = 1.0 / ( 1.0 + torch.cdist(feats, feats).pow(2) )   # (2B,E),(E,2B) -> (2B,2B)

        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)

        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        # Attraction
        qii = q[pos_mask].clone()
        attractive_forces = - torch.log(qii)

        # Repulsion
        momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho

        Z_hat = ( self.s_inv[feats_idx] / self.N.pow(1) * ( 2 * B - 1) ).repeat(2)    # (B,) -> (2B,)
        Z_i = torch.sum(q, dim=1)

        repulsive_forces = torch.log( (1-momentum) * Z_i + momentum * Z_hat ) * 1.0 / (1.0 - momentum)

        loss = attractive_forces.mean() + repulsive_forces.mean()

        self.update_s(qii[0:B], q[0:B], q[B:], feats_idx)

        return loss


class CosineLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=1.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.temp = 0.5

    def forward(self, feats, feats_idx):

        feats = F.normalize(feats, dim=1, p=2)

        B = feats.shape[0] // 2

        q = torch.exp(torch.mm(feats, feats.T) / self.temp)  # (2B,E),(E,2B) -> (2B,2B)

        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)

        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        # Attraction
        qii = q[pos_mask].clone()
        attractive_forces = - torch.log(qii)

        # Repulsion
        momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho

        Z_hat = ( self.s_inv[feats_idx] / self.N.pow(1) * ( 2 * B - 1) ).repeat(2)    # (B,) -> (2B,)
        Z_i = torch.sum(q, dim=1)  # (2B,)

        repulsive_forces = torch.log( (1.0-momentum) * Z_i + momentum * Z_hat ) * 1.0 / (1.0 - momentum)

        loss = attractive_forces.mean() + repulsive_forces.mean()

        self.update_s(qii[0:B], q[0:B], q[B:], feats_idx)

        return loss


class GaussianLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=1.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        var = 1.0
        self.gamma = 1.0 / (2.0 * var)

    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2

        q = torch.exp( - torch.cdist(feats, feats).pow(2) * self.gamma )  # (2B,E),(E,2B) -> (2B,2B)

        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)

        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        # Attraction
        qii = q[pos_mask].clone()
        attractive_forces = - torch.log(qii).mean()

        # Repulsion
        momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho

        Z_hat = (self.s_inv[feats_idx] / self.N.pow(1) * (2 * B - 1)).repeat(2)  # (B,) -> (2B,)
        Z_i = torch.sum(q, dim=1)

        repulsive_forces = torch.log( (1-momentum) * Z_i + momentum * Z_hat ) * 1.0 / (1.0 - momentum)

        loss = attractive_forces + repulsive_forces

        self.update_s(qii[0:B], q[0:B], q[B:], feats_idx)

        return loss
