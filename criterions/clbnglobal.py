import torch
from torch import nn
from torch.nn import functional as F

# Contrastive Learning by Stochastic Approximation with Momentum
# CLSAM
class CLSAv3GlobalLoss(nn.Module):
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
        return self.criterion(x)


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
    def update_s(self, qii, qij):
        #####################
        self.qii = qii.mean()
        self.qij = qij.mean()
        self.qcoeff = self.N.pow(1) / self.s_inv
        #######################

        self.xi = torch.zeros(1, ).to(qii.device)
        self.omega = torch.zeros(1, ).to(qij.device)

        # Attraction
        Bii = qii.shape[0]
        self.xi = self.xi + torch.sum(self.alpha * qii.detach())
        self.omega = self.omega + self.alpha * Bii

        # Repulsion
        Bij = qij.shape[0]
        self.xi = self.xi + torch.sum( (1 - self.alpha) * qij.detach().sum(dim=1, keepdim=True) / (Bij - 1) )  # remove self-mask
        self.omega = self.omega + (1 - self.alpha) * Bij

        # Automatically set rho or constant
        momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho
        weighted_sum_count = self.xi / self.omega
        self.s_inv = momentum * self.s_inv + (1 - momentum) * self.N.pow(1) * weighted_sum_count


class CauchyLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=1.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)

    def forward(self, feats):
        B = feats.shape[0] // 2

        q = 1.0 / ( 1.0 + torch.cdist(feats, feats).pow(2) )   # (2B,E),(E,2B) -> (2B,2B)

        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)

        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        qii = q[pos_mask].clone()

        with torch.no_grad():
            momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho
            Z_hat = self.s_inv.unsqueeze(1) / self.N.pow(1) * (2 * B - 1)  # (B,1) -> (2B,1)
            Z_i = torch.sum(q, dim=1, keepdim=True)
            Z = (1-momentum) * Z_i + momentum * Z_hat

        loss = - qii / Z.detach()

        self.update_s(qii, q)

        return loss


class CosineLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=1.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.temp = 0.5

    def forward(self, feats):

        feats = F.normalize(feats, dim=1, p=2)

        B = feats.shape[0] // 2

        q = torch.exp(torch.matmul(feats, feats.T) / self.temp)  # (2B,E),(E,2B) -> (2B,2B)

        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)

        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        qii = q[pos_mask].clone()

        with torch.no_grad():
            momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho
            Z_hat = self.s_inv.unsqueeze(1) / self.N.pow(1) * (2 * B - 1)  # (B,1) -> (2B,1)
            Z_i = torch.sum(q, dim=1, keepdim=True)
            Z = (1-momentum) * Z_i + momentum * Z_hat

        loss = - qii / Z.detach()

        self.update_s(qii, q)

        return loss


class GaussianLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=1.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)

    def forward(self, feats):
        B = feats.shape[0] // 2

        q = torch.exp( - torch.cdist(feats, feats).pow(2) * self.gamma )  # (2B,E),(E,2B) -> (2B,2B)

        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)

        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        qii = q[pos_mask].clone()

        with torch.no_grad():
            momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho
            Z_hat = self.s_inv.unsqueeze(1) / self.N.pow(1) * (2 * B - 1)  # (B,1) -> (2B,1)
            Z_i = torch.sum(q, dim=1, keepdim=True)
            Z = (1-momentum) * Z_i + momentum * Z_hat

        loss = - qii / Z.detach()

        self.update_s(qii, q)

        return loss


