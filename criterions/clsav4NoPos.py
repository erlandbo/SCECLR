import torch
from torch import nn
from torch.nn import functional as F


class CLSAv4NoPosLoss(nn.Module):
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
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super().__init__()
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1, ))  # weighted sum q
        self.register_buffer("omega", torch.zeros(1, ))  # count q
        self.register_buffer("N", torch.zeros(1, ) + N)  # N samples in dataset
        self.register_buffer("s_inv", torch.zeros(1, ) + N**S_init)
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
        self.qcoeff = self.N.pow(2) / self.s_inv
        #######################

        self.xi = torch.zeros(1, ).to(qii.device)
        self.omega = torch.zeros(1, ).to(qij.device)

        # Attraction
        Bii = qii.shape[0]
        self.xi = self.xi + torch.sum(self.alpha * qii.detach())
        self.omega = self.omega + self.alpha * Bii

        # Repulsion
        Bij = qij.shape[0]
        self.xi = self.xi + torch.sum( (1 - self.alpha) * qij.detach().sum(dim=1) / (Bij - 2) )  # remove self-mask
        self.omega = self.omega + (1 - self.alpha) * Bij

        # Automatically set rho or constant
        momentum = self.N.pow(2) / (self.N.pow(2) + self.omega) if self.rho < 0 else self.rho
        weighted_sum_count = self.xi / self.omega
        self.s_inv = momentum * self.s_inv + (1 - momentum) * self.N.pow(2) * weighted_sum_count


class CauchyLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)

    def forward(self, feats):
        B = feats.shape[0] // 2

        q = 1.0 / ( torch.cdist(feats, feats, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)

        self_mask = torch.eye(2*B, device=feats.device, dtype=torch.bool)
        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        # Attraction
        qii = q[pos_mask].clone()  # (B,1)
        attractive_forces = - torch.log(qii)

        # Repulsion
        q.masked_fill(pos_mask, 0.0)

        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = torch.log( torch.sum(q, dim=1) ) * s_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()

        self.update_s(qii, q)

        return loss


class CosineLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.temp = 0.5

    def forward(self, feats):
        B = feats.shape[0] // 2

        feats = F.normalize(feats, dim=1, p=2)

        sim = torch.mm(feats, feats.T)   # (2B,E),(E,2B) -> (2B,2B)
        q = torch.exp(sim / self.temp)

        self_mask = torch.eye(2*B, device=feats.device, dtype=torch.bool)
        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        # Attraction
        qii = q[pos_mask].clone()  # (B,)
        attractive_forces = - torch.log(qii)

        # Repulsion
        q.masked_fill(pos_mask, 0.0)
        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = torch.log(torch.sum(q, dim=1)) * s_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()

        self.update_s(qii.detach() / self.N.pow(2), q.detach() / self.N.pow(2) )

        return loss


class GaussianLoss(CLSABase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        var = 0.5
        self.gamma = 1.0 / (2.0 * var)

    def forward(self, feats):
        B = feats.shape[0] // 2

        q = torch.exp( -torch.cdist(feats, feats, p=2).pow(2) )

        self_mask = torch.eye(2*B, device=feats.device, dtype=torch.bool)
        pos_mask = torch.roll(self_mask, shifts=B, dims=1)

        q.masked_fill(self_mask, 0.0)

        # Attraction
        qii = q[pos_mask].clone()  # (B,1)
        attractive_forces = - torch.log(qii)

        # Repulsion
        q.masked_fill(pos_mask, 0.0)
        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = torch.log(torch.sum(q, dim=1)) * s_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()

        self.update_s(qii, q)

        return loss
