import torch
from torch import nn
from torch.nn import functional as F


class SCELoss(nn.Module):
    def __init__(self, metric, **kwargs):
        super().__init__()
        if metric == 'cauchy':
            self.criterion = CauchyLoss(**kwargs)
        elif metric == 'cosine':
            self.criterion = CosineLoss(**kwargs)
        elif metric == 'gaussian':
            self.criterion = GaussianLoss(**kwargs)
        else:
            raise ValueError(f'Undefined similarity metric in SCELoss: {metric}')

    def forward(self, x, idx):
        return self.criterion(x)


class SCEBase(nn.Module):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super().__init__()
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1, ))  # weighted sum q
        self.register_buffer("omega", torch.zeros(1, ))  # count q
        self.register_buffer("N", torch.zeros(1, ) + N)  # N samples in dataset
        self.register_buffer("s_inv", torch.zeros(1, ) + N**S_init)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.register_buffer("rho", torch.zeros(1, ) + rho)  # Automatically set rho or constant

        # TODO remove
        ########### Debug
        #self.register_buffer("qii", torch.nn.Parameter(torch.tensor(0.0), requires_grad=False))
        #self.register_buffer("qii", torch.tensor(0.0) )
        #self.register_buffer("qij", torch.nn.Parameter(torch.tensor(0.0), requires_grad=False))
        #self.register_buffer("qij", torch.tensor(0.0) )
        #self.register_buffer("qcoeff", torch.zeros(1, ) )
        ##################

    @torch.no_grad()
    def update_s(self, qii, qij):
        #####################
        #self.qii = torch.nn.Parameter(qii.clone().detach().mean(), requires_grad=False)
        #self.qij = torch.nn.Parameter(qij.clone().detach().mean(), requires_grad=False)
        # self.qii = qii.mean()
        # self.qij = qij.mean()
        # self.qcoeff = self.N.pow(2) / self.s_inv
        #######################
        self.xi = torch.zeros(1, ).to(qii.device)
        self.omega = torch.zeros(1, ).to(qij.device)
        # Attraction
        Bii = qii.shape[0]
        self.xi = self.xi + torch.sum(self.alpha * qii.detach())
        self.omega = self.omega + self.alpha * Bii
        # Repulsion
        Bij = qij.shape[0]
        self.xi = self.xi + torch.sum((1 - self.alpha) * qij.detach())
        self.omega = self.omega + (1 - self.alpha) * Bij
        # Automatically set rho or constant
        momentum = self.N.pow(1) / (self.N.pow(1) + self.omega * 0.01 ) if self.rho < 0 else self.rho
        weighted_sum_count = self.xi / self.omega
        self.s_inv = momentum * self.s_inv + (1 - momentum) * self.N.pow(2) * weighted_sum_count


class CauchyLoss(SCEBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super(CauchyLoss, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        z1, z2 = torch.cat([zi, zj], dim=0), torch.cat([zj, zi], dim=0)

        # Attraction
        pairdist_ii = F.pairwise_distance(z1, z2, p=2).pow(2)
        qii = 1.0 / (1.0 + pairdist_ii)
        attractive_forces = - torch.log(qii)

        # Repulsion
        pairdist_ij = F.pairwise_distance(z1, torch.roll(z2, shifts=-1, dims=0), p=2).pow(2)
        #pairdist_ij = F.pairwise_distance(z1, z2, p=2).pow(2)
        qij = 1.0 / (1.0 + pairdist_ij)
        Z_hat = self.s_inv / self.N.pow(2)
        repulsive_forces = qij / Z_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()

        self.update_s(qii, qij)

        return loss


class GaussianLoss(SCEBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super(GaussianLoss, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        var = 0.5
        self.gamma = 1.0 / (2.0 * var)

    def forward(self, z):

        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        z1, z2 = torch.cat([zi, zj], dim=0), torch.cat([zj, zi], dim=0)

        # Attraction
        pairdist_ii = F.pairwise_distance(z1, z2, p=2).pow(2)
        qii = torch.exp(F.pairwise_distance(z1, z2, p=2).pow(2) * self.gamma)
        attractive_forces = - pairdist_ii * self.gamma

        # Repulsion
        pairdist_ij = F.pairwise_distance(z1, torch.roll(z2, shifts=-1, dims=0), p=2).pow(2)
        qij = - torch.exp(pairdist_ij * self.gamma)
        Z_hat = self.s_inv / self.N.pow(2)
        repulsive_forces = qij / Z_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()

        self.update_s(qii, qij)

        return loss


class CosineLoss(SCEBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super(CosineLoss, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.temp = 0.5

    def forward(self, z):

        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        z1, z2 = torch.cat([zi, zj], dim=0), torch.cat([zj, zi], dim=0)

        # Attraction
        qii = torch.exp( F.cosine_similarity(z1, z2, dim=1) / self.temp )

        attractive_forces = - torch.log(qii)

        # Repulsion
        qij = torch.exp( F.cosine_similarity(z1, torch.roll(z2, shifts=-1, dims=0)) / self.temp )
        Z_hat = self.s_inv / self.N.pow(2)
        repulsive_forces = qij / Z_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()

        self.update_s(qii / self.N.pow(2), qij / self.N.pow(2))

        # import pdb; pdb.set_trace()

        return loss
