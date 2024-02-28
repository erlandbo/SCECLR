import torch
from torch import nn
from torch.nn import functional as F
from criterions.koleoloss import KoLeoLoss


class SCELoss(nn.Module):
    def __init__(self, metric, **kwargs):
        super().__init__()
        if metric == 'cauchy':
            self.criterion = CauchyLoss(**kwargs)
        elif metric == 'heavy-tailed':
            self.criterion = HeavyTailedLoss(**kwargs)
        elif metric == 'gaussian':
            self.criterion = GaussianLoss(**kwargs)
        else:
            raise ValueError(f'Undefined similarity metric in SCELoss: {metric}')

    def forward(self, x):
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
        self.xi = self.xi + torch.sum((1 - self.alpha) * qij.detach())
        self.omega = self.omega + (1 - self.alpha) * Bij
        # Automatically set rho or constant
        momentum = self.N.pow(1) / (self.N.pow(1) + self.omega * 0.01 ) if self.rho < 0 else self.rho
        weighted_sum_count = self.xi / self.omega
        self.s_inv = momentum * self.s_inv + (1 - momentum) * self.N.pow(2) * weighted_sum_count


class CauchyLoss(SCEBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super(CauchyLoss, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.koleoloss = KoLeoLoss()

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        # Attraction
        pairdist_ii = F.pairwise_distance(zi, zj, keepdim=True, eps=1e-8)
        qii = 1.0 / ( pairdist_ii.pow(2)  + 1.0 )  # (B,1)
        attractive_forces = - torch.log(qii)

        # Repulsion
        pairdist_ij = F.pairwise_distance(zi, torch.roll(zj, shifts=-1, dims=0), keepdim=True, eps=1e-8)
        qij = 1.0 / ( pairdist_ij.pow(2)  + 1.0 )  # (B,1)

        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = qij * s_hat

        # koleoloss = (self.koleoloss(zi) + self.koleoloss(zj)) / 2 * 0.000


        #koleoloss1 = self.koleoloss(zi, zi) * 0.05
        #koleoloss2 = self.koleoloss(zj, zj) * 0.05
        koleoloss = self.koleoloss(zi, zj) * 0.1
        #koleoloss = (koleoloss1 + koleoloss2) / 2

        loss = attractive_forces.mean() + repulsive_forces.mean()

        # print(koleoloss, loss)

        self.update_s(qii, qij)

        return loss + koleoloss


class HeavyTailedLoss(SCEBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, v=2.0):
        super(HeavyTailedLoss, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.v = v  # alpha in https://proceedings.neurips.cc/paper/2009/file/2291d2ec3b3048d1a6f86c2c4591b7e0-Paper.pdf
        # https://arxiv.org/pdf/1902.05804.pdf
        self.koleoloss = KoLeoLoss()

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        # Attraction
        pairdist_ii = F.pairwise_distance(zi, zj, keepdim=True, eps=1e-8)
        qii = 1 / ( pairdist_ii.pow(2) / self.v + 1.0 )**self.v  # (B,1)
        attractive_forces = - torch.log(qii)

        # Repulsion
        pairdist_ij = F.pairwise_distance(zi, torch.roll(zj, shifts=-1, dims=0), keepdim=True, eps=1e-8)
        qij = 1 / ( pairdist_ij.pow(2) / self.v + 1.0 )**self.v   # (B,1)

        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = qij * s_hat

        #koleoloss = self.koleoloss(zi, zj) * 0.01
        koleoloss1 = self.koleoloss(zi, zi) * 0.0001
        koleoloss2 = self.koleoloss(zj, zj) * 0.0001
        koleoloss = (koleoloss1 + koleoloss2) / 2

        loss = attractive_forces.mean() + repulsive_forces.mean()

        print(koleoloss, loss)

        self.update_s(qii, qij)

        return loss + koleoloss


class GaussianLoss(SCEBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, var=0.5):
        super(GaussianLoss, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.var = var

    def forward(self, z):
        self.xi = torch.zeros(1, ).to(z.device)
        self.omega = torch.zeros(1, ).to(z.device)

        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        # TODO add var
        # Attraction
        pairdist_ii = F.pairwise_distance(zi, zj, keepdim=True)
        qii = torch.exp(-pairdist_ii.pow(2))  # (B,1)
        qii = qii.clamp(torch.finfo(float).eps)

        attractive_forces = torch.mean(- pairdist_ii)  # log cancels exp

        # Repulsion
        pairdist_ij = F.pairwise_distance(zi, torch.roll(zj, shifts=-1, dims=0), keepdim=True)
        qij = torch.exp(-pairdist_ij.pow(2))
        qij = qij.clamp(torch.finfo(float).eps)

        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = torch.mean(qij * s_hat)

        loss = attractive_forces + repulsive_forces
        self.update_s(qii, qij)
        return loss



# class CosineSCELoss(SCELoss):
#     def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, tau=0.1):
#         super(SCELoss, self).__init__(N=60_000, rho=-1, alpha=0.5, S_init=2.0)
#         self.register_buffer("tau", torch.zeros(1, ) + tau)
#
#     def forward(self, z):
#         self.xi = torch.zeros(1, ).to(z.device)
#         self.omega = torch.zeros(1, ).to(z.device)
#
#         B = z.shape[0] // 2
#         zi, zj = z[0:B], z[B:]
#         # Attraction
#         cossim_ii = F.cosine_similarity(zi, zj, dim=1).unsqueeze(1)  # (B,1)
#         qii = torch.exp(cossim_ii)
#
#         attractive_forces = torch.mean(cossim_ii / self.tau)  # log cancels exp
#
#         # Repulsion
#         cossim_ij = F.cosine_similarity(zi, torch.roll(zj, shifts=-1, dims=0), dim=1).unsqueeze(1)  # (B,1)
#         qij = torch.exp(cossim_ij / self.tau)
#
#         s_hat = self.N.pow(2) / self.s_inv
#         repulsive_forces = torch.mean(qij * s_hat)
#
#         loss = attractive_forces + repulsive_forces
#         self.update_s(qii, qij)
#         return loss
#
#
# class DotProdL2SCELoss(SCELoss):
#     def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, tau=0.1):
#         super(SCELoss, self).__init__(N=60_000, rho=-1, alpha=0.5, S_init=2.0)
#         self.register_buffer("tau", torch.zeros(1, ) + tau)
#
#     def forward(self, z):
#         self.xi = torch.zeros(1, ).to(z.device)
#         self.omega = torch.zeros(1, ).to(z.device)
#
#         B = z.shape[0] // 2
#         zi, zj = z[0:B], z[B:]
#         # Attraction
#         dotprod_ii = torch.sum(zi * zj, dim=1, keepdim=True)  # (B,1)
#         qii = torch.exp(dotprod_ii)
#
#         attractive_forces = torch.mean(dotprod_ii)  # log cancels exp
#
#         # Repulsion
#         dotprod_ii = torch.sum(zi * torch.roll(zj, shifts=-1, dims=0), dim=1, keepdim=True)  # (B,1)
#         qii = torch.exp(dotprod_ii)
#
#         attractive_forces = torch.mean(dotprod_ii)  # log cancels exp
#
#         cossim_ij = F.cosine_similarity(zi, torch.roll(zj, shifts=-1, dims=0), dim=1).unsqueeze(1)  # (B,1)
#         qij = torch.exp(cossim_ij / self.tau)
#
#         s_hat = self.N.pow(2) / self.s_inv
#         repulsive_forces = torch.mean(qij * s_hat)
#
#         loss = attractive_forces + repulsive_forces
#         self.update_s(qii, qij)
#         return loss

