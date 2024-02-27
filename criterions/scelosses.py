import torch
from torch import nn
from torch.nn import functional as F


class SCELoss(nn.Module):
    def __init__(self, metric, **kwargs):
        super().__init__()
        if metric == 'student-t':
            self.criterion = StudenttLoss(**kwargs)
        elif metric == 'gaussian':
            self.criterion = GaussianLoss(**kwargs)
        else:
            raise ValueError(f'Undefined similarity metric in SCECLRLoss: {metric}')

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

    @torch.no_grad()
    def update_s(self, qii, qij):
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
        momentum = self.N.pow(2) / (self.N.pow(2) + self.omega) if self.rho < 0 else self.rho
        weighted_sum_count = self.xi / self.omega
        self.s_inv = momentum * self.s_inv + (1 - momentum) * self.N.pow(2) * weighted_sum_count


class StudenttLoss(SCEBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, dof=1.0):
        super(StudenttLoss, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.dof = dof

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        # TODO add dof
        # Attraction
        pairdist_ii = F.pairwise_distance(zi, zj, keepdim=True)
        qii = 1.0 / ( pairdist_ii.pow(2) + 1.0 )  # (B,1)
        attractive_forces = - torch.log(qii)

        # Repulsion
        pairdist_ij = F.pairwise_distance(zi, torch.roll(zj, shifts=-1, dims=0), keepdim=True)
        qij = 1.0 / ( pairdist_ij.pow(2) + 1.0 )  # (B,1)

        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = qij * s_hat
        loss = attractive_forces.mean() + repulsive_forces.mean()
        self.update_s(qii, qij)

        return loss


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

