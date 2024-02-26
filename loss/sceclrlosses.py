import torch
from torch import nn
from torch.nn import functional as F


class SCELoss(nn.Module):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, metric="student-t"):
        super().__init__()
        self.metric = metric
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1, ))  # weighted sum q
        self.register_buffer("omega", torch.zeros(1, ))  # count q
        self.register_buffer("N", torch.zeros(1, ) + N)  # N samples in dataset
        self.register_buffer("s_inv", torch.zeros(1, ) + N**S_init)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.register_buffer("rho", torch.zeros(1, ) + rho)  # Automatically set rho or constant

        self.l2dist = nn.PairwiseDistance(p=2, keepdim=True)

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        #za, zr = torch.chunk(z, 2)
        #za_i, za_j = torch.chunk(za, 2)
        #zr_i, zr_j = torch.chunk(zr, 2)
        self.xi = torch.zeros(1, ).to(zi.device)
        self.omega = torch.zeros(1, ).to(zi.device)
        # Process batch
        # B = za_i.size(0)
        # Positive forces
        qii = self.sim_function(zi, zj, metric=self.metric)  # (B,1)
        positive_forces = torch.mean(- torch.log(qii))
        self.xi = self.xi + torch.sum(self.alpha * qii).detach()
        self.omega = self.omega + self.alpha * B
        # Negative forces
        qij = self.sim_function(zi, torch.roll(zj, shifts=-1, dims=0), metric=self.metric)  # (B,1)
        negative_forces = torch.mean(qij * (self.N**2 / self.s_inv))
        self.xi = self.xi + torch.sum((1 - self.alpha) * qij).detach()
        self.omega = self.omega + (1 - self.alpha) * B
        # Automatically set rho or constant
        rho = self.N ** 2 / (self.N ** 2 + self.omega) if self.rho > 0 else self.rho
        self.s_inv = rho * self.s_inv + (1 - rho) * self.N ** 2 * self.xi / self.omega
        loss = positive_forces + negative_forces
        return loss

    def sim_function(self, z1, z2, metric="student-t"):
        # TODO add temperature (std)
        if metric == "student-t":
            sim = 1 / (self.l2dist(z1, z2)).square().add(1)
        elif metric == "gaussian":
            sim = torch.exp(-(self.l2dist(z1, z2)).square())
            sim = torch.clamp(sim, min=torch.finfo(float).eps)  # TODO make more numerical stable?
        elif metric == "cosine":
            z1 = z1.norm(p=2, dim=1, keepdim=True)
            z2 = z2.norm(p=2, dim=1, keepdim=True)
            sim = torch.sum(z1 * z2, dim=1)
        return sim


class SCECLRLoss(nn.Module):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super().__init__()
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1, ))  # weighted sum q
        self.register_buffer("omega", torch.zeros(1, ))  # count q
        self.register_buffer("N", torch.zeros(1, ) + N)  # N samples in dataset
        self.register_buffer("s_inv", torch.zeros(1, ) + N**S_init)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.register_buffer("rho", torch.zeros(1, ) + rho)  # Automatically set rho or constant

    # def forward(self, z):
    #     B = z.shape[0] // 2
    #     zi, zj = z[0:B], z[B:]
    #     self.xi = torch.zeros(1, ).to(zi.device)
    #     self.omega = torch.zeros(1, ).to(zi.device)
    #     # Positive forces
    #     q = 1 / (1 + torch.cdist(zi, zj, p=2)**2)
    #     qii = torch.diag(q)
    #     s = torch.sum(qii, dim=1, keepdim=True)
    #     positive_forces = torch.mean(- torch.log(qii))
    #     self.xi = self.xi + torch.sum(self.alpha * qii).detach()
    #     self.omega = self.omega + self.alpha * B
    #     # Negative forces
    #     qij = torch.sum((q / s.detach()), dim=1, keepdim=True)
    #     #qij = self.sim_function(zi, torch.roll(zj, shifts=-1, dims=0), metric=self.metric)  # (B,1)
    #     negative_forces = torch.mean(qij * (self.N**2 / self.s_inv))
    #     self.xi = self.xi + torch.sum((1 - self.alpha) * qij).detach()
    #     self.omega = self.omega + (1 - self.alpha) * B
    #     # Automatically set rho or constant
    #     rho = self.N ** 2 / (self.N ** 2 + self.omega) if self.rho > 0 else self.rho
    #     self.s_inv = rho * self.s_inv + (1 - rho) * self.N ** 2 * self.xi / self.omega
    #     loss = positive_forces + negative_forces
    #     return loss
    #
    # def sim_function(self, z1, z2, metric="student-t"):
    #     # TODO add temperature (std)
    #     if metric == "student-t":
    #         sim = 1 / (torch.cdist(z1, z2).pow(2) + 1)
    #     elif metric == "gaussian":
    #         sim = torch.exp(-(torch.cdist(z1, z2)).pow(2))
    #     elif metric == "cosine":
    #         z1 = z1.norm(p=2, dim=1, keepdim=True)
    #         z2 = z2.norm(p=2, dim=1, keepdim=True)
    #         sim = torch.sum(z1 * z2, dim=1)
    #     sim = torch.clamp(sim, min=torch.finfo(float).eps)  # TODO make more numerical stable?
    #     return sim


class StudenttLoss(SCECLRLoss):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super(StudenttLoss, self).__init__(N=60_000, rho=-1, alpha=0.5, S_init=2.0)

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        self.xi = torch.zeros(1, ).to(zi.device)
        self.omega = torch.zeros(1, ).to(zi.device)
        # Positive forces
        q = 1 / (1 + torch.cdist(zi, zj, p=2)**2)
        qii = torch.diag(q)
        s = torch.sum(qii, dim=1, keepdim=True)
        positive_forces = torch.mean(- torch.log(qii))
        self.xi = self.xi + torch.sum(self.alpha * qii).detach()
        self.omega = self.omega + self.alpha * B
        # Negative forces
        qij = torch.sum((q / s.detach()), dim=1, keepdim=True)
        #qij = self.sim_function(zi, torch.roll(zj, shifts=-1, dims=0), metric=self.metric)  # (B,1)
        negative_forces = torch.mean(qij * (self.N**2 / self.s_inv))
        self.xi = self.xi + torch.sum((1 - self.alpha) * qij).detach()
        self.omega = self.omega + (1 - self.alpha) * B
        # Automatically set rho or constant
        rho = self.N ** 2 / (self.N ** 2 + self.omega) if self.rho > 0 else self.rho
        self.s_inv = rho * self.s_inv + (1 - rho) * self.N ** 2 * self.xi / self.omega
        loss = positive_forces + negative_forces
        return loss


class GaussianLoss(SCECLRLoss):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super(GaussianLoss, self).__init__(N=60_000, rho=-1, alpha=0.5, S_init=2.0)

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        self.xi = torch.zeros(1, ).to(zi.device)
        self.omega = torch.zeros(1, ).to(zi.device)
        # Positive forces
        q = -torch.cdist(zi, zj).pow(2)
        qii = torch.diag(q)  # log cancels exp
        s = torch.sum(qii, dim=1, keepdim=True)
        positive_forces = torch.mean(- torch.log(qii))
        self.xi = self.xi + torch.sum(self.alpha * qii).detach()
        self.omega = self.omega + self.alpha * B
        # Negative forces
        qij = torch.sum((q / s.detach()), dim=1, keepdim=True)
        negative_forces = torch.mean(qij * (self.N**2 / self.s_inv))
        self.xi = self.xi + torch.sum((1 - self.alpha) * qij).detach()
        self.omega = self.omega + (1 - self.alpha) * B
        # Automatically set rho or constant
        rho = self.N ** 2 / (self.N ** 2 + self.omega) if self.rho > 0 else self.rho
        self.s_inv = rho * self.s_inv + (1 - rho) * self.N ** 2 * self.xi / self.omega
        loss = positive_forces + negative_forces
        return loss


class CosineLoss(SCECLRLoss):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, tau=0.1):
        super(CosineLoss, self).__init__(N=60_000, rho=-1, alpha=0.5, S_init=2.0)
        self.tau = tau

    def forward(self, z):
        z = F.normalize(z, dim=1)
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:2*B]
        self.xi = torch.zeros(1, ).to(zi.device)
        self.omega = torch.zeros(1, ).to(zi.device)
        # Positive forces
        q = torch.matmul(zi, zj.T) / self.tau  # (B,E) @ (E,B) -> (N,B)
        qii = torch.diag(q)  # log cancels exp
        s = torch.sum(qii, dim=1, keepdim=True)
        positive_forces = torch.mean(- torch.log(qii))
        self.xi = self.xi + torch.sum(self.alpha * qii).detach()
        self.omega = self.omega + self.alpha * B
        # Negative forces
        qij = torch.sum((q / s.detach()), dim=1, keepdim=True)
        negative_forces = torch.mean(qij * (self.N**2 / self.s_inv))
        self.xi = self.xi + torch.sum((1 - self.alpha) * qij).detach()
        self.omega = self.omega + (1 - self.alpha) * B
        # Automatically set rho or constant
        rho = self.N ** 2 / (self.N ** 2 + self.omega) if self.rho > 0 else self.rho
        self.s_inv = rho * self.s_inv + (1 - rho) * self.N ** 2 * self.xi / self.omega
        loss = positive_forces + negative_forces
        return loss


class DotProdL2Loss(SCECLRLoss):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, l2_reg=0.02):
        super(DotProdL2Loss, self).__init__(N=60_000, rho=-1, alpha=0.5, S_init=2.0)
        self.l2_reg = l2_reg

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        self.xi = torch.zeros(1, ).to(zi.device)
        self.omega = torch.zeros(1, ).to(zi.device)
        # Positive forces
        q = torch.matmul(zi, zj.T)  # (B,E) @ (E,B) -> (N,B)
        qii = torch.diag(q)  # log cancels exp
        s = torch.sum(qii, dim=1, keepdim=True)
        positive_forces = torch.mean(- torch.log(qii))
        self.xi = self.xi + torch.sum(self.alpha * qii).detach()
        self.omega = self.omega + self.alpha * B
        # Negative forces
        qij = torch.sum((q / s.detach()), dim=1, keepdim=True)
        negative_forces = torch.mean(qij * (self.N**2 / self.s_inv))
        self.xi = self.xi + torch.sum((1 - self.alpha) * qij).detach()
        self.omega = self.omega + (1 - self.alpha) * B
        # Automatically set rho or constant
        rho = self.N ** 2 / (self.N ** 2 + self.omega) if self.rho > 0 else self.rho
        self.s_inv = rho * self.s_inv + (1 - rho) * self.N ** 2 * self.xi / self.omega
        loss = positive_forces + negative_forces
        l2_z = torch.sum(z.pow(2)) * self.l2_reg
        loss = loss + l2_z
        return loss
