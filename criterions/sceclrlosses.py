import torch
from torch import nn
from torch.nn import functional as F


class SCECLRLoss(nn.Module):
    def __init__(self, metric, **kwargs):
        super().__init__()
        if metric == 'student-t':
            self.criterion = StudenttLoss(**kwargs)
        elif metric == 'gaussian':
            self.criterion = GaussianLoss(**kwargs)
        elif metric == 'cosine':
            self.criterion = CosineLoss(**kwargs)
        elif metric == 'dotprodwl2':
            self.criterion = DotProdLoss(**kwargs)

    def forward(self, x):
        return self.criterion(x)


class SCECLRBase(nn.Module):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0):
        super().__init__()
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1, ))  # weighted sum q
        self.register_buffer("omega", torch.zeros(1, ))  # count q
        self.register_buffer("N", torch.zeros(1, ) + N)  # N samples in dataset
        self.register_buffer("s_inv", torch.zeros(1, ) + N**S_init)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.register_buffer("rho", torch.zeros(1, ) + rho)  # Automatically set rho or constant

    def __str__(self):
        string = ""
        for name, param in self.named_buffers():
            string += "{}: {} ".format( name, param)
        string += "S-coeff: {}".format(self.N.pow(2)/self.s_inv)
        return string

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


class StudenttLoss(SCECLRBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, dof=1):
        super(SCECLRBase, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.dof = dof

    def forward(self, z):

        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        # TODO add dof

        q = 1.0 / ( torch.cdist(zi, zj, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)
        Z = torch.sum(q.detach(), dim=1, keepdim=True).requires_grad_(False)  # (B,B) -> (B,1)
        Q = q / Z

        # Attraction
        Qii = torch.diag(Q).unsqueeze(1)  # (B,1)
        attractive_forces = - torch.log(Qii)

        # Repulsion
        Qij = Q[~torch.eye(B, dtype=torch.bool)]  # off diagonal
        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = torch.sum(Q, dim=1, keepdim=True) * s_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()
        self.update_s(Qii, Qij)

        return loss


class GaussianLoss(SCECLRBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, var=0.5):
        super(SCECLRBase, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.var = var

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:]
        # TODO add var
        allpairnegdist = - torch.cdist(zi, zj, p=2).pow(2)  # (B,E),(B,E) -> (B,B)

        q = torch.exp(allpairnegdist)
        # Attraction
        allpairnegdist_ii = torch.diag(allpairnegdist).unsqueeze(1)  # (B,1)
        attractive_forces = - allpairnegdist_ii   # log cancels exp
        qii = torch.diag(q).unsqueeze(1)
        # Repulsion
        qij = q[~torch.eye(B, dtype=torch.bool)]  # off diagonal
        Z = torch.sum(q, dim=1, keepdim=True).detach().clone().requires_grad_(False)  # (B,B) -> (B,1)
        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = torch.sum(q / Z, dim=1, keepdim=True) * s_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()
        self.update_s(qii, qij)
        return loss


class CosineLoss(SCECLRBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, tau=0.1):
        super(SCECLRBase, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.tau = tau

    def forward(self, z):
        B = z.shape[0] // 2
        z = F.normalize(z, dim=1)
        zi, zj = z[0:B], z[B:2*B]

        # TODO add var
        cossim = torch.matmul(zi, zj.T) / self.tau  # (B,E) @ (E,B) -> (N,B)

        q = torch.exp(cossim)

        Z = torch.sum(q.detach(), dim=1, keepdim=True).requires_grad_(False)  # (B,B) -> (B,1)
        q = q / Z
        # Attraction
        #cossim_ii = torch.diag(cossim).unsqueeze(1)  # (B,1)
        qii = torch.diag(q).unsqueeze(1)  # (B,1)
        #attractive_forces = cossim_ii   # log cancels exp
        attractive_forces = -torch.log(qii)   # log cancels exp
        #qii = torch.diag(q).unsqueeze(1)

        # Repulsion
        qij = q[~torch.eye(B, dtype=torch.bool)]  # off diagonal
        #Z = torch.sum(q.detach(), dim=1, keepdim=True).requires_grad_(False)  # (B,B) -> (B,1)
        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = torch.sum(q , dim=1, keepdim=True) * s_hat

        loss = attractive_forces.mean() + repulsive_forces.mean()
        self.update_s(qii, qij)
        return loss


class DotProdLoss(SCECLRBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, l2_reg=0.01):
        super(SCECLRBase, self).__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.l2_reg = l2_reg

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:2*B]

        # TODO add var
        dotsim = torch.matmul(zi, zj.T)  # (B,E) @ (E,B) -> (N,B)

        q = torch.exp(dotsim)

        # Attraction
        dotsim_ii = torch.diag(dotsim).unsqueeze(1)  # (B,1)
        attractive_forces = dotsim_ii   # log cancels exp
        qii = torch.diag(q).unsqueeze(1)

        # Repulsion
        qij = q[~torch.eye(B, dtype=torch.bool)]  # off diagonal
        Z = torch.sum(q.detach(), dim=1, keepdim=True).requires_grad_(False)  # (B,B) -> (B,1)
        s_hat = self.N.pow(2) / self.s_inv
        repulsive_forces = torch.sum(q / Z, dim=1, keepdim=True) * s_hat

        loss = attractive_forces.mean() + repulsive_forces.mean() + self.l2_reg * z.sum()
        self.update_s(qii, qij)
        return loss

