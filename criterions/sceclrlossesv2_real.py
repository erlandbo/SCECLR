import torch
from torch import nn
from torch.nn import functional as F


class SCECLRV2RealLoss(nn.Module):
    def __init__(self, metric, **kwargs):
        super().__init__()
        if metric == 'cauchy':
            self.criterion = CauchyLoss(**kwargs)
        elif metric == 'gaussian':
            self.criterion = GaussianLoss(**kwargs)
        elif metric == 'cosine':
            self.criterion = CosineLoss(**kwargs)
        elif metric == 'dotprod':
            self.criterion = DotProdLoss(**kwargs)

    def forward(self, x, idx):
        return self.criterion(x, idx)


class SCECLRBase(nn.Module):
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
        # self.qcoeff = self.N.pow(2) / self.s_inv
        #######################
        B = feats_idx.size(0)

        self.xi = torch.zeros(B, ).to(qii.device)
        self.omega = torch.zeros(B, ).to(qii.device)

        # Attraction
        self.xi = self.xi + torch.sum(self.alpha * qii.detach())
        self.omega = self.omega + self.alpha * B

        # Repulsion
        qij_hat = (torch.sum(qij.detach(), dim=1) + torch.sum(qji.detach(), dim=1)) / (2 * 2 * B)
        self.xi = self.xi + torch.sum((1 - self.alpha) * qij_hat )
        self.omega = self.omega + (1 - self.alpha) * B

        # Automatically set rho or constant
        momentum = self.N.pow(1) / (self.N.pow(1) + self.omega) if self.rho < 0 else self.rho
        weighted_sum_count = self.xi / self.omega
        self.s_inv[feats_idx] = (1 - momentum) * self.s_inv[feats_idx] + momentum * self.N.pow(1) * weighted_sum_count


class CauchyLoss(SCECLRBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=1.0):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)

    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2
        feats_u, feats_v = feats[:B], feats[B:]

        q_uv = 1.0 / ( torch.cdist(feats_u, feats_v, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)
        q_uu = 1.0 / ( torch.cdist(feats_u, feats_u, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)
        q_vv = 1.0 / ( torch.cdist(feats_v, feats_v, p=2).pow(2) + 1.0 )  # (B,E),(B,E) -> (B,B)

        self_mask = torch.eye(B, device=feats.device, dtype=torch.bool)

        qii = torch.diag(q_uv.clone())

        q_uv.masked_fill(self_mask, 0.0)  # Pos mask

        q_uu.masked_fill(self_mask, 0.0)
        q_vv.masked_fill(self_mask, 0.0)

        qij = torch.cat([q_uu, q_uv], dim=1)   # (B,B), (B,B) -> (B, 2B)
        qji = torch.cat([q_uv.T, q_vv], dim=1)  # (B,B), (B,B) -> (B, 2B)

        # Z = torch.sum(q.detach(), dim=1, keepdim=True).requires_grad_(False)  # (B,B) -> (B,1)

        # import pdb; pdb.set_trace()

        # Attraction
        # Qii = Q[pos_mask].unsqueeze(1)  # (B,1)
        attractive_forces = - torch.log(qii).mean()

        # Repulsion
        #s_hat = self.N.pow(1) / self.s_inv[feats_idx].unsqueeze(1)
        s_hat = self.s_inv[feats_idx].unsqueeze(1) / self.N.pow(1)
        moment = 0.9
        #Qij = qij / ( torch.mean(qij.detach(), dim=1, keepdim=True) * moment + (1.0 - moment) * 1.0 / s_hat )
        #Qji = qji / ( torch.mean(qji.detach(), dim=1, keepdim=True) * moment + (1.0 - moment) * 1.0 / s_hat )


        # repulsive_forces_1 = torch.sum(Qij, dim=1, keepdim=True) / (2.0 * B)
        # repulsive_forces_2 = torch.sum(Qji, dim=1, keepdim=True) / (2.0 * B)

        pos_sim = qii.unsqueeze(1).detach().clone().requires_grad_(False)  # detach() from computation graph

        repulsive_forces_1 = torch.log( ((torch.mean(qij, dim=1, keepdim=True) + pos_sim) * moment + (1.0 - moment) * s_hat.detach()) * 2*B ) * 1.0 / moment
        repulsive_forces_2 = torch.log( ((torch.mean(qji, dim=1, keepdim=True) + pos_sim) * moment + (1.0 - moment) * s_hat.detach()) * 2*B ) * 1.0 / moment
        repulsive_forces = ( repulsive_forces_1.mean() + repulsive_forces_2.mean() ) / 2.0

        # repulsive_forces = ( torch.sum(Qij, dim=1, keepdim=True).mean() + torch.sum(Qji, dim=1, keepdim=True).mean() ) / (2.0 * 2.0 * B)

        loss = attractive_forces + repulsive_forces

        self.update_s(qii, qij, qji, feats_idx)

        # with torch.no_grad():
        #    real_rep_loss = (torch.log(B*(torch.mean(qij, dim=1, keepdim=True) * 0.9 + 0.1 / s_hat ))).mean().detach()

        # import pdb; pdb.set_trace()

        # print("att", attractive_forces.mean())
        # print("rep", repulsive_forces.mean())
        #
        # print("Qij", Qij.mean())
        # print("Qij", Qij.mean())
        # print("qij", qij.mean())
        # print("qji", qji.mean())

        return loss  # + real_rep_loss.detach()


class GaussianLoss(SCECLRBase):
    def __init__(self, N=60_000, rho=-1, alpha=0.5, S_init=2.0, var=0.5):
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
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
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
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
        super().__init__(N=N, rho=rho, alpha=alpha, S_init=S_init)
        self.l2_reg = l2_reg

    def forward(self, z):
        B = z.shape[0] // 2
        zi, zj = z[0:B], z[B:2*B]

        # TODO add tau
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

