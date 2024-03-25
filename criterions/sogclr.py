from torch import nn
from torch.nn import functional as F
import torch


class SogCLR(nn.Module):
    def __init__(self, N):
        super(SogCLR, self).__init__()
        # # sogclr
        # if not device:
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # else:
        #     self.device = device
        #
        #     # for DCL
        self.u = torch.zeros(N).reshape(-1, 1)  # .to(self.device)
        self.LARGE_NUM = 1e9
        self.T = 0.5

    def forward(self, hidden, index=None, gamma=0.9, distributed=False):
        B = hidden.shape[0] // 2
        hidden1, hidden2 = hidden[0:B], hidden[B:]
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if distributed:
            hidden1_large = torch.cat(all_gather_layer.apply(hidden1), dim=0)  # why concat_all_gather()
            hidden2_large = torch.cat(all_gather_layer.apply(hidden2), dim=0)
            enlarged_batch_size = hidden1_large.shape[0]

            labels_idx = (torch.arange(batch_size, dtype=torch.long) + batch_size * torch.distributed.get_rank()).to(
                self.device)
            labels = F.one_hot(labels_idx, enlarged_batch_size * 2).to(self.device)
            masks = F.one_hot(labels_idx, enlarged_batch_size).to(self.device)
            batch_size = enlarged_batch_size
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(hidden.device)
            masks = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(hidden.device)

        logits_aa = torch.matmul(hidden1, hidden1_large.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)
        logits_ba = torch.matmul(hidden2, hidden1_large.T)

        #  SogCLR
        neg_mask = 1 - labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa], 1)
        logits_ba_bb = torch.cat([logits_ba, logits_bb], 1)

        neg_logits1 = torch.exp(logits_ab_aa / self.T) * neg_mask  # (B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb / self.T) * neg_mask

        # u init
        if self.u[index.cpu()].sum() == 0:
            gamma = 1

        u1 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits1, dim=1, keepdim=True) / (
                    2 * (batch_size - 1))
        u2 = (1 - gamma) * self.u[index.cpu()].cuda() + gamma * torch.sum(neg_logits2, dim=1, keepdim=True) / (
                    2 * (batch_size - 1))

        # this sync on all devices (since "hidden" are gathering from all devices)
        if distributed:
            u1_large = concat_all_gather(u1)
            u2_large = concat_all_gather(u2)
            index_large = concat_all_gather(index)
            self.u[index_large.cpu()] = (u1_large.detach().cpu() + u2_large.detach().cpu()) / 2
        else:
            self.u[index.cpu()] = (u1.detach().cpu() + u2.detach().cpu()) / 2

        p_neg_weights1 = (neg_logits1 / u1).detach()
        p_neg_weights2 = (neg_logits2 / u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights * logits, dim=1, keepdim=True) / (2 * (batch_size - 1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss