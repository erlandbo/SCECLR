# Copied from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copied from
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, xi, xj):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        # dots = torch.mm(x, x.t())
        dots = torch.cdist(xi, xj)
        n = xi.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, xi, xj, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        # student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(xi, xj)  # noqa: E741
        distances = self.pdist(xi, xj[I])**2  # BxD, BxD -> B
        loss = 1 / ( distances / 2.0 + 1 )**2.0
        return torch.log(loss + eps).mean()
        # loss = -torch.log(distances + eps).mean()
        # return I


class KoLeoLossOld(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, xi, xj):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        # dots = torch.mm(x, x.t())
        dots = torch.cdist(xi, xj)
        n = xi.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, xi, xj, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        # student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(xi, xj)  # noqa: E741
        distances = self.pdist(xi, xj[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        return loss