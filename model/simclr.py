from argparse import Namespace
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from model.base import BaseSelfSupervisedModel


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NTXentLoss(nn.Module):
    def __init__(self, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    @lru_cache(maxsize=4)
    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 *
                    batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 *
                    batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        batch_size = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)
        device = representations.device

        similarity_matrix = self.similarity_function(
            representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask = self._get_correlated_mask(batch_size).to(device)
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


class SimCLR(BaseSelfSupervisedModel):

    def __init__(self, backbone: nn.Module, params: Namespace):
        super().__init__(backbone, params)
        self.head = ProjectionHead(backbone.final_feat_dim, out_dim=params.model_simclr_projection_dim)
        self.ssl_loss_fn = NTXentLoss(temperature=params.model_simclr_temperature, use_cosine_similarity=True)
        self.final_feat_dim = self.backbone.final_feat_dim

    def compute_ssl_loss(self, x1, x2=None, return_features=False):
        if x2 is None:
            x = x1
        else:
            x = torch.cat([x1, x2])
        batch_size = int(x.shape[0] / 2)

        f = self.backbone(x)
        f1, f2 = f[:batch_size], f[batch_size:]
        p1 = self.head(f1)
        p2 = self.head(f2)
        loss = self.ssl_loss_fn(p1, p2)

        if return_features:
            if x2 is None:
                return loss, f
            else:
                return loss, f1, f2
        else:
            return loss
