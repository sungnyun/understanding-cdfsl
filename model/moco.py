import copy
from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from model.base import BaseSelfSupervisedModel


class MoCo(BaseSelfSupervisedModel):
    def __init__(self, backbone: nn.Module, params: Namespace):
        super().__init__(backbone, params)

        dim = 128
        mlp = False
        self.K = 1024
        self.m = 0.999
        self.T = 1.0

        self.encoder_q = self.backbone
        self.encoder_k = copy.deepcopy(self.backbone)

        if not mlp:
            self.projector_q = nn.Linear(self.encoder_q.final_feat_dim, dim)
            self.projector_k = nn.Linear(self.encoder_k.final_feat_dim, dim)
        else:
            mlp_dim = self.encoder_q.feature.final_feat_dim
            self.projector_q = nn.Sequential(nn.Linear(mlp_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, dim))
            self.projector_k = nn.Sequential(nn.Linear(mlp_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, dim))

        self.encoder_k.requires_grad_(False)
        self.projector_k.requires_grad_(False)
        # Just in case (copied from old code)
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False
        for param_k in self.projector_k.parameters():
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.ce_loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q_, param_k_ in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k_.data = param_k_.data * self.m + param_q_.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def compute_ssl_loss(self, x1, x2=None, return_features=False):
        if x2 is None:
            x = x1
            batch_size = int(x.shape[0] / 2)
            im_q = x[:batch_size]
            im_k = x[batch_size:]
        else:
            im_q = x1
            im_k = x2

        q_features = self.encoder_q(im_q)
        q = self.projector_q(q_features)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k_features = self.encoder_k(im_k)
            k = self.projector_k(k_features)  # keys: NxC
            k = F.normalize(k, dim=1)

        # compute logits (Einstein sum is more intuitive)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # positive logits: Nx1
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # negative logits: NxK

        logits = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
        logits /= self.T  # apply temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # labels: positive key indicators

        self._dequeue_and_enqueue(k)

        loss = self.ce_loss(logits, labels)

        if return_features:
            if x2 is None:
                return loss, torch.cat([q_features, k_features])
            else:
                return loss, q_features, k_features
        else:
            return loss
