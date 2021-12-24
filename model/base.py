from abc import abstractmethod
from argparse import Namespace
from typing import Tuple, Union

import torch
from torch import nn


class BaseModel(nn.Module):
    """
    BaseModel subclasses self-contain all modules and losses required for pre-training.
    """

    def __init__(self, backbone: nn.Module, params: Namespace):
        super().__init__()
        self.backbone = backbone
        self.params = params
        self.classifier = nn.Linear(backbone.final_feat_dim, params.num_classes)
        self.classifier.bias.data.fill_(0)
        self.cls_loss_function = nn.CrossEntropyLoss()

    def forward_features(self, x):
        """
        You'll likely need to override this method for SSL models.
        """
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def compute_cls_loss_and_accuracy(self, x, y, return_predictions=False) -> Tuple:
        scores = self.forward(x)
        _, predicted = torch.max(scores.data, 1)
        accuracy = predicted.eq(y.data).cpu().sum() / x.shape[0]
        if return_predictions:
            return self.cls_loss_function(scores, y), accuracy, predicted
        else:
            return self.cls_loss_function(scores, y), accuracy


class BaseSelfSupervisedModel(BaseModel):
    @abstractmethod
    def compute_ssl_loss(self, x1, x2=None, return_features=False):
        """
        If SSL is based on paired input:
            By default: x1, x2 represent the input pair.
            If x2=None: x1 alone contains the full concatenated input pair.
        Else:
            x1 contains the input.
        """
        raise NotImplementedError()
