from argparse import Namespace

from torch import nn

from model import BYOL


class SimSiam(BYOL):
    def __init__(self, backbone: nn.Module, params: Namespace):
        super().__init__(backbone, params, use_momentum=False)
