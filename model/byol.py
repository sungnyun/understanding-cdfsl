import copy
import random
from argparse import Namespace
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn

from model.base import BaseSelfSupervisedModel


def _singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def _get_module_device(module):
    return next(module.parameters()).device


def _set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def _loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def _update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size,
                 layer=-1):  # default layer = -2 since network includes classifier. Ours does not have classifier.
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)  # flatten

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @_singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection=True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


class BYOL(BaseSelfSupervisedModel):
    def __init__(self, backbone: nn.Module, params: Namespace, use_momentum=True):
        super().__init__(backbone, params)

        image_size = 224
        hidden_layer = -1
        projection_size = 256
        projection_hidden_size = 4096
        moving_average_decay = 0.99
        use_momentum = use_momentum

        self.online_encoder = NetWrapper(self.backbone, projection_size, projection_hidden_size, layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = _get_module_device(backbone)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.compute_ssl_loss(torch.randn(2, 3, image_size, image_size, device=device),
                              torch.randn(2, 3, image_size, image_size, device=device))

    @_singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        _set_requires_grad(target_encoder, False)
        return target_encoder

    def _reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def _update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        _update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def compute_ssl_loss(self, x1, x2=None, return_features=False):
        if x2 is None:
            x = x1
            batch_size = int(x.shape[0] / 2)
            x1 = x[:batch_size]
            x2 = x[batch_size:]

        assert not (self.training and x1.shape[
            0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        online_proj_one, _ = self.online_encoder(x1)
        online_proj_two, _ = self.online_encoder(x2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(x1)
            target_proj_two, _ = target_encoder(x2)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = _loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = _loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        loss = loss.mean()

        if return_features:
            if x2 is None:
                return loss, torch.cat([online_proj_one, online_proj_two])
            else:
                return loss, online_proj_one, online_proj_two
        else:
            return loss

    def on_step_end(self):
        if self.use_momentum:
            self._update_moving_average()

