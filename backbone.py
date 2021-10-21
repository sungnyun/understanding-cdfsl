import torch
import torchvision
from torch import Tensor
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.nn.utils.weight_norm import WeightNorm

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# For meta-learning based algorithms (task-specific weight)
class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    def __init__(self, method, indim, outdim, half_res, track_bn):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim, track_running_stats=track_bn)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim, track_running_stats=track_bn)
            
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim, track_running_stats=track_bn)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, method, block, list_of_num_layers, list_of_out_dims, flatten, track_bn, reinit_bn_stats):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'

        self.reinit_bn_stats = reinit_bn_stats
        
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64, track_running_stats=track_bn)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(method, indim, list_of_out_dims[i], half_res, track_bn)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        if self.reinit_bn_stats:
            self._reinit_running_batch_statistics()
        out = self.trunk(x)
        return out
        
    def _reinit_running_batch_statistics(self):
        with torch.no_grad():
            self.trunk[1].running_mean.data.fill_(0.)
            self.trunk[1].running_var.data.fill_(1.)

            self.trunk[4].BN1.running_mean.data.fill_(0.)
            self.trunk[4].BN1.running_var.data.fill_(1.)
            self.trunk[4].BN2.running_mean.data.fill_(0.)
            self.trunk[4].BN2.running_var.data.fill_(1.)

            self.trunk[5].BN1.running_mean.data.fill_(0.)
            self.trunk[5].BN1.running_var.data.fill_(1.)
            self.trunk[5].BN2.running_mean.data.fill_(0.)
            self.trunk[5].BN2.running_var.data.fill_(1.)
            self.trunk[5].BNshortcut.running_mean.data.fill_(0.)
            self.trunk[5].BNshortcut.running_var.data.fill_(1.)

            self.trunk[6].BN1.running_mean.data.fill_(0.)
            self.trunk[6].BN1.running_var.data.fill_(1.)
            self.trunk[6].BN2.running_mean.data.fill_(0.)
            self.trunk[6].BN2.running_var.data.fill_(1.)
            self.trunk[6].BNshortcut.running_mean.data.fill_(0.)
            self.trunk[6].BNshortcut.running_var.data.fill_(1.)

            self.trunk[7].BN1.running_mean.data.fill_(0.)
            self.trunk[7].BN1.running_var.data.fill_(1.)
            self.trunk[7].BN2.running_mean.data.fill_(0.)
            self.trunk[7].BN2.running_var.data.fill_(1.)
            self.trunk[7].BNshortcut.running_mean.data.fill_(0.)
            self.trunk[7].BNshortcut.running_var.data.fill_(1.)
    
    def return_features(self, x, return_avg=False):
        flat = Flatten()
        m = nn.AdaptiveAvgPool2d((1,1))

        with torch.no_grad():
            block1_out = self.trunk[4](self.trunk[3](self.trunk[2](self.trunk[1](self.trunk[0](x)))))
            block2_out = self.trunk[5](block1_out)
            block3_out = self.trunk[6](block2_out)
            block4_out = self.trunk[7](block3_out)
            
        if return_avg:
            return flat(m(block1_out)), flat(m(block2_out)), flat(m(block3_out)), flat(m(block4_out))
        else:
            return flat(block1_out), flat(block2_out), flat(block3_out), flat(block4_out)
    
    def forward_bodyfreeze(self,x):
        flat = Flatten()
        m = nn.AdaptiveAvgPool2d((1,1))

        with torch.no_grad():
            block1_out = self.trunk[4](self.trunk[3](self.trunk[2](self.trunk[1](self.trunk[0](x)))))
            block2_out = self.trunk[5](block1_out)
            block3_out = self.trunk[6](block2_out)
            
            out = self.trunk[7].C1(block3_out)
            out = self.trunk[7].BN1(out)
            out = self.trunk[7].relu1(out)
        
        out = self.trunk[7].C2(out)
        out = self.trunk[7].BN2(out)
        short_out = self.trunk[7].BNshortcut(self.trunk[7].shortcut(block3_out))
        out = out + short_out
        out = self.trunk[7].relu2(out)
        
        return flat(m(out))

def ResNet10(method, track_bn, reinit_bn_stats):
    return ResNet(method, block=SimpleBlock, list_of_num_layers=[1,1,1,1], list_of_out_dims=[64,128,256,512], flatten=True, track_bn=track_bn, reinit_bn_stats=reinit_bn_stats)

# -*- coding: utf-8 -*-
# https://github.com/ElementAI/embedding-propagation/blob/master/src/models/backbones/resnet12.py

class Block(torch.nn.Module):
    def __init__(self, ni, no, stride, dropout, track_bn, reinit_bn_stats):
        super().__init__()
        self.reinit_bn_stats = reinit_bn_stats
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else lambda x: x
        self.C0 = nn.Conv2d(ni, no, 3, stride, padding=1, bias=False)
        self.BN0 = nn.BatchNorm2d(no, track_running_stats=track_bn)
        self.C1 = nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(no, track_running_stats=track_bn)
        self.C2 = nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(no, track_running_stats=track_bn)
        if stride == 2 or ni != no:
            self.shortcut = nn.Conv2d(ni, no, 1, stride=1, padding=0, bias=False)
            self.BNshortcut = nn.BatchNorm2d(no, track_running_stats=track_bn)

    def get_parameters(self):
        return self.parameters()

    def forward(self, x):
        if self.reinit_bn_stats:
            self._reinit_running_batch_statistics()
        
        out = self.C0(x)
        out = self.BN0(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.C1(out)
        out = self.BN1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.C2(out)
        out = self.BN2(out)
        out += self.BNshortcut(self.shortcut(x))
        out = F.relu(out)
        
        return out
    
    def _reinit_running_batch_statistics(self):
        with torch.no_grad():
            self.BN0.running_mean.data.fill_(0.)
            self.BN0.running_var.data.fill_(1.)
            self.BN1.running_mean.data.fill_(0.)
            self.BN1.running_var.data.fill_(1.)
            self.BN2.running_mean.data.fill_(0.)
            self.BN2.running_var.data.fill_(1.)
            self.BNshortcut.running_mean.data.fill_(0.)
            self.BNshortcut.running_var.data.fill_(1.)

class ResNet12(torch.nn.Module):
    def __init__(self, track_bn, reinit_bn_stats, width=1, dropout=0):
        super().__init__()
        self.final_feat_dim = 512
        assert(width == 1) # Comment for different variants of this model
        self.widths = [x * int(width) for x in [64, 128, 256]]
        self.widths.append(self.final_feat_dim * width)
        # self.bn_out = nn.BatchNorm1d(self.final_feat_dim)

        start_width = 3
        for i in range(len(self.widths)):
            setattr(self, "group_%d" %i, Block(start_width, self.widths[i], 1, dropout, track_bn, reinit_bn_stats))
            start_width = self.widths[i]

    def add_classifier(self, nclasses, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.final_feat_dim, nclasses))

    def up_to_embedding(self, x):
        """ Applies the four residual groups
        Args:
            x: input images
            n: number of few-shot classes
            k: number of images per few-shot class
        """
        for i in range(len(self.widths)):
            x = getattr(self, "group_%d" % i)(x)
            x = F.max_pool2d(x, 3, 2, 1)
        return x

    def forward(self, x):
        """Main Pytorch forward function
        Returns: class logits
        Args:
            x: input mages
        """
        *args, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.up_to_embedding(x)
        # return F.relu(self.bn_out(x.mean(3).mean(2)), True)
        return F.relu(x.mean(3).mean(2), True)


class ResNet18(torchvision.models.resnet.ResNet):
    def __init__(self, track_bn):
        def norm_layer(*args, **kwargs):
            return nn.BatchNorm2d(*args, **kwargs, track_running_stats=track_bn)
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer)
        del self.fc

    def load_imagenet_weights(self, progress=True):
        state_dict = load_state_dict_from_url(torchvision.models.resnet.model_urls['resnet18'],
                                              progress=progress)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            raise AssertionError('Model code may be incorrect')

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x
