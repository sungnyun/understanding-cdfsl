import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain

from io_utils import parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, tieredImageNet_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot


class apply_twice:
    '''
        A wrapper for torchvision transform. The transform is applied twice for 
        SimCLR training
    '''
    def __init__(self, transform, transform2=None):
        self.transform = transform

        if transform2 is not None:
            self.transform2 = transform2
        else:
            self.transform2 = transform

    def __call__(self, img):
        return self.transform(img), self.transform2(img)


class MoCo(nn.Module):
    def __init__(self, backbone, num_class, base_encoder=BaselineTrain, dim=128, K=1024, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(backbone, num_class, loss_type='softmax')
        self.encoder_k = base_encoder(backbone, num_class, loss_type='softmax')
        self.projector = nn.Linear(self.encoder_q.feature.final_feat_dim, dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)        

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        q = self.projector(self.encoder_q.feature(im_q)) # queries: NxC
        q = F.normalize(q, dim=1)
            
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.projector(self.encoder_k.feature(im_k))  # keys: NxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # apply temperature
        logits /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


def set_labeled_source_loader(dataset_name, aug_mode, batch_size):
    if dataset_name == 'miniImageNet':
        transform = miniImageNet_few_shot.TransformLoader(image_size=224).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = miniImageNet_few_shot.SimpleDataset(transform, train=True)
        labeled_source_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=False) # batch size is originally 16
    elif dataset_name == 'tieredImageNet':
        transform = tieredImageNet_few_shot.TransformLoader(image_size=84).get_composed_transform(aug=False) # Do no augmentation for tieredImageNet to be consisitent with the literature
        dataset = tieredImageNet_few_shot.SimpleDataset(transform, train=True)
        labeled_source_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=False)
    return labeled_source_loader

def set_unlabeled_source_loader(dataset_name, aug_mode, batch_size):
    if dataset_name == 'miniImageNet':
        transform = miniImageNet_few_shot.TransformLoader(image_size=224).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = miniImageNet_few_shot.SimpleDataset(apply_twice(transform), train=True)
    elif dataset_name == 'tieredImageNet':
        transform = tieredImageNet_few_shot.TransformLoader(image_size=84).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = tieredImageNet_few_shot.SimpleDataset(apply_twice(transform), train=True)
    unlabeled_source_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
    return unlabeled_source_loader

def set_unlabeled_target_loader(dataset_name, aug_mode, batch_size):
    if dataset_name == 'miniImageNet':
        transform = miniImageNet_few_shot.TransformLoader(image_size=224).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = miniImageNet_few_shot.SimpleDataset(apply_twice(transform), train=False, split=True)
    elif dataset_name == 'tieredImageNet':
        transform = tieredImageNet_few_shot.TransformLoader(image_size=84).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = tieredImageNet_few_shot.SimpleDataset(apply_twice(transform), train=False, split=True)
    elif dataset_name == 'CropDisease':
        transform = CropDisease_few_shot.TransformLoader(image_size=224).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = CropDisease_few_shot.SimpleDataset(apply_twice(transform), split=True)
    elif dataset_name == 'EuroSAT':
        transform = EuroSAT_few_shot.TransformLoader(image_size=224).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = EuroSAT_few_shot.SimpleDataset(apply_twice(transform), split=True)
    elif dataset_name == 'ISIC':
        transform = ISIC_few_shot.TransformLoader(image_size=224).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = ISIC_few_shot.SimpleDataset(apply_twice(transform), split=True)
    elif dataset_name == 'ChestX':
        transform = Chest_few_shot.TransformLoader(image_size=224).get_composed_transform(aug=True, aug_mode=aug_mode)
        dataset = Chest_few_shot.SimpleDataset(apply_twice(transform), split=True)
    unlabeled_target_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
    return unlabeled_target_loader

def train(model, checkpoint_dir, pretrain_type, dataset_name=None,
          labeled_source_loader=None, unlabeled_source_loader=None, unlabeled_target_loader=None):
    
    if labeled_source_loader is None and unlabeled_source_loader is None and unlabeled_target_loader is None:
        raise ValueError('Invalid unlabeled loaders')

    start_epoch = 0
    stop_epoch = 1000
    freq_epoch = 100

    if pretrain_type in [6, 7, 8]:
        first_pretrained_model_dir = '%s/checkpoints/miniImageNet/ResNet10_baseline/type1_strong' %(configs.save_dir)
        modelfile = get_resume_file(first_pretrained_model_dir)
        if not os.path.exists(modelfile):
            raise Exception('Invalid model path: "{}" (no such file found)'.format(modelfile))
        print ('Pre-training from the model weights path {}'.format(modelfile))

        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state[newkey] = state.pop(key)
        model.encoder_q.feature.load_state_dict(state, strict=True)
        model.encoder_k.feature.load_state_dict(state, strict=True) 

    model.train()
    model.cuda()
    opt_params = [{'params': model.parameters()}]

    if pretrain_type != 1:
        criterion = nn.CrossEntropyLoss().cuda()

    if pretrain_type != 1 and labeled_source_loader is not None:
        labeled_source_loader_iter = iter(labeled_source_loader)
        nll_criterion = nn.NLLLoss(reduction='mean').cuda()

    if pretrain_type != 2 and unlabeled_source_loader is not None:
        unlabeled_source_loader_iter = iter(unlabeled_source_loader)
    
    optimizer = torch.optim.SGD(opt_params,
            lr=0.1, momentum=0.9,
            weight_decay=1e-4,
            nesterov=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[400,600,800],
                                                     gamma=0.1)

    print ("Learning setup is set!")

    if pretrain_type == 1:
        raise NotImplementedError
 
    elif pretrain_type == 2:
        for epoch in range(start_epoch, stop_epoch):
            epoch_loss = 0
            if epoch == 0:
                outfile = os.path.join(checkpoint_dir, 'initial.tar')
                torch.save({'epoch':epoch, 'state':model.encoder_q.state_dict(), 'projector':model.projector.state_dict()}, outfile)

            for i, (X, y) in enumerate(unlabeled_source_loader): # For pre-training 2
                outputs = model(im_q=X[0].cuda(non_blocking=True), im_k=X[1].cuda(non_blocking=True))
                loss = criterion(*outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            print ('epoch: {}, loss: {}'.format(epoch, epoch_loss/len(unlabeled_source_loader)))

            if (epoch%freq_epoch==0) or (epoch==stop_epoch-1):
                outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch':epoch, 'state':model.encoder_q.state_dict(), 'projector':model.projector.state_dict()}, outfile)
    else:
        for epoch in range(start_epoch, stop_epoch):
            epoch_loss = 0
            if epoch == 0:
                outfile = os.path.join(checkpoint_dir, '{}_initial.tar'.format(dataset_name))
                torch.save({'epoch':epoch, 'state':model.encoder_q.state_dict(), 'projector':model.projector.state_dict()}, outfile)

            for i, (X, y) in enumerate(unlabeled_target_loader):
                outputs = model(im_q=X[0].cuda(non_blocking=True), im_k=X[1].cuda(non_blocking=True))

                if labeled_source_loader is None and unlabeled_source_loader is None: # For pre-training 3, 6
                    loss = criterion(*outputs)

                elif labeled_source_loader is not None: # For pre-training 4, 7
                    try:
                        X_base, y_base = labeled_source_loader_iter.next()
                    except StopIteration:
                        labeled_source_loader_iter = iter(labeled_source_loader)
                        X_base, y_base = labeled_source_loader_iter.next()

                    features_base = model.encoder_q.feature(X_base.cuda())
                    logits_base = model.encoder_q.classifier(features_base)
                    log_probability_base = F.log_softmax(logits_base, dim=1)

                    gamma = 0.50
                    loss = gamma * criterion(*outputs) + (1-gamma) * nll_criterion(log_probability_base, y_base.cuda())
                    
                elif unlabeled_source_loader is not None: # For pre-training 5, 8
                    try:
                        X_base, y_base = unlabeled_source_loader_iter.next()
                    except StopIteration:
                        unlabeled_source_loader_iter = iter(unlabeled_source_loader)
                        X_base, y_base = unlabeled_source_loader_iter.next()

                    outputs_base = model(im_q=X_base[0].cuda(non_blocking=True), im_k=X_base[1].cuda(non_blocking=True))
                    loss = criterion(torch.cat([outputs[0], outputs_base[0]], dim=0),
                                     torch.cat([outputs[1], outputs_base[1]], dim=0))
                
                else:
                    raise Exception('Invalid loader settings')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            print ('epoch: {}, loss: {}'.format(epoch, epoch_loss/len(unlabeled_target_loader)))

            if (epoch%freq_epoch==0) or (epoch==stop_epoch-1):
                outfile = os.path.join(checkpoint_dir, '{}_{:d}.tar'.format(dataset_name, epoch))
                torch.save({'epoch':epoch, 'state':model.encoder_q.state_dict(), 'projector':model.projector.state_dict()}, outfile)

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    ##################################################################
    if params.model == 'ResNet10':
        model_dict = {params.model: backbone.ResNet10(method=params.method, track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    elif params.model == 'ResNet18-84':
        model_dict = {params.model: backbone.ResNet18_84x84(track_bn=params.track_bn)}
    elif params.model == 'ResNet18':
        model_dict = {params.model: backbone.ResNet18(track_bn=params.track_bn)}
    else:
        raise ValueError('Invalid `model` argument: {}'.format(params.model))

    if params.dataset == 'miniImageNet':
        params.num_classes = 64
    elif params.dataset == 'tieredImageNet':
        params.num_classes = 351
    elif params.dataset == 'ImageNet':
        params.num_classes = 1000
    elif params.dataset == 'none':
        params.num_classes = 5
    else:
        raise ValueError('Invalid `dataset` argument: {}'.format(params.dataset))

    if params.method == 'baseline':
        model = MoCo(model_dict[params.model], num_class=params.num_classes)
    else:
        raise ValueError('Invalid `method` argument: {}'.format(params.method))

    if params.aug_mode is None:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s_moco/type%s' %(configs.save_dir, params.dataset, params.model, params.method, str(params.pretrain_type))
    else:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s_moco/type%s_%s' %(configs.save_dir, params.dataset, params.model, params.method, str(params.pretrain_type), params.aug_mode)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    ##################################################################
    if params.pretrain_type == 1: # Pretrained on labeled source data (Transfer)
        raise NotImplementedError 

    elif params.pretrain_type == 2: # Pretrained on unlabeled source data (MoCo (base))
        unlabeled_source_loader = set_unlabeled_source_loader(params.dataset, params.aug_mode, batch_size=64)
        print('Data loader initialized successfully! unlabeled source {}'.format(params.dataset))
        train(model, checkpoint_dir, pretrain_type=params.pretrain_type, dataset_name=None,
             labeled_source_loader=None, unlabeled_source_loader=unlabeled_source_loader, unlabeled_target_loader=None)

    elif params.pretrain_type in [3, 6]: # 3: Pretrained on unlabeled target data (MoCo)
                                         # 6: Pretrained on labeled source data -> unlabeled target data (Transfer+MoCo) (Pre-trained by type 1 and then type 3)
        dataset_names = params.dataset_names
        for dataset_name in dataset_names:
            unlabeled_target_loader = set_unlabeled_target_loader(dataset_name, params.aug_mode, batch_size=64)
            print('Data loader initialized successfully! unlabeled target {}'.format(dataset_name))
            train(model, checkpoint_dir, pretrain_type=params.pretrain_type, dataset_name=dataset_name,
                  labeled_source_loader=None, unlabeled_source_loader=None, unlabeled_target_loader=unlabeled_target_loader)

    elif params.pretrain_type in [4, 7]: # 4: Pretrained on labeled source data + unlabeled target data
                                         # 7: Pretrained on labeled source data -> labeled source data + unlabeled target data (Pre-trained by type 1 and then type 4)
        labeled_source_loader = set_labeled_source_loader(params.dataset, params.aug_mode, batch_size=64)
        dataset_names = params.dataset_names
        for dataset_name in dataset_names:
            unlabeled_target_loader = set_unlabeled_target_loader(dataset_name, params.aug_mode, batch_size=64)
            print('Data loader initialized successfully! unlabeled target {} with labeled {}'.format(dataset_name, params.dataset))
            train(model, checkpoint_dir, pretrain_type=params.pretrain_type, dataset_name=dataset_name,
                  labeled_source_loader=labeled_source_loader, unlabeled_source_loader=None, unlabeled_target_loader=unlabeled_target_loader)

    elif params.pretrain_type in [5, 8]: # 5: Pretrained on unlabeled source data + unlabeled target data
                                         # 8: Pretrained on labeled source data -> unlabeled source data + unlabeled target data (Pre-trained by type 1 and then type 5)
        unlabeled_source_loader = set_unlabeled_source_loader(params.dataset, params.aug_mode, batch_size=32)
        dataset_names = params.dataset_names
        for dataset_name in dataset_names:
            unlabeled_target_loader = set_unlabeled_target_loader(dataset_name, params.aug_mode, batch_size=32)
            print('Data loader initialized successfully! unlabeled target {} with unlabeled {}'.format(dataset_name, params.dataset))
            train(model, checkpoint_dir, pretrain_type=params.pretrain_type, dataset_name=dataset_name,
                  labeled_source_loader=None, unlabeled_source_loader=unlabeled_source_loader, unlabeled_target_loader=unlabeled_target_loader)
