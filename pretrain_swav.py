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
from datasets.dataloader import get_dataloader, get_unlabeled_dataloader
from methods.baselinetrain import BaselineTrain

from io_utils import parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, tieredImageNet_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot


class SwAV(nn.Module): # wrapper for BaselineTrain
    def __init__(self, base_encoder, normalize=False, output_dim=128, hidden_mlp=2048, nmb_prototypes=3000):    # based on the default values of SwAV code
        super(SwAV, self).__init__()
        assert isinstance(base_encoder, BaselineTrain)
        self.base_encoder = base_encoder
        
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(base_encoder.feature.final_feat_dim, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(base_encoder.feature.final_feat_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            ) 

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            raise NotImplementedError('for MultiPrototypes')
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
    
    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)
        if self.l2norm:
            x = F.normalize(x, dim=1, p=2)
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        # not multi-crop setting for now...
        outputs = self.base_encoder.feature(torch.cat([inputs[0], inputs[1]], dim=0).cuda(non_blocking=True))
        return self.forward_head(outputs)


@torch.no_grad()
def distributed_sinkhorn(out, params):
    Q = torch.exp(out / params['epsilon']).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(params['sinkhorn_iterations']):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def train(model, checkpoint_dir, pretrain_type, dataset_name=None,
          labeled_source_loader=None, unlabeled_source_loader=None, unlabeled_target_loader=None):

    ################################################
    # default swav setting for non-multi-crop
    nmb_crops = [2] # for multi-crop, use eg. [2,6]
    size_crops = [224] # for multi-crop, use eg. [224, 84]
    crops_for_assign = [0, 1]
    temperature = 0.1
    queue_length = 0
    freeze_prototypes_niters = 313
    sinkhorn_params = {'epsilon': 0.05, 
                       'sinkhorn_iterations': 3}
    ################################################
    
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
        model.base_encoder.feature.load_state_dict(state, strict=True)

    model.train()
    model.cuda()
    opt_params = [{'params': model.parameters()}]

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
                torch.save({'epoch':epoch, 'state':model.base_encoder.state_dict(), 'projection_head':model.projection_head.state_dict()}, outfile)
                queue = None
                if queue_length > 0:
                    queue = torch.zeros(len(crops_for_assign), queue_length, 128).cuda()

            for i, (X, y) in enumerate(unlabeled_source_loader): # For pre-training 2

                iteration = epoch * len(unlabeled_source_loader) + i
                embedding, output = model(X)
                embedding = embedding.detach()
                bs = X[0].size(0)
                    
                # ============ swav loss ... ============
                loss = 0
                for j, crop_id in enumerate(crops_for_assign):
                    with torch.no_grad():
                        out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                        # time to use the queue
                        if queue is not None:
                            if not torch.all(queue[j, -1, :] == 0):
                                out = torch.cat((torch.mm(
                                    queue[j],
                                    model.prototypes.weight.t()
                                ), out))
                            # fill the queue
                            queue[j, bs:] = queue[j, :-bs].clone()
                            queue[j, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                        # get assignments
                        q = distributed_sinkhorn(out, sinkhorn_params)[-bs:]

                    # cluster assignment prediction
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id): # crop_id -> 0, v -> 1
                        x = output[bs * v: bs * (v + 1)] / temperature
                        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    loss += subloss / (np.sum(nmb_crops) - 1)
                loss /= len(crops_for_assign)

                optimizer.zero_grad()
                loss.backward()
                # cancel gradients for the prototypes
                if iteration < freeze_prototypes_niters:
                    for name, p in model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None

                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            print ('epoch: {}, loss: {}'.format(epoch, epoch_loss/len(unlabeled_source_loader)))

            if (epoch%freq_epoch==0) or (epoch==stop_epoch-1):
                outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch':epoch, 'state':model.base_encoder.state_dict(), 'projection_head':model.projection_head.state_dict()}, outfile)
    else:
        for epoch in range(start_epoch, stop_epoch):
            epoch_loss = 0
            if epoch == 0:
                outfile = os.path.join(checkpoint_dir, '{}_initial.tar'.format(dataset_name))
                torch.save({'epoch':epoch, 'state':model.base_encoder.state_dict(), 'projection_head':model.projection_head.state_dict()}, outfile)
                queue = None
                if queue_length > 0:
                    queue = torch.zeros(len(crops_for_assign), queue_length, 128).cuda()

            for i, (X, y) in enumerate(unlabeled_target_loader):

                iteration = epoch * len(unlabeled_target_loader) + i
                embedding, output = model(X)
                embedding = embedding.detach()
                bs = X[0].size(0)
                    
                # ============ swav loss ... ============
                swav_loss = 0
                for j, crop_id in enumerate(crops_for_assign):
                    with torch.no_grad():
                        out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                        # time to use the queue
                        if queue is not None:
                            if not torch.all(queue[j, -1, :] == 0):
                                out = torch.cat((torch.mm(
                                    queue[j],
                                    model.prototypes.weight.t()
                                ), out))
                            # fill the queue
                            queue[j, bs:] = queue[j, :-bs].clone()
                            queue[j, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                        # get assignments
                        q = distributed_sinkhorn(out, sinkhorn_params)[-bs:]

                    # cluster assignment prediction
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id): # crop_id -> 0, v -> 1
                        x = output[bs * v: bs * (v + 1)] / temperature
                        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    swav_loss += subloss / (np.sum(nmb_crops) - 1)
                swav_loss /= len(crops_for_assign)

                if labeled_source_loader is None and unlabeled_source_loader is None: # For pre-training 3, 6
                    loss = swav_loss

                elif labeled_source_loader is not None: # For pre-training 4, 7
                    try:
                        X_base, y_base = labeled_source_loader_iter.next()
                    except StopIteration:
                        labeled_source_loader_iter = iter(labeled_source_loader)
                        X_base, y_base = labeled_source_loader_iter.next()

                    features_base = model.base_encoder.feature(X_base.cuda())
                    logits_base = model.base_encoder.classifier(features_base)
                    log_probability_base = F.log_softmax(logits_base, dim=1)

                    gamma = 0.50
                    loss = gamma * swav_loss + (1-gamma) * nll_criterion(log_probability_base, y_base.cuda())
                    
                elif unlabeled_source_loader is not None: # For pre-training 5, 8
                    try:
                        X_base, y_base = unlabeled_source_loader_iter.next()
                    except StopIteration:
                        unlabeled_source_loader_iter = iter(unlabeled_source_loader)
                        X_base, y_base = unlabeled_source_loader_iter.next()

                        embedding, output = model(X_base)
                        embedding = embedding.detach()
                        bs = X_base[0].size(0)

                        # ============ swav loss ... ============
                        swav_loss_base = 0
                        for k, crop_id in enumerate(crops_for_assign):
                            with torch.no_grad():
                                out = output[bs * crop_id: bs * (crop_id + 1)].detach()
                                # not use the queue
                                # get assignments
                                q = distributed_sinkhorn(out, sinkhorn_params)[-bs:]

                            # cluster assignment prediction
                            subloss = 0
                            for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id): # crop_id -> 0, v -> 1
                                x = output[bs * v: bs * (v + 1)] / temperature
                                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                            swav_loss_base += subloss / (np.sum(nmb_crops) - 1)
                        swav_loss_base /= len(crops_for_assign)
                    loss = 0.5 * swav_loss + 0.5 * swav_loss_base
                
                else:
                    raise Exception('Invalid loader settings')

                optimizer.zero_grad()
                loss.backward()
                # cancel gradients for the prototypes
                if iteration < freeze_prototypes_niters:
                    for name, p in model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None

                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            print ('epoch: {}, loss: {}'.format(epoch, epoch_loss/len(unlabeled_target_loader)))

            if (epoch%freq_epoch==0) or (epoch==stop_epoch-1):
                outfile = os.path.join(checkpoint_dir, '{}_{:d}.tar'.format(dataset_name, epoch))
                torch.save({'epoch':epoch, 'state':model.base_encoder.state_dict(), 'projection_head':model.projection_head.state_dict()}, outfile)

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
        baseline = BaselineTrain(model_dict[params.model], num_class=params.num_classes)
        model = SwAV(baseline)
    else:
        raise ValueError('Invalid `method` argument: {}'.format(params.method))

    if params.aug_mode is None:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s_swav/type%s' %(configs.save_dir, params.dataset, params.model, params.method, str(params.pretrain_type))
    else:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s_swav/type%s_%s' %(configs.save_dir, params.dataset, params.model, params.method, str(params.pretrain_type), params.aug_mode)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    """
    pretrain_type
        1: Source L
        2: Source U

        3: Target U
        6: (Source L) > Target U

        4: Source L + Target U
        7: (Source L) > Source L + Target U

        5: Source U + Target U
        8: (Source L) > Source U + Target U
    """
    if params.pretrain_type == 1:
        raise NotImplementedError

    labeled_source_loader = None
    unlabeled_source_loader = None
    unlabeled_target_loader = None

    base_batch_size = 64
    labeled_source_bs = base_batch_size
    unlabeled_source_bs = base_batch_size
    unlabeled_target_bs = base_batch_size
    if params.pretrain_type in [5, 8]:
        unlabeled_source_bs //= 2
        unlabeled_target_bs //= 2

    if params.pretrain_type in [1, 4, 7]:
        labeled_source_loader = get_dataloader(dataset_name=params.dataset, augmentation=params.aug_mode,
                                               batch_size=labeled_source_bs)
        print('Source labeled ({}) data loader initialized.'.format(params.dataset))
    if params.pretrain_type in [2, 5, 8]:
        print('Source unlabeled ({}) data loader initialized.'.format(params.dataset))
        unlabeled_source_loader = get_dataloader(dataset_name=params.dataset, augmentation=params.aug_mode,
                                                 batch_size=unlabeled_source_bs,
                                                 siamese=True)  # important

    if params.pretrain_type in [1, 2]:
        print('Start pretraining type {}'.format(params.pretrain_type).center(60).center(80, '#'))
        train(model, checkpoint_dir, pretrain_type=params.pretrain_type, dataset_name=None,
              labeled_source_loader=labeled_source_loader, unlabeled_source_loader=unlabeled_source_loader,
              unlabeled_target_loader=unlabeled_target_loader)
    else:
        for i, dataset_name in enumerate(params.dataset_names):
            unlabeled_target_loader = get_unlabeled_dataloader(dataset_name=dataset_name, augmentation=params.aug_mode,
                                                               batch_size=unlabeled_target_bs, siamese=True,
                                                               unlabeled_ratio=params.unlabeled_ratio)
            print('Target unlabeled ({}) data loader initialized.'.format(params.dataset))
            print('Start pretraining type {}'.format(params.pretrain_type).center(60).center(80, '#'))
            print('Target {}/{}: {}'.format(i + 1, len(params.dataset_names), dataset_name).center(60).center(80, '#'))
            train(model, checkpoint_dir, pretrain_type=params.pretrain_type, dataset_name=dataset_name,
                  labeled_source_loader=labeled_source_loader, unlabeled_source_loader=unlabeled_source_loader,
                  unlabeled_target_loader=unlabeled_target_loader)
