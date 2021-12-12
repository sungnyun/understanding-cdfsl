import numpy as np
import torch
import torch.nn as nn
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
from methods.maml import MAML
from methods.boil import BOIL
from methods.protonet import ProtoNet

from io_utils import parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, tieredImageNet_few_shot


def train(base_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        if params.method == 'baseline_body':
            head_params = [p for name, p in model.named_parameters() if 'classifier' in name]
            body_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
            optimizer = torch.optim.Adam([{'params': head_params, 'lr': 0},
                                          {'params': body_params, 'lr': 1e-3}])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)
        scheduler = None
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    for epoch in range(start_epoch, stop_epoch):
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
            
        if epoch == 0:
            outfile = os.path.join(params.checkpoint_dir, 'initial.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
            
        model.train()
        model.train_loop(epoch, base_loader, optimizer, scheduler)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    if params.dataset == 'miniImageNet':
        model_dict = {params.model: backbone.ResNet10(method=params.method, track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    # elif params.model == 'ResNet12':
    #     model_dict = {params.model: backbone.ResNet12(track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    elif params.dataset == 'tieredImageNet':
        if params.reinit_bn_stats:
            raise NotImplementedError('Not supported')
        model_dict = {params.model: backbone.ResNet18_84x84(track_bn=params.track_bn)}
    # elif params.dataset  == 'ImageNet':
    #     if params.reinit_bn_stats:
    #         raise NotImplementedError('Not supported')
    #     model_dict = {params.model: backbone.ResNet18(track_bn=params.track_bn)}
    #     image_size = 224
    else:
        raise ValueError('Unknown extractor')

    optimization = 'Adam'

    if params.method in ['baseline', 'baseline++', 'baseline_body'] :
        if params.dataset == "miniImageNet":
            image_size = 224
            bsize = 16 # Original
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size=bsize)
            base_loader = datamgr.get_data_loader(aug=params.train_aug)
            params.num_classes = 64
        elif params.dataset == 'tieredImageNet':
            image_size = 84
            bsize = 256
            datamgr = tieredImageNet_few_shot.SimpleDataManager(image_size, batch_size=bsize)
            base_loader = datamgr.get_data_loader(aug=False) # Do no augmentation for tiered imagenet to be consisitent with the literature
            params.num_classes = 351
#         elif params.dataset == 'ImageNet':
#             image_size = 224
#             bsize = 256
#             datamgr = ImageNet_few_shot.SimpleDataManager(image_size, batch_size=bsize)
#             base_loader = datamgr.get_data_loader(aug=params.train_aug, num_workers=2)
#             params.num_classes = 1000
        else:
            raise ValueError('Unknown dataset')
            
        if params.method == 'baseline' or params.method == 'baseline_body':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax')
        elif params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')
            
    elif params.method in ['maml', 'boil', 'protonet']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        if params.dataset == "miniImageNet":
            datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query, **train_few_shot_params)
            base_loader = datamgr.get_data_loader(aug = params.train_aug)
            params.num_classes = 64
        else:
            raise ValueError('Unknown dataset')

        if params.method == 'maml':
            model = MAML(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'boil':
            model = BOIL(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)

    else:
        raise ValueError('Unknown method')

    model = model.cuda()
    save_dir = configs.save_dir

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if params.track_bn:
        params.checkpoint_dir += '_track'
    if params.reinit_bn_stats:
        params.checkpoint_dir += '_Restats'
    if not params.method in ['baseline', 'baseline++', 'baseline_body']:
        params.checkpoint_dir += '_%dway_%dshot'%(params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params)
