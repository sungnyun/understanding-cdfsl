from typing import List, Dict

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import math
import time
import os
import glob
import warnings
from itertools import combinations
from tqdm import tqdm

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain

from io_utils import parse_args, get_init_file, get_resume_file, get_best_file, get_assigned_file
from utils import *
from datasets import miniImageNet_few_shot, tieredImageNet_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

class projector_SIMCLR(nn.Module):
    '''
        The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    '''
    def __init__(self, in_dim, out_dim):
        super(projector_SIMCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def finetune(params, dataset_name, novel_loader, pretrained_dataset, pretrained_model, checkpoint_dir, n_query=15, n_way=5, n_support=5):
    iter_num = len(novel_loader)

    df = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['accuracy'])

    basename = '{}_{}way{}shot_nil_results.csv'.format(dataset_name, n_way, n_support)
    result_path = os.path.join(checkpoint_dir, basename)
    print('Saving results to {}'.format(result_path))

    if params.pretrain_type == 1 or params.pretrain_type == 2:
        modelfile = get_resume_file(checkpoint_dir)
    else:
        modelfile = get_resume_file(checkpoint_dir, dataset_name)
    if not os.path.exists(modelfile):
        raise Exception('Invalid model path: "{}" (no such file found)'.format(modelfile))
    print ('Using model weights path {}'.format(modelfile))

    ##### Fine-tuning and Evaluation #####
    print ('NIL-testing start!')
    for task_num, (x, y) in tqdm(enumerate(novel_loader)):
        task_all = []

        # Load pre-trained state dict
        tmp = torch.load(modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state[newkey] = state.pop(key)
        pretrained_model.feature.load_state_dict(state, strict=True)
        pretrained_model.cuda()

        # Set both support and query set
        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25 (5-way * 5-n_support), 3, 224, 224)
        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,  *x.size()[2:]) # (75 (5-way * 15-n_qeury), 3, 224, 224)
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)
        y_b_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_query ) )).cuda()

        # Find the template using support set
        pretrained_model.eval()
        support_output = pretrained_model.feature(x_a_i)
        support_template = torch.zeros([n_way, support_output.shape[1]]).cuda()
        for label in range(n_way):
            support_template[label] = torch.mean(support_output[torch.where(y_a_i==label)], dim=0)
        
        # NIL testing
        query_output = pretrained_model.feature(x_b_i)
        
        distance_mtx = torch.zeros([len(query_output), len(support_template)])
        cos = nn.CosineSimilarity()
        for i, q in enumerate(query_output):
            distance_mtx[i] = cos(torch.cat([q.unsqueeze(0)]*len(support_template)), support_template)
        query_pred = torch.argmax(distance_mtx, dim=1)
        
        task_all.append(100*(sum(query_pred==y_b_i.cpu())/float(len(query_pred))).item())

        df.loc[task_num+1] = task_all
        df.to_csv(result_path)
    print ('{:4.2f} +- {:4.2f}'.format(df.mean()[-1], 1.96*df.std()[-1]/np.sqrt(iter_num)))

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

    pretrained_dataset = params.dataset
    if pretrained_dataset == 'miniImageNet':
        params.num_classes = 64
    elif pretrained_dataset == 'tieredImageNet':
        params.num_classes = 351
    elif pretrained_dataset == 'ImageNet':
        params.num_classes = 1000
    elif pretrained_dataset == 'none':
        params.num_classes = 5
    else:
        raise ValueError('Invalid `dataset` argument: {}'.format(pretrained_dataset))

    if params.method == 'baseline':
        pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax')
    elif params.method == 'baseline++':
        pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')
    else:
        raise ValueError('Invalid `method` argument: {}'.format(params.method))

    checkpoint_dir = '%s/checkpoints/%s/%s_%s/type%s_%s' %(configs.save_dir, params.dataset, params.model, params.method, str(params.pretrain_type), params.aug_mode)
    ##################################################################
    image_size = 224  # for every evaluation dataset except tieredImageNet
    iter_num = 600
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    dataset_names = params.dataset_names
    split = params.startup_split
    
    for dataset_name in dataset_names:
        print (dataset_name)
        print ('Initializing data loader...')
        if dataset_name == "miniImageNet_test":
            datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_episode=iter_num, n_query=15, split=split, **few_shot_params)
        if dataset_name == "tieredImageNet":
            image_size = 84
            datamgr = tieredImageNet_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, split=split, **few_shot_params)
        elif dataset_name == "CropDisease":
            datamgr = CropDisease_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, split=split, **few_shot_params)
        elif dataset_name == "EuroSAT":
            datamgr = EuroSAT_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, split=split, **few_shot_params)
        elif dataset_name == "ISIC":
            datamgr = ISIC_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, split=split, **few_shot_params)
        elif dataset_name == "ChestX":
            datamgr = Chest_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, split=split, **few_shot_params)
        
        if dataset_name == "miniImageNet_test" or dataset_name == "tieredImageNet":
            novel_loader = datamgr.get_data_loader(aug=False, train=False)
        else:
            novel_loader = datamgr.get_data_loader(aug=False)
        print('Data loader initialized successfully!')

        finetune(params, dataset_name, novel_loader, pretrained_dataset, pretrained_model, checkpoint_dir=checkpoint_dir, n_query=15, **few_shot_params)