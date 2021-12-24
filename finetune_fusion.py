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

class FeatureFusionModule(nn.Module):
    def __init__(self, fusion_method, feature_dim):
        super(FeatureFusionModule, self).__init__()
        self.fusion_method = fusion_method

        if fusion_method == 'concat':
            pass
        elif fusion_method == 'adaptive_weight_scalar':
            self.alpha = nn.Parameter(torch.tensor(0.0))
            self.sigmoid = nn.Sigmoid()
        elif fusion_method == 'adaptive_weight_vector':
            self.source_alpha = nn.Parameter(torch.zeros(feature_dim))
            self.target_alpha = nn.Parameter(torch.zeros(feature_dim))
            self.sigmoid = nn.Sigmoid()

    def forward(self, source_feature, target_feature):
        if self.fusion_method == 'concat':
            return torch.cat([source_feature, target_feature], dim=1)
        elif self.fusion_method == 'adaptive_weight_scalar':
            source_weight = self.sigmoid(self.alpha)
            target_weight = 1-source_weight
            return source_weight*source_feature + target_weight*target_feature
        elif self.fusion_method == 'adaptive_weight_vector':
            source_weight = self.sigmoid(self.source_alpha)
            target_weight = self.sigmoid(self.target_alpha)
            return source_weight*source_feature + target_weight*target_feature

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

def finetune(params, dataset_name, novel_loader, pretrained_model, checkpoint_dir, fusion_method,
            finetune_epoch=100, batch_size=4, finetune_parts='head', n_query=15, n_way=5, n_support=5):
    iter_num = len(novel_loader)

    df = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['epoch{}'.format(e+1) for e in range(finetune_epoch)])
    df_train = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['epoch{}'.format(e+1) for e in range(finetune_epoch)])

    basename = '{}_{}way{}shot_{}_ft{}_bs{}.csv'.format(
        dataset_name, n_way, n_support, finetune_parts, finetune_epoch, batch_size)
    result_path = os.path.join(checkpoint_dir, basename)
    basename_train = '{}_{}way{}shot_{}_ft{}_bs{}_train.csv'.format(
        dataset_name, n_way, n_support, finetune_parts, finetune_epoch, batch_size)
    result_path_train = os.path.join(checkpoint_dir, basename_train)
    print('Saving results to {}'.format(result_path))

    # Determine model weights path
    source_pretrained_model = copy.deepcopy(pretrained_model)
    target_pretrained_model = copy.deepcopy(pretrained_model)

    source_modelfile = get_resume_file(checkpoint_dir.replace('fusion', 'type1_strong'))
    target_modelfile = get_resume_file(checkpoint_dir.replace('fusion', 'type3_strong'), dataset_name)
    print ('Fine-tuning from the two pre-trained model')
    print ('Using model weights path; {} and {}'.format(source_modelfile, target_modelfile))

    ##### Fine-tuning and Evaluation #####
    print ('Fine-tuning start! Fine-tuned part is {}.'.format(finetune_parts))
    for task_num, (x, y) in tqdm(enumerate(novel_loader)):
        task_all = []
        task_all_train = []

        # Load pre-trained state dict
        tmp = torch.load(source_modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state[newkey] = state.pop(key)
        source_pretrained_model.feature.load_state_dict(state, strict=True)

        tmp = torch.load(target_modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state[newkey] = state.pop(key)
        target_pretrained_model.feature.load_state_dict(state, strict=True)

        source_pretrained_model.cuda()
        target_pretrained_model.cuda()

        # Set a new classifier for fine-tuning and optimizers according to the fine-tuning parts and loss function
        if params.method in ['baseline', 'baseline++']:
            if fusion_method == 'concat':
                classifier = Classifier(pretrained_model.feature.final_feat_dim*2, n_way)
                classifier.cuda()
            else:
                classifier = Classifier(pretrained_model.feature.final_feat_dim, n_way)
                classifier.cuda()
            feature_fusion = FeatureFusionModule(fusion_method=fusion_method, feature_dim=pretrained_model.feature.final_feat_dim)
            feature_fusion.cuda()

            if finetune_parts == 'head':
                classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                if fusion_method != 'concat':
                    # Need to lr optimization for feature fusion?
                    feature_fusion_opt = torch.optim.SGD(feature_fusion.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                source_delta_opt = torch.optim.SGD(source_pretrained_model.parameters(), lr = 0)
                target_delta_opt = torch.optim.SGD(target_pretrained_model.parameters(), lr = 0)
            elif finetune_parts == 'full':
                classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                if fusion_method != 'concat':
                    feature_fusion_opt = torch.optim.SGD(feature_fusion.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                source_delta_opt = torch.optim.SGD(source_pretrained_model.parameters(), lr = 1e-2)
                target_delta_opt = torch.optim.SGD(target_pretrained_model.parameters(), lr = 1e-2)
            else:
                raise ValueError('Invalid `finetune_parts` argument: {}'.format(finetune_parts))

        loss_fn = nn.CrossEntropyLoss().cuda()

        # Set both support and query set
        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25 (5-way * 5-n_support), 3, 224, 224)
        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,  *x.size()[2:]) # (75 (5-way * 15-n_qeury), 3, 224, 224)
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)

        support_size = n_way * n_support

        # Start fine-tuning and evaluation in an episodic manner
        for epoch in range(finetune_epoch):
            source_pretrained_model.train()
            target_pretrained_model.train()
            classifier.train()
            feature_fusion.train()
                
            if finetune_parts == 'head':
                source_pretrained_model.eval()
                target_pretrained_model.eval()

            # Fine-tuning
            rand_id = np.random.permutation(support_size)
            for j in range(0, support_size, batch_size):
                # Divide few samples into fewer batches
                selected_id = torch.from_numpy(rand_id[j:min(j+batch_size, support_size)]).cuda()
                x_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id]
                
                source_output = source_pretrained_model.feature(x_batch)
                target_output = target_pretrained_model.feature(x_batch)

                # Make feature fusion module
                output = feature_fusion(source_output, target_output)
                scores = classifier(output)
                loss = loss_fn(scores, y_batch)
                
                classifier_opt.zero_grad()
                source_delta_opt.zero_grad()
                target_delta_opt.zero_grad()
                if fusion_method != 'concat':
                    feature_fusion_opt.zero_grad()
                
                loss.backward()

                classifier_opt.step()
                source_delta_opt.step()
                target_delta_opt.step()
                if fusion_method != 'concat':
                    feature_fusion_opt.step()

            # Evaluation
            if (not params.no_tracking) or (epoch+1 == finetune_epoch):
                with torch.no_grad():
                    source_pretrained_model.eval()
                    target_pretrained_model.eval()
                    classifier.eval()
                    feature_fusion.eval()

                    ### Train
                    y_support = np.repeat(range( n_way ), n_support )

                    source_output = source_pretrained_model.feature(x_a_i.cuda())
                    target_output = target_pretrained_model.feature(x_a_i.cuda())
                    output = feature_fusion(source_output, target_output)
                    scores = classifier(output)
                    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.cpu().numpy()

                    top1_correct = np.sum(topk_ind[:,0] == y_support)
                    correct_this, count_this = float(top1_correct), len(y_support)
                    train_acc = correct_this/count_this*100
                    task_all_train.append(train_acc)

                    ### Test
                    y_query = np.repeat(range( n_way ), n_query )
                    
                    source_output = source_pretrained_model.feature(x_b_i.cuda())
                    target_output = target_pretrained_model.feature(x_b_i.cuda())
                    output = feature_fusion(source_output, target_output)
                    scores = classifier(output)
                    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.cpu().numpy()
                    
                    top1_correct = np.sum(topk_ind[:,0] == y_query)
                    correct_this, count_this = float(top1_correct), len(y_query)
                    test_acc = correct_this/count_this*100
                    task_all.append(test_acc)
                if (epoch+1 == finetune_epoch):
                    print('task: {}, train acc: {}, test acc: {}'.format(task_num, train_acc, test_acc))
            else:
                task_all_train.append(0.0)
                task_all.append(0.0)

        df.loc[task_num+1] = task_all
        df.to_csv(result_path)
        df_train.loc[task_num+1] = task_all_train
        df_train.to_csv(result_path_train)
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

    if params.method == 'baseline':
        pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax')
    elif params.method == 'baseline++':
        pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')
    else:
        raise ValueError('Invalid `method` argument: {}'.format(params.method))

    checkpoint_dir = '%s/checkpoints/%s/%s_%s/fusion' %(configs.save_dir, params.dataset, params.model, params.method)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ##################################################################
    image_size = 224  # for every evaluation dataset except tieredImageNet
    iter_num = 600
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    finetune_parts = params.finetune_parts # 'head', 'body', 'full'
    dataset_names = params.dataset_names
    split = params.startup_split
    fusion_method = params.fusion_method
    
    if fusion_method is None:
        raise ValueError('Invalid `fusion_method` argument: {}'.format(params.fusion_method))

    for dataset_name in dataset_names:
        print (dataset_name)
        print ('Initializing data loader...')
        if dataset_name == "miniImageNet":
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
        
        if dataset_name == "miniImageNet" or dataset_name == "tieredImageNet":
            novel_loader = datamgr.get_data_loader(aug=False, train=False)
        else:
            novel_loader = datamgr.get_data_loader(aug=False)
        print('Data loader initialized successfully!')

        finetune(params, dataset_name, novel_loader, pretrained_model, checkpoint_dir=checkpoint_dir, fusion_method=fusion_method,
                finetune_epoch=100, batch_size=4, finetune_parts=finetune_parts, n_query=15, **few_shot_params)
