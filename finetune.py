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

def finetune(params, dataset_name, novel_loader, pretrained_dataset, pretrained_model, checkpoint_dir, use_simclr_clf,
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
    if pretrained_dataset == 'none':
        modelfile = None
        init_state = copy.deepcopy(pretrained_model.feature.state_dict())
        print ('Fine-tuning from scratch')
    else:
        if params.pretrain_type == 1 or params.pretrain_type == 2:
            modelfile = get_resume_file(checkpoint_dir)
        else:
            modelfile = get_resume_file(checkpoint_dir, dataset_name)
        if not os.path.exists(modelfile):
            raise Exception('Invalid model path: "{}" (no such file found)'.format(modelfile))
        print ('Fine-tuning from the pre-trained model')
        print ('Using model weights path {}'.format(modelfile))

    ##### Fine-tuning and Evaluation #####
    print ('Fine-tuning start! Fine-tuned part is {}.'.format(finetune_parts))
    for task_num, (x, y) in tqdm(enumerate(novel_loader)):
        task_all = []
        task_all_train = []

        # Load pre-trained state dict
        if modelfile is not None:
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
            if use_simclr_clf:
                clf_SIMCLR = projector_SIMCLR(pretrained_model.feature.final_feat_dim, out_dim=128)
                state = tmp['simCLR']
                clf_SIMCLR.load_state_dict(state, strict=True)
                clf_SIMCLR.cuda()
        else:
            pretrained_model.feature.load_state_dict(init_state, strict=True)
        pretrained_model.cuda()

        # Set a new classifier for fine-tuning and optimizers according to the fine-tuning parts and loss function
        if params.method in ['baseline', 'baseline++']:
            if use_simclr_clf:
                classifier = Classifier(128, n_way)
            else:
                classifier = Classifier(pretrained_model.feature.final_feat_dim, n_way)
            classifier.cuda()

            if use_simclr_clf:
                if finetune_parts == 'head':
                    classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                    clf_SIMCLR_opt = torch.optim.SGD(clf_SIMCLR.parameters(), lr = 0)
                    delta_opt = torch.optim.SGD(pretrained_model.parameters(), lr = 0)
                if finetune_parts == 'simclr_head':
                    classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                    clf_SIMCLR_opt = torch.optim.SGD(clf_SIMCLR.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                    delta_opt = torch.optim.SGD(pretrained_model.parameters(), lr = 0)
                if finetune_parts == 'full':
                    classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                    clf_SIMCLR_opt = torch.optim.SGD(clf_SIMCLR.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                    delta_opt = torch.optim.SGD(pretrained_model.parameters(), lr = 1e-2)
            else:
                if finetune_parts == 'head':
                    classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                    delta_opt = torch.optim.SGD(pretrained_model.parameters(), lr = 0)
                elif finetune_parts == 'full':
                    classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
                    delta_opt = torch.optim.SGD(pretrained_model.parameters(), lr = 1e-2)
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
            pretrained_model.train()
            classifier.train()
            if use_simclr_clf:
                clf_SIMCLR.train()
                
            if use_simclr_clf:
                if finetune_parts == 'head':
                    pretrained_model.eval()
                    clf_SIMCLR.eval()
                elif finetune_parts == 'simclr_head':
                    pretrained_model.eval()
            else:
                if finetune_parts == 'head':
                    pretrained_model.eval()

            # Fine-tuning
            rand_id = np.random.permutation(support_size)
            for j in range(0, support_size, batch_size):
                # Divide few samples into fewer batches
                selected_id = torch.from_numpy(rand_id[j:min(j+batch_size, support_size)]).cuda()
                x_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id]
                
                if use_simclr_clf:
                    output = pretrained_model.feature(x_batch)
                    scores = classifier(clf_SIMCLR(output))
                    loss = loss_fn(scores, y_batch)
                    
                    classifier_opt.zero_grad()
                    clf_SIMCLR_opt.zero_grad()
                    delta_opt.zero_grad()

                    loss.backward()

                    classifier_opt.step()
                    clf_SIMCLR_opt.step()
                    delta_opt.step()
                else:
                    output = pretrained_model.feature(x_batch)
                    scores = classifier(output)
                    loss = loss_fn(scores, y_batch)
                    
                    classifier_opt.zero_grad()
                    delta_opt.zero_grad()

                    loss.backward()

                    classifier_opt.step()
                    delta_opt.step()

            # Evaluation
            if (not params.no_tracking) or (epoch+1 == finetune_epoch):
                with torch.no_grad():
                    pretrained_model.eval()
                    classifier.eval()
                    if use_simclr_clf:
                        clf_SIMCLR.eval()

                    ### Train
                    y_support = np.repeat(range( n_way ), n_support )

                    if use_simclr_clf:
                        scores = classifier(clf_SIMCLR(pretrained_model.feature(x_a_i.cuda())))
                    else:
                        scores = classifier(pretrained_model.feature(x_a_i.cuda()))
                    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.cpu().numpy()

                    top1_correct = np.sum(topk_ind[:,0] == y_support)
                    correct_this, count_this = float(top1_correct), len(y_support)
                    train_acc = correct_this/count_this*100
                    task_all_train.append(train_acc)

                    ### Test
                    y_query = np.repeat(range( n_way ), n_query )
                    
                    if use_simclr_clf:
                        scores = classifier(clf_SIMCLR(pretrained_model.feature(x_b_i.cuda())))
                    else:
                        scores = classifier(pretrained_model.feature(x_b_i.cuda()))
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

    finetune_parts = params.finetune_parts # 'head', 'body', 'full'
    dataset_names = params.dataset_names
    split = params.startup_split
    use_simclr_clf = params.use_simclr_clf
    
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

        finetune(params, dataset_name, novel_loader, pretrained_dataset, pretrained_model, checkpoint_dir=checkpoint_dir, use_simclr_clf=use_simclr_clf,
                finetune_epoch=100, batch_size=4, finetune_parts=finetune_parts, n_query=15, **few_shot_params)