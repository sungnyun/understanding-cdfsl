from typing import List

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
from itertools import combinations
from tqdm import tqdm

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.maml import MAML
from methods.boil import BOIL
from methods.protonet import ProtoNet

from io_utils import parse_args, get_resume_file, get_best_file, get_assigned_file

from utils import *

from datasets import miniImageNet_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

# [CVPR2022] Re-initialization (hard-coded arguments, overrides CLI arguments)
MANUAL_REINIT = False
MANUAL_REINIT_BLOCKS = [1]
MANUAL_REINIT_BN_STATS = False


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(dim, n_way)

        with torch.no_grad():
            self.fc.bias.data.fill_(0.)

    def forward(self, x):
        # x = self.relu(x)
        x = self.fc(x)
        return x


def reinit_blocks(model, block_indices: List[int]):
    """
    block_indices should be subset of { 1, 2, 3, 4 }

    :param model:
    :param block_indices:
    :return:
    """
    trunk_indices = [i + 3 for i in block_indices]

    for name, p in model.named_parameters():
        for i in trunk_indices:
            if 'trunk.{}'.format(i) in name:
                if 'BN' in name:
                    if 'weight' in name:
                        p.data.fill_(1.)
                    else:
                        p.data.fill_(0.)
                else:
                    # p.data[48:,:,:,:].fill_(0.)
                    # half = p.data.shape[0] // 2
                    nn.init.kaiming_uniform_(p.data, a=math.sqrt(5)) # p.data[half:,:,:,:]

    return model


def reinit_running_batch_statistics(model, var_init=0.1):
    model.feature.trunk[1].running_mean.data.fill_(0.)
    model.feature.trunk[1].running_var.data.fill_(var_init)

    model.feature.trunk[4].BN1.running_mean.data.fill_(0.)
    model.feature.trunk[4].BN1.running_var.data.fill_(var_init)
    model.feature.trunk[4].BN2.running_mean.data.fill_(0.)
    model.feature.trunk[4].BN2.running_var.data.fill_(var_init)

    model.feature.trunk[5].BN1.running_mean.data.fill_(0.)
    model.feature.trunk[5].BN1.running_var.data.fill_(var_init)
    model.feature.trunk[5].BN2.running_mean.data.fill_(0.)
    model.feature.trunk[5].BN2.running_var.data.fill_(var_init)
    model.feature.trunk[5].BNshortcut.running_mean.data.fill_(0.)
    model.feature.trunk[5].BNshortcut.running_var.data.fill_(var_init)

    model.feature.trunk[6].BN1.running_mean.data.fill_(0.)
    model.feature.trunk[6].BN1.running_var.data.fill_(var_init)
    model.feature.trunk[6].BN2.running_mean.data.fill_(0.)
    model.feature.trunk[6].BN2.running_var.data.fill_(var_init)
    model.feature.trunk[6].BNshortcut.running_mean.data.fill_(0.)
    model.feature.trunk[6].BNshortcut.running_var.data.fill_(var_init)

    model.feature.trunk[7].BN1.running_mean.data.fill_(0.)
    model.feature.trunk[7].BN1.running_var.data.fill_(var_init)
    model.feature.trunk[7].BN2.running_mean.data.fill_(0.)
    model.feature.trunk[7].BN2.running_var.data.fill_(var_init)
    model.feature.trunk[7].BNshortcut.running_mean.data.fill_(0.)
    model.feature.trunk[7].BNshortcut.running_var.data.fill_(var_init)

    return model

def finetune(dataset_name, novel_loader, pretrained_model, checkpoint_dir, freeze_backbone=False, n_query=15, n_way=5, n_support=5):
    iter_num = len(novel_loader)
    finetune_epoch = 100
    batch_size = 4

    if params.method in ['baseline', 'baseline++', 'baseline_body']:
        df = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['epoch{}'.format(e+1) for e in range(finetune_epoch)])
        df_nil = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['epoch{}'.format(e+1) for e in range(finetune_epoch)])
    elif params.method in ['maml', 'boil']:
        df = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['Accuracy'])
        acc_all = []

    # [CVPR2022] Determine re-init parameters
    if MANUAL_REINIT:
        print('Using manual hard-coded re-init arguments')
        params.reinit_blocks = MANUAL_REINIT_BLOCKS
        params.reinit_bn_stats = MANUAL_REINIT_BN_STATS
    if params.reinit_blocks:
        print('Re-initializing blocks {} (one-index)'.format(params.reinit_blocks))
    if params.reinit_bn_stats:
        print('Re-initializing all running batch statistics')

    # [CVPR2022] Build result_path
    suffixes = ['']
    if freeze_backbone:
        suffixes.append('freeze')
    if params.reinit_blocks:
        blocks_string = ''.join([str(i) for i in params.reinit_blocks])
        suffixes.append('reinitblock{}'.format(blocks_string))  # changed from LBreinit
    if params.reinit_bn_stats:
        suffixes.append('reinitbn'.format(params.reinit_bn_stats))
    # suffixes.append('nil'.format(params.reinit_bn_stats))
    suffix = '_'.join(suffixes)
    basename = '{}_{}way{}shot_ft{}_bs{}{}.csv'.format(
        dataset_name, n_way, n_support, finetune_epoch, batch_size, suffix)
    result_path = os.path.join(checkpoint_dir, basename)
    print('Saving results to {}'.format(result_path))

    for task_num, (x, y) in tqdm(enumerate(novel_loader)):
        ###############################################################################################
        if params.method in ['baseline', 'baseline++', 'baseline_body']:
            task_all = []
            task_all_nil = []

        # load pretrained model on miniImageNet
        params.save_iter = -1
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir, params.save_iter)
        elif params.method in ['baseline', 'baseline++', 'baseline_body'] :
            modelfile   = get_resume_file(checkpoint_dir)
        else:
            modelfile   = get_best_file(checkpoint_dir)

        if not modelfile or not os.path.exists(modelfile):
            raise Exception('Invalid model path: "{}" (no such file found)'.format(modelfile))
        tmp = torch.load(modelfile)
        state = tmp['state']

        # state_keys = list(state.keys())
        # for _, key in enumerate(state_keys):
        #     if "feature." in key:
        #         newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
        #         state[newkey] = state.pop(key)
        #     else:
        #         state[newkey] = state.pop(key)

        pretrained_model.load_state_dict(state, strict=True)
        pretrained_model.cuda()
        pretrained_model.train()

        ###############################################################################################

        if params.method in ['baseline', 'baseline++', 'baseline_body']:
            classifier = Classifier(pretrained_model.feature.final_feat_dim, n_way)
            classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
            classifier.cuda()
            classifier.train()

            # with torch.no_grad():
            #     classifier.fc.weight.data = torch.stack([torch.mean(pretrained_model.classifier.weight.data[:64], dim=0)]*n_way)

            if freeze_backbone is False:
                # [CVPR2022] Re-initialization
                if params.reinit_bn_stats:
                    reinit_running_batch_statistics(pretrained_model, var_init=0.1)
                if params.reinit_blocks:
                    reinit_blocks(pretrained_model, block_indices=params.reinit_blocks)

                delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001) # 기본코드에는 이거 그냥 1e-2만 있음

        loss_fn = nn.CrossEntropyLoss().cuda()

        ###############################################################################################

        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)

        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25 (5-way * 5-n_support), 3, 224, 224)
        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,  *x.size()[2:]) # (75 (5-way * 15-n_qeury), 3, 224, 224)
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)

        ###############################################################################################

        if params.method in ['baseline', 'baseline++', 'baseline_body']:
            support_size = n_way * n_support

            for epoch in range(finetune_epoch):
                pretrained_model.train()
                classifier.train()

                if freeze_backbone:
                    pretrained_model.eval()

                rand_id = np.random.permutation(support_size)
                for j in range(0, support_size, batch_size):
                    classifier_opt.zero_grad()
                    if freeze_backbone is False:
                        delta_opt.zero_grad()
                    #####################################
                    selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
                    z_batch = x_a_i[selected_id]
                    y_batch = y_a_i[selected_id]
                    #####################################
                    output = pretrained_model.feature(z_batch)
                    scores = classifier(output)
                    loss = loss_fn(scores, y_batch)
                    #####################################
                    loss.backward()
                    classifier_opt.step()
                    if freeze_backbone is False:
                        delta_opt.step()

                with torch.no_grad():
                    pretrained_model.eval()
                    classifier.eval()
                    y_query = np.repeat(range( n_way ), n_query )

                    scores = classifier(pretrained_model.feature(x_b_i.cuda()))
                    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.cpu().numpy()

                    top1_correct = np.sum(topk_ind[:,0] == y_query)
                    correct_this, count_this = float(top1_correct), len(y_query)
                    # print (correct_this/ count_this *100)
                    task_all.append((correct_this/count_this*100))

#                     nil_cls = torch.zeros([5, 512])
#                     for i in range(5):
#                         cls_idx = y_a_i==i
#                         nil_cls[i] = torch.mean(pretrained_model.feature(x_a_i)[cls_idx].cpu(), dim=0)
#                     scores = torch.mm(pretrained_model.feature(x_b_i).cpu(), nil_cls.T)
#                     topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
#                     topk_ind = topk_labels.cpu().numpy()

#                     top1_correct = np.sum(topk_ind[:,0] == y_query)
#                     correct_this, count_this = float(top1_correct), len(y_query)
#                     # print (correct_this/ count_this *100)
#                     task_all_nil.append((correct_this/count_this*100))

            df.loc[task_num+1] = task_all
            df.to_csv(result_path)
            # df_nil.loc[task_num+1] = task_all_nil
            # df_nil.to_csv(result_nil_path)

        elif params.method in ['maml', 'boil']:                
            scores = pretrained_model.set_forward(x)

            y_query = np.repeat(range( n_way ), n_query )
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()

            top1_correct = np.sum(topk_ind[:,0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            # print (correct_this/ count_this *100)
            acc_all.append((correct_this/ count_this *100))

        ###############################################################################################

#     acc_all  = np.asarray(acc_all)
#     acc_mean = np.mean(acc_all)
#     acc_std  = np.std(acc_all)

#     if freeze_backbone:
#         result_path = os.path.join(checkpoint_dir, dataset_name + '_{}way{}shot_ft{}_bs{}_freeze.csv'.format(n_way, n_support, finetune_epoch, batch_size))
#     else:
#         result_path = os.path.join(checkpoint_dir, dataset_name + '_{}way{}shot_ft{}_bs{}.csv'.format(n_way, n_support, finetune_epoch, batch_size))
#     df['Accuracy'] = list(acc_all)
#     df.to_csv(result_path)
#     print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    if params.model == 'ResNet10':
        model_dict = {params.model: backbone.ResNet10(method=params.method)}
    else:
        raise ValueError('Unknown extractor')

    ##################################################################
    image_size = 224
    iter_num = 600

    params.n_shot = 5
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    if params.method == 'baseline' or params.method == 'baseline_body':
        # params.num_classes = 64
        pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax')
    elif params.method == 'baseline++':
        pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')
    elif params.method == 'maml':
        pretrained_model = MAML(model_dict[params.model], **few_shot_params)
    elif params.method == 'boil':
        pretrained_model = BOIL(model_dict[params.model], **few_shot_params)
    elif params.method == 'protonet':
        pretrained_model = ProtoNet(model_dict[params.model], **few_shot_params)

    pretrained_dataset = params.dataset
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++', 'baseline_body']: 
        checkpoint_dir += '_%dway_%dshot'%(params.train_n_way, params.n_shot)

    freeze_backbone = params.freeze_backbone
    #########################################################################
    dataset_names = ["miniImageNet", "CropDisease", "EuroSAT", "ISIC", "ChestX"]
    for dataset_name in dataset_names:
        print (dataset_name)
        if dataset_name == "miniImageNet":
            datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_episode=iter_num, n_query=15, **few_shot_params)
        elif dataset_name == "CropDisease":
            datamgr = CropDisease_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
        elif dataset_name == "EuroSAT":
            datamgr = EuroSAT_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
        elif dataset_name == "ISIC":
            datamgr = ISIC_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
        elif dataset_name == "ChestX":
            datamgr = Chest_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)

        if dataset_name == "miniImageNet":
            novel_loader = datamgr.get_data_loader(aug=False, train=False)
        else:
            novel_loader = datamgr.get_data_loader(aug=False)

        # replace finetine() with your own method
        finetune(dataset_name, novel_loader, pretrained_model, checkpoint_dir=checkpoint_dir, freeze_backbone=freeze_backbone, n_query=15, **few_shot_params)