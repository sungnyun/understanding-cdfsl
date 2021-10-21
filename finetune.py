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
from itertools import combinations
from tqdm import tqdm

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.maml import MAML
from methods.boil import BOIL
from methods.protonet import ProtoNet

from io_utils import parse_args, get_init_file, get_resume_file, get_best_file, get_assigned_file

from utils import *

from datasets import miniImageNet_few_shot, tieredImageNet_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

STARTUP_METHODS = [
    'startup',
    'startup_both_body',  # teacher body + student body
    'startup_student_body',  # teacher full + student body
]

def tracking_off(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False

def change_momentum(m):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 1.0

def print_momentum(m):
    if isinstance(m, nn.BatchNorm2d):
        print (m.momentum)

def print_BNstats(m):
    if isinstance(m, nn.BatchNorm2d):
        print (m.running_mean, m.running_var)


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
#         self.pre_fc = nn.Linear(dim, dim)
#         self.relu = nn.ReLU()
        self.fc = nn.Linear(dim, n_way)

        with torch.no_grad():
#             self.pre_fc.bias.data.fill_(0.)
            self.fc.bias.data.fill_(0.)

    def forward(self, x):
#         x = self.relu(self.pre_fc(x))
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
                    nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))

    return model


SUPPORTED_LAYERS = ['C0', 'BN0', 'C1', 'BN1', 'C2' , 'BN2', 'shortcut', 'BNshortcut']


def convert_layer_names(model_name, layers) -> Dict:
    """
    Currently only supports last block
    :param model_name:
    :param layers:

    Return example: {
        'trunk.7.BN1': 'bn',
        'trunk.7.C1': 'conv',
    }
    """
    targets = dict()

    for layer in layers:
        if layer not in SUPPORTED_LAYERS:
            raise ValueError('Unsupported layer name {}'.format(layer))

    for layer in layers:
        layer_type = 'bn' if 'BN' in layer else 'conv'
        if model_name == 'ResNet10':
            if layer in ['C0', 'BN0']:
                raise ValueError('Unsupported layer name {} for ResNet10'.format(layer))
            targets['trunk.7.{}'.format(layer)] = layer_type
        elif model_name == 'ResNet12':
            targets['group_3.{}'.format(layer)] = layer_type
        elif model_name == 'ResNet18':
            if layer in ['C0', 'BN0', 'shortcut', 'BNshortcut']:
                raise ValueError('Unsupported layer name {} for ResNet10'.format(layer))
            mapping = {
                'C1': 'conv1',
                'BN1': 'bn1',
                'C2': 'conv2',
                'BN2': 'bn2',
            }
            targets['layer4.1.{}'.format(mapping[layer])] = layer_type

    return targets


def partial_reinit(model, model_name, layers, lottery_checkpoint_dir=None, dataset_name=None):
    """
    Partially re
    :param model:
    :param model_name:
    :param layers:
    :param lottery_checkpoint_dir:
    :return:
    """
    lottery_state = None
    if lottery_checkpoint_dir:
        assert(dataset_name is not None)
        if params.simclr_finetune:
            lottery_model_path = os.path.join(checkpoint_dir, 'unlabeled', '{}_initial.tar'.format(dataset_name.split('_')[0]))
        else:
            lottery_model_path = get_init_file(checkpoint_dir)
        if not lottery_model_path or not os.path.exists(lottery_model_path):
            raise Exception('Invalid model path: "{}" (no such file found)'.format(lottery_model_path))
        lottery_state = torch.load(lottery_model_path)['state']  # TODO: optimize repeated loads

    targets = convert_layer_names(model_name, layers) # Dict[target_name, layer_type]

    consumed = set()
    with torch.no_grad():
        for name, p in model.named_parameters():
            for target, layer_type in targets.items():
                if target in name:
                    if lottery_state:
                        p.data = lottery_state[name]
                    else:
                        if layer_type == 'bn':
                            if 'weight' in name:
                                p.data.fill_(1.)
                            else:
                                p.data.fill_(0.)
                        else:
                            nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
                    consumed.add(target)

    remaining = set(targets.keys()) - consumed
    if remaining:
        raise AssertionError('Missing layers during partial_reinit: {}'.format(remaining))


def mv_init(model):
    """
    Re-randomize all layers with existing mean-var
    :param model:
    :return:
    """
    for name, p in model.named_parameters():
        if 'classifier' not in name:
            mean, var = p.data.mean(), p.data.var()
            nn.init.normal_(p.data, mean, var)

    return model


def reinit_stem(model):
    """
    :param model:
    :return:
    """
    
    for name, p in model.named_parameters():
        if 'trunk.0' in name:
            nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
        elif 'trunk.1' in name:
            if 'weight' in name:
                p.data.fill_(1.)
            else:
                p.data.fill_(0.)
        else:
            pass

    return model


def finetune(dataset_name, novel_loader, pretrained_model, checkpoint_dir, simclr_epoch=None, freeze_backbone=False, n_query=15, n_way=5, n_support=5):
    iter_num = len(novel_loader)
    finetune_epoch = 100
    batch_size = 4

    if params.method in ['baseline', 'baseline++', 'baseline_body'] + STARTUP_METHODS:
        df = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['epoch{}'.format(e+1) for e in range(finetune_epoch)])
        df_nil = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['epoch{}'.format(e+1) for e in range(finetune_epoch)])
    elif params.method in ['maml', 'boil']:
        df = pd.DataFrame(None, index=list(range(1, iter_num+1)), columns=['Accuracy'])
        acc_all = []

    reinit_arguments = [
        bool(params.reinit_blocks), bool(params.partial_reinit), bool(params.lottery_reinit)
    ]
    if sum(reinit_arguments) > 1:
        raise Exception('Cannot apply multiple reinit arguments at the same time')

    if params.mv_init:
        print('Mean-var re-init (full network)')
    if params.reinit_stem:
        print('Re-initializing stem')
    if params.reinit_blocks:
        print('Re-initializing blocks {} (one-index)'.format(params.reinit_blocks))
    if params.partial_reinit:
        print('Re-randomizing specific layers from last block (partial reinit)')
    if params.lottery_reinit:
        print('Re-initializing specific layers from last block (lottery reinit)')

    # [CVPR2022] Build result_path
    suffixes = ['']
    if freeze_backbone:
        suffixes.append('freeze')
    if params.mv_init:
        suffixes.append('mvinit')
    if params.reinit_stem:
        suffixes.append('reinitstem')
    if params.reinit_blocks:
        blocks_string = ''.join([str(i) for i in params.reinit_blocks])
        suffixes.append('reinitblock{}'.format(blocks_string))  # changed from LBreinit
    if params.partial_reinit:
        suffixes.append('pr')
    if params.lottery_reinit:
        suffixes.append('lottery')
    if params.full_supp_stats:
        suffixes.append('fullStats')
        
    suffix = '_'.join(suffixes)
    
    if params.simclr_finetune:
        basename = '{}_{}way{}shot_ft{}_bs{}{}_se{}.csv'.format(
            dataset_name, n_way, n_support, finetune_epoch, batch_size, suffix, simclr_epoch)
        result_path = os.path.join(checkpoint_dir, 'unlabeled', basename)
    else:
        basename = '{}_{}way{}shot_ft{}_bs{}{}.csv'.format(
            dataset_name, n_way, n_support, finetune_epoch, batch_size, suffix)
        result_path = os.path.join(checkpoint_dir, basename)
    print('Saving results to {}'.format(result_path))

    # Determine model weights path
    params.save_iter = -1
    if params.simclr_finetune:
        if simclr_epoch == 0:
            modelfile = os.path.join(checkpoint_dir, 'unlabeled', '{}_initial.tar'.format(dataset_name.split('_')[0]))
        elif simclr_epoch == 1000:
            modelfile = os.path.join(checkpoint_dir, 'unlabeled', '{}_999.tar'.format(dataset_name.split('_')[0]))
        else:
            modelfile = os.path.join(checkpoint_dir, 'unlabeled', '{}_{}.tar'.format(dataset_name.split('_')[0], simclr_epoch))
    else:
        if params.method in STARTUP_METHODS:  # startup methods apply pre-training separately to each target dataset
            if '_split' in dataset_name:  # hotfix -- startup always uses split
                assert ('_split' == dataset_name[-6:])
                dataset_name = dataset_name[:-6]
            model_dir = '{}_unlabeled_20'.format(dataset_name)
            modelfile = os.path.join(checkpoint_dir, model_dir, 'checkpoint_best.pkl')
        else:
            if params.save_iter != -1:
                modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
            elif params.method in ['baseline', 'baseline++', 'baseline_body']:
                modelfile = get_resume_file(checkpoint_dir)
            else:
                modelfile = get_best_file(checkpoint_dir)

    if params.model != 'ResNet18' and not (modelfile and os.path.exists(modelfile)):
        raise Exception('Invalid model path: "{}" (no such file found)'.format(modelfile))
    print('Using model weights path {}'.format(modelfile))

    for task_num, (x, y) in tqdm(enumerate(novel_loader)):
        ###############################################################################################
        if params.method in ['baseline', 'baseline++', 'baseline_body'] + STARTUP_METHODS:
            task_all = []
            task_all_nil = []

        if params.model == 'ResNet18':
            try:
                tmp = torch.load(modelfile)
                state = tmp['state']  # state dict
            except:
                state = None
        elif params.method in STARTUP_METHODS:
            tmp = torch.load(modelfile)  # note, tmp is only used to load the model weights
            state = tmp['model']  # state dict of *backbone* (from STARTUP student .pkl file)
        else:
            tmp = torch.load(modelfile)
            state = tmp['state']  # state dict

        # state_keys = list(state.keys())
        # for _, key in enumerate(state_keys):
        #     if "feature." in key:
        #         newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
        #         state[newkey] = state.pop(key)
        #     else:
        #         state[newkey] = state.pop(key)

        if params.model == 'ResNet18':
            if state:
                pretrained_model.load_state_dict(state, strict=True)
            else:
                pretrained_model.load_imagenet_weights()
        elif params.method in STARTUP_METHODS:  # extractor state_dict is saved separately in startup code
            pretrained_model.feature.load_state_dict(state, strict=True)
        else:
            pretrained_model.load_state_dict(state, strict=True)
        pretrained_model.cuda()
        pretrained_model.train()
        
        ###############################################################################################

        if params.method in ['baseline', 'baseline++', 'baseline_body'] + STARTUP_METHODS:
            classifier = Classifier(pretrained_model.feature.final_feat_dim, n_way)
            classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 1e-2, momentum=0.9, dampening=0.9, weight_decay=0.001)
            classifier.cuda()
            classifier.train()
            
            if freeze_backbone is False:
                # [CVPR2022] Re-initialization
                if params.mv_init:
                    mv_init(pretrained_model)
                if params.reinit_stem:
                    reinit_stem(pretrained_model)
                if params.reinit_bn_stats:
                    reinit_running_batch_statistics(pretrained_model, var_init=0.1)
                if params.reinit_blocks:
                    reinit_blocks(pretrained_model, block_indices=params.reinit_blocks)
                if params.partial_reinit:
                    partial_reinit(pretrained_model, params.model, params.partial_reinit)
                if params.lottery_reinit:
                    partial_reinit(pretrained_model, params.model, params.lottery_reinit, lottery_checkpoint_dir=checkpoint_dir, dataset_name=dataset_name)
                # TODO [Add] Lottery ticket (?)
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

        if params.method in ['baseline', 'baseline++', 'baseline_body'] + STARTUP_METHODS:
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
                    # output = pretrained_model.feature.forward_bodyfreeze(z_batch)
                    scores = classifier(output)
                    loss = loss_fn(scores, y_batch)
                    #####################################
                    loss.backward()
                    classifier_opt.step()
                    if freeze_backbone is False:
                        delta_opt.step()

                if (not params.no_tracking) or (epoch+1 == finetune_epoch):
                    with torch.no_grad():
                        if params.full_supp_stats:
                            # print ("before")
                            # pretrained_model.apply(print_momentum)
                            # pretrained_model.apply(print_BNstats)

                            pretrained_model.apply(change_momentum)
                            # print ("after")
                            # pretrained_model.apply(print_momentum)
                            _ = pretrained_model.feature(x_a_i)
                            # pretrained_model.apply(print_BNstats)
                        
                        # Evaluation
                        pretrained_model.eval()
                        classifier.eval()
                        y_query = np.repeat(range( n_way ), n_query )

                        scores = classifier(pretrained_model.feature(x_b_i.cuda()))
                        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                        topk_ind = topk_labels.cpu().numpy()

                        top1_correct = np.sum(topk_ind[:,0] == y_query)
                        correct_this, count_this = float(top1_correct), len(y_query)
                        task_all.append((correct_this/count_this*100))

                else:
                    task_all.append(0.0)

            df.loc[task_num+1] = task_all
            df.to_csv(result_path)

        elif params.method in ['maml', 'boil']:                
            scores = pretrained_model.set_forward(x)

            y_query = np.repeat(range( n_way ), n_query )
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()

            top1_correct = np.sum(topk_ind[:,0] == y_query)
            correct_this, count_this = float(top1_correct), len(y_query)
            acc_all.append((correct_this/ count_this *100))

        ###############################################################################################


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    if params.model == 'ResNet10':
        model_dict = {params.model: backbone.ResNet10(method=params.method, track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    elif params.model == 'ResNet12':
        model_dict = {params.model: backbone.ResNet12(track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    elif params.model == 'ResNet18':
        if params.reinit_bn_stats:
            raise AssertionError('Not supported')
        model_dict = {params.model: backbone.ResNet18(track_bn=params.track_bn)}
    else:
        raise ValueError('Unknown extractor')

    ##################################################################
    if params.dataset == 'miniImageNet':
        image_size = 224
    elif params.dataset == 'tieredImageNet':
        image_size = 84
    iter_num = 600

    # params.n_shot = 5
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    if params.method in ['baseline', 'baseline_body'] + STARTUP_METHODS:
        if params.dataset == 'miniImageNet':
            params.num_classes = 64
        elif params.dataset == 'tieredImageNet':
            params.num_classes = 351
        pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax')
    elif params.method == 'baseline++':
        pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')
    elif params.method == 'maml':
        pretrained_model = MAML(model_dict[params.model], **few_shot_params)
    elif params.method == 'boil':
        pretrained_model = BOIL(model_dict[params.model], **few_shot_params)
    elif params.method == 'protonet':
        pretrained_model = ProtoNet(model_dict[params.model], **few_shot_params)
    else:
        raise ValueError('Invalid `method` argument: {}'.format(params.method))

    pretrained_dataset = params.dataset
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if params.track_bn:
        checkpoint_dir += '_track'
    if not params.method in ['baseline', 'baseline++', 'baseline_body'] + STARTUP_METHODS:
        checkpoint_dir += '_%dway_%dshot'%(params.train_n_way, params.n_shot)
        
    freeze_backbone = params.freeze_backbone
    #########################################################################
    
    dataset_names = params.dataset_names
    
    if params.simclr_finetune:
        split = True
    else:
        split = params.startup_split
    
    for dataset_name in dataset_names:
        print (dataset_name)
        print('Initializing data loader...')
        if dataset_name == "miniImageNet":
            datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_episode=iter_num, n_query=15, split=split, **few_shot_params)
        if dataset_name == "tieredImageNet":
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
            
        if split:
            dataset_name += '_split'
        
        print('Data loader initialized successfully!')

        if params.simclr_finetune:
            for simclr_epoch in [1000, 800, 600, 400, 200, 0]:
                finetune(dataset_name, novel_loader, pretrained_model, checkpoint_dir=checkpoint_dir, simclr_epoch=simclr_epoch, freeze_backbone=freeze_backbone, n_query=15, **few_shot_params)
        else:
            finetune(dataset_name, novel_loader, pretrained_model, checkpoint_dir=checkpoint_dir, freeze_backbone=freeze_backbone, n_query=15, **few_shot_params)
