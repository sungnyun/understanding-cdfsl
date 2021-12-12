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
from methods.maml import MAML
from methods.boil import BOIL
from methods.protonet import ProtoNet

from io_utils import parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, tieredImageNet_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

def partial_reinit(model, model_name, pretrained_dataset):
    """
    Re-initialize {Conv2, BN2, ShortCutConv, ShortCutBN} from last block

    :param model:
    :return:
    """
    if model_name == 'ResNet10':
        targets = {  # ResNet10 - block 4
            'trunk.7.C2',
            'trunk.7.BN2',
            'trunk.7.shortcut',
            'trunk.7.BNshortcut',
        }
    elif model_name == 'ResNet12':
        targets = {  # ResNet12 - block 4
            'group_3.C2',
            'group_3.BN2',
            'group_3.shortcut',
            'group_3.BNshortcut',
        }
    elif model_name == 'ResNet18':
        if pretrained_dataset == 'tieredImageNet':
            targets = {
                'layer4.1.conv3',
                'layer4.1.bn3',
            }
        elif pretrained_dataset == 'ImageNet':
            targets = {
                'layer4.1.conv2',
                'layer4.1.bn2',
            }
        
    consumed = set()
    for name, p in model.named_parameters():
        for target in targets:
            if target in name:
                if 'BN' in name or 'bn' in name:
                    if 'weight' in name:
                        p.data.fill_(1.)
                    else:
                        p.data.fill_(0.)
                else:
                    nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
                consumed.add(target)

    remaining = targets - consumed
    if remaining:
        raise AssertionError('Missing layers during partial_reinit: {}'.format(remaining))

    return model

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
    
class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 *
                    self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 *
                    self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(
            representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

def train_unlabeled(dataset_name, loader, model, clf_SIMCLR, criterion_SIMCLR,
                    checkpoint_dir, start_epoch, stop_epoch, params, base_loader, clf, unlabeled_base_loader):
    
    model.train()
    clf_SIMCLR.train()

    model.cuda()
    clf_SIMCLR.cuda()
    criterion_SIMCLR.cuda()

    opt_params = [
            {'params': model.parameters()},
            {'params': clf_SIMCLR.parameters()}
        ]
    
    if base_loader is not None and clf is not None:
        clf.train()
        clf.cuda()

        base_loader_iter = iter(base_loader)
        nll_criterion = nn.NLLLoss(reduction='mean').cuda()
        opt_params.append({'params': clf.parameters()})
        
    if unlabeled_base_loader is not None:
        unlabeled_base_loader_iter = iter(unlabeled_base_loader)
        
    optimizer = torch.optim.SGD(opt_params,
            lr=0.1, momentum=0.9,
            weight_decay=1e-4,
            nesterov=False)
    

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[400,600,800],
                                                     gamma=0.1)

    print ("Learning setup is set!")

    for epoch in range(start_epoch, stop_epoch):
        epoch_loss = 0
        if epoch == 0:
            outfile = os.path.join(checkpoint_dir, '{}_initial.tar'.format(dataset_name))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        for i, (X, y) in enumerate(loader):
            f1 = model.feature(X[0].cuda())
            f2 = model.feature(X[1].cuda())
            loss = criterion_SIMCLR(clf_SIMCLR(f1), clf_SIMCLR(f2))

            if base_loader is not None:
                try:
                    X_base, y_base = base_loader_iter.next()
                except StopIteration:
                    base_loader_iter = iter(base_loader)
                    X_base, y_base = base_loader_iter.next()

                features_base = model.feature(X_base.cuda())
                logits_base = clf(features_base)
                log_probability_base = F.log_softmax(logits_base, dim=1)
                loss += nll_criterion(log_probability_base, y_base.cuda())
                
            if unlabeled_base_loader is not None:
                try:
                    X_base, y_base = unlabeled_base_loader_iter.next()
                except StopIteration:
                    unlabeled_base_loader_iter = iter(unlabeled_base_loader)
                    X_base, y_base = unlabeled_base_loader_iter.next()

                f1_base = model.feature(X_base[0].cuda())
                f2_base = model.feature(X_base[1].cuda())
                loss += criterion_SIMCLR(clf_SIMCLR(f1_base), clf_SIMCLR(f2_base))
                print ("hihihi")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()
        print ('epoch: {}, loss: {}'.format(epoch, epoch_loss/len(loader)))

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(checkpoint_dir, '{}_{:d}.tar'.format(dataset_name, epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    batch_size = 32
    temperature = 1

    if params.dataset == 'miniImageNet':
        model_dict = {params.model: backbone.ResNet10(method=params.method, track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    # elif params.model == 'ResNet12':
    #     model_dict = {params.model: backbone.ResNet12(track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    elif params.dataset == 'tieredImageNet':
        if params.reinit_bn_stats:
            raise NotImplementedError('Not supported')
        model_dict = {params.model: backbone.ResNet18_84x84(track_bn=params.track_bn)}
    elif params.dataset  == 'ImageNet':
        if params.reinit_bn_stats:
            raise NotImplementedError('Not supported')
        model_dict = {params.model: backbone.ResNet18(track_bn=params.track_bn)}
    else:
        raise ValueError('Unknown extractor')

    pretrained_dataset = params.dataset
    
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, pretrained_dataset, params.model, params.method)
    if params.train_aug:
        checkpoint_dir += '_aug'
    if params.track_bn:
        checkpoint_dir += '_track'
    if not params.method in ['baseline', 'baseline++', 'baseline_body']:
        checkpoint_dir += '_%dway_%dshot'%(params.train_n_way, params.n_shot)
    
    if pretrained_dataset == 'miniImageNet':
        params.num_classes = 64
    elif pretrained_dataset == 'tieredImageNet':
        params.num_classes = 351
    elif pretrained_dataset == 'ImageNet':
        params.num_classes = 1000
    pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax')

    if not params.no_base_pretraining:
        if pretrained_dataset in ['miniImageNet', 'tieredImageNet']:
            params.save_iter = -1
            if params.save_iter != -1:
                modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
            elif params.method in ['baseline', 'baseline++', 'baseline_body']:
                modelfile = get_resume_file(checkpoint_dir)
            else:
                modelfile = get_best_file(checkpoint_dir)
            if not modelfile or not os.path.exists(modelfile):
                raise Exception('Invalid model path: "{}" (no such file found)'.format(modelfile))
            print('Using model weights path {}'.format(modelfile))
            state = torch.load(modelfile)['state']  # state dict
            pretrained_model.load_state_dict(state, strict=True)
        elif pretrained_dataset == 'ImageNet':
            pretrained_model.feature.load_imagenet_weights()
        
    # Re-randomization
    if not params.no_rerand:
        partial_reinit(pretrained_model, params.model, pretrained_dataset)
        
    # Make checkpoint_dir
    checkpoint_dir += '/unlabeled'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    dataset_names = params.dataset_names
    
    image_size = 224 # for all unlabeled target dataset except tieredImageNet
    for dataset_name in dataset_names:
        print (dataset_name)
        print('Initializing data loader...')
        # If you use base classes, prepare supervised learning based on source domain
        if params.use_base_classes:
            print ('Using base classes!')
            if pretrained_dataset == 'miniImageNet':
                datamgr = miniImageNet_few_shot.SimpleDataManager(image_size=224, batch_size=batch_size)
                base_loader = datamgr.get_data_loader(aug=params.train_aug)
                params.num_classes = 64
                clf = nn.Linear(pretrained_model.feature.final_feat_dim, params.num_classes)
            elif pretrained_dataset == 'tieredImageNet':
                datamgr = tieredImageNet_few_shot.SimpleDataManager(image_size=84, batch_size=batch_size)
                base_loader = datamgr.get_data_loader(aug=params.train_aug)
                params.num_classes = 351
                clf = nn.Linear(pretrained_model.feature.final_feat_dim, params.num_classes)
            else:
                base_loader = None
                unlabeled_base_loader = None
                clf = None
        elif params.use_base_classes_as_unlabeled:
            print ('Using base classes as unlabeled data!')
            if pretrained_dataset == 'miniImageNet':
                transform = miniImageNet_few_shot.TransformLoader(
                    image_size=224).get_composed_transform(aug=True, aug_mode=params.aug_mode)
                dataset = miniImageNet_few_shot.SimpleDataset(
                    apply_twice(transform), train=True)
                unlabeled_base_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                                    num_workers=2, shuffle=True, drop_last=True)
            elif pretrained_dataset == 'tieredImageNet':
                transform = tieredImageNet_few_shot.TransformLoader(
                    image_size=84).get_composed_transform(aug=True, aug_mode=params.aug_mode)
                dataset = tieredImageNet_few_shot.SimpleDataset(
                    apply_twice(transform), train=True)
                unlabeled_base_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                                    num_workers=2, shuffle=True, drop_last=True)
            base_loader = None
            clf = None
        else:
            base_loader = None
            unlabeled_base_loader = None
            clf = None

        if dataset_name == "miniImageNet":
            transform = miniImageNet_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True, aug_mode=params.aug_mode)
            dataset = miniImageNet_few_shot.SimpleDataset(
                apply_twice(transform), train=False, split=True)
        elif dataset_name == "tieredImageNet":
            image_size = 84
            transform = tieredImageNet_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True, aug_mode=params.aug_mode)
            dataset = tieredImageNet_few_shot.SimpleDataset(
                apply_twice(transform), train=False, split=True)
        elif dataset_name == "CropDisease":
            transform = CropDisease_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True, aug_mode=params.aug_mode)
            dataset = CropDisease_few_shot.SimpleDataset(
                apply_twice(transform), split=True)
        elif dataset_name == "EuroSAT":
            transform = EuroSAT_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True, aug_mode=params.aug_mode)
            dataset = EuroSAT_few_shot.SimpleDataset(
                apply_twice(transform), split=True)
        elif dataset_name == "ISIC":
            transform = ISIC_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True, aug_mode=params.aug_mode)
            dataset = ISIC_few_shot.SimpleDataset(
                apply_twice(transform), split=True)
        elif dataset_name == "ChestX":
            transform = Chest_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True, aug_mode=params.aug_mode)
            dataset = Chest_few_shot.SimpleDataset(
                apply_twice(transform), split=True)

        # Prepare SimCLR
        clf_SIMCLR = projector_SIMCLR(pretrained_model.feature.final_feat_dim, out_dim=128) # Projection dimension is fixed to 128
        criterion_SIMCLR = NTXentLoss('cuda', batch_size=batch_size, temperature=temperature, use_cosine_similarity=True)

        novel_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   num_workers=2, shuffle=True, drop_last=True)

        print('Data loader initialized successfully!, length: {}'.format(len(dataset)))

        train_unlabeled(dataset_name, novel_loader, pretrained_model, clf_SIMCLR, criterion_SIMCLR,
                        checkpoint_dir, start_epoch, stop_epoch, params, base_loader, clf, unlabeled_base_loader)
