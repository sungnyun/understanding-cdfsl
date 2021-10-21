import math
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
from datasets import miniImageNet_few_shot, tieredImageNet_few_shot, ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, DTD_few_shot

def partial_reinit(model, model_name):
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
        pass
        
    consumed = set()
    for name, p in model.named_parameters():
        for target in targets:
            if target in name:
                if 'BN' in name:
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
                    checkpoint_dir, start_epoch, stop_epoch, params):

    model.train()
    clf_SIMCLR.train()

    model.cuda()
    clf_SIMCLR.cuda()
    criterion_SIMCLR.cuda()

    optimizer = torch.optim.SGD([
            {'params': pretrained_model.parameters()},
            {'params': clf_SIMCLR.parameters()}
        ],
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
            loss_SIMCLR = criterion_SIMCLR(clf_SIMCLR(f1), clf_SIMCLR(f2))

            optimizer.zero_grad()
            loss_SIMCLR.backward()
            optimizer.step()

            epoch_loss += loss_SIMCLR.item()
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

    # Load pre-trained model
    if params.model == 'ResNet10':
        model_dict = {params.model: backbone.ResNet10(method=params.method, track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    elif params.model == 'ResNet12':
        model_dict = {params.model: backbone.ResNet12(track_bn=params.track_bn, reinit_bn_stats=params.reinit_bn_stats)}
    elif params.model == 'ResNet18':
        pass
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
    
    if pretrained_dataset == 'miniImageNet':
        params.num_classes = 64
        image_size = 224
    elif pretrained_dataset == 'tieredImageNet':
        params.num_classes = 351
        image_size = 84
    pretrained_model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='softmax')
    pretrained_model.load_state_dict(state, strict=True)
    
    # Re-randomization
    partial_reinit(pretrained_model, params.model)
        
    # Make checkpoint_dir
    checkpoint_dir += '/unlabeled'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    dataset_names = params.dataset_names

    for dataset_name in dataset_names:
        print (dataset_name)
        print('Initializing data loader...')
        if dataset_name == "miniImageNet":
            transform = miniImageNet_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True)
            dataset = miniImageNet_few_shot.SimpleDataset(
                apply_twice(transform), train=False, split=True)
        if dataset_name == "tieredImageNet":
            transform = tieredImageNet_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True)
            dataset = tieredImageNet_few_shot.SimpleDataset(
                apply_twice(transform), train=False, split=True)
        elif dataset_name == "CropDisease":
            transform = CropDisease_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True)
            dataset = CropDisease_few_shot.SimpleDataset(
                apply_twice(transform), split=True)
        elif dataset_name == "EuroSAT":
            transform = EuroSAT_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True)
            dataset = EuroSAT_few_shot.SimpleDataset(
                apply_twice(transform), split=True)
        elif dataset_name == "ISIC":
            transform = ISIC_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True)
            dataset = ISIC_few_shot.SimpleDataset(
                apply_twice(transform), split=True)
        elif dataset_name == "ChestX":
            transform = Chest_few_shot.TransformLoader(
                image_size).get_composed_transform(aug=True)
            dataset = Chest_few_shot.SimpleDataset(
                apply_twice(transform), split=True)

        # Prepare SimCLR
        clf_SIMCLR = projector_SIMCLR(pretrained_model.feature.final_feat_dim, out_dim=128) # Projection dimension is fixed to 128
        criterion_SIMCLR = NTXentLoss('cuda', batch_size=batch_size, temperature=temperature, use_cosine_similarity=True)

        novel_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   num_workers=2, shuffle=True, drop_last=True)

        print('Data loader initialized successfully!, length: {}'.format(len(dataset)))

        train_unlabeled(dataset_name, novel_loader, pretrained_model, clf_SIMCLR, criterion_SIMCLR,
                        checkpoint_dir, start_epoch, stop_epoch, params)
