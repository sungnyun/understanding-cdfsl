import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import backbone
import configs
from datasets import miniImageNet_few_shot, tieredImageNet_few_shot, ISIC_few_shot, EuroSAT_few_shot, \
    CropDisease_few_shot, Chest_few_shot
from datasets.dataloader import get_labeled_dataloader, get_unlabeled_dataloader, get_dataloader
from io_utils import parse_args, get_resume_file
from methods.baselinetrain import BaselineTrain
from methods.byol import BYOL


def train(model, checkpoint_dir, pretrain_type, dataset_name=None,
          labeled_source_loader=None, unlabeled_source_loader=None, unlabeled_target_loader=None):
    if labeled_source_loader is None and unlabeled_source_loader is None and unlabeled_target_loader is None:
        raise ValueError('Invalid unlabeled loaders')

    start_epoch = 0
    stop_epoch = 1000
    freq_epoch = 100

    if pretrain_type in [6, 7, 8]:
        first_pretrained_model_dir = '%s/checkpoints/miniImageNet/ResNet10_baseline/type1_strong' % (configs.save_dir)
        modelfile = get_resume_file(first_pretrained_model_dir)
        if not os.path.exists(modelfile):
            raise Exception('Invalid model path: "{}" (no such file found)'.format(modelfile))
        print('Pre-training from the model weights path {}'.format(modelfile))

        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.",
                                     "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                state[newkey] = state.pop(key)
            else:
                state[newkey] = state.pop(key)
        model.online_encoder.net.load_state_dict(state, strict=True)

    model.train()
    model.cuda()
    opt_params = [{'params': model.parameters()}]

    # if pretrain_type != 1:
    #     criterion = nn.CrossEntropyLoss().cuda()

    if pretrain_type != 1 and labeled_source_loader is not None:
        labeled_source_loader_iter = iter(labeled_source_loader)
        nll_criterion = nn.NLLLoss(reduction='mean').cuda()

    if pretrain_type != 2 and unlabeled_source_loader is not None:
        unlabeled_source_loader_iter = iter(unlabeled_source_loader)

    optimizer = torch.optim.Adam(opt_params, lr=3e-4)
    # torch.optim.SGD(opt_params,
    # lr=0.1, momentum=0.9,
    # weight_decay=1e-4,
    # nesterov=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[400, 600, 800],
                                                     gamma=0.1)

    print("Learning setup is set!")

    if pretrain_type == 1:
        raise NotImplementedError

    elif pretrain_type == 2:
        for epoch in range(start_epoch, stop_epoch):
            epoch_loss = 0
            if epoch == 0:
                outfile = os.path.join(checkpoint_dir, 'initial.tar')
                torch.save({'epoch': epoch, 'state': model.net.state_dict()}, outfile)

            for i, (X, y) in enumerate(unlabeled_source_loader):  # For pre-training 2
                loss = model(X[0].cuda(non_blocking=True), X[1].cuda(non_blocking=True))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_moving_average()
                epoch_loss += loss.item()
            scheduler.step()
            print('epoch: {}, loss: {}'.format(epoch, epoch_loss / len(unlabeled_source_loader)))

            if (epoch % freq_epoch == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.net.state_dict()}, outfile)
    else:
        for epoch in range(start_epoch, stop_epoch):
            epoch_loss = 0
            if epoch == 0:
                outfile = os.path.join(checkpoint_dir, '{}_initial.tar'.format(dataset_name))
                torch.save({'epoch': epoch, 'state': model.net.state_dict()}, outfile)

            for i, (X, y) in enumerate(unlabeled_target_loader):
                loss = model(X[0].cuda(non_blocking=True), X[1].cuda(non_blocking=True))

                if labeled_source_loader is None and unlabeled_source_loader is None:  # For pre-training 3, 6
                    total_loss = loss

                elif labeled_source_loader is not None:  # For pre-training 4, 7
                    try:
                        X_base, y_base = labeled_source_loader_iter.next()
                    except StopIteration:
                        labeled_source_loader_iter = iter(labeled_source_loader)
                        X_base, y_base = labeled_source_loader_iter.next()

                    features_base = model.net.feature(X_base.cuda())
                    logits_base = model.net.classifier(features_base)
                    log_probability_base = F.log_softmax(logits_base, dim=1)

                    gamma = 0.50
                    total_loss = gamma * loss + (1 - gamma) * nll_criterion(log_probability_base, y_base.cuda())

                elif unlabeled_source_loader is not None:  # For pre-training 5, 8
                    try:
                        X_base, y_base = unlabeled_source_loader_iter.next()
                    except StopIteration:
                        unlabeled_source_loader_iter = iter(unlabeled_source_loader)
                        X_base, y_base = unlabeled_source_loader_iter.next()

                    loss_base = model(X_base[0].cuda(non_blocking=True), X_base[1].cuda(non_blocking=True))
                    total_loss = 0.5 * loss + 0.5 * loss_base

                else:
                    raise Exception('Invalid loader settings')

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                model.update_moving_average()
                epoch_loss += total_loss.item()

            scheduler.step()
            print('epoch: {}, loss: {}'.format(epoch, epoch_loss / len(unlabeled_target_loader)))

            if (epoch % freq_epoch == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(checkpoint_dir, '{}_{:d}.tar'.format(dataset_name, epoch))
                torch.save({'epoch': epoch, 'state': model.net.state_dict()}, outfile)


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
        if params.dataset == 'tieredImageNet':
            image_size = 84
        else:
            image_size = 224
        baseline = BaselineTrain(model_dict[params.model], num_class=params.num_classes)
        model = BYOL(net=baseline, image_size=image_size)
    else:
        raise ValueError('Invalid `method` argument: {}'.format(params.method))

    if params.aug_mode is None:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s_byol/type%s' % (
            configs.save_dir, params.dataset, params.model, params.method, str(params.pretrain_type))
    else:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s_byol/type%s_%s' % (
            configs.save_dir, params.dataset, params.model, params.method, str(params.pretrain_type), params.aug_mode)
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
