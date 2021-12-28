import json
import os

import numpy as np
import torch
import torch.optim
from tqdm import tqdm

from backbone import get_backbone_class
from datasets.dataloader import get_dataloader, get_unlabeled_dataloader
from io_utils import parse_args
from model import get_model_class
from paths import get_output_directory, get_final_pretrain_state_path, get_pretrain_state_path, get_pretrain_params_path


def _get_dataloaders(params):
    batch_size = params.batch_size
    labeled_source_bs = batch_size
    unlabeled_source_bs = batch_size
    unlabeled_target_bs = batch_size

    if params.us and params.ut:
        unlabeled_source_bs //= 2
        unlabeled_target_bs //= 2

    ls, us, ut = None, None, None
    if params.ls:
        print('Using source data {} (labeled)'.format(params.source_dataset))
        ls = get_dataloader(dataset_name=params.source_dataset, augmentation=params.augmentation,
                            batch_size=labeled_source_bs)

    if params.us:
        print('Using source data {} (unlabeled)'.format(params.source_dataset))
        us = get_dataloader(dataset_name=params.source_dataset, augmentation=params.augmentation,
                            batch_size=unlabeled_source_bs,
                            siamese=True)  # important

    if params.ut:
        print('Using target data {} (unlabeled)'.format(params.target_dataset))
        ut = get_unlabeled_dataloader(dataset_name=params.target_dataset, augmentation=params.augmentation,
                                      batch_size=unlabeled_target_bs, siamese=True,
                                      unlabeled_ratio=params.unlabeled_ratio)

    return ls, us, ut


def main():
    params = parse_args('pretrain')

    backbone = get_backbone_class(params.backbone)()
    model = get_model_class(params.model)(backbone, params)
    output_dir = get_output_directory(params)
    labeled_source_loader, unlabeled_source_loader, unlabeled_target_loader = _get_dataloaders(params)

    params_path = get_pretrain_params_path(output_dir)
    with open(params_path, 'w') as f:
        json.dump(vars(params), f, indent=4)
    print('Saving pretrain params to {}'.format(params_path))

    if params.pls:
        # Load previous pre-trained weights for second-step pre-training
        previous_output_dir = get_output_directory(params, previous_step=True)
        state_path = get_final_pretrain_state_path(previous_output_dir)
        if not os.path.exists(state_path):
            raise Exception('Pre-train state path not found: {}'.format(state_path))
        print('Loading previous state for second-step pre-training:')
        print(state_path)

        # Note, override model.load_state_dict to change this behavior.
        state = torch.load(state_path)
        model.load_state_dict(state, strict=True)

    model.train()
    model.cuda()

    if params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.1, momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=False)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    else:
        raise ValueError('Invalid value for params.optimizer: {}'.format(params.optimizer))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[400, 600, 800],
                                                     gamma=0.1)

    for epoch in range(params.epochs):
        print('EPOCH {}'.format(epoch).center(40).center(80, '#'))

        if epoch == 0:
            state_path = get_pretrain_state_path(output_dir, epoch=0)
            print('Saving pre-train state to:')
            print(state_path)
            torch.save(model.state_dict(), state_path)

        if params.ls and not params.us and not params.ut:  # only ls (type 1)
            for x, y in tqdm(labeled_source_loader):
                optimizer.zero_grad()
                loss, _ = model.compute_cls_loss_and_accuracy(x.cuda(), y.cuda())
                loss.backward()
                optimizer.step()
        elif not params.ls and params.us and not params.ut:  # only us (type 2)
            for x, _ in tqdm(unlabeled_source_loader):
                optimizer.zero_grad()
                loss = model.compute_ssl_loss(x[0].cuda(), x[1].cuda())
                loss.backward()
                optimizer.step()
        elif params.ut:  # ut (epoch is based on unlabeled target)
            for x, _ in tqdm(unlabeled_target_loader):
                optimizer.zero_grad()
                target_loss = model.compute_ssl_loss(x[0].cuda(), x[1].cuda())  # UT loss
                source_loss = None
                if params.ls:  # type 4, 7
                    try:
                        sx, sy = labeled_source_loader_iter.next()
                    except (StopIteration, NameError):
                        labeled_source_loader_iter = iter(labeled_source_loader)
                        sx, sy = labeled_source_loader_iter.next()
                    source_loss = model.compute_cls_loss_and_accuracy(sx.cuda(), sy.cuda())[0]  # LS loss
                if params.us:  # type 5, 8
                    try:
                        sx, sy = unlabeled_source_loader_iter.next()
                    except (StopIteration, NameError):
                        unlabeled_source_loader_iter = iter(unlabeled_source_loader)
                        sx, sy = unlabeled_source_loader_iter.next()
                    source_loss = model.compute_ssl_loss(sx[0].cuda(), sx[1].cuda())  # US loss

                if source_loss:
                    loss = source_loss * params.gamma + target_loss * (1 - params.gamma)
                else:
                    loss = target_loss
                loss.backward()
                optimizer.step()
        else:
            raise AssertionError('Unknown training combination.')

        if scheduler is not None:
            scheduler.step()

        epoch += 1
        if epoch % params.model_save_interval == 0 or epoch == params.epochs:
            state_path = get_pretrain_state_path(output_dir, epoch=epoch)
            print('Saving pre-train state to:')
            print(state_path)
            torch.save(model.state_dict(), state_path)


if __name__ == '__main__':
    np.random.seed(10)
    main()
