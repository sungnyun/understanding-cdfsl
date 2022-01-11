import json
import os

import numpy as np
import pandas as pd
import torch
import torch.optim
from tqdm import tqdm

from backbone import get_backbone_class
from datasets.dataloader import get_dataloader, get_unlabeled_dataloader
from io_utils import parse_args
from model import get_model_class
from paths import get_output_directory, get_final_pretrain_state_path, get_pretrain_state_path, \
    get_pretrain_params_path, get_pretrain_history_path


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
                            batch_size=labeled_source_bs, num_workers=params.num_workers)

    if params.us:
        print('Using source data {} (unlabeled)'.format(params.source_dataset))
        us = get_dataloader(dataset_name=params.source_dataset, augmentation=params.augmentation,
                            batch_size=unlabeled_source_bs, num_workers=params.num_workers,
                            siamese=True)  # important

    if params.ut:
        print('Using target data {} (unlabeled)'.format(params.target_dataset))
        ut = get_unlabeled_dataloader(dataset_name=params.target_dataset, augmentation=params.augmentation,
                                      batch_size=unlabeled_target_bs, num_workers=params.num_workers, siamese=True,
                                      unlabeled_ratio=params.unlabeled_ratio)

    return ls, us, ut


def main(params):
    backbone = get_backbone_class(params.backbone)()
    model = get_model_class(params.model)(backbone, params)
    output_dir = get_output_directory(params)
    labeled_source_loader, unlabeled_source_loader, unlabeled_target_loader = _get_dataloaders(params)

    params_path = get_pretrain_params_path(output_dir)
    with open(params_path, 'w') as f:
        json.dump(vars(params), f, indent=4)
    pretrain_history_path = get_pretrain_history_path(output_dir)
    print('Saving pretrain params to {}'.format(params_path))
    print('Saving pretrain history to {}'.format(pretrain_history_path))

    if params.pls:
        # Load previous pre-trained weights for second-step pre-training
        previous_base_output_dir = get_output_directory(params, pls_previous=True)
        state_path = get_final_pretrain_state_path(previous_base_output_dir)
        print('Loading previous state for second-step pre-training:')
        print(state_path)

        # Note, override model.load_state_dict to change this behavior.
        state = torch.load(state_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(unexpected):
            raise Exception("Unexpected keys from previous state: {}".format(unexpected))

    model.train()
    model.cuda()

    if params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=params.lr, momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=False)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        raise ValueError('Invalid value for params.optimizer: {}'.format(params.optimizer))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[400, 600, 800],
                                                     gamma=0.1)

    pretrain_history = {
        'loss': [0] * params.epochs,
        'source_loss': [0] * params.epochs,
        'target_loss': [0] * params.epochs,
    }

    for epoch in range(params.epochs):
        print('EPOCH {}'.format(epoch).center(40).center(80, '#'))

        epoch_loss = 0
        epoch_source_loss = 0
        epoch_target_loss = 0
        steps = 0

        if epoch == 0:
            state_path = get_pretrain_state_path(output_dir, epoch=0)
            print('Saving pre-train state to:')
            print(state_path)
            torch.save(model.state_dict(), state_path)

        model.on_epoch_start()
        model.train()

        if params.ls and not params.us and not params.ut:  # only ls (type 1)
            for x, y in tqdm(labeled_source_loader):
                model.on_step_start()
                optimizer.zero_grad()
                loss, _ = model.compute_cls_loss_and_accuracy(x.cuda(), y.cuda())
                loss.backward()
                optimizer.step()
                model.on_step_end()

                epoch_loss += loss.item()
                epoch_source_loss += loss.item()
                steps += 1
        elif not params.ls and params.us and not params.ut:  # only us (type 2)
            for x, _ in tqdm(unlabeled_source_loader):
                model.on_step_start()
                optimizer.zero_grad()
                loss = model.compute_ssl_loss(x[0].cuda(), x[1].cuda())
                loss.backward()
                optimizer.step()
                model.on_step_end()

                epoch_loss += loss.item()
                epoch_source_loss += loss.item()
                steps += 1
        elif params.ut:  # ut (epoch is based on unlabeled target)
            for x, _ in tqdm(unlabeled_target_loader):
                model.on_step_start()
                optimizer.zero_grad()
                target_loss = model.compute_ssl_loss(x[0].cuda(), x[1].cuda())  # UT loss
                epoch_target_loss += target_loss.item()
                source_loss = None
                if params.ls:  # type 4, 7
                    try:
                        sx, sy = labeled_source_loader_iter.next()
                    except (StopIteration, NameError):
                        labeled_source_loader_iter = iter(labeled_source_loader)
                        sx, sy = labeled_source_loader_iter.next()
                    source_loss = model.compute_cls_loss_and_accuracy(sx.cuda(), sy.cuda())[0]  # LS loss
                    epoch_source_loss += source_loss.item()
                if params.us:  # type 5, 8
                    try:
                        sx, sy = unlabeled_source_loader_iter.next()
                    except (StopIteration, NameError):
                        unlabeled_source_loader_iter = iter(unlabeled_source_loader)
                        sx, sy = unlabeled_source_loader_iter.next()
                    source_loss = model.compute_ssl_loss(sx[0].cuda(), sx[1].cuda())  # US loss
                    epoch_source_loss += source_loss.item()

                if source_loss:
                    loss = source_loss * (1 - params.gamma) + target_loss * params.gamma
                else:
                    loss = target_loss
                loss.backward()
                optimizer.step()
                model.on_step_end()

                epoch_loss += loss.item()
                steps += 1
        else:
            raise AssertionError('Unknown training combination.')

        if scheduler is not None:
            scheduler.step()
        model.on_epoch_end()

        mean_loss = epoch_loss / steps
        mean_source_loss = epoch_source_loss / steps
        mean_target_loss = epoch_target_loss / steps
        fmt = 'Epoch {:04d}: loss={:6.4f} source_loss={:6.4f} target_loss={:6.4f}'
        print(fmt.format(epoch, mean_loss, mean_source_loss, mean_target_loss))

        pretrain_history['loss'][epoch] = mean_loss
        pretrain_history['source_loss'][epoch] = mean_source_loss
        pretrain_history['target_loss'][epoch] = mean_target_loss

        pd.DataFrame(pretrain_history).to_csv(pretrain_history_path)

        epoch += 1
        if epoch % params.model_save_interval == 0 or epoch == params.epochs:
            state_path = get_pretrain_state_path(output_dir, epoch=epoch)
            print('Saving pre-train state to:')
            print(state_path)
            torch.save(model.state_dict(), state_path)


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('pretrain')

    targets = params.target_dataset
    if targets is None:
        targets = [targets]
    elif len(targets) > 1:
        print('#' * 80)
        print("Running pretrain iteratively for multiple target datasets: {}".format(targets))
        print('#' * 80)

    for target in targets:
        params.target_dataset = target
        main(params)
