import glob
import os
import re
from argparse import Namespace

import configs

DATASET_KEYS = {
    'miniImageNet': 'mini',
    'miniImageNet_test': 'mini_test',
    'tieredImageNet': 'tiered',
    'tieredImageNet_test': 'tiered_test',
    'ImageNet': 'imagenet',
    'CropDisease': 'crop',
    'EuroSAT': 'euro',
    'ISIC': 'isic',
    'ChestX': 'chest',
    'cars': 'cars',
    'cub': 'cub',
    'places': 'places',
    'plantae': 'plantae',
}

BACKBONE_KEYS = {
    'resnet10': 'resnet10',
    'resnet18': 'resnet18',
    'resnet50': 'resnet50',
}

MODEL_KEYS = {
    'base': 'base',
    'simclr': 'simclr',
    'simsiam': 'simsiam',
    'moco': 'moco',
    'swav': 'swav',
    'byol': 'byol',
}


def get_output_directory(params: Namespace, pls_previous=False, pmsl_previous=False, makedirs=True):
    """
    :param params:
    :param pls_previous: get previous output directory for pls mode
    :param pmsl_previous: get previous output directory for pmsl mode
    :return:
    """
    if pls_previous and not params.pls:
        raise Exception('Should not get pls_previous when params.pls is False')
    if pmsl_previous and not params.pmsl:
        raise Exception('Should not get pmsl_previous when params.pmsl is False')

    path = configs.save_dir
    path = os.path.join(path, 'output')
    path = os.path.join(path, DATASET_KEYS[params.source_dataset])

    pretrain_specifiers = []
    pretrain_specifiers.append(BACKBONE_KEYS[params.backbone])
    if pls_previous:
        pretrain_specifiers.append(MODEL_KEYS['base'])
        pretrain_specifiers.append('LS')
        pretrain_specifiers.append(params.pls_tag)
    elif pmsl_previous:
        pretrain_specifiers.append(MODEL_KEYS[params.model])
        pretrain_specifiers.append('LS_UT')
        pretrain_specifiers.append(params.pmsl_tag)
    else:
        pretrain_specifiers.append(MODEL_KEYS[params.model])
        if params.pls:
            pretrain_specifiers.append('PLS')
        if params.pmsl:
            pretrain_specifiers.append('PMSL')
        if params.ls:
            pretrain_specifiers.append('LS')
        if params.us:
            pretrain_specifiers.append('US')
        if params.ut:
            pretrain_specifiers.append('UT')
        pretrain_specifiers.append(params.tag)
    path = os.path.join(path, '_'.join(pretrain_specifiers))

    if pmsl_previous:
        path = os.path.join(path, DATASET_KEYS[params.target_dataset])
    if params.ut and not (pls_previous or pmsl_previous):
        path = os.path.join(path, DATASET_KEYS[params.target_dataset])

    if makedirs:
        os.makedirs(path, exist_ok=True)

    return path


def get_pretrain_history_path(output_directory):
    basename = 'pretrain_history.csv'
    return os.path.join(output_directory, basename)


def get_pretrain_state_path(output_directory, epoch=0):
    """
    :param output_directory:
    :param epoch: Number of completed epochs. I.e., 0 = initial.
    :return:
    """
    basename = 'pretrain_state_{:04d}.pt'.format(epoch)
    return os.path.join(output_directory, basename)


def get_final_pretrain_state_path(output_directory):
    glob_pattern = os.path.join(output_directory, 'pretrain_state_*.pt')
    paths = glob.glob(glob_pattern)

    pattern = re.compile('pretrain_state_(\d{4}).pt')
    paths_by_epoch = dict()
    for path in paths:
        match = pattern.search(path)
        if match:
            paths_by_epoch[match.group(1)] = path

    if len(paths_by_epoch) == 0:
        raise FileNotFoundError('Could not find valid pre-train state file in {}'.format(output_directory))

    max_epoch = max(paths_by_epoch.keys())
    return paths_by_epoch[max_epoch]


def get_pretrain_params_path(output_directory):
    return os.path.join(output_directory, 'pretrain_params.json')


def get_ft_output_directory(params, makedirs=True):
    path = get_output_directory(params, makedirs=makedirs)
    if not params.ut:
        path = os.path.join(path, params.target_dataset)
    ft_basename = '{:02d}way_{:03d}shot_{}_{}'.format(params.n_way, params.n_shot, params.ft_parts, params.ft_tag)
    path = os.path.join(path, ft_basename)

    if makedirs:
        os.makedirs(path, exist_ok=True)

    return path


def get_ft_params_path(output_directory):
    return os.path.join(output_directory, 'params.json')


def get_ft_train_history_path(output_directory):
    return os.path.join(output_directory, 'train_history.csv')


def get_ft_test_history_path(output_directory):
    return os.path.join(output_directory, 'test_history.csv')
