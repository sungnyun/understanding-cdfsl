import copy
import os
from typing import List, Tuple

import pandas as pd
from numpy.random import RandomState
from torchvision.datasets import ImageFolder

DIRNAME = os.path.dirname(os.path.abspath(__file__))

DATASETS_WITH_DEFAULT_SPLITS = [
    "miniImageNet",
    "miniImageNet_test",
    "tieredImageNet",
    "tieredImageNet_test",
    "CropDisease",
    "EuroSAT",
    "ISIC",
    "ChestX",
]


def split_dataset(dataset: ImageFolder, ratio=20, seed=1):
    """
    :param dataset:
    :param ratio: Ratio of unlabeled portion
    :param seed:
    :return: unlabeled_dataset, labeled_dataset
    """
    assert (0 <= ratio <= 100)

    # Check default splits
    unlabeled_path = _get_split_path(dataset, ratio, seed, True)
    labeled_path = _get_split_path(dataset, ratio, seed, False)
    for path in [unlabeled_path, labeled_path]:
        if ratio == 20 and seed == 1 and dataset.name in DATASETS_WITH_DEFAULT_SPLITS and not os.path.exists(path):
            raise Exception("Default split file missing: {}".format(path))

    if os.path.exists(unlabeled_path) and os.path.exists(labeled_path):
        print("Loading unlabeled split from {}".format(unlabeled_path))
        print("Loading labeled split from {}".format(labeled_path))
        unlabeled = _load_split(unlabeled_path)
        labeled = _load_split(labeled_path)
    else:
        unlabeled, labeled = _get_split(dataset, ratio, seed)
        print("Generating unlabeled split to {}".format(unlabeled_path))
        print("Generating labeled split to {}".format(labeled_path))
        _save_split(unlabeled, unlabeled_path)
        _save_split(labeled, labeled_path)

    ud = copy.deepcopy(dataset)
    ld = copy.deepcopy(dataset)

    _apply_split(ud, unlabeled)
    _apply_split(ld, labeled)

    return ud, ld


def _get_split(dataset: ImageFolder, ratio: int, seed: int) -> Tuple[List[str], List[str]]:
    img_paths = []
    for path, label in dataset.samples:
        root_with_slash = os.path.join(dataset.root, "")
        img_paths.append(path.replace(root_with_slash, ""))
    img_paths.sort()
    # Assert uniqueness
    assert (len(img_paths) == len(set(img_paths)))

    rs = RandomState(seed)
    unlabeled_count = len(img_paths) * ratio // 100
    unlabeled_paths = set(rs.choice(img_paths, unlabeled_count, replace=False))
    labeled_paths = set(img_paths) - unlabeled_paths

    return sorted(list(unlabeled_paths)), sorted(list(labeled_paths))


def _save_split(split: List, path):
    df = pd.DataFrame({
        "img_path": split
    })
    df.to_csv(path)


def _load_split(path) -> List[str]:
    df = pd.read_csv(path)
    return df["img_path"].values


def _get_split_path(dataset: ImageFolder, ratio: int, seed=1, unlabeled=True, makedirs=True):
    if unlabeled:
        basename = '{}_unlabeled_{}.csv'.format(dataset.name, ratio)
    else:
        basename = '{}_labeled_{}.csv'.format(dataset.name, 100 - ratio)
    path = os.path.join(DIRNAME, 'split_seed_{}'.format(seed), basename)
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _apply_split(dataset: ImageFolder, split: List[str]):
    img_paths = []
    for path, label in dataset.samples:
        root_with_slash = os.path.join(dataset.root, "")
        img_paths.append(path.replace(root_with_slash, ""))

    split_set = set(split)
    samples = []
    for path, sample in zip(img_paths, dataset.samples):
        if len(split) > 0 and '.jpg' not in split[0] and dataset.name == 'ISIC':  # HOTFIX (paths in ISIC's default split file don't have ".jpg")
            path = path.replace('.jpg', '')
        if path in split_set:
            samples.append(sample)

    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [s[1] for s in samples]
