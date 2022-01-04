from typing import Tuple, MutableMapping
from weakref import WeakValueDictionary

import torch
import torch.utils.data
from torch.utils.data import Dataset

from datasets.datasets import dataset_class_map
from datasets.sampler import EpisodicBatchSampler
from datasets.split import split_dataset
from datasets.transforms import get_composed_transform

_unlabeled_dataset_cache: MutableMapping[Tuple[str, str, int, bool, int], Dataset] = WeakValueDictionary()

DEFAULT_IMAGE_SIZE = 224


class ToSiamese:
    """
    A wrapper for torchvision transform. The transform is applied twice for
    SimCLR training
    """

    def __init__(self, transform, transform2=None):
        self.transform = transform

        if transform2 is not None:
            self.transform2 = transform2
        else:
            self.transform2 = transform

    def __call__(self, img):
        return self.transform(img), self.transform2(img)


def get_default_dataset(dataset_name: str, augmentation: str, image_size: int = None, siamese=False,
                        force_tiered_augmentation=False):
    """
    :param augmentation: One of {'base', 'strong', None}
    """

    # Set tieredImageNet parameters in line with existing literature (STARTUP)
    if dataset_name == 'tieredImageNet' and not force_tiered_augmentation:
        print('Disabling augmentation for tieredImageNet as per literature.')
        augmentation = None
    if image_size is None:
        print('Using default image size: {}'.format(DEFAULT_IMAGE_SIZE))
        image_size = DEFAULT_IMAGE_SIZE

    try:
        dataset_cls = dataset_class_map[dataset_name]
    except KeyError as e:
        raise ValueError('Unsupported dataset: {}'.format(dataset_name))

    transform = get_composed_transform(augmentation, image_size=image_size)
    if siamese:
        transform = ToSiamese(transform)
    return dataset_cls(transform=transform)


def get_split_dataset(dataset_name: str, augmentation: str, image_size: int = None,
                      siamese=False, force_tiered_augmentation=False, unlabeled_ratio: int = 20, seed=1):
    # If cache details change, just remove the cache – it's not worth the maintenance TBH.
    cache_key = (dataset_name, augmentation, image_size, siamese, unlabeled_ratio)
    if cache_key not in _unlabeled_dataset_cache:
        dataset = get_default_dataset(dataset_name=dataset_name, augmentation=augmentation, image_size=image_size,
                                      siamese=siamese, force_tiered_augmentation=force_tiered_augmentation)
        unlabeled, labeled = split_dataset(dataset, ratio=unlabeled_ratio, seed=seed)
        # Cross-reference so that strong ref persists if either split is currently referenced
        unlabeled.counterpart = labeled
        labeled.counterpart = unlabeled
        _unlabeled_dataset_cache[cache_key] = unlabeled

    unlabeled = _unlabeled_dataset_cache[cache_key]
    labeled = unlabeled.counterpart

    return unlabeled, labeled


def get_dataloader(dataset_name: str, augmentation: str, batch_size: int, image_size: int = None, siamese=False,
                   force_tiered_augmentation=False, num_workers=2, shuffle=True,
                   drop_last=False):
    dataset = get_default_dataset(dataset_name=dataset_name, augmentation=augmentation, image_size=image_size,
                                  siamese=siamese, force_tiered_augmentation=force_tiered_augmentation)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                       shuffle=shuffle, drop_last=drop_last)


def get_split_dataloader(dataset_name: str, augmentation: str, batch_size: int, image_size: int = None, siamese=False,
                         force_tiered_augmentation=False, unlabeled_ratio: int = 20, num_workers=2, shuffle=True,
                         drop_last=False, seed=1):
    unlabeled, labeled = get_split_dataset(dataset_name, augmentation, image_size=image_size, siamese=siamese,
                                           force_tiered_augmentation=force_tiered_augmentation,
                                           unlabeled_ratio=unlabeled_ratio, seed=seed)
    dataloaders = []
    for dataset in [unlabeled, labeled]:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                                 shuffle=shuffle, drop_last=drop_last)
        dataloaders.append(dataloader)
    return dataloaders


def get_labeled_dataloader(dataset_name: str, augmentation: str, batch_size: int, image_size: int = None, siamese=False,
                           force_tiered_augmentation=False, unlabeled_ratio: int = 20, num_workers=2, shuffle=True,
                           drop_last=False, split_seed=1):
    unlabeled, labeled = get_split_dataloader(dataset_name, augmentation, batch_size, image_size, siamese=siamese,
                                              force_tiered_augmentation=force_tiered_augmentation,
                                              unlabeled_ratio=unlabeled_ratio,
                                              num_workers=num_workers, shuffle=shuffle, drop_last=drop_last,
                                              seed=split_seed)
    return labeled


def get_unlabeled_dataloader(dataset_name: str, augmentation: str, batch_size: int, image_size: int = None,
                             siamese=False, force_tiered_augmentation=False, unlabeled_ratio: int = 20, num_workers=2,
                             shuffle=True, drop_last=True, split_seed=1):
    unlabeled, labeled = get_split_dataloader(dataset_name, augmentation, batch_size, image_size, siamese=siamese,
                                              force_tiered_augmentation=force_tiered_augmentation,
                                              unlabeled_ratio=unlabeled_ratio,
                                              num_workers=num_workers, shuffle=shuffle, drop_last=drop_last,
                                              seed=split_seed)
    return unlabeled


def get_episodic_dataloader(dataset_name: str, n_way: int, n_shot: int, support: bool, n_episodes=600, n_query_shot=15,
                            augmentation: str = None, image_size: int = None, num_workers=2, n_epochs=1,
                            episode_seed=0):
    dataset = get_default_dataset(dataset_name=dataset_name, augmentation=augmentation, image_size=image_size,
                                  siamese=False, force_tiered_augmentation=True)
    sampler = EpisodicBatchSampler(dataset, n_way=n_way, n_shot=n_shot, n_query_shot=n_query_shot,
                                   n_episodes=n_episodes, support=support, n_epochs=n_epochs, seed=episode_seed)
    return torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=sampler)


def get_labeled_episodic_dataloader(dataset_name: str, n_way: int, n_shot: int, support: bool, n_episodes=600,
                                    n_query_shot=15, n_epochs=1, augmentation: str = None, image_size: int = None,
                                    unlabeled_ratio: int = 20, num_workers=2, split_seed=1, episode_seed=0):
    unlabeled, labeled = get_split_dataset(dataset_name, augmentation, image_size=image_size, siamese=False,
                                           force_tiered_augmentation=True, unlabeled_ratio=unlabeled_ratio,
                                           seed=split_seed)
    sampler = EpisodicBatchSampler(labeled, n_way=n_way, n_shot=n_shot, n_query_shot=n_query_shot,
                                   n_episodes=n_episodes, support=support, n_epochs=n_epochs, seed=episode_seed)
    return torch.utils.data.DataLoader(labeled, num_workers=num_workers, batch_sampler=sampler)
