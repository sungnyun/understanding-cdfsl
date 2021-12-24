from abc import abstractmethod

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset

from datasets.transforms import parse_transform, get_composed_transform


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=None):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        img = self.transform(self.sub_meta[i])
        target = self.cl
        if self.target_transform is not None:
            target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class TransformLoader:
    """
    Deprecated class. Refer to datasets.dataloader
    """
    def __init__(self, image_size):
        self.image_size = image_size

    def parse_transform(self, transform_type):
        return parse_transform(transform_type, image_size=self.image_size)

    def get_composed_transform(self, aug=False, aug_mode='base'):
        if aug == False:
            aug_mode = None
        return get_composed_transform(aug_mode, image_size=self.image_size)


class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass
