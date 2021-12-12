# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import copy
import os

import pandas as pd
import torch
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from datasets.common import SubDataset, DataManager, TransformLoader, EpisodicBatchSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys

sys.path.append("../")
from configs import *


def construct_subset(dataset, labeled):
    if labeled:
        split = './datasets/split_seed_1/miniImageNet_test_labeled_80.csv'
    else:
        split = './datasets/split_seed_1/miniImageNet_test_unlabeled_20.csv'
    split = pd.read_csv(split)['img_path'].values
    root = dataset.root

    class_to_idx = dataset.class_to_idx
    targets = [class_to_idx[os.path.dirname(i)] for i in split]

    # image_names = np.array([i[0] for i in dataset.imgs])
    # # ind 
    # ind = np.concatenate([np.where(image_names == os.path.join(root, j))[0] for j in split])
    image_names = [os.path.join(root, j) for j in split]
    dataset_subset = copy.deepcopy(dataset)

    dataset_subset.samples = [j for j in zip(image_names, targets)]
    dataset_subset.imgs = dataset_subset.samples
    dataset_subset.targets = targets
    return dataset_subset


identity = lambda x: x


class SimpleDataset:
    def __init__(self, transform, train, split=False, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        if train:
            self.d = ImageFolder(miniImageNet_path)
        else:
            self.d = ImageFolder(miniImageNet_test_path)

        if split:
            print("Using unlabeled split: ", split)
            self.d = construct_subset(self.d, labeled=False)

        for i, (data, label) in enumerate(self.d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, batch_size, transform, train, split):

        self.sub_meta = {}
        if train:
            self.cl_list = range(64)
        else:
            self.cl_list = range(20)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        if train:
            d = ImageFolder(miniImageNet_path)
        else:
            d = ImageFolder(miniImageNet_test_path)

        if split:
            print("Using labeled split: ", split)
            d = construct_subset(d, labeled=True)

        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

        #         for key, item in self.sub_meta.items():
        #             print (len(self.sub_meta[key]))

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug, train=True):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform, train)

        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_episode=100, split=False):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)
        self.split = split

    def get_data_loader(self, aug, train=True):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform, train, self.split)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=2, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


if __name__ == '__main__':
    pass
