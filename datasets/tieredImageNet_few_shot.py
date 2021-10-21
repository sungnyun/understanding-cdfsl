# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import os
import copy
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *

def construct_subset(dataset, split):
    split = './datasets/split_seed_1/tieredImageNet_test_labeled_80.csv'
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


identity = lambda x:x

class SimpleDataset:
    def __init__(self, transform, train, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}
        self.meta['image_names'] = []
        self.meta['image_labels'] = []
        
        if train:
            d = ImageFolder(tieredImageNet_path)
        else:
            d = ImageFolder(tieredImageNet_test_path)
            
        for i, (data, label) in enumerate(d):
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
            self.cl_list = range(351)
        else:
            self.cl_list = range(160)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        if train:
            d = ImageFolder(tieredImageNet_path)
        else:
            d = ImageFolder(tieredImageNet_test_path)

        if split:
            print("Using split: ", split)
            d = construct_subset(d, split)
            
        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

#         for key, item in self.sub_meta.items():
#             print (len(self.sub_meta[key]))
    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)
    
class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        img = self.transform(self.sub_meta[i])
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
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        if transform_type == 'Scale_original' or transform_type == 'Resize_original':
            return transforms.Resize([int(self.image_size), int(self.image_size)])
            
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop' or transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Scale' or transform_type == 'Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter',
                              'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize_original', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, split=False):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.split = split

    # parameters that would change on train/val set
    def get_data_loader(self, aug, train=True):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform, train)

        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers = 12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide=100, split=False):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)
        self.split = split

    # parameters that would change on train/val set
    def get_data_loader(self, aug, train=True):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform, train, self.split)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader