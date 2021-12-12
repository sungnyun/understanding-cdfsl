from abc import abstractmethod

import torch
import torchvision.transforms as transforms


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
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'RandomColorJitter':
            return transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0)
        elif transform_type == 'RandomGrayscale':
            return transforms.RandomGrayscale(p=0.1)
        elif transform_type == 'RandomGaussianBlur':
            return transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5))], p=0.3)
        elif transform_type == 'RandomCrop':
            return transforms.RandomCrop(self.image_size)
        elif transform_type == 'RandomResizedCrop':
            return transforms.RandomResizedCrop(self.image_size)
        elif transform_type == 'CenterCrop':
            return transforms.CenterCrop(self.image_size)
        elif transform_type == 'Resize_up':
            return transforms.Resize(
                [int(self.image_size * 1.15),
                 int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return transforms.Normalize(**self.normalize_param)
        elif transform_type == 'Resize':
            return transforms.Resize(
                [int(self.image_size),
                 int(self.image_size)])
        elif transform_type == 'RandomRotation':
            return transforms.RandomRotation(degrees=10)
        else:
            method = getattr(transforms, transform_type)
            return method()

    def get_composed_transform(self, aug=False, aug_mode='base'):
        if aug:
            if aug_mode == 'base':
                transform_list = ['RandomResizedCrop', 'RandomColorJitter', 'RandomHorizontalFlip', 'ToTensor',
                                  'Normalize']
            elif aug_mode == 'strong':
                transform_list = ['RandomResizedCrop', 'RandomColorJitter', 'RandomGrayscale', 'RandomGaussianBlur',
                                  'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass
