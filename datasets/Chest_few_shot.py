# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

from datasets.common import SubDataset, DataManager, TransformLoader, EpisodicBatchSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys

sys.path.append("../")
from configs import *


class CustomDatasetFromImages(Dataset):
    def __init__(self, split, labeled, csv_path=ChestX_path + "/Data_Entry_2017.csv", \
                 image_path=ChestX_path + "/images/"):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                            "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4,
                            "Nodule": 5, "Pneumothorax": 6}

        labels_set = []

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name = []
        self.labels = []

        for name, label in zip(self.image_name_all, self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[
                0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)

        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)

        if split:
            if labeled:
                print("Using labeled Split: ", split)
                split = './datasets/split_seed_1/ChestX_labeled_80.csv'
            else:
                print("Using unlabeled Split: ", split)
                split = './datasets/split_seed_1/ChestX_unlabeled_20.csv'
            split = pd.read_csv(split)['img_path'].values
            # construct the index
            ind = np.concatenate(
                [np.where(self.image_name == j)[0] for j in split])
            self.image_name = self.image_name[ind]
            self.labels = self.labels[ind]
            self.data_len = len(split)

            assert len(self.image_name) == len(split)
            assert len(self.labels) == len(split)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Open image
        img_as_img = Image.open(self.img_path + single_image_name).resize((256, 256)).convert('RGB')
        img_as_img.load()

        # Transform image to tensor
        # img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


identity = lambda x: x


class SimpleDataset:
    def __init__(self, transform, split=False, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        self.d = CustomDatasetFromImages(split=split, labeled=False)

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
    def __init__(self, batch_size, transform, split):

        self.sub_meta = {}
        self.cl_list = range(7)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = CustomDatasetFromImages(split=split, labeled=True)

        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

        for key, item in self.sub_meta.items():
            print(len(self.sub_meta[key]))

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)

        for cl in self.cl_list:
            print(cl)
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

    def get_data_loader(self, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
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

    def get_data_loader(self, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform, self.split)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=2, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


if __name__ == '__main__':
    base_datamgr = SetDataManager(224, n_query=16, n_support=5)
    base_loader = base_datamgr.get_data_loader(aug=True)
