from collections import defaultdict

import numpy as np
import torch
from torchvision.datasets import ImageFolder


class EpisodicBatchSampler(object):
    def __init__(self, dataset: ImageFolder, n_way, n_shot, n_query_shot, n_episodes):
        self.n_classes = len(dataset.classes)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query_shot = n_query_shot
        self.n_episodes = n_episodes

        self.indices_by_class = defaultdict(list)
        for index, (path, label) in enumerate(dataset.samples):
            self.indices_by_class[label].append(index)

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            selected_classes = torch.randperm(self.n_classes)[:self.n_way].numpy()
            indices = []
            for cls in selected_classes:
                indices.append(np.random.choice(self.indices_by_class[cls], self.n_shot + self.n_query_shot))
            yield np.concatenate(indices)



