from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler
from torchvision.datasets import ImageFolder


class EpisodeSampler:
    """
    Stable sampler for support and query indices. Used by episodic batch sampler, so that the support and query sets
    can be sampled from independent data loaders using the same splits, i.e., such that support and query do not overlap.
    """

    def __init__(self, dataset: ImageFolder, n_way: int, n_shot: int, n_query_shot: int, n_episodes: int,
                 seed: int = 0):
        self.dataset = dataset
        self.n_classes = len(dataset.classes)
        self.w = n_way
        self.s = n_shot
        self.q = n_query_shot
        self.n_episodes = n_episodes
        self.seed = seed

        rs = np.random.RandomState(seed)
        self.episode_seeds = []
        for i in range(n_episodes):
            self.episode_seeds.append(rs.randint(2 ** 32 - 1))

        self.indices_by_class = defaultdict(list)
        for index, (path, label) in enumerate(dataset.samples):
            self.indices_by_class[label].append(index)

    def __getitem__(self, index):
        """
        :param index:
        :return: support: ndarray[w, s], query: ndarray[w ,q]
        """
        rs = np.random.RandomState(self.episode_seeds[index])
        selected_classes = rs.permutation(self.n_classes)[:self.w]
        indices = []
        for cls in selected_classes:
            indices.append(
                rs.choice(self.indices_by_class[cls], self.s + self.q, replace=False))
        episode = np.stack(indices)
        support = episode[:, :self.s]
        query = episode[:, self.s:]
        return support, query

    def __len__(self):
        return self.n_episodes


class EpisodicBatchSampler(Sampler):
    """
    For each epoch, the same batch is yielded repeatedly. For batch-training within episodes, you need to divide up the
    sampled data (from the dataloader) into further smaller batches.

    For classification-based training, note that you need to reset the class indices to [0, 0, ..., 1, ..., w-1]. Note
    that this is why inter-episode batches are not supported by the sampler: it's harder to reset the class indices.
    """

    def __init__(self, dataset: ImageFolder, n_way: int, n_shot: int, n_query_shot: int, n_episodes: int, support: bool,
                 n_epochs=1, seed=0):
        super().__init__(dataset)
        self.dataset = dataset

        self.w = n_way
        self.s = n_shot
        self.q = n_query_shot
        self.episode_sampler = EpisodeSampler(dataset, n_way, n_shot, n_query_shot, n_episodes, seed)

        self.n_episodes = n_episodes
        self.n_epochs = n_epochs
        self.support = support

    def __len__(self):
        return self.n_episodes * self.n_epochs

    def __iter__(self):
        for i in range(self.n_episodes):
            try:  # HOTFIX TO GET CAR RESULTS
                support, query = self.episode_sampler[i]
                indices = support if self.support else query
                indices = indices.flatten()
                for j in range(self.n_epochs):
                    yield indices
            except:
                yield None
