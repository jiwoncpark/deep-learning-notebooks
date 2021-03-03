"""Generates toy regression data

"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Dataset, Data


class ToyDataset(Dataset):
    def __init__(self, root, size, anisotropic=False, seed=123,
                 transform=None, pre_transform=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.size = size
        self.anisotropic = anisotropic
        super(ToyDataset, self).__init__(root, transform, pre_transform)
        if self.anisotropic:
            self._mean = [14.977728,  7.357365,  5.883]
            self._std = [8.514355,  5.6329246, 2.718779]
        else:
            self._mean = [-1.3628578e-01, -9.4296522e-03,  1.1839000e+01]
            self._std = [3.493968,   0.29279956, 4.2775073]
        self._mean = torch.FloatTensor([self._mean])
        self._std = torch.FloatTensor([self._std])

    @property
    def mean_std(self):
        return self._mean, self._std

    @mean_std.setter
    def mean_std(self, mean_std):
        mean, std = mean_std
        self._mean = mean
        self._std = std

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        pass

    def len(self):
        return self.size

    def get(self, idx):
        gen = torch.Generator().manual_seed(idx)
        N = torch.randint(low=5, high=20, size=[], generator=gen)
        # 1. Nodes
        neighbor_features = torch.randn(N, 3, generator=gen)
        node_features = torch.cat([torch.zeros(1, 3),
                                   neighbor_features], dim=0)
        # 2. Labels
        if self.anisotropic:
            dist = torch.sum(node_features[:, :2]**2.0, dim=1)**0.5
            weighted_sum = torch.sum(1.0/(dist[1:] + 1.e-7))
            dist = dist[node_features[:, 2] > 0.0]
            weighted_sum_with_cut = torch.sum(1.0/(dist + 1.e-7))
            sum_with_cut = len(dist)
            y = torch.FloatTensor([[weighted_sum, weighted_sum_with_cut,
                                  sum_with_cut]])
        else:
            sum_1 = torch.sum(node_features[:, 1])
            mean_2 = torch.mean(node_features[:, 2])
            y = torch.FloatTensor([[sum_1, mean_2, N]])
        mean, std = self.mean_std
        y = (y - mean)/std  # standardize
        # 3. Edges
        # Undirected edge between pairs of galaxies that are close enough
        kd_tree = cKDTree(node_features[:, :2].numpy())
        edges_close = kd_tree.query_pairs(r=0.5, p=2,
                                          eps=0.5/5.0,
                                          output_type='set')
        edges_close_reverse = [(b, a) for a, b in edges_close]
        # Edge from every neighbor to central node
        edges_to_center = set(zip(np.arange(N+1), np.zeros(N+1)))
        edge_index = edges_to_center.union(edges_close)
        edge_index = edge_index.union(edges_close_reverse)
        edge_index = torch.LongTensor(list(edge_index)).transpose(0, 1)
        data = Data(x=node_features,
                    edge_index=edge_index,
                    y=y)
        return data


if __name__ == '__main__':
    toy_dataset = ToyDataset(root='.', size=7, device_type='cpu')