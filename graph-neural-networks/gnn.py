"""Various GNN models

"""
import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GravNetConv
import torch.nn as nn
import torch.nn.functional as F


def get_zero_nodes(batch_idx):
    """Get indices of the zeroth nodes in the batch

    """
    batch_idx = torch.cat([torch.zeros(1, device=batch_idx.device), batch_idx])
    diff = batch_idx[1:] - batch_idx[:-1]
    diff[0] = 1
    return diff.bool()


class GCNNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=256, n_layers=3, dropout=0.0):
        super(GCNNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.conv1 = GCNConv(self.in_channels, self.hidden_channels,
                             aggr='add',
                             add_self_loops=False)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels,
                             aggr='add',
                             add_self_loops=False)
        self.conv3 = GCNConv(self.hidden_channels, self.out_channels,
                             aggr='add',
                             add_self_loops=False)
        # self.fc = nn.Linear(self.hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv3(x, edge_index)
        #x = F.leaky_relu(x)
        zero_idx_mask = get_zero_nodes(batch)
        x = x[zero_idx_mask, :]
        # x = self.fc(x)
        # x = F.dropout(x, p=self.dropout, training=True)
        return x


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=256, n_layers=3, dropout=0.0):
        super(GATNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.conv1 = GATConv(self.in_channels, self.hidden_channels,
                             aggr='add',
                             dropout=self.dropout,
                             add_self_loops=False)
        self.conv2 = GATConv(self.hidden_channels, self.hidden_channels,
                             aggr='add',
                             dropout=self.dropout,
                             add_self_loops=False)
        self.conv3 = GATConv(self.hidden_channels, self.out_channels,
                             aggr='add',
                             dropout=self.dropout,
                             add_self_loops=False)
        # self.fc = nn.Linear(self.hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv3(x, edge_index)
        #x = F.leaky_relu(x)
        zero_idx_mask = get_zero_nodes(batch)
        x = x[zero_idx_mask, :]
        # x = self.fc(x)
        # x = F.dropout(x, p=self.dropout, training=True)
        return x


class SageNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=256, n_layers=3, dropout=0.0):
        super(SageNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.conv1 = SAGEConv(self.in_channels, self.hidden_channels,
                              aggr='add', normalize=True,
                              root_weight=False)
        self.conv2 = SAGEConv(self.hidden_channels, self.hidden_channels,
                              aggr='add', normalize=True,
                              root_weight=False)
        self.conv3 = SAGEConv(self.hidden_channels, self.out_channels,
                              aggr='add', normalize=True,
                              root_weight=False)
        # self.fc = nn.Linear(self.hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv3(x, edge_index)
        #x = F.leaky_relu(x)
        zero_idx_mask = get_zero_nodes(batch)
        x = x[zero_idx_mask, :]
        # x = self.fc(x)
        # x = F.dropout(x, p=self.dropout, training=True)
        return x


class GravNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 hidden_channels=256, n_layers=3, dropout=0.0):
        super(GravNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.conv1 = GravNetConv(self.in_channels,
                                 self.hidden_channels,
                                 aggr='add',
                                 space_dimensions=3,
                                 propagate_dimensions=2,
                                 # num_workers=4,
                                 k=20)
        self.conv2 = GravNetConv(self.hidden_channels,
                                 self.hidden_channels,
                                 aggr='add',
                                 space_dimensions=3,
                                 propagate_dimensions=2,
                                 # num_workers=4,
                                 k=20)
        self.conv3 = GravNetConv(self.hidden_channels,
                                 self.out_channels,
                                 aggr='add',
                                 space_dimensions=3,
                                 propagate_dimensions=2,
                                 # num_workers=4,
                                 k=20)
        # self.fc = nn.Linear(self.hidden_channels, out_channels)

    def forward(self, data):
        x, batch = data.x, data.batch
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv3(x)
        #x = F.leaky_relu(x)
        zero_idx_mask = get_zero_nodes(batch)
        x = x[zero_idx_mask, :]
        # x = self.fc(x)
        # x = F.dropout(x, p=self.dropout, training=True)
        return x
