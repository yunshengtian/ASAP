import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch


class PN(nn.Module):
    def __init__(self, feat_len=1024, channels=4, batch_norm=True):
        super(PN, self).__init__()
        self.batch_norm = batch_norm
        bn_layer = nn.BatchNorm1d if batch_norm else nn.Identity

        self.conv1 = nn.Conv1d(channels, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, feat_len, kernel_size=1, bias=False)
        self.bn1 = bn_layer(64)
        self.bn2 = bn_layer(64)
        self.bn3 = bn_layer(64)
        self.bn4 = bn_layer(128)
        self.bn5 = bn_layer(feat_len)

    def forward(self, pcd, normals=None):
        # From (batch_size, num_points, 3)
        # to (batch_size, 3, num_points)
        x = pcd.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        return x


class MLP(nn.Module):
    def __init__(self, feat_len, max_num_parts=20):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(feat_len, 64)
        self.mlp2 = nn.Linear(64, 64)
        self.mlp3 = nn.Linear(64, max_num_parts)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        x = torch.sigmoid(self.mlp3(x))
        softmax_x = self.softmax(x)
        return softmax_x, x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        feat_len = args.feat_len
        max_num_parts = args.max_num_parts
        self.encoder = PN(feat_len, batch_norm=bool(args.batch_norm))
        self.classifier = MLP(feat_len, max_num_parts=max_num_parts)
        self.criterion = nn.BCELoss()

    def forward(self, batch):
        pc, gt = batch
        # pc: point cloud with shape (batch_size, max_pc_size, 4)
        # where the 4 is xyz followed by a counter for each part index
        feat = self.encoder(pc)
        label, logits = self.classifier(feat)
        # try MSE loss: 
        # loss = ((gt - logits)**2).mean()
        # CE loss
        loss = self.criterion(logits, gt)
        # TODO: Implement accuracy metric
        acc = torch.tensor(0.0, device=pc.device).float()
        return loss, acc


class MLPMultiFeature(nn.Module):
    def __init__(self, feat_len, hand_feat_len, batch_norm=False):
        super(MLPMultiFeature, self).__init__()
        bn_layer = nn.BatchNorm1d if batch_norm else nn.Identity
        self.mlp1 = nn.Linear(hand_feat_len, feat_len)
        self.mlp2 = nn.Linear(feat_len * 2, feat_len)
        self.relu = nn.ReLU()
        self.bn1 = bn_layer(feat_len)
        self.bn2 = bn_layer(feat_len)

    def forward(self, pc_feat, volume, distance, edge_count):
        # Combine the hand crafted features and
        # make them the same size as the point cloud features
        vol_dist = torch.column_stack([volume, distance, edge_count])
        vol_dist_feat = self.relu(self.bn1(self.mlp1(vol_dist)))
        # Combine the hand crafted and point cloud features
        # then scale them to the desired feature size
        vol_dist_pc = torch.cat([pc_feat, vol_dist_feat], dim=1)
        vol_dist_pc_feat = self.relu(self.bn2(self.mlp2(vol_dist_pc)))
        return vol_dist_pc_feat


class GAT(torch.nn.Module):
    def __init__(self, hidden_dim, dropout=0.0, batch_norm=False):
        super(GAT, self).__init__()
        bn_layer = nn.BatchNorm1d if batch_norm else nn.Identity
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
        self.relu = nn.ReLU()
        self.bn1 = bn_layer(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edges_idx):
        x = self.dropout(x)
        x = self.relu(self.bn1(self.conv1(x, edges_idx)))
        x = self.dropout(x)
        x = self.conv2(x, edges_idx)
        return x


class MLPNodeClassifier(nn.Module):
    def __init__(self, feat_len, batch_norm=False):
        super(MLPNodeClassifier, self).__init__()
        bn_layer = nn.BatchNorm1d if batch_norm else nn.Identity
        self.mlp1 = nn.Linear(feat_len, 64)
        self.mlp2 = nn.Linear(64, 64)
        self.mlp3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.bn1 = bn_layer(64)
        self.bn2 = bn_layer(64)

    def forward(self, x):
        x = self.relu(self.bn1(self.mlp1(x)))
        x = self.relu(self.bn2(self.mlp2(x)))
        x = self.mlp3(x)
        return x


class ModelGraph(nn.Module):
    def __init__(self, args):
        super(ModelGraph, self).__init__()
        bn = bool(args.batch_norm)
        self.pointnet = PN(args.feat_len, channels=3, batch_norm=bn)
        self.multi_feat = MLPMultiFeature(args.feat_len, 3, batch_norm=bn)
        self.gnn = GAT(args.feat_len, args.dropout, batch_norm=bn)
        self.classifier = MLPNodeClassifier(args.feat_len, batch_norm=bn)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, graph):
        # Pass the graph and get the logits
        logits = self.forward_logits(graph)        
        # Calculate hits for each graph in the batch
        top1_hits = torch.zeros(graph.num_graphs)
        for graph_index in range(graph.num_graphs):
            # Find the indices of the graph to slice
            graph_mask = (graph.batch == graph_index).float()
            mask_indices = torch.nonzero(graph_mask)
            mask_min = mask_indices.min()
            mask_max = mask_indices.max()
            # See if we have a top-1 hit in just this graph
            top1_index = torch.argmax(logits[mask_min:mask_max])
            gt_index = torch.argmax(graph.y[mask_min:mask_max])
            top1_hits[graph_index] = top1_index == gt_index

        # Calculate loss from logits
        loss = self.criterion(logits, graph.y)
        return loss, torch.mean(top1_hits)
        
    def forward_logits(self, graph):
        # graph.x contains an xyz point cloud for each node in the graph
        # i.e. part in the assembly, with shape (num_nodes, max_pc_size, 3)
        x = self.pointnet(graph.x)
        # shape: (num_nodes, feat_len)
        # Add our hand crafted features, i.e. volume and distance,
        # these are just scalar values calculated for each part
        x = self.multi_feat(x, graph.volume, graph.distance, graph.edge_count)
        # shape: (num_nodes, feat_len)
        x = self.gnn(x, graph.edge_index)
        # shape: (num_nodes, feat_len)
        logits = self.classifier(x).squeeze()
        # shape: (num_nodes)
        # probs = torch.sigmoid(logits).squeeze()
        return logits

        