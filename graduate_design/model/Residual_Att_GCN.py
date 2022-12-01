'''
note:
    1. Residual-Att模型结构完成，

todo:
    1. 组合部分未确定！！
'''
from collections import OrderedDict
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphNorm, global_max_pool
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from graduate_design.utils import debatch
from graduate_design.graph_dataset import GNNDataset
from torch_geometric.data import DataLoader


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = GCNConv(-1, out_channels)
        self.norm = GraphNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        output = self.conv(x, edge_index)
        output = self.norm(output)
        data.x = F.relu(output)

        return data


class ResidualLayer(nn.Module):
    def __init__(self, num_input_features):
        super(ResidualLayer, self).__init__()
        self.conv1 = GraphConvBn(num_input_features, 128)
        self.conv2 = GraphConvBn(128, 32)

    def forward(self, data):
        if isinstance(data.x, torch.Tensor):
            data.x = [data.x]
        data.x = torch.cat(data.x, 1)
        output = self.conv1(data)
        output = self.conv2(output)

        return output


class ResidualBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features):
        super(ResidualBlock, self).__init__()

        for i in range(num_layers):
            layers = ResidualLayer(num_input_features)
            self.add_module('layer %d' % (i + 1), layers)

    def forward(self, data):
        features = data.x
        for name, layer in self.items():
            output = layer(data)
            features += output.x
            data.x = F.relu(features)

        return data


class ResidualGCN(nn.Module):
    def __init__(self, num_input_features, out_dim, block_config=(3, 3, 3, 3)):
        super(ResidualGCN, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        for i, num_layers in enumerate(block_config):
            block = ResidualBlock(num_layers, 32)
            self.features.add_module('block%d' % (i + 1), block)

        self.linear = nn.Linear(32, out_dim)
        self.multihead_attn = nn.MultiheadAttention(32, 1)

    def node_feature_attention(self, x, graph_feature_max_pool):
        sim = torch.matmul(x.double(), graph_feature_max_pool.double())
        sim = F.softmax(sim, dim=1).unsqueeze(-1)
        node_att_featrue = sim * x
        graph_att_featrues = torch.sum(node_att_featrue, dim=0)
        return graph_att_featrues

    def Att(self, data, graph_max_features):
        attn_outputs = []
        graphdata = debatch(data)
        for node_features, graph_feature in zip(graphdata, graph_max_features):
            # note 这个自己写的att机制运算不对，可能是计算方式错误了，建议之间用封装好的！！
            # output = self.node_feature_attention(node_features, graph_feature)
            nf = node_features.reshape(-1, 1, 32).float()
            gf = graph_feature.reshape(1, 1, 32).float()

            # Todo 考虑设置多头att机制，后期实验可再改善！！！
            output = self.multihead_attn(nf, gf, gf, need_weights=False)
            attn_outputs.append(output[0])
        return attn_outputs

    def forward(self, data):
        data = self.features(data)
        output = global_max_pool(data.x, data.batch)

        attn_output = self.Att(data, output)

        return attn_output


def main():
    drug_encoder = ResidualGCN(num_input_features=78, out_dim=96, block_config=[8, 8, 8])
    davis_path = '../data/Davis/'
    davis_trainset = GNNDataset(davis_path, train=True)
    train_loader = DataLoader(davis_trainset, batch_size=54)

    for batch in train_loader:
        pred = drug_encoder(batch)
        print(pred)


if __name__ == '__main__':
    main()
