import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: 输入节点特征 [num_nodes, feature_dim]
        # edge_index: 图的边的索引，表示图的连接关系
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x


class ConvTranWithGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, seq_len, num_nodes):
        super(ConvTranWithGNN, self).__init__()

        # GCN Layer
        self.gcn = GCNLayer(in_channels, hidden_channels)

        # ConvTran 部分
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_channels, nhead=4)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=6)

        # 最后的输出层
        self.fc = nn.Linear(hidden_channels * seq_len, out_channels)

    def forward(self, x, edge_index):
        # x: [num_nodes, seq_len, feature_dim]
        # edge_index: 图的边的索引

        # 1. 使用 GNN 层提取图结构特征
        batch_size, seq_len, num_nodes, feature_dim = x.shape
        x = x.view(batch_size * num_nodes, seq_len, feature_dim)  # 变形为 [batch_size * num_nodes, seq_len, feature_dim]
        gnn_out = self.gcn(x, edge_index)  # 输出 [batch_size * num_nodes, seq_len, hidden_channels]

        # 2. 将 GNN 输出传入 Transformer 编码器
        gnn_out = gnn_out.view(batch_size, num_nodes, seq_len,
                               -1)  # 重塑为 [batch_size, num_nodes, seq_len, hidden_channels]
        gnn_out = gnn_out.view(batch_size, num_nodes * seq_len,
                               -1)  # 扁平化 [batch_size, num_nodes * seq_len, hidden_channels]

        transformer_out = self.transformer_encoder(gnn_out)  # 输出 [batch_size, num_nodes * seq_len, hidden_channels]

        # 3. 最后的全连接层
        output = self.fc(transformer_out.view(batch_size, -1))  # [batch_size, out_channels]
        return output
