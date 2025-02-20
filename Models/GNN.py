import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from math import ceil


# 多层浅层嵌入模型
class MultiShallowEmbedding(nn.Module):
    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs
        self.emb_s = Parameter(torch.Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(torch.Tensor(num_graphs, 1, num_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)

    def forward(self, device):
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')
        adj_flat = adj.view(self.num_graphs, -1)

        # 限制 self.k 不超过邻居数量
        k = min(self.k, adj_flat.size(-1))
        indices = adj_flat.topk(k=k, dim=-1)[1]

        adj_flat.fill_(0)
        adj_flat.scatter_(1, indices, 1.0)
        adj = adj_flat.view_as(adj)
        return adj


# 分组线性层
class GroupLinear(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()
        self.out_channels = out_channels
        self.groups = groups
        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups,
                                   bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.group_mlp.reset_parameters()

    def forward(self, x: Tensor, is_reshape=False):
        B, C, N = x.size(0), x.size(1), x.size(-2)
        G = self.groups
        if not is_reshape:
            x = x.view(B, C, N, G, -1).transpose(2, 3)
        x = x.transpose(1, 2).reshape(B, G * C, N, -1)
        out = self.group_mlp(x)
        out = out.view(B, G, self.out_channels, N, -1).transpose(1, 2)
        return out


# 密集GCN卷积层
class DenseGCNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = GroupLinear(in_channels, out_channels, groups, bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            init.zeros_(self.bias)

    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            idx = torch.arange(adj.size(-1), device=adj.device)
            adj = adj.clone()  # Clone the tensor to avoid in-place operation on expanded tensor
            adj[:, :, idx, idx] += 1
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        return adj

    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        adj = self.norm(adj, add_loop).unsqueeze(1)
        x = self.lin(x, False)
        B, C, G, N, F = x.size()
        adj = adj.expand(B, G, N, N)
        x = x.view(B, G, C // G, N, F).transpose(2, 3)
        out = torch.matmul(adj, x)
        out = out.transpose(2, 3).view(B, C, N, F)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out


# 时间差池化层
class DenseTimeDiffPool2d(nn.Module):
    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))
        self.re_param = Parameter(torch.Tensor(kern_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')

    def forward(self, x: Tensor, adj: Tensor):
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))
        return out, out_adj


# GNN堆栈模型
class GNNStack(nn.Module):
    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size,
                 in_dim, hidden_dim, out_dim, seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU()):
        super().__init__()
        self.num_nodes = num_nodes
        self.g_constr = MultiShallowEmbedding(num_nodes, num_nodes, groups)
        gnn_model, heads = self.build_gnn_model(gnn_model_type)
        paddings = [(k - 1) // 2 for k in kern_size]
        self.tconvs = nn.ModuleList(
            [nn.Conv2d(1, in_dim, (1, kern_size[0]), padding=(0, paddings[0]))] +
            [nn.Conv2d(heads * in_dim, hidden_dim, (1, kern_size[i]), padding=(0, paddings[i])) for i in range(1, num_layers)]
        )
        self.gconvs = nn.ModuleList(
            [gnn_model(in_dim, heads * in_dim, groups)] +
            [gnn_model(hidden_dim, heads * hidden_dim, groups) for _ in range(num_layers - 1)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(heads * out_dim, num_classes)
        self.dropout = dropout
        self.activation = activation

    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1

    def forward(self, inputs: Tensor):
        x = inputs
        adj = self.g_constr(x.device)
        print(f"adj shape before unsqueeze: {adj.shape}")
        adj = adj.unsqueeze(0)
        print(f"adj shape after unsqueeze: {adj.shape}")
        print(f"x shape: {x.shape}")
        adj = adj.expand(x.size(0), adj.size(1), adj.size(2), adj.size(3))
        print(f"adj shape after expand: {adj.shape}")
        adj = adj.unsqueeze(2).expand(-1, -1, x.size(1), -1, -1)
        print(f"adj shape after second unsqueeze and expand: {adj.shape}")
        adj = adj.contiguous().view(x.size(0), -1, adj.size(-2), adj.size(-1))
        print(f"adj shape after view: {adj.shape}")
        for tconv, gconv in zip(self.tconvs, self.gconvs):
            x = tconv(x)
            x = self.activation(gconv(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.linear(x)


# 示例代码
if __name__ == "__main__":
    inputs = torch.randn(8, 1, 10, 16)  # (batch_size, channels, num_nodes, seq_len)
    model = GNNStack(
        gnn_model_type='dyGCN2d',
        num_layers=3,
        groups=2,
        pool_ratio=0.5,
        kern_size=[3, 3, 3],
        in_dim=32,
        hidden_dim=64,
        out_dim=128,
        seq_len=16,
        num_nodes=10,
        num_classes=3
    )
    output = model(inputs)
    print(output.shape)