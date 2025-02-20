import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter


class multi_shallow_embedding(nn.Module):

    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()

        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.emb_s)
        torch.nn.init.xavier_uniform_(self.emb_t)

    def forward(self, batch_size, num_channels, num_nodes, device):
        # adj: [G, N, N]
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)

        # remove self-loop
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')

        # top-k-edge adj
        adj_flat = adj.reshape(self.num_graphs, -1)
        indices = adj_flat.topk(k=self.k)[1].reshape(-1)

        idx = torch.tensor([i // self.k for i in range(indices.size(0))], device=device)

        adj_flat = torch.zeros_like(adj_flat).clone()
        adj_flat[idx, indices] = 1.
        adj = adj_flat.reshape_as(adj)

        # Adjust adj's node dimension to match num_nodes
        adj = adj[:, :num_nodes, :num_nodes]

        # expand adj to match batch size and number of channels
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Adjust adj's channel dimension to match num_channels
        if adj.size(1) != num_channels:
            adj = adj.mean(dim=1, keepdim=True).expand(-1, num_channels, -1, -1)

        # Print adj's shape for debugging
        print(f"adj.shape after expand and adjust: {adj.shape}")

        return adj


class Group_Linear(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()

        self.out_channels = out_channels
        self.groups = groups

        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups,
                                   bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.group_mlp.reset_parameters()

    def forward(self, x: Tensor, is_reshape: False):
        """
        Args:
            x (Tensor): [B, C, N, F] (if not is_reshape), [B, C, G, N, F//G] (if is_reshape)
        """
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups

        if not is_reshape:
            # x: [B, C_in, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C_in, N, F//G]
        x = x.transpose(1, 2).reshape(B, G * C, N, -1)

        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)

        # out: [B, C_out, G, N, F//G]
        return out


class DenseGCNConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Group_Linear(in_channels, out_channels, groups, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        init.zeros_(self.bias)

    def norm(self, adj: Tensor, add_loop: bool):
        print(f"adj.shape before norm: {adj.shape}")
        print(f"adj values: {adj[:2, :2]}")  # 打印部分张量值以检查是否有非法值

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            print(f"idx: {idx}")

            # 检查 adj 的形状是否正确
            if adj.dim() != 4:
                raise ValueError(f"adj must have 4 dimensions, but got {adj.dim()} dimensions.")
            if adj.size(-1) != adj.size(-2):
                raise ValueError(f"adj must be square matrix, but got shape {adj.shape}")

            adj[:, idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        return adj

    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [B, G, N, N]
        """
        # Normalize adjacency matrix
        adj = self.norm(adj, add_loop)

        # Linear transformation of x
        x = self.lin(x, False)  # Ensure x.shape is [B, C, G, N, F//G]

        # Print shapes for debugging
        print(f"adj.shape before matmul: {adj.shape}")
        print(f"x.shape before matmul: {x.shape}")

        # Graph convolution operation
        out = torch.matmul(adj, x)

        print(f"out.shape after matmul: {out.shape}")

        # Adjust output shape
        B, C, G, N, F = out.size()
        out = out.transpose(2, 3).reshape(B, C, N, -1)

        # Add bias if available
        if self.bias is not None:
            out = out.transpose(1, -1) + self.bias
            out = out.transpose(1, -1)

        return out


class DenseGINConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, eps=0, train_eps=True):
        super().__init__()

        # TODO: Multi-layer model
        self.mlp = Group_Linear(in_channels, out_channels, groups, bias=False)

        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)

    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj

    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [B, G, N, N]
        """
        B, C, N, _ = x.size()
        G = adj.size(0)

        # adj-norm
        adj = self.norm(adj, add_loop=False)

        # x: [B, C, G, N, F//G]
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)

        out = torch.matmul(adj, x)

        # DYNAMIC
        x_pre = x[:, :, :-1, ...]

        # out = x[:, :, 1:, ...] + x_pre
        out[:, :, 1:, ...] = out[:, :, 1:, ...] + x_pre
        # out = torch.cat( [x[:, :, 0, ...].unsqueeze(2), out], dim=2 )

        if add_loop:
            out = (1 + self.eps) * x + out

        # out: [B, C, G, N, F//G]
        out = self.mlp(out, True)

        # out: [B, C, N, F]
        C = out.size(1)
        out = out.transpose(2, 3).reshape(B, C, N, -1)

        return out


class Dense_TimeDiffPool2d(nn.Module):

    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()

        # TODO: add Normalization
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))

        self.re_param = Parameter(Tensor(kern_size, 1))

    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')

    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [B, G, N, N]
        """
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)

        # s: [ N^(l+1), N^l, 1, K ]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # TODO: fully-connect, how to decrease time complexity
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))

        return out, out_adj