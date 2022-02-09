import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from Transformer_encoder import EncoderBlock

class GCN_Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 2):
        super(GCN_Encoder, self).__init__()

        assert k >= 2
        self.k = k
        self.conv = [GCNConv(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(GCNConv(2 * out_channels, 2 * out_channels))
        self.conv.append(GCNConv(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, mask=None):
        for i in range(self.k):
            x = F.elu(self.conv[i](x, edge_index))

        inner_product = x.mm(x.T)
        inner_product = inner_product.masked_fill(mask == 1, -9e15)
        adjacency = torch.sigmoid(inner_product)

        return x, adjacency

    def recons_loss(self, z, adj):
        return F.binary_cross_entropy(z, adj)


class MLPCluster(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, num_comm, dropout=0.0):
        super(MLPCluster, self).__init__()
        self.mlp1 = nn.Linear(input_dim, hid_dim)
        self.mlp2 = nn.Linear(hid_dim, num_comm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, mask=None):
        assignment = F.elu(self.mlp1(z))
        assignment = self.mlp2(self.dropout(assignment))
        assignment = F.softmax(assignment, dim=-1)

        adjacency_logit = torch.matmul(assignment, assignment.transpose(0,1))

        if mask is not None:
            adjacency_logit = adjacency_logit.masked_fill(mask == 1, 0)

        return assignment, adjacency_logit

    def recons_loss(self, c_sim, adj):
        return F.binary_cross_entropy(c_sim, adj)

    def reg_term(self, c, n_nodes, n_class):
        return (torch.linalg.norm(c.sum(1)) / n_nodes * math.sqrt(n_class) - 1)

    def modul_loss(self, c, adj):
        degrees = adj.sum(0)
        w = adj.sum()
        mod = torch.sum(torch.mul(c, torch.matmul(adj, c)))
        C_d = torch.matmul(degrees, c).unsqueeze(0)
        mod2 = torch.matmul(C_d, C_d.transpose(0,1))
        mod = mod + mod2 / w
        mod /= w
        return -mod

    def soft_cross_entropy(self, pred, true):
        return F.kl_div(pred, true, reduction='sum')


class TransformerEncoder(nn.Module):

    def __init__(self, raw_dim, input_dim, num_layers, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.init_linear = nn.Linear(raw_dim, input_dim)
        self.layers = nn.ModuleList([EncoderBlock(input_dim, num_heads, dim_feedforward, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, attn_bias=None, mask=None):

        # unwrap data features
        x = x.unsqueeze(0)
        if attn_bias is not None:
            graph_attn_bias = attn_bias
            graph_attn_bias = graph_attn_bias.unsqueeze(0).repeat(self.num_heads, 1, 1) # [n_heads, n_nodes, n_nodes]
        else:
            graph_attn_bias = None
        # Generate Node Embedding
        node_feature = F.elu(self.init_linear(x))
        for layer in self.layers:
            node_feature = layer(node_feature, graph_attn_bias, mask)

        inner_product = node_feature[0].mm(node_feature[0].T)
        inner_product = inner_product.masked_fill(mask == 1, -9e15)
        adjacency = torch.sigmoid(inner_product)

        return node_feature, adjacency

    def recons_loss(self, z_sim, adj):
        return F.binary_cross_entropy(z_sim, adj)