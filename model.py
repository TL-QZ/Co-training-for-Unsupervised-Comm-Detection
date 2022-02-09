# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Layers
from layers.Transformer_encoder import EncoderBlock


class TransformerEncoder(nn.Module):

    def __init__(self, raw_dim, input_dim, num_layers, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.init_linear = nn.Linear(raw_dim, input_dim)
        self.layers = nn.ModuleList([EncoderBlock(input_dim, num_heads, dim_feedforward, dropout)
                                     for _ in range(num_layers)])

    def forward(self, data, attn_bias=None, mask=None):

        # unwrap data features
        x = data.x
        adj = data.Adj
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
        inner_product = inner_product.masked_fill(adj == 1, -9e15)
        adjacency = torch.sigmoid(inner_product)

        return node_feature, adjacency
