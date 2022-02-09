# Standard libraries
import math

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v, attn_bias=None, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    if attn_bias is not None:
        attn_logits = attn_logits + attn_bias

    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    attn_ratio = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attn_ratio, v)
    return values, attn_ratio


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_bias=None, mask=None):
        batch_size, num_node, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, num_node, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, Node, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attn_ratio = scaled_dot_product(q, k, v, attn_bias, mask)

        values = values.permute(0, 2, 1, 3)  # [Batch, Node, Head, Dims]
        values = values.reshape(batch_size, num_node, embed_dim)
        o = self.o_proj(values)

        return o, attn_ratio


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ELU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, mask=None):
        # Attention
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, attn_bias, mask)
        x = x + self.dropout(attn_out)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x
