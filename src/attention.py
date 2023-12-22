import functools as ft # pyright: ignore
import itertools as it # pyright: ignore
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .combinators import ResidualShortcut
from .basic import MLP

# mostly copied and adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# for some reason, the code in pytorch wasn’t adapted to my use 
# (iirc the way they handle the dimensions is different from the way I do
# or maybe the projections)
# so I had to use this
# def scaled_dot_product(q, k, v, mask=None):
#     d_k = q.size()[-1]
#     attn_logits = torch.matmul(q, k.transpose(-2, -1))
#     attn_logits = attn_logits / math.sqrt(d_k)
#     if mask is not None:
#         attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
#     attention = F.softmax(attn_logits, dim=-1)
#     values = torch.matmul(attention, v)
#     return values, attention

class MultiheadAttention(nn.Module):
    """
    Multihead Self-attention module.
    Has some limitations that I could workaround, mostly the 
    value dimension is the same as the key/query dimension
    """

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, input_dim) #get back our original dimension

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        #self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None):
        *batch, seq_length, in_dim = x.shape
        assert in_dim == self.input_dim, "Input dimension does not match."

        qkv = self.qkv_proj(x) #(*batch, seq_length, 3*embed_dim)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(*batch, seq_length, self.num_heads, self.head_dim, 3)
        # *batch, seq_length, num_heads, head_dim, 3
        #using einsum instead of permute bc I find it more readable
        qkv = torch.einsum("...lhit->...hlit", qkv)#*b, num_heads, seq_length, head_dim, 3
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2] #(*batch,num_heads, seq_length head_dim)

        # Determine value outputs
        values = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # *batch, num_heads, seq_length, head_dim
        values = torch.einsum("...hli->...lhi", values) #*batch, seq_length, num_heads, head_dim
        values = values.reshape(*batch, seq_length, self.embed_dim)
        o = self.o_proj(values)#*batch, seq_length, input_dim
        return o

class TransformerBlock(nn.Sequential):
    
    def __init__(self, input_dim, embed_dim=250, feed_forward_depth=1, activation=nn.LeakyReLU):
        super().__init__()
        
        self.add_module("attention", ResidualShortcut(
            MultiheadAttention(input_dim, embed_dim=embed_dim, num_heads=5)
        ))
        
        self.add_module("norm1", nn.LayerNorm(input_dim))
        
        self.add_module("feed_forward", ResidualShortcut(
            
            MLP(input_dim, *[embed_dim] * feed_forward_depth, input_dim, batchnorm=False, 
                activation=activation)
            ))
        
        self.add_module("norm2", nn.LayerNorm(input_dim))