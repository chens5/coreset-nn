import torch.nn as nn
import torch
from torch import nn, Tensor
from .basic import MLP
from .data_representation import Batch, BatchIndicator

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, num_heads=8, **kwargs):
        super(Transformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                        nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x: Tensor|Batch):
        data = x.data
        indicator = x.n_nodes
        order = x.order

        data = self.encoder(data)

        return Batch.from_batched(data=data, order=order, n_nodes=indicator)


class ConvexHullNNTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, transformer_output_dim, 
                output_dim, depth, num_heads, *args):
        super().__init__()
        self.initial = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.transformer = Transformer(input_dim = embedding_dim, hidden_dim=hidden_dim, 
                            output_dim = transformer_output_dim, num_layers=depth, num_heads=num_heads)
       

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
       
    def forward(self, x: Tensor|Batch):
        out = self.initial(x)
        out = self.transformer(out)
        # print(f'size after transformer {x.data.shape}')
        out = self.mlp(out)

        # print(f'out data shape (forward): {out.data.shape}')
        
        return out
