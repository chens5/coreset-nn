from typing import Literal
import torch
from torch import nn
import torch.nn.functional as F

from .basic import MLP
from .combinators import Repeat, ResidualShortcut


class AverageAggregator(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()
        self.keepdim = keepdim
    def forward(self, node_embeddings):
        return node_embeddings.mean(dim=-2, keepdim=self.keepdim)

class SumAggregator(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()
        self.keepdim = keepdim
        
    def forward(self, node_embeddings):
        return node_embeddings.sum(dim=-2, keepdim=self.keepdim)

class MaxAggregator(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()
        self.keepdim = keepdim
        
    def forward(self, node_embeddings):
        return node_embeddings.max(dim=-2, keepdim=self.keepdim)[0]

class CombinedAggregator(nn.Module):
    aggregators: nn.ModuleList
    combine: nn.Linear

    def __init__(self, average=True, sum=True, max=True, keepdim=False):
        super().__init__()
        self.aggregators = nn.ModuleList()
        if average:
            self.aggregators.append(AverageAggregator(keepdim=keepdim))
        if sum:
            self.aggregators.append(SumAggregator(keepdim=keepdim))
        if max:
            self.aggregators.append(MaxAggregator(keepdim=keepdim))
        self.combine = nn.Linear(len(self.aggregators), 1)

    def forward(self, node_embeddings):
        aggregations = [aggregator(node_embeddings) for aggregator in self.aggregators]
        aggregations = torch.stack(aggregations, dim=-1)
        return self.combine(aggregations).squeeze(-1)

class SumformerInnerBlock(nn.Module):
    """
    Here we implement the sumformer "attention" block (in quotes, because it is not really attention)
    It is permutation-equivariant
    and almost equivalent to a 2-step MPNN on a disconnected graph with a single witness node.

    We implement the MLP-sumformer (not the polynomial sumformer). Why?
        1. Simpler.
        2. They do say that polynomial tends to train better at the beginning, but the MLP catches up, 
            and it’s on synthetic functions which may perform very differently from real data 
            (and gives an advantage to the polynomial sumformer, which has fewer parameters).

    """

    input_dim: int
    """dimension of the input features"""

    key_dim: int
    """Dimesion of the aggregate sigma"""

    hidden_dim: int
    """Dimension of the hidden layers of the MLPs"""

    node_embed: MLP
    r"""The MLP that changes the input features to be summed (\phi in the paper)"""

    node_embed_activation: nn.Module
    r"""The last activation after that MLP"""

    aggregation: nn.Module
    r"""The aggregation function (sum or average)"""

    input_linear: nn.Linear
    
    aggreg_linear: nn.Linear

    psi: MLP

    def __init__(self, input_dim, hidden_dim=512, key_dim = 256 , aggregation:Literal["average","sum"] = "average", 
                 node_embed_n_layers=3, output_n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.node_embed = MLP(input_dim, *[hidden_dim]*node_embed_n_layers, key_dim, 
                              batchnorm=False, activation=nn.LeakyReLU)
        self.node_embed_activation = nn.LeakyReLU()
        match aggregation:
            case "average":
                self.aggregation = AverageAggregator(keepdim=True)
            case "sum":
                self.aggregation = SumAggregator(keepdim=True)
            case "max":
                self.aggregation = MaxAggregator(keepdim=True)
            case "combined":
                self.aggregation = CombinedAggregator(keepdim=True)
            case _:
                raise ValueError(f"Invalid aggregation {aggregation}")

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.aggreg_linear = nn.Linear(key_dim, hidden_dim)
        self.psi = MLP(hidden_dim, *[hidden_dim]*output_n_layers, input_dim, 
                          batchnorm=False, activation=nn.LeakyReLU)

    def forward(self, x):
        """This is a faster, equivalent formulation of the sumformer attention block.
        See my notes for the derivation (that i’ll transcribe to here at some point)

        Caution! This approximation may not be exact (but should still be universal)
        if the aggregation is not linear (ie sum or average).
        """
        *batch, n_nodes, n_features = x.shape
        assert n_features == self.input_dim
        node_embeddings = self.node_embed_activation(self.node_embed(x)) #*batch, n_nodes, key_dim
        sigma = self.aggregation(node_embeddings) #*batch, 1, key_dim
        
        psi_input = self.input_linear(x) + self.aggreg_linear(sigma) #*batch, n_nodes, hidden_dim

        return self.psi(psi_input) #*batch, n_nodes, input_dim

class SumformerBlock(nn.Sequential):
    """
    Inner SumformerBlock, with a residual connection and a layer norm.
    """
    
    def __init__(self, *block_args, **block_kwargs):
        super().__init__()
        block = SumformerInnerBlock(*block_args, **block_kwargs)
        residual_block = ResidualShortcut(block)
        self.add_module("residual_block", residual_block)
        self.add_module("norm", nn.LayerNorm(block.input_dim))

class Sumformer(Repeat):
    def __init__(self, num_blocks: int, *block_args, **block_kwargs):
        make_block = lambda: SumformerBlock(*block_args, **block_kwargs)
        super().__init__(num_blocks, make_block)