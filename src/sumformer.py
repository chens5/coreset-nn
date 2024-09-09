"""
Batched version of the sumformer
"""
from typing import Literal
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# from ml_lib.models.layers import MLP, Repeat, ResidualShortcut
from .basic import MLP
from .combinators import ResidualShortcut, Repeat
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.nn.pool import global_max_pool
from .data_representation import Batch


class GlobalEmbedding(nn.Module):

    input_dim: int
    embed_dim: int

    mlp: MLP
    r"""The MLP that changes the input features to be summed (\phi in the paper)"""
    activation: nn.Module
    r"""The last activation after that MLP"""
    aggregation: nn.Module
    r"""The aggregation function (sum or mean. resolved using torch_geometric.nn.resolver.aggregation_resolver, so the choices are the same as in torch_geometric.nn.aggr.Multi)"""

    def __init__(self, input_dim, embed_dim, hidden_dim=256, n_layers = 3, aggregation:str = "mean", aggregation_args={}):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.mlp = MLP(input_dim, *[hidden_dim]*n_layers, embed_dim, batchnorm=False, activation=nn.LeakyReLU)
        self.activation = nn.LeakyReLU()
        if "multi" in aggregation.lower(): 
            aggregation_args["mode"]= "proj"
            aggregation_args["mode_kwargs"] = {"in_channels": embed_dim, "out_channels": embed_dim, **aggregation_args.get("mode_kwargs", {})}
        self.aggregation = aggregation_resolver(aggregation, **aggregation_args)

    def forward(self, x: Batch):
        node_embeddings = self.activation(self.mlp(x.data)) #n_nodes_total, key_dim
        return self.aggregation(node_embeddings, ptr=x.ptr) #batch_size, key_dim


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

    aggreg_linear: nn.Linear

    psi: MLP

    def __init__(self, input_dim, hidden_dim=512, key_dim = 256 , aggregation:str = "mean", aggregation_args={}, 
                 node_embed_n_layers=3, output_n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.global_embedding = GlobalEmbedding(
                input_dim=input_dim, embed_dim=key_dim, 
                hidden_dim=hidden_dim, n_layers=node_embed_n_layers, 
                aggregation=aggregation, aggregation_args=aggregation_args
        ) 

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.aggreg_linear = nn.Linear(key_dim, hidden_dim)
        self.psi = MLP(hidden_dim, *[hidden_dim]*output_n_layers, input_dim, 
                          batchnorm=False, activation=nn.LeakyReLU)

    def forward(self, x: Tensor|Batch):
        """This is a faster, equivalent formulation of the sumformer attention block.
        See my notes for the derivation (that i’ll transcribe to here at some point)

        Caution! This approximation may not be exact (but should still be universal)
        if the aggregation is not linear (ie sum or average).
        """
        if isinstance(x, Tensor): x = Batch.from_unbatched(x)
        assert isinstance(x, Batch)
        assert x.n_features == self.input_dim
        sigma = self.global_embedding(x)

        sigma_hiddendim = self.aggreg_linear(sigma) #batch_size, hidden_dim
        x_hiddendim = self.input_linear(x.data) #n_nodes_total, hidden_dim
        
        psi_input = x_hiddendim + sigma_hiddendim[x.batch, :] #n_nodes_total, hidden_dim
        psi_input = F.leaky_relu(psi_input) #n_nodes_total, hidden_dim

        return Batch.from_other(self.psi(psi_input), x) #n_nodes_total, input_dim

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

class PointEncoder(nn.Module):
    def __init__(self, dimension, mlp_params: dict, phi_params: dict, bn=False, mean=False,max=False, activation='relu'):
        super(PointEncoder, self).__init__()
        mlp_hdim = mlp_params['hidden']
        mlp_output = mlp_params['output']
        mlp_layers = mlp_params['layers']

        phi_hdim = phi_params['hidden']
        phi_layers = phi_params['layers']
        phi_output = phi_params['output']
        self.mlp = initialize_mlp(dimension, mlp_hdim, mlp_output, mlp_layers, activation=activation)
        self.phi = initialize_mlp(mlp_output, phi_hdim, phi_output, phi_layers, activation=activation)
        self.mean = mean
        self.max = max
    
    def forward(self, input):
        
        out = self.mlp(input)
        if self.mean:
            out = torch.mean(out, dim=0)
        elif self.max:
            out = torch.max(out, dim=0)[0]
        else:
            out = torch.sum(out, dim=0)
        out = self.phi(out)
        return out
    
class ConvexHullNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, depth, *args):
        super().__init__()
        self.initial = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.sumformer = Sumformer(num_blocks=depth, input_dim = embedding_dim, hidden_dim=hidden_dim)
        # self.ff1 = nn.Linear(in_features = embedding_dim, out_features=output_dim) #old feedforward
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
       
    def forward(self, x: Tensor|Batch):
        out = self.initial(x)
        out = self.sumformer(out)
        # out = self.ff1(out)
        # print(f'out data shape (initial): {out.data.shape}')
        # print(f'indicator shape (initial): {out.batch1.shape}')
        out = self.mlp(out)

        # print(f'out data shape (forward): {out.data.shape}')
        # print(f'indicator shape (forward): {out.batch1.shape}')

        return out


class ConvexHullNN_new(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, depth, n_layers, *args):
        super().__init__()
        self.output_dim = output_dim
        self.initial = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.sumformer = Sumformer(num_blocks=depth, input_dim = embedding_dim, hidden_dim=hidden_dim)

        self.mlp = MLP(embedding_dim, *[hidden_dim]*n_layers, embedding_dim, batchnorm=False, activation=nn.LeakyReLU)
        self.final = nn.Linear(in_features=embedding_dim, out_features=output_dim * 2)
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(2 * output_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_dim, output_dim * 2)
        # )

    def forward(self, x: Tensor|Batch):

        out = self.initial(x)
        out = self.sumformer(out) #shape: [50 * batch_size, 16]
        out = out.unsqueeze(0)  # Shape: [1, 50 * batch_size, 16]

    
        out = global_max_pool(x = out.data, batch = out.batch1).squeeze(dim=0) #shape: [batch_size, 2 * output_dim]
        # out =  Batch.from_batched(out, n_nodes = n_nodes, order = 1) # switch n nodes or delete this line

        
        out = self.mlp(out)
        out = self.final(out)

        batch_len = out.data.shape[0]
        out = out.reshape(batch_len * self.output_dim, 2)
       
        return out #return as a tensor


class encoder_process_decoder(nn.Module):
    def __init__(self, encoder, processor, decoder, input_dim, output_dim, *args):
        self.encoder = encoder()
        self.processor = processor()
        self.decoder = decoder() 

    def forward(self, x: Tensor|Batch):
        out = self.encoder(x)
        out = self.processor(x)
        out = self.decoder(x)