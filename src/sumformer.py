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
        #return here
        x_hiddendim = self.input_linear(x.data) #n_nodes_total, hidden_dim
        
        psi_input = x_hiddendim + sigma_hiddendim[x.batch, :] #n_nodes_total, hidden_dim
        psi_input = F.leaky_relu(psi_input) #n_nodes_total, hidden_dim

        return Batch.from_other(self.psi(psi_input), x) #n_nodes_total, input_dim

    #add getter method for aggregation?

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
    def __init__(self, input_dim, embed_dim, mlp_hdim, mlp_out_dim, mlp_layers, phi_hdim, phi_out_dim, phi_layers, batchnorm=False, mean=False, use_max=False, activation=nn.LeakyReLU):
        super(PointEncoder, self).__init__()

        self.mlp = MLP(input_dim, *[mlp_hdim]*mlp_layers, embed_dim, batchnorm=batchnorm, activation = activation)
        self.phi = MLP(embed_dim, *[phi_hdim]*phi_layers, phi_out_dim, batchnorm = batchnorm, activation = activation)

        self.mean = mean
        self.max = use_max
    
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

    def get_approx_chull(self, x: Tensor|Batch):
        
        hulls = []
        start = 0
        for num in batch.n_nodes:
            end = start + num
            ptset = batch.data[start:end]
            ptset_probs = probabilities.data[start:end]
            # print(f'probs (from softmax): {ptset_probs}')
            hull_approx = torch.mm(ptset_probs.T, ptset)
            hulls.append(hull_approx)
            start = end
        
        return hulls


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
        out = out.reshape(batch_len * self.output_dim, 2) #need to update, hardcoding for 2d
       
        return out #return as a tensor

        def get_approx_chull(self, x: Tensor|Batch):
            return x


class ConvexHullEncoder(nn.Module):
    def __init__(self, input_dim, encoder_depth, encoder_width, encoder_output_dim,
                processor_depth, processor_embedding_dim, processor_hidden_dim, processor_output_dim, **config):
        super(ConvexHullEncoder, self).__init__()

        self.encoder = MLP(input_dim, *[encoder_width]*encoder_depth, encoder_output_dim, 
                            batchnorm=False, activation=nn.LeakyReLU)
        self.processor = ConvexHullNN(depth=processor_depth, embedding_dim=processor_embedding_dim, 
                                                    hidden_dim=processor_hidden_dim, input_dim=encoder_output_dim, 
                                                    output_dim=processor_output_dim)

    def forward(self, x):
        out = self.encoder(x)
        out = self.processor(out)
        return out



class EncoderProcessDecoder(nn.Module):
    def __init__(self, *args):
        super(EncoderProcessDecoder, self).__init__()

        self.encoder = globals()[encoder_layer](*encoder_configs)
        self.processor = globals()[processor_layer](*processor_configs)
        #self.include_decoder = include_decoder
        print(decoder_configs)
        if include_decoder:
            self.decoder = globals()[decoder_layer](*decoder_configs)


    def forward(self, include_decoder, x: Tensor|Batch):
        out = self.encoder(x) #check if this is desired behavior
        out = self.processor(out)
        out = self.processor.get_approx_chull(out)
        out = self.decoder(out)