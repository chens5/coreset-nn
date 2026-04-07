import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from .basic import MLP
from .data_representation import Batch, BatchIndicator
from rotary_embedding_torch import RotaryEmbedding
from torch_geometric.nn.pool import global_max_pool



class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, num_heads=8, **kwargs):
        super(Transformer, self).__init__()
       
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            batch_first=True)
            for _ in range(num_layers)
        ])

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                         nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, return_attn=False):
        data = x.data
        indicator = x.n_nodes
        order = x.order

        if return_attn == True:
            attention_weights = []

            for layer in self.layers:
                data, weights = layer.self_attn(data, data, data, need_weights=True)
                attention_weights.append(weights)

            return Batch.from_batched(data=data, order=order, n_nodes=indicator), attention_weights

        else:
            data = self.encoder(data)
            return Batch.from_batched(data=data, order=order, n_nodes=indicator)

# class RotaryTransformer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, num_heads=8, **kwargs):
#         super(Transformer, self).__init__()
       
#         self.layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
#             batch_first=True)
#             for _ in range(num_layers)
#         ])

#         encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
#                          nhead=num_heads, dim_feedforward=hidden_dim)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

#     def forward(self, x, return_attn=False):
#         data = x.data
#         indicator = x.n_nodes
#         order = x.order

#         if return_attn == True:
#             attention_weights = []

#             for layer in self.layers:
#                 data, weights = layer.self_attn(data, data, data, need_weights=True)
#                 attention_weights.append(weights)

#             return Batch.from_batched(data=data, order=order, n_nodes=indicator), attention_weights

#         else:
#             #apply rotary encoding here
#             data = self.encoder(data)
#             return Batch.from_batched(data=data, order=order, n_nodes=indicator)


class ConvexHullNNTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, transformer_output_dim, 
                 output_dim, depth, num_heads, return_attn=False, *args):
        super().__init__()
        self.return_attn = return_attn
        self.initial = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.transformer = Transformer(input_dim=embedding_dim, hidden_dim=hidden_dim, 
                                       output_dim=transformer_output_dim, num_layers=depth, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.initial(x)
        if self.return_attn:
            out, attention_maps = self.transformer(out, self.return_attn)  # Retrieve both output and attention maps
            out = self.mlp(out)
            return out, attention_maps  # Return attention maps along with the model output
        else:
            out = self.transformer(out, self.return_attn)
            out = self.mlp(out)
            return out
    def get_approx_chull(self, probabilities, x: Tensor|Batch):

        n = x.n_nodes[0].item() #todo: assuming constant ptset size throughout batch
        
        probabilities = probabilities.view(-1, n, probabilities.data.size(-1))
        probabilities = F.softmax(probabilities, dim=1)
        probabilities = probabilities.view(-1, probabilities.data.size(-1))
        
        hulls = []
        start = 0
        for num in x.n_nodes:
            end = start + num
            ptset = x.data[start:end]
            ptset_probs = probabilities.data[start:end]
            # print(f'probs (from softmax): {ptset_probs}')
            hull_approx = torch.mm(ptset_probs.T, ptset)
            hulls.append(hull_approx)
            start = end
          
   
        # # out =  Batch.from_batched(hulls, n_nodes = x.n_nodes, order = 1)
        # # print(out.data.shape)
        # print(len(hulls))
        # print(len(hulls[0]))
        return hulls

class ConvexHullNNTransformer_L1(ConvexHullNNTransformer):
    def get_approx_chull(self, probabilities, x: Tensor|Batch):

        n = x.n_nodes[0].item() #todo: assuming constant ptset size throughout batch
        
        probabilities = probabilities.view(-1, n, probabilities.data.size(-1))
        probabilities = F.leaky_relu(probabilities) ##only for L1 norm
        probabilities = F.normalize(probabilities, p = 1.0, dim = 1)
        probabilities = probabilities.view(-1, probabilities.data.size(-1))
        
        hulls = []
        start = 0
        for num in x.n_nodes:
            end = start + num
            ptset = x.data[start:end]
            ptset_probs = probabilities.data[start:end]
            # print(f'probs (from softmax): {ptset_probs}')
            hull_approx = torch.mm(ptset_probs.T, ptset)
            hulls.append(hull_approx)
            start = end
          
        return hulls


class ConvexHullNNTransformer_Direct(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, transformer_output_dim, 
                 output_dim, depth, num_heads, return_attn=False, *args):
        super().__init__()
        self.initial = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.transformer = Transformer(input_dim=embedding_dim, hidden_dim=hidden_dim, 
                                       output_dim=transformer_output_dim, num_layers=depth, num_heads=num_heads)
        # self.ff1 = nn.Linear(in_features = embedding_dim, out_features=output_dim) #old feedforward
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim * input_dim)
        )
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.transformer_output_dim = transformer_output_dim

    def forward(self, x: Tensor|Batch):
        # print(type(x))
        out = self.initial(x)
        out = self.transformer(out)


        # print(out.data.shape)

        out = global_max_pool(x = out.data, batch = x.batch1)#.squeeze(dim=0) ##pooling setwise
        n_nodes_out = torch.full_like(x.n_nodes, fill_value = self.output_dim)


        out =  Batch.from_batched(out, n_nodes = n_nodes_out, order = 1)
        # print(type(out))

        out = self.mlp(out)

        # print(f'before reshape: {out.data.shape}') #torch.Size([B, K * input_dim])

        out = out.view(-1, self.output_dim, self.input_dim)  # [B, K, input_dim]
        out = out.view(-1, self.input_dim)  # [B * K, input_dim]

        # print(f'after reshape: {out.data.shape}')

        return out
        

    def get_approx_chull(self, x: Tensor|Batch):

        return x
       

class ShapeFittingDirectTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, transformer_output_dim, 
                 output_dim, depth, num_heads, return_attn=False, *args):
        super().__init__()
        self.initial = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.transformer = Transformer(input_dim=embedding_dim, hidden_dim=hidden_dim, 
                                       output_dim=transformer_output_dim, num_layers=depth, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, x: Tensor|Batch):
        # print(type(x))
        out = self.initial(x)
        out = self.transformer(out)

        # n_nodes_out = torch.full_like(x.n_nodes, fill_value = self.output_dim)

        out = global_max_pool(x = out.data, batch = x.batch1).squeeze(dim=0) ##pooling setwise
        # out =  Batch.from_batched(out, n_nodes = n_nodes_out, order = 1)
        # print(type(out))

        out = self.mlp(out)

        return out



class RotaryMultiheadAttention(nn.Module):
    def __init__(self, dim, heads, rotary_emb, shuffle = False):
        super().__init__()
        self.shuffle = shuffle
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)



    def forward(self, x):
        ### reshaping
        N = x.n_nodes[0].item()
        dim = x.data.shape[-1]
        B = x.data.shape[0] // N
        x_reshaped = x.data.view(B, N, dim)

        qkv = self.to_qkv(x_reshaped)
        q, k, v = qkv.chunk(3, dim=-1)

        q_rotated = []
        k_rotated = []
        v_shuffled = []

        for i in range(B):
            if self.shuffle:
                # Setwise shuffle
                perm = torch.randperm(N, device=x.data.device)
                q_set = q[i][perm].unsqueeze(0)  # (1, N, head_dim * heads)
                k_set = k[i][perm].unsqueeze(0)
                v_set = v[i][perm]               # no need to unsqueeze for v
            else:
                q_set = q[i].unsqueeze(0)  # (1, N, head_dim * heads)
                k_set = k[i].unsqueeze(0)
                v_set = v[i]   

            # Apply rotary embeddings
            q_rot = self.rotary_emb.rotate_queries_or_keys(q_set)
            k_rot = self.rotary_emb.rotate_queries_or_keys(k_set)

            q_rotated.append(q_rot.squeeze(0))
            k_rotated.append(k_rot.squeeze(0))
            v_shuffled.append(v_set)

        q = torch.stack(q_rotated, dim=0)
        k = torch.stack(k_rotated, dim=0)
        v = torch.stack(v_shuffled, dim=0)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        out = torch.matmul(attn_weights, v)
        out = self.out_proj(out)

        out_flat = out.reshape(B * N, dim)
        return Batch.from_other(out_flat, x)


class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim = d_model // nhead)
        self.self_attn = RotaryMultiheadAttention(d_model, nhead,
                                                     rotary_emb = self.rotary_emb)


        # Feedforward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        # Self-attention block
        src2 = self.self_attn(src)
        src = self.norm1(src)

        # Feedforward block
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src
        src = self.norm2(src)
        
        return src

class RotaryTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, num_heads=8, **kwargs):
        super(RotaryTransformer, self).__init__()

        self.layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x, return_attn=False):
        data = x#.data
        indicator = x.n_nodes
        order = x.order

        n = indicator[0]

        if return_attn:
            attention_weights = []

            for layer in self.layers:
                data, weights = layer(data, return_attn=True)
                attention_weights.append(weights)

            return data, attention_weights

        else:
            for layer in self.layers:
                data = layer(data)
                
            return data


class ConvexHullNNRotaryTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, transformer_output_dim, 
                 output_dim, depth, num_heads, return_attn=False, *args):
        super().__init__()
        self.return_attn = return_attn
        self.initial = nn.Linear(in_features=input_dim, out_features=embedding_dim)
        self.transformer = RotaryTransformer(input_dim=embedding_dim, hidden_dim=hidden_dim, 
                                       output_dim=transformer_output_dim, num_layers=depth, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.initial(x)
        if self.return_attn:
            out, attention_maps = self.transformer(out, self.return_attn)  # Retrieve both output and attention maps
            out = self.mlp(out)
            return out, attention_maps  # Return attention maps along with the model output
        else:
            out = self.transformer(out, self.return_attn)
            out = self.mlp(out)
            return out

            
    def get_approx_chull(self, probabilities, x: Tensor|Batch):

        n = x.n_nodes[0].item() #todo: assuming constant ptset size throughout batch
        
        probabilities = probabilities.view(-1, n, probabilities.data.size(-1))
        probabilities = F.softmax(probabilities, dim=1)
        probabilities = probabilities.view(-1, probabilities.data.size(-1))
        
        hulls = []
        start = 0
        for num in x.n_nodes:
            end = start + num
            ptset = x.data[start:end]
            ptset_probs = probabilities.data[start:end]
            # print(f'probs (from softmax): {ptset_probs}')
            hull_approx = torch.mm(ptset_probs.T, ptset)
            hulls.append(hull_approx)
            start = end
          
   
        # # out =  Batch.from_batched(hulls, n_nodes = x.n_nodes, order = 1)
        # # print(out.data.shape)
        # print(len(hulls))
        # print(len(hulls[0]))
        return hulls


    


class ConvexHullEncoderTransformer(nn.Module):
    def __init__(self, input_dim, encoder_depth, encoder_width, encoder_output_dim,
                transformer_depth, num_heads, transformer_od, processor_embedding_dim, 
                processor_hidden_dim, processor_output_dim, return_attn=False, **config):
        super(ConvexHullEncoderTransformer, self).__init__()

        self.return_attn = return_attn

        self.encoder = MLP(input_dim, *[encoder_width]*encoder_depth, encoder_output_dim, 
                            batchnorm=False, activation=nn.LeakyReLU)
        self.processor = ConvexHullNNTransformer(input_dim = encoder_output_dim, embedding_dim=processor_embedding_dim, hidden_dim=processor_hidden_dim, 
        transformer_output_dim=transformer_od, output_dim=processor_output_dim, depth=transformer_depth, num_heads=num_heads,
        return_attn = self.return_attn)


    def forward(self, x):
        out = self.encoder(x)
       
        if self.return_attn == True:
            out, attn_maps = self.processor(out)
            return out, attn_maps
        else:
            out =  self.processor(out)
            return out
