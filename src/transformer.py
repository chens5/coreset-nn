import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(Transformer, self).__init__()
    
    def forward(self, **kwargs):
        raise NotImplementedError("TODO: finish transformer implementation")
