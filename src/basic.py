from typing import Sequence

import functools as ft
import itertools as it # pyright: ignore
from inspect import signature

import torch.nn as nn
import torch.nn.functional as F


def MLP_rectangular(n_layers, dim_in, dim_hidden, dim_out, activation=nn.ELU, activation_out=None):
    if activation_out is None:
        activation_out = activation
    if "dim" in signature(activation).parameters:
         #trick to pass dim if needed
        activation = ft.partial(activation, dim=-1)
    if "dim" in signature(activation_out).parameters:
         #trick to pass dim if needed
        activation_out = ft.partial(activation_out, dim=-1)

    layers = []
    n_in = dim_in
    n_out = dim_hidden
    for i in range(n_layers - 1):
        layers.append(nn.Linear(n_in, n_out))
        layers.append(activation())
        n_in = n_out
    n_out = dim_out
    layers.append(nn.Linear(n_in, n_out))
    layers.append(activation_out())
    return nn.Sequential(*layers)

class MLP(nn.Sequential):
    def __init__(self, *dims: int, batchnorm=True, end_activation=False, activation:type[nn.Module]=nn.ReLU):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP must have at least 2 dimensions (input and output))")
        for i, input_dim, output_dim in zip(it.count(), dims, dims[1:]):
            self.add_module(f"linear_{i}", nn.Linear(input_dim, output_dim))
            if i == len(dims)-2 and not end_activation:
                continue #skip batchnorm and activation at last layer
            if batchnorm:
                self.add_module(f"norm_{i}", nn.BatchNorm1d(output_dim))
            self.add_module(f"activation_{i}", activation())

    def forward(self, x):
        *batch, input_dim = x.shape
        x = x.view(-1, input_dim)
        y = super().forward(x)
        return y.view(*batch, -1)

class MultiInputLinear(nn.Module):
    """
    MultiInputLinear is a module that performs a 
    linear transformation on the concatenation of multiple inputs.
    if 

    ..code-block:: python
        mil = MultiInputLinear((input_dim1, input_dim2, input_dim3), output_dim)
        linear = nn.Linear(input_dim1 + input_dim2 + input_dim3 , output_dim)

    `mil(a, b, c)` is strictly equivalent to `linear(torch.cat([a, b, c], dim=-1))`

    but there is a computational advantage: the `torch.cat([a, b, c], dim=-1)` 
    operation is never done in memory
    and in particular if a, b and c don’t have the same batch shape (the batch shapes should be broadcastable)
    then this is computationally more efficient
    """

    linears: nn.ModuleList

    def __init__(self, input_dims: Sequence[int], output_dim: int, bias=True):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=bias) for input_dim in input_dims
        ])

    def forward(self, *inputs):
        """
        :param inputs: a list of tensors of shape (batch_size, input_dim)
        :return: a tensor of shape (batch_size, output_dim)
        """
        assert len(inputs) == len(self.linears)
        return sum(linear(input) for linear, input in zip(self.linears, inputs))