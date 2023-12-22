from typing import Callable
import functools as ft
import itertools as it # pyright: ignore
from inspect import signature

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualShortcut(nn.Module):
    """Residual shortcut as used in ResNet.

    A module that adds the input to the output of another module.
    So if inner module is f, the output is x + f(x).
    
    This is useful to implement residual blocks as they were 
    originally used in resnet (and are used in a lot of modern architectures)
    """
    inner_module: nn.Module
    def __init__(self, inner_module):
        super().__init__()
        self.inner_module = inner_module
    
    def forward(self, x):
        y = self.inner_module(x)
        return x + y
    
class Repeat(nn.Sequential):
    """A module that repeats copies of a module n times.
    Takes a module fatcory as input (a function that returns a module),
    and creates n modules by calling the factory n times.
    and puts them in a sequential module.

    This is for convenience, when an architecture reads like
    " block defined above × 5"
    """
    def __init__(self, n:int , module_factory: Callable[[], nn.Module]):
        super().__init__(*[module_factory() for _ in range(n)])

class Split(nn.Module):
    """
    A module that splits the input into n parts and returns them as a tuple.
    """

    def __init__(self, n=2):
        super().__init__()
        self.n = n

    def forward(self, x):
        return tuple(x.chunk(self.n, dim=-1))
    

class Sequential(nn.Module):
    """Sequential implementation that allows for more than one input and output."""
    sub_modules: nn.ModuleList
    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.sub_modules = nn.ModuleList(modules) 

    def forward(self, *args, **kwargs):
        output = self.sub_modules[0](*args, **kwargs)
        for module in self.sub_modules[1:]:
            prev_output = output
            match prev_output, signature(module.forward).parameters.items():
                case x, [(_, p),] if p.kind in [Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD]:
                    #only one parameter, and it's positional
                    output = module(x)
                case {**kwargs}, _:
                    # the dict is interpreted as kwargs
                    output = module(**kwargs)
                case t, _ if hasattr(t, "_asdict"):
                    # the namedtuple is interpreted as kwargs
                    output = module(**prev_output._asdict())
                case (tuple(args), {**kwargs}), _:
                    output = module(*args, **kwargs)
                case tuple(args), _  :
                    output = module(*args)
                case x, _:
                    output = module(x)
        return output

class Concat(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)