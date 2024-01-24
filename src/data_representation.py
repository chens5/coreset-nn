"""
Batch support

Batched input should be given in the following format:
    input2 = (sum_i n_nodes_i^2 , n_features) float
        contains the features of all matrices flattened and concatenated
        must be in order (ie the batch2 vector must be in ascending order)
    batch2  = (sum_i n_nodes_i^2) long indicating which batch each nodes belongs to. 
        must be in ascending order.
        Optional, will be recomputed otherwise.
    n_nodes  = (batch_size) long indicating the number of nodes in each batch (optional, will be recomputed otherwise)

either batch2 or n_nodes must be given if the output is batched 
(otherwise it will be assumed not batched). 
If both are given, they must be consistent.
"""
from typing import Literal, Final, Callable, ClassVar, Any
from typing_extensions import Self
from logging import getLogger; log = getLogger(__name__)
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
from dataclasses import dataclass

def ptr_from_sizes(size: Tensor) -> Tensor:
    """returns a ptr vector from a sizes vector"""
    ptr = torch.zeros(size.size(0) + 1, dtype=torch.long)
    ptr[1:] = size.cumsum(0)
    return ptr

def all_equal(*args):
    """
    tests whether all the passed arguments are equal. 
    useful for checking dimensions for a lot of vectors for ex
    """
    match args:
        case ():
            return True
        case (x0, *rest):
            return all(i == x0 for i in rest)

@dataclass
class BatchIndicator:
    """Represents a batch of set or 2-tensor data"""

    n_nodes: Tensor
    """n_nodes: (batch_size) long indicating the number of nodes in each batch"""
    n_edges: Tensor|None = None
    """n_edges: (batch_size) long indicating the number of edges in each batch (optional, will be recomputed otherwise)"""

    ptr1: Tensor|None = None
    """ptr1: (batch_size + 1) long indicating the offset of each batch in the batch1 vector (optional, will be recomputed otherwise)"""
    ptr2: Tensor|None = None
    """ptr2: (batch_size + 1) long indicating the offset of each batch in the batch2 vector (optional, will be recomputed otherwise)"""

    batch1: Tensor|None = None
    """batch1: (sum_i n_nodes_i) long indicating the batch of each node (optional, will be recomputed otherwise)"""
    batch2: Tensor|None = None
    """batch2: (sum_i n_nodes_i^2) long indicating the batch of each edge (optional, will be recomputed otherwise)"""

    diagonal: Tensor|None = None
    """diagonal: (sum_i n_nodes_i) long indicating the diagonal of each matrix (optional, will be recomputed otherwise)"""
    transpose_indices: Tensor|None = None

    def __getattr__(self, name) -> Callable[[], Any]:
        if not name.startswith("get_"):
            raise AttributeError(f"BatchIndicator has no attribute {name}")
        attr_name = name[4:]
        if not hasattr(self, attr_name):
            raise AttributeError(f"BatchIndicator has no attribute {attr_name}")

        def get_attr():
            attr = getattr(self, attr_name)
            if attr is None:
                with torch.no_grad(), self.n_nodes.device:
                    attr = getattr(self, f"_compute_{attr_name}")()
                setattr(self, attr_name, attr)
            return attr
        return get_attr

    def _compute_n_edges(self) -> LongTensor:
        return self.n_nodes.square()#type: ignore

    def _compute_batch1(self):
        return torch.cat([i * torch.ones(n_nodes_i.item(), dtype=torch.long) for i, n_nodes_i in enumerate(self.n_nodes)])#type: ignore

    def _compute_batch2(self):
        return torch.cat([i * torch.ones(n_nodes_i.item() * n_nodes_i.item() , dtype=torch.long) for i, n_nodes_i in enumerate(self.n_nodes)])#type: ignore

    def _compute_ptr1(self):
        return ptr_from_sizes(self.n_nodes)

    def _compute_ptr2(self):
        return ptr_from_sizes(self.get_n_edges())

    def _compute_diagonal(self):
        diagonals = []
        for start, end, n_nodes in zip(self.get_ptr2(), self.get_ptr2()[1:], self.n_nodes):
            start = start.item()
            end= end.item()
            n_nodes = n_nodes.item()
            diagonal = torch.arange(start, end + 1, n_nodes + 1 )
            diagonals.append(diagonal)
        return torch.cat(diagonals, dim=0)

    def _compute_transpose_indices(self):
        indices = []
        for start, end, n_nodes in zip(self.get_ptr2(), self.get_ptr2()[1:], self.n_nodes):
            start = start.item()
            end= end.item()
            n_nodes = n_nodes.item()
            indices.append(torch.arange(start, end).reshape(n_nodes, n_nodes).T.reshape(-1))
        return torch.cat(indices, dim=0)

    def get_batch_size(self):
        return self.n_nodes.size(0)

    def __eq__(self, other: Self):
        return (self.n_nodes == other.n_nodes).all().item()

    def to(self, *args, **kwargs):
        return BatchIndicator(self.n_nodes.to(*args, **kwargs))


    def grid(self):
        ptr1 = self.get_ptr1()
        subgrids_x = []
        subgrids_y = []
        for n, ptr in zip(self.n_nodes, ptr1):
            n = n.item()
            ptr = ptr.item()
            arange = torch.arange(n, device = ptr1.device)
            arange = arange + ptr
            arange_x = arange[:, None].expand(n, n).reshape(-1)
            arange_y = arange[None, :].expand(n, n).reshape(-1)
            subgrids_x.append(arange_x)
            subgrids_y.append(arange_y)
        return torch.cat(subgrids_x, dim=0), torch.cat(subgrids_y, dim=0)

@dataclass
class Batch:
    data: Tensor
    """data: (sum_i n_nodes_i^order, n_features) float tensor"""
    order: Literal[1, 2]
    """1 or 2, depending on whether the data is a 1-tensor or a 2-tensor"""

    indicator: BatchIndicator

    @property
    def n_features(self):
        return self.data.shape[-1]

    @property
    def n(self):
        match self.order:
            case 1: return self.n_nodes
            case 2: return self.n_edges

    @property
    def batch(self):
        match self.order:
            case 1: return self.batch1
            case 2: return self.batch2
    @property
    def ptr(self):
        match self.order:
            case 1: return self.ptr1
            case 2: return self.ptr2

    @property
    def n_nodes(self) -> Tensor:
        return self.indicator.n_nodes
    @property
    def n_edges(self)-> Tensor:
        return self.indicator.get_n_edges()
        
    @property
    def batch1(self)-> Tensor:
        return self.indicator.get_batch1()

    @property
    def batch2(self)-> Tensor:
        return self.indicator.get_batch2()

    @property
    def ptr1(self)-> Tensor:
        return self.indicator.get_ptr1()

    @property
    def ptr2(self)-> Tensor:
        return self.indicator.get_ptr2()

    @property
    def batch_size(self) -> int:
        return self.indicator.get_batch_size()

    @property
    def diagonal(self) -> Tensor:
        return self.indicator.get_diagonal()

    @classmethod
    def from_unbatched(cls, data: Tensor):
        if data.ndim == 2:
            order = 1
            n_nodes, n_features = data.shape
            n_nodes = torch.as_tensor([n_nodes], dtype=torch.long)
            batched_data = data #(n_nodes, n_features)
        else:
            assert data.ndim == 3
            order = 2
            n_nodes, n_nodes_, n_features = data.shape
            assert n_nodes == n_nodes_
            n_nodes = torch.as_tensor([n_nodes], dtype=torch.long)
            batched_data = data

        return cls(data=batched_data, 
                   order=order, 
                   indicator=BatchIndicator(n_nodes=n_nodes))

    @classmethod
    def from_batched(cls, data: Tensor, n_nodes: Tensor, order: Literal[1, 2]):
        return cls(data=data,
                   order=order,
                   indicator=BatchIndicator(n_nodes=n_nodes))

    @classmethod
    def from_other(cls, data: Tensor, other: Self, order=None):
        return cls(data=data,
                   order=order or other.order,
                   indicator=other.indicator)

    @classmethod
    def from_list(cls, data: list[Tensor], order: Literal[1, 2]):
        n_nodes = torch.as_tensor([d.shape[0] for d in data], dtype=torch.long)
        all_n_features = []
        all_n_nodes = []
        if order == 1:
            for d in data:
                assert d.ndim == 2
                n_nodes,  n_features_ = d.shape
                all_n_nodes.append(n_nodes)
                all_n_features.append(n_features_)
        elif order == 2:
            for d in data:
                assert d.ndim == 3
                n_nodes, n_nodes_, n_features_ = d.shape
                assert n_nodes == n_nodes_
                all_n_nodes.append(n_nodes)
                all_n_features.append(n_features_)
            data = [d.reshape(-1, d.shape[-1]) for d in data]
        assert all_equal(*all_n_features)
        concat_data = torch.cat(data, dim=0)
        all_n_nodes = torch.as_tensor(all_n_nodes, dtype=torch.long)
        return cls.from_batched(concat_data, all_n_nodes, order)

    def batch_split(self):
        ptr = self.ptr
        order = self.order
        
        for begin, end, n_nodes in zip(ptr, ptr[1:], self.n_nodes):
            begin = begin.item()
            end = end.item()
            n_nodes = n_nodes.item()
            d = self.data[begin:end, :]
            if order == 1:
                yield d
            else:
                yield d.reshape(n_nodes, n_nodes, self.n_features)#type: ignore


    def same_batch(self, other: Self):
        return self.indicator == other.indicator

    def normalize(self):
        l1norm = torch.norm(self.data, p=1, dim=1)
        return self.data/l1norm.unsqueeze(-1)

    def multiply_prob(self, p):
        return 0

    HANDLED_FUNCTIONS: ClassVar[dict[Callable, None|Callable[..., bool]]] = {
        torch.add: None, 
        F.layer_norm: None, 
        torch.square: None, 
        torch.sub: None, 
        F.linear: None, 
        F.leaky_relu: None,
        F.relu: None,
        F.sigmoid: None,
        torch.sigmoid: None,
        torch.gt: None, 
        torch.mul: None,
        F.softmax: None
        }

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Magic code that makes torch functions work directly on Batch objects"""
        if kwargs is None:
            kwargs = {}
        new_args = []
        new_kwargs = {}
        indicators = []
        orders = []

        ########################### check if the function is supported
        if func not in cls.HANDLED_FUNCTIONS:
            log.warning(f"Function {func} called on a Batch object directly, but not in the supported list")
            return NotImplemented
        is_handled = cls.HANDLED_FUNCTIONS[func]
        if callable(is_handled) and not is_handled(*args, **kwargs):
            log.warning(f"Function {func} called on a Batch object directly, but not supported with the given arguments")
            return NotImplemented

        ############################# extract the data from the batches
        for arg in args:
            if not isinstance(arg, Batch):
                new_args.append(arg)
                continue
            indicators.append(arg.indicator)
            orders.append(arg.order)
            new_args.append(arg.data)
        for k, v in kwargs.items():
            if not isinstance(v, Batch):
                new_kwargs[k] = v
                continue
            indicators.append(v.indicator)
            orders.append(v.order)
            new_kwargs[k] = v.data

        
        #################################check that all batches have the same structure
        if not all_equal(indicators):
            raise ValueError(f"Calling function {func} on Batch elements with different structure\n maybe you want to use the data attribute?")
        if not all_equal(orders):
            raise ValueError(f"Calling function {func} on Batch elements with different order\n maybe you want to use the data attribute?")
        indicator = indicators[0]
        order = orders[0]
        
        ##############################actual call
        result_data = func(*new_args, **new_kwargs)

        ##############################return
        assert result_data.ndim == 2
        return cls(data=result_data, order=order, indicator=indicator)

    def __add__(self, b):
        return torch.add(self, b)#type: ignore

    def __sub__(self, b):
        return torch.sub(self, b)#type: ignore

    def __mul__(self, b):
        return torch.mul(self, b)#type: ignore

    def to(self, *args, **kwargs):
        return type(self)(data=self.data.to(*args, **kwargs), order=self.order, indicator=self.indicator.to(*args, **kwargs))

    @property
    def T(self):
        """Transpose"""
        assert self.order==2
        transpose_indices = self.indicator.get_transpose_indices()
        return Batch(self.data[transpose_indices], self.order, self.indicator)