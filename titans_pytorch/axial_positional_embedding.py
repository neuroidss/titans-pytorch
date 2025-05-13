from __future__ import annotations

from math import ceil, prod
from typing import Callable, Tuple, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, pack, unpack

# helpers

def exists(val):
    return val is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# continuous axial positional embedding

class ContinuousAxialPositionalEmbedding(Module):
    def __init__(
        self,
        dim: int,
        num_axial_dims = 2,
        max_seq_len_derive_axial_from: Optional[int] = None, # If axial_dims not given, derive from this
        axial_dims: Optional[Tuple[int, ...]] = None,
        learned_sin_div_factor = True,
        learned_sin_mult_factor = False,
        init_div_factor = 1.,
        init_mult_factor = 1.
    ):
        super().__init__()
        self.dim = dim
        self.num_axial_dims = num_axial_dims

        self.axial_dims: Optional[Tuple[int, ...]] = axial_dims
        self._max_seq_len_for_axial_dims: Optional[int] = max_seq_len_derive_axial_from

        if self.axial_dims is None and self._max_seq_len_for_axial_dims is None:
            print("Warning: ContinuousAxialPositionalEmbedding initialized without explicit axial_dims or max_seq_len_derive_axial_from. Axial dimensions will be derived on first use based on input sequence length.")

        self.learned_sin_div_factor = learned_sin_div_factor
        self.learned_sin_mult_factor = learned_sin_mult_factor

        self.weights = ModuleList([])
        self.div_factors = []
        self.mult_factors = []

        for _ in range(num_axial_dims):
            self.weights.append(nn.Linear(1, dim // num_axial_dims))
            self.div_factors.append(nn.Parameter(torch.tensor(init_div_factor), requires_grad = learned_sin_div_factor))
            self.mult_factors.append(nn.Parameter(torch.tensor(init_mult_factor), requires_grad = learned_sin_mult_factor))

    def derive_axial_dims_from_seq_len(self, seq_len: int) -> Tuple[int, ...]:
        dims = []
        remaining_seq_len = seq_len

        for i in range(self.num_axial_dims - 1):
            # Calculate the size of the current dimension
            # Ensure that the dimension size is at least 1
            dim_size = max(1, int(remaining_seq_len ** (1 / (self.num_axial_dims - i))))
            dims.append(dim_size)
            # Update remaining_seq_len, ensuring it's at least 1 for ceil
            remaining_seq_len = ceil(remaining_seq_len / dim_size) if dim_size > 0 else 1
        
        dims.append(max(1, remaining_seq_len)) # Ensure the last dimension is also at least 1
        return tuple(dims)

    def get_axial_dims(self, current_seq_len: Optional[int] = None) -> Tuple[int, ...]:
        if self.axial_dims is not None:
            return self.axial_dims
        
        derive_from_len = current_seq_len
        if derive_from_len is None:
            derive_from_len = self._max_seq_len_for_axial_dims
        
        if derive_from_len is None:
            raise ValueError("axial_dims not set and no seq_len provided for derivation for ContinuousAxialPositionalEmbedding.")
            
        # Derive and cache if not already set explicitly during init
        # This makes the axial_dims fixed after the first derivation based on a relevant seq_len
        self.axial_dims = self.derive_axial_dims_from_seq_len(derive_from_len)
        return self.axial_dims

    def forward(
        self,
        seq_len_or_axial_dims: Union[int, Tuple[int, ...]],
        *,
        device = None,
        dtype: Optional[torch.dtype] = None,
        return_factorized = False
    ):
        if isinstance(seq_len_or_axial_dims, int):
            axial_dims_to_use = self.get_axial_dims(seq_len_or_axial_dims)
        else:
            axial_dims_to_use = seq_len_or_axial_dims

        assert len(axial_dims_to_use) == self.num_axial_dims
        
        if device is None and len(self.weights) > 0: # Get device from parameters if available
            device = self.weights[0].weight.device
        if dtype is None and len(self.weights) > 0:
             dtype = self.weights[0].weight.dtype


        factorized_abs_pos_emb = []

        for ind, (cache_len, weight, div_factor, mult_factor) in enumerate(zip(axial_dims_to_use, self.weights, self.div_factors, self.mult_factors)):
            pos = torch.arange(cache_len, device = device, dtype = dtype)
            pos = (pos / div_factor) * mult_factor

            abs_pos_emb = torch.sin(weight(rearrange(pos, '... -> ... 1')))
            factorized_abs_pos_emb.append(abs_pos_emb)

        if return_factorized:
            return tuple(factorized_abs_pos_emb)

        output_dim = self.dim
        abs_pos_emb = torch.zeros((prod(axial_dims_to_use), output_dim), device = device, dtype = dtype)

        for i in range(prod(axial_dims_to_use)):
            coords = []
            temp = i
            for cache_len in reversed(axial_dims_to_use):
                coords.append(temp % cache_len)
                temp //= cache_len
            
            coords = list(reversed(coords)) # Correct order of coordinates

            emb_slice = torch.cat([factorized_abs_pos_emb[j][coords[j]] for j in range(self.num_axial_dims)])
            abs_pos_emb[i] = emb_slice
        
        return abs_pos_emb

    def forward_with_seq_len(
        self,
        seq_len: int,
        factorized: Optional[Tuple[Tensor, ...]] = None,
        device = None,
        dtype: Optional[torch.dtype] = None,
    ):
        axial_dims_to_use = self.get_axial_dims(seq_len)

        if not exists(factorized):
            factorized = self.forward(axial_dims_to_use, return_factorized = True, device=device, dtype=dtype)

        # reconstruct from factorized

        abs_pos_emb = torch.zeros((seq_len, self.dim), device = factorized[0].device, dtype=factorized[0].dtype)

        for i in range(seq_len):
            coords = []
            temp = i
            for cache_len in reversed(axial_dims_to_use):
                coords.append(temp % cache_len)
                temp //= cache_len
            
            coords = list(reversed(coords))

            emb_slice = torch.cat([factorized[j][coords[j]] for j in range(self.num_axial_dims)])
            abs_pos_emb[i] = emb_slice
        
        return abs_pos_emb

    def forward_tokens_at_indices(
        self,
        indices: Tensor, # (batch, num_tokens_to_get_embed_for)
        factorized_cache: Tuple[Tensor, ...]
    ):
        # factorized_cache is the tuple of (factorized_abs_pos_emb_x, factorized_abs_pos_emb_y, ...)
        # It was created using specific axial_dims.
        # The axial_dims used to create factorized_cache are implicitly defined by its shape.
        # We need to derive these axial_dims to correctly map flat indices to multi-dim coords.
        
        cached_axial_dims = tuple(cache_t.shape[0] for cache_t in factorized_cache)
        assert len(cached_axial_dims) == self.num_axial_dims, "Factorized cache does not match number of axial dimensions"

        batch_size, num_indices = indices.shape
        output_embeds = torch.zeros(batch_size, num_indices, self.dim, device=indices.device, dtype=factorized_cache[0].dtype)

        for b in range(batch_size):
            for i_idx, flat_index in enumerate(indices[b]):
                coords = []
                temp = flat_index.item() # Ensure it's a Python int for modulo arithmetic
                for cache_len in reversed(cached_axial_dims):
                    coords.append(temp % cache_len)
                    temp //= cache_len
                
                coords = list(reversed(coords))

                try:
                    emb_slice = torch.cat([factorized_cache[j][coords[j]] for j in range(self.num_axial_dims)])
                    output_embeds[b, i_idx] = emb_slice
                except IndexError as e:
                    print(f"IndexError in forward_tokens_at_indices: batch {b}, index_in_prompt {i_idx}, flat_token_index {flat_index.item()}")
                    print(f"Derived coords: {coords}, for cached_axial_dims: {cached_axial_dims}")
                    # This can happen if flat_index is out of bounds for the product of cached_axial_dims
                    # Or if a coord is out of bounds for its respective factorized_cache tensor.
                    # For now, fill with zeros and raise error or warning.
                    output_embeds[b, i_idx] = 0.0 
                    # raise e # Or handle more gracefully

        return output_embeds
