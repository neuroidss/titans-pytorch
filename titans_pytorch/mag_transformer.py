# FILE: titans_pytorch/mag_transformer.py

from __future__ import annotations
from typing import Callable, Tuple, Optional, List, Union

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, Parameter

from einops import rearrange, repeat, pack, unpack, einsum
from einops.layers.torch import Rearrange

# Assuming these are in the same directory or accessible via titans_pytorch
from titans_pytorch.axial_positional_embedding import ContinuousAxialPositionalEmbedding
from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState, mem_state_detach
from titans_pytorch.mac_transformer import SegmentedAttention, FeedForward, gumbel_sample, min_p_filter, log # Reusing components
from titans_pytorch.memory_models import MemoryMLP # Default NM model

# helpers (can be moved to a common utils.py later)
def exists(v): return v is not None
def default(v, d): return v if exists(v) else d
def identity(t): return t
def divisible_by(num, den): return (num % den) == 0
def round_up_multiple(seq, mult): return ceil(seq / mult) * mult

# Cache structure for MAG
# (current_token_idx_for_state, List_of_LayerCaches)
# LayerCache: (swa_kv_cache, nm_state)
MAGCache = namedtuple('MAGCache', ['current_token_idx', 'layer_caches'])
MAGLayerCache = namedtuple('MAGLayerCache', ['swa_kv_cache', 'nm_state'])

class MAGLayer(Module):
    def __init__(
        self,
        dim: int,
        swa_module: Module, # Pre-configured SegmentedAttention with sliding=True
        nm_module: Module,  # Pre-configured NeuralMemory
        ff_module: Module,
        norm_eps: float = 1e-6,
        gate_activation_fn: Callable = nn.SiLU
    ):
        super().__init__()
        self.dim = dim
        self.swa = swa_module
        self.nm = nm_module
        self.ff = ff_module

        self.norm_input = nn.RMSNorm(dim, eps=norm_eps) # Norm before branches

        # Gating components
        self.norm_for_swa_gate = nn.RMSNorm(dim, eps=norm_eps)
        self.norm_for_nm_gate = nn.RMSNorm(dim, eps=norm_eps)
        self.swa_gate_scale = Parameter(torch.ones(dim))
        self.nm_gate_scale = Parameter(torch.ones(dim))
        self.gate_activation = gate_activation_fn()

        self.norm_after_gate = nn.RMSNorm(dim, eps=norm_eps) # Norm before FF

    def forward(
        self,
        x: Tensor, # Input sequence (batch, seq_len, dim), already has persist_mem + pos_emb
        nm_state_in: Optional[NeuralMemState] = None,
        swa_kv_cache_in: Optional[Tuple[Tensor, Tensor]] = None,
        disable_ttl: bool = False,
        is_causal_swa: bool = True # Typically true for SWA
    ):
        residual = x
        x_normed_for_branches = self.norm_input(x)

        # Sliding Window Attention Branch
        y_swa, swa_attn_intermediates = self.swa(
            x_normed_for_branches,
            cache=swa_kv_cache_in,
        )
        swa_kv_cache_out = swa_attn_intermediates.cached_key_values

        # Neural Memory Branch
        y_nm_retrieved, nm_state_out, _ = self.nm(
            x_normed_for_branches.unsqueeze(0), # Add view dim for NM
            state=nm_state_in,
            return_surprises=True, 
            disable_ttl=disable_ttl
        )

        # Gating
        y_swa_g = self.norm_for_swa_gate(y_swa) * self.swa_gate_scale
        y_nm_g = self.norm_for_nm_gate(y_nm_retrieved) * self.nm_gate_scale
        
        gated_combination = self.gate_activation(y_swa_g + y_nm_g)

        x = residual + gated_combination

        # FeedForward
        residual_ff = x
        x_normed_for_ff = self.norm_after_gate(x)
        x = residual_ff + self.ff(x_normed_for_ff)
        
        layer_cache_out = MAGLayerCache(swa_kv_cache=swa_kv_cache_out, nm_state=nm_state_out)
        return x, layer_cache_out

class MAGTransformer(Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        swa_window_size: int = 256, # Sliding window size for SWA
        num_persist_mem_tokens: int = 0,
        ff_mult: int = 4,
        neural_memory_model: Optional[Module] = None,
        neural_memory_kwargs: dict = dict(),
        token_emb: Optional[Module] = None,
        max_seq_len_for_axial_dims: Optional[int] = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        use_flex_attn_for_swa: bool = False 
    ):
        super().__init__()
        self.dim = dim
        self.num_persist_mem_tokens = num_persist_mem_tokens

        self.token_emb = default(token_emb, nn.Embedding(num_tokens, dim))
        
        if num_persist_mem_tokens > 0:
            self.persistent_mems = Parameter(torch.randn(num_persist_mem_tokens, dim))
        else:
            self.persistent_mems = None

        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(
            dim=dim,
            num_axial_dims=2, 
            max_seq_len_derive_axial_from=max_seq_len_for_axial_dims
        )

        self.layers = ModuleList([])
        for _ in range(depth):
            swa_module = SegmentedAttention(
                dim=dim,
                segment_len=swa_window_size,
                num_persist_mem_tokens=num_persist_mem_tokens, 
                num_longterm_mem_tokens=0, 
                dim_head=dim_head,
                heads=heads,
                sliding=True, 
                use_flex_attn=use_flex_attn_for_swa,
                rope_theta=rope_theta,
                norm_eps=norm_eps
            )

            nm_model_instance = default(deepcopy(neural_memory_model), MemoryMLP(dim=dim_head, depth=2))
            
            current_nm_kwargs = deepcopy(neural_memory_kwargs)
            current_nm_kwargs.setdefault('dim', dim) 
            current_nm_kwargs.setdefault('dim_head', dim_head)
            current_nm_kwargs.setdefault('heads', heads)
            current_nm_kwargs.setdefault('norm_eps', norm_eps)
            # chunk_size for NM in MAG can be different from swa_window_size
            current_nm_kwargs.setdefault('chunk_size', swa_window_size) # Default to SWA window for simplicity

            nm_module = NeuralMemory(
                model=nm_model_instance,
                **current_nm_kwargs
            )

            ff_module = FeedForward(dim=dim, mult=ff_mult, norm_eps=norm_eps)

            self.layers.append(MAGLayer(
                dim=dim,
                swa_module=swa_module,
                nm_module=nm_module,
                ff_module=ff_module,
                norm_eps=norm_eps
            ))

        self.final_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.to_logits = Linear(dim, num_tokens)

    def forward(
        self,
        x: Tensor, 
        return_loss: bool = False,
        cache: Optional[MAGCache] = None, 
        return_cache: bool = False,
        factorized_pos_emb: Optional[Tuple[Tensor, ...]] = None,
        disable_ttl: bool = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch_size, data_seq_len = x.shape
        
        current_token_idx = 0
        layer_caches_in_list = [None] * len(self.layers)

        if exists(cache):
            current_token_idx, layer_caches_in_list = cache.current_token_idx, cache.layer_caches
            if layer_caches_in_list is None: layer_caches_in_list = [None] * len(self.layers)


        x_emb = self.token_emb(x)

        current_input_for_layers = x_emb
        num_prepended_persist = 0
        if exists(self.persistent_mems):
            pm = repeat(self.persistent_mems, 'n d -> b n d', b=batch_size)
            current_input_for_layers = torch.cat((pm, current_input_for_layers), dim=1)
            num_prepended_persist = self.num_persist_mem_tokens
        
        seq_len_with_persist = current_input_for_layers.shape[1]

        indices_for_pos = torch.arange(
            current_token_idx,
            current_token_idx + seq_len_with_persist,
            device=x.device
        ).unsqueeze(0) 

        pos_emb = self.axial_pos_emb.forward_tokens_at_indices(
            indices=indices_for_pos, 
            factorized_cache=factorized_pos_emb
        )
        current_input_for_layers = current_input_for_layers + pos_emb

        next_layer_caches_list = []
        for i, layer in enumerate(self.layers):
            layer_cache_in = layer_caches_in_list[i]
            nm_state_in_this_layer = layer_cache_in.nm_state if exists(layer_cache_in) else None
            swa_kv_cache_in_this_layer = layer_cache_in.swa_kv_cache if exists(layer_cache_in) else None
            
            if exists(nm_state_in_this_layer):
                 nm_effective_start_idx = current_token_idx 
                 nm_state_in_this_layer = nm_state_in_this_layer._replace(seq_index=nm_effective_start_idx)


            current_input_for_layers, layer_cache_out = layer(
                current_input_for_layers,
                nm_state_in=nm_state_in_this_layer,
                swa_kv_cache_in=swa_kv_cache_in_this_layer,
                disable_ttl=disable_ttl
            )
            next_layer_caches_list.append(layer_cache_out)

        output_from_layers = current_input_for_layers[:, num_prepended_persist:, :] 

        final_normed_output = self.final_norm(output_from_layers)
        logits = self.to_logits(final_normed_output)

        if return_cache:
            next_current_token_idx = current_token_idx + data_seq_len
            cache_out = MAGCache(current_token_idx=next_current_token_idx, layer_caches=next_layer_caches_list)
            return logits, cache_out

        if return_loss:
            if logits.shape[1] == 0: 
                return torch.tensor(0.0, device=x.device, requires_grad=True)
            return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)

        return logits

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor, 
        seq_len: int,   
        temperature: float = 1.0,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(min_p=0.01),
        show_progress: bool = True,
        disable_ttl: bool = False
    ):
        was_training = self.training
        self.eval()

        batch_size, prompt_len = prompt.shape
        out = prompt.clone() 

        max_conceptual_len_for_pos_emb = seq_len + self.num_persist_mem_tokens
        
        factorized_pos_emb = None
        if hasattr(self, 'axial_pos_emb') and self.axial_pos_emb is not None:
            axial_dims_for_sample = self.axial_pos_emb.get_axial_dims(max_conceptual_len_for_pos_emb)
            factorized_pos_emb = self.axial_pos_emb(axial_dims_for_sample, return_factorized=True)

        current_cache = MAGCache(current_token_idx=0, layer_caches=[None] * len(self.layers))

        if prompt_len > 0:
            _, current_cache = self.forward(
                out,
                return_loss=False,
                cache=current_cache,
                return_cache=True,
                factorized_pos_emb=factorized_pos_emb,
                disable_ttl=disable_ttl
            )

        num_tokens_to_generate = seq_len - prompt_len

        for _ in range(num_tokens_to_generate):
            current_token_input = out[:, -1:] 

            logits, current_cache = self.forward(
                current_token_input,
                return_loss=False,
                cache=current_cache,
                return_cache=True,
                factorized_pos_emb=factorized_pos_emb,
                disable_ttl=disable_ttl
            )

            next_token_logits = logits[:, -1, :]
            next_token_logits = filter_fn(next_token_logits, **filter_kwargs)
            sample_token = gumbel_sample(next_token_logits, temperature=temperature)
            out = torch.cat((out, sample_token), dim=-1)

            if out.shape[1] >= seq_len:
                break
        
        self.train(was_training)
        return out[..., prompt_len:]
