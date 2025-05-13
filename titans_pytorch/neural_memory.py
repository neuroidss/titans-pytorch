# FILE: titans_pytorch/neural_memory.py

from __future__ import annotations
from typing import Callable, Dict # Added Dict

import math
from functools import partial
from itertools import zip_longest
from collections import namedtuple

import torch
from torch import nn, stack, cat, is_tensor, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList, ParameterDict
from torch.func import functional_call, vmap, grad
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from tensordict import TensorDict

from assoc_scan import AssocScan

# Assuming ResidualNorm is either defined here or correctly in memory_models.py
class ResidualNorm(Module):
    def __init__(
        self,
        dim: int,
        model: Module,
        norm_eps: float = 1e-6
    ):
        super().__init__()
        self.model = model
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.norm(x + self.model(x, **kwargs))


from titans_pytorch.memory_models import(
    MemoryMLP
)

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

LinearNoBias = partial(Linear, bias = False)

NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
    'memory_weights_updated_in_call' # Added field
])

def mem_state_detach(
    state: NeuralMemState
):
    assert isinstance(state, NeuralMemState)
    # Make sure to handle the new field if it's a tensor (it's a bool, so fine)
    state_tuple = tuple(state)
    detached_elements = []
    for item in state_tuple:
        if is_tensor(item):
            detached_elements.append(item.detach())
        elif isinstance(item, TensorDict):
            detached_elements.append(item.detach())
        else:
            detached_elements.append(item)
    return NeuralMemState(*detached_elements)


def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def safe_cat(inputs, dim = -2):
    inputs = tuple(filter(exists, inputs))
    if len(inputs) == 0: return None
    if len(inputs) == 1: return inputs[0]
    return cat(inputs, dim = dim)

def is_empty_tensor(t):
    return t.numel() == 0

def dict_get_value_shapes(td):
    return [v.shape for k, v in td.items()]

def rearrange_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: rearrange(t, pattern, **kwargs))

def repeat_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: repeat(t, pattern, **kwargs))

def pair(v):
    return (v, v) if not isinstance(v, tuple) else v

def round_down_multiple(seq, mult):
    return seq // mult * mult

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)
    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]
    return packed, inverse

def Sequential(*modules):
    modules = [*filter(exists, modules)]
    if len(modules) == 0: return nn.Identity()
    if len(modules) == 1: return modules[0]
    return nn.Sequential(*modules)

def softclamp_max(t, max_value):
    half_max_value = max_value / 2
    return ((t / half_max_value).tanh() * half_max_value) + half_max_value

def softclamp_grad_norm(t, max_value):
    if is_empty_tensor(t): return t
    t_packed, inverse = pack_one_with_inverse(t, 'bn *')
    norm = t_packed.norm(dim = -1, keepdim = True)
    clamped_norm = softclamp_max(norm, max_value)
    t_packed = t_packed * (clamped_norm / norm.clamp(min=1e-6)) # Added clamp for stability
    return inverse(t_packed)

def l2norm(t, eps=1e-12, dim=-1):
    return t * torch.rsqrt(t.pow(2).sum(dim=dim, keepdim=True) + eps)

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads, eps=1e-6): # Added eps
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, eps=eps, elementwise_affine = False) # Use eps
        self.gamma = Parameter(torch.zeros(heads, 1, dim))
    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

class AveragePool(Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
    def forward(self, x, chunk_size = None):
        chunk_size = default(chunk_size, self.chunk_size)
        if x.ndim == 4: x = x.squeeze(0) # Assuming first dim is view if 4D
        assert x.ndim == 3, f"Input to AveragePool must be 3D (b n d), got {x.ndim}D"
        if x.shape[1] == 0 : return x # Handle empty sequence
        if x.shape[1] % chunk_size != 0: # Pad if not divisible
            padding = chunk_size - (x.shape[1] % chunk_size)
            x = pad_at_dim(x, (0, padding), dim=1)
        return reduce(x, 'b (n c) d -> b n d', 'mean', c = chunk_size)

class AttentionPool(Module):
    def __init__(self, dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.to_attn_logits = nn.Linear(dim, dim)
        nn.init.zeros_(self.to_attn_logits.weight)
        nn.init.zeros_(self.to_attn_logits.bias)
    def forward(self, x, chunk_size = None):
        chunk_size = default(chunk_size, self.chunk_size)
        if x.ndim == 4: x = x.squeeze(0) # Assuming first dim is view if 4D
        assert x.ndim == 3, f"Input to AttentionPool must be 3D (b n d), got {x.ndim}D"
        if x.shape[1] == 0 : return x # Handle empty sequence
        if x.shape[1] % chunk_size != 0: # Pad if not divisible
            padding = chunk_size - (x.shape[1] % chunk_size)
            x = pad_at_dim(x, (0, padding), dim=1)
        x_rearranged = rearrange(x, 'b (n c) d -> b n c d', c = chunk_size)
        attn_logits = self.to_attn_logits(x_rearranged)
        attn = attn_logits.softmax(dim = -2)
        return reduce(x_rearranged * attn, 'b n c d -> b n d', 'sum')

def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)

class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        chunk_size: int | tuple[int, int] = 1,
        batch_size = None,
        dim_head = None,
        heads = 1,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr = 1e-2,
        per_parameter_lr_modulation = False,
        max_mem_layer_modulation = 1.,
        per_head_learned_parameters = True,
        attn_pool_chunks = False,
        momentum = True,
        momentum_order = 1,
        learned_momentum_combine = False,
        learned_combine_include_zeroth = False,
        num_kv_per_token = 1,
        qkv_receives_diff_views = False,
        norm_eps: float = 1e-6,
        pre_rmsnorm = True,
        post_rmsnorm = False,
        qk_rmsnorm = False,
        qk_l2norm = True,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        activation: Module | None = None,
        init_adaptive_step_bias = None,
        init_momentum_bias = None,
        init_decay_bias = None,
        accept_weight_residual = False,
        gated_transition = False,
        init_transition_gate_bias: float = -5.0, # Added parameter with default
        mem_model_norm_add_residual = True,
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
        )
    ):
        super().__init__()
        self.dim = dim
        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)

        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)

        if exists(batch_size):
            assert divisible_by(batch_size, self.store_chunk_size)
        self.batch_size = batch_size

        self.assoc_scan = AssocScan(use_accelerated = use_accelerated_scan)

        self.qkv_receives_diff_views = qkv_receives_diff_views

        self.retrieve_norm = nn.RMSNorm(dim, eps=norm_eps) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim, eps=norm_eps) if pre_rmsnorm else nn.Identity()
        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads, eps=norm_eps) if post_rmsnorm else nn.Identity()

        self.qk_l2norm = qk_l2norm
        self.qk_rmsnorm = qk_rmsnorm
        if self.qk_rmsnorm and not self.qk_l2norm:
            self.q_norm_module = MultiheadRMSNorm(dim_head, heads, eps=norm_eps)
            self.k_norm_module = MultiheadRMSNorm(dim_head, heads, eps=norm_eps)
        else:
            self.q_norm_module = nn.Identity()
            self.k_norm_module = nn.Identity()


        dim_inner = dim_head * heads
        self.heads = heads
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_kv_heads = Rearrange('b n (h u d) -> b h (n u) d', h = heads, u = num_kv_per_token)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = Linear(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrieve_gate = Sequential(
            Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if heads > 1 else None

        if not exists(model):
            model = MemoryMLP(dim=dim_head, **default_model_kwargs)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'
        test_shape = (3, 2, dim_head)
        with torch.no_grad():
            try:
                test_input = torch.randn(test_shape)
                mem_model_output = model(test_input)
            except:
                raise RuntimeError(f'memory model unable to accept a tensor of shape {test_shape}')
            assert mem_model_output.shape == test_shape, 'output of memory model needs to be same shape as input'

        if mem_model_norm_add_residual and not isinstance(model, ResidualNorm):
            model = ResidualNorm(dim = dim_head, model = model, norm_eps=norm_eps)

        self.memory_model = model
        mem_model_params_dict = dict(self.memory_model.named_parameters())
        self.num_memory_parameter_tensors = len(mem_model_params_dict)
        self.memory_model_parameter_names = [*mem_model_params_dict.keys()]

        cloned_params_for_list = []
        for name, p_original_shape in mem_model_params_dict.items():
            p_data = p_original_shape.data.clone()
            if per_head_learned_parameters:
                p_data = repeat(p_data, '... -> h ...', h=heads)
            cloned_params_for_list.append(Parameter(p_data))

        self.memory_model_parameters = ParameterList(cloned_params_for_list)
        self.init_weight_shape = [p.shape for p in self.memory_model_parameters]
        self.per_head_learned_parameters = per_head_learned_parameters

        self.activation_for_qkv = default(activation, nn.SiLU())

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), self.activation_for_qkv)
        assert num_kv_per_token > 0
        self.to_keys = Sequential(LinearNoBias(dim, dim_inner * num_kv_per_token), self.activation_for_qkv)
        self.to_values = Sequential(LinearNoBias(dim, dim_inner * num_kv_per_token), self.activation_for_qkv)

        self.store_memory_loss_fn = store_memory_loss_fn
        self.num_kv_per_token = num_kv_per_token

        store_chunk_size_val = self.store_chunk_size
        assert not (attn_pool_chunks and store_chunk_size_val == 1), '`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1'
        if not attn_pool_chunks:
            self.reduce_to_chunk_rep = AveragePool(chunk_size = store_chunk_size_val)
        else:
            self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size = store_chunk_size_val)

        self.to_adaptive_step = Sequential(
            nn.Linear(dim, heads * num_kv_per_token),
            Rearrange('b n (h u) -> (b h) (n u)', u = num_kv_per_token)
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr = default_step_transform_max_lr)
        self.adaptive_step_transform = adaptive_step_transform

        self.to_momentum = Sequential(
            nn.Linear(dim, heads * momentum_order),
            Rearrange('b n (h o) -> o (b h) n 1', o = momentum_order)
        ) if momentum else None
        self.momentum_order = momentum_order
        self.to_learned_momentum_combine = None
        if learned_momentum_combine:
            assert momentum
            momentum_order_comb = momentum_order
            if learned_combine_include_zeroth:
                momentum_order_comb += 1
            self.to_learned_momentum_combine = Sequential(
                nn.Linear(dim, heads * momentum_order_comb),
                Rearrange('b n (h o) -> o (b h) n', h = heads, o=momentum_order_comb),
                nn.Softmax(dim = 0),
            )
            self.learned_combine_include_zeroth = learned_combine_include_zeroth

        self.to_layer_modulation = Sequential(
            nn.Linear(dim, heads * self.num_memory_parameter_tensors),
            Rearrange('b n (h w) -> w (b h) n', h = heads),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None
        self.max_mem_layer_modulation = max_mem_layer_modulation

        self.to_learned_weight_residual_mix = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n'),
            nn.Sigmoid()
        ) if accept_weight_residual else None

        self.max_grad_norm = max_grad_norm

        self.to_decay_factor = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        self.transition_gate = nn.Parameter(tensor(init_transition_gate_bias)) if gated_transition else None # Use init_transition_gate_bias

        if exists(init_adaptive_step_bias):
            if isinstance(self.to_adaptive_step[0], nn.Linear):
                linear = self.to_adaptive_step[0]
                nn.init.zeros_(linear.weight)
                nn.init.constant_(linear.bias, init_adaptive_step_bias)

        if exists(self.to_momentum) and exists(init_momentum_bias):
             if isinstance(self.to_momentum[0], nn.Linear):
                linear = self.to_momentum[0]
                nn.init.zeros_(linear.weight)
                nn.init.constant_(linear.bias, init_momentum_bias)

        if exists(init_decay_bias):
             if isinstance(self.to_decay_factor[0], nn.Linear):
                linear = self.to_decay_factor[0]
                nn.init.zeros_(linear.weight)
                nn.init.constant_(linear.bias, init_decay_bias)

        self.use_accelerated_scan = use_accelerated_scan
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def memory_model_parameter_dict(self):
        return TensorDict(dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)))

    def init_weights(self, batch):
        source_param_dict = self.memory_model_parameter_dict
        if self.per_head_learned_parameters:
            weights = repeat_dict_values(source_param_dict, 'h ... -> (b h) ...', b = batch)
        else:
            weights = repeat_dict_values(source_param_dict, '... -> bh ...', bh = batch * self.heads)
        return weights

    def init_momentum(self, batch):
        zeros_template = self.memory_model_parameter_dict.apply(lambda t: torch.zeros_like(t))
        if self.per_head_learned_parameters:
            zeros = repeat_dict_values(zeros_template, 'h ... -> o (b h) ...', b = batch, o = self.momentum_order)
        else:
            zeros = repeat_dict_values(zeros_template, '... -> o bh ...', bh = batch * self.heads, o = self.momentum_order)
        return zeros

    def _forward_and_loss_for_grad(self, params_slice: Dict[str, Tensor], inputs: Tensor, loss_weights: Tensor, target: Tensor):
        pred = functional_call(self.memory_model, params_slice, (inputs,))
        loss = self.store_memory_loss_fn(pred, target)
        weighted_loss = loss * loss_weights
        return weighted_loss.sum(), loss

    def store_memories(
        self,
        seq,
        keys_values_views_seq=None,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0,
        prev_weights = None,
        mask: Tensor | None = None,
        return_surprises = True,
        disable_ttl: bool = False
    ):
        assert seq.ndim == 3, f"Input 'seq' to store_memories must be 3D, got {seq.ndim}D"
        batch, seq_len_local = seq.shape[:2]

        heads, configured_chunk_size_for_internal_grad, num_updates = self.heads, self.store_chunk_size, self.num_kv_per_token

        current_processing_chunk_size_for_grad_calc = configured_chunk_size_for_internal_grad
        if not disable_ttl and seq_len_local > 0 and seq_len_local < configured_chunk_size_for_internal_grad:
            current_processing_chunk_size_for_grad_calc = seq_len_local
        
        effective_round_down_seq_len = round_down_multiple(seq_len_local, current_processing_chunk_size_for_grad_calc)

        if disable_ttl or effective_round_down_seq_len == 0:
             remainder_if_skipped = seq
             if effective_round_down_seq_len == 0 and not disable_ttl and seq_len_local > 0:
                 if seq_len_local == 0:
                    remainder_if_skipped = torch.empty_like(seq[:, :0, :])


             if not exists(weights): weights = self.init_weights(batch)
             weights = TensorDict(weights)

             current_past_last_applied_deltas, current_past_last_momentum = default(
                 past_state,
                 (weights.apply(lambda t: torch.zeros_like(t) if t.numel() > 0 else t),
                  self.init_momentum(batch))
             )
             if not isinstance(current_past_last_applied_deltas, TensorDict): current_past_last_applied_deltas = TensorDict(current_past_last_applied_deltas)
             if not isinstance(current_past_last_momentum, TensorDict): current_past_last_momentum = TensorDict(current_past_last_momentum)

             updates_data = weights.apply(lambda t: torch.zeros_like(t).unsqueeze(1) if t.numel() > 0 else t.unsqueeze(1))
             next_store_state = NeuralMemState(seq_index + seq_len_local, weights, remainder_if_skipped, (current_past_last_applied_deltas, current_past_last_momentum), updates_data, False)


             if not return_surprises: return updates_data, next_store_state
             zero_loss_shape = (batch, heads, seq_len_local * num_updates) if seq_len_local > 0 else (batch, heads, 0)
             zero_loss = torch.zeros(zero_loss_shape, device=seq.device, dtype=seq.dtype)
             zero_lr_shape = (batch, heads, seq_len_local * num_updates) if seq_len_local > 0 else (batch, heads, 0)
             zero_lr = torch.zeros(zero_lr_shape, device=seq.device, dtype=seq.dtype)
             return updates_data, next_store_state, (zero_loss, zero_lr)

        seq_proc, remainder = seq[:, :effective_round_down_seq_len, :], seq[:, effective_round_down_seq_len:, :]
        num_chunks_for_grad_calc = effective_round_down_seq_len // current_processing_chunk_size_for_grad_calc
        next_seq_len_index = seq_index + effective_round_down_seq_len

        if not exists(weights):
            weights = self.init_weights(batch)
        weights = TensorDict(weights)

        weights_for_surprise = repeat_dict_values(weights, 'bh ... -> (bh n) ...', n = num_chunks_for_grad_calc)

        normed_seq_proc = self.store_norm(seq_proc)
        raw_adaptive_step_values = self.to_adaptive_step(normed_seq_proc)
        adaptive_lr_transformed = self.adaptive_step_transform(raw_adaptive_step_values)

        chunked_seq_for_params = self.reduce_to_chunk_rep(normed_seq_proc, chunk_size = current_processing_chunk_size_for_grad_calc)
        decay_factor = self.to_decay_factor(chunked_seq_for_params).sigmoid()

        need_layer_lr_mod = exists(self.to_layer_modulation) and num_chunks_for_grad_calc > 0
        has_momentum = exists(self.to_momentum)

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq_for_params).sigmoid()
            learned_combine = exists(self.to_learned_momentum_combine)
            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq_for_params)

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq_for_params) * self.max_mem_layer_modulation

        keys_seq_for_proj = normed_seq_proc
        values_seq_for_proj = normed_seq_proc

        if exists(keys_values_views_seq):
            kv_views_proc = keys_values_views_seq[:, :, :effective_round_down_seq_len, :]
            if kv_views_proc.shape[0] >= 1:
                keys_seq_for_proj = self.store_norm(kv_views_proc[0])
            if kv_views_proc.shape[0] >= 2:
                values_seq_for_proj = self.store_norm(kv_views_proc[1])
            elif kv_views_proc.shape[0] == 1:
                 values_seq_for_proj = self.store_norm(kv_views_proc[0])

        keys = self.to_keys(keys_seq_for_proj)
        values = self.to_values(values_seq_for_proj)

        keys, values = map(self.split_kv_heads, (keys, values))

        if self.qk_l2norm:
            keys = l2norm(keys)
        elif self.qk_rmsnorm:
            keys = self.k_norm_module(keys)

        keys_for_grad = rearrange(keys, 'b h (n c u) d -> (b h n) (c u) d', c=current_processing_chunk_size_for_grad_calc, u=num_updates)
        values_for_grad = rearrange(values, 'b h (n c u) d -> (b h n) (c u) d', c=current_processing_chunk_size_for_grad_calc, u=num_updates)
        adaptive_lr_for_grad_as_loss_weights = rearrange(adaptive_lr_transformed, '(bh) (n c u) -> (bh n) (c u)', c=current_processing_chunk_size_for_grad_calc, u=num_updates, n=num_chunks_for_grad_calc)

        if exists(mask):
            mask_proc = mask[:, :effective_round_down_seq_len]
            mask_rearranged = repeat(mask_proc, 'b (n c) -> (b h n) (c u)', h=heads, u=num_updates, c=current_processing_chunk_size_for_grad_calc)
            adaptive_lr_for_grad_as_loss_weights = torch.where(mask_rearranged, adaptive_lr_for_grad_as_loss_weights, 0.)

        weights_for_grad_vmap_prep = weights_for_surprise
        if exists(prev_weights):
            start_idx_global_chunk = math.ceil(seq_index / configured_chunk_size_for_internal_grad)
            end_idx_global_chunk = start_idx_global_chunk + num_chunks_for_grad_calc
            
            prev_weights_chunk = prev_weights.apply(
                lambda t: t[:, start_idx_global_chunk:end_idx_global_chunk] if t.ndim > 1 and t.shape[1] >= end_idx_global_chunk else torch.zeros_like(t[:, :num_chunks_for_grad_calc] if t.ndim > 1 and t.shape[1] >=num_chunks_for_grad_calc else t.unsqueeze(1).expand(-1, num_chunks_for_grad_calc, *([-1]*(t.ndim-1))).contiguous() )
            )

            if exists(self.to_learned_weight_residual_mix) and num_chunks_for_grad_calc > 0:
                mix = self.to_learned_weight_residual_mix(chunked_seq_for_params)
                mix = rearrange(mix, 'b h n -> (b h) n')
                prev_weights_chunk = prev_weights_chunk.apply(lambda t: einx.multiply('bh n, bh n ... -> bh n ...', mix, t))

            prev_weights_reshaped = rearrange_dict_values(prev_weights_chunk, 'bh n ... -> (bh n) ...')
            weights_for_grad_vmap_prep = weights_for_surprise.apply(lambda cur, prev_r: cur + prev_r, prev_weights_reshaped)
        
        vmap_grad_fn = grad(self._forward_and_loss_for_grad, has_aux=True)
        per_sample_grad_fn_local = vmap(vmap_grad_fn, in_dims=(0, 0, 0, 0))

        grads_dict_vmapped, unweighted_loss_vmapped = per_sample_grad_fn_local(
            dict(weights_for_grad_vmap_prep), keys_for_grad, adaptive_lr_for_grad_as_loss_weights, values_for_grad
        )
        grads = TensorDict(grads_dict_vmapped)

        adaptive_lr_reshaped_for_surprise = rearrange(adaptive_lr_transformed, '(b h) (n c u) -> b h (n c u)', b=batch, h=heads, n=num_chunks_for_grad_calc, c=current_processing_chunk_size_for_grad_calc, u=num_updates)
        unweighted_mem_model_loss_reshaped = rearrange(
            unweighted_loss_vmapped,
            '(b h n) (c u) -> b h (n c u)', 
            b=batch, h=heads, n=num_chunks_for_grad_calc, c=current_processing_chunk_size_for_grad_calc, u=num_updates
        )

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        grads = rearrange_dict_values(grads, '(bh n) ... -> bh n ...', bh=batch*heads, n=num_chunks_for_grad_calc)

        if need_layer_lr_mod :
            new_grads_dict = {}
            for i, (param_name, grad_tensor) in enumerate(grads.items()):
                current_mod = layer_lr_mod[i]
                mod_unsqueezed = current_mod.view(current_mod.shape[0], current_mod.shape[1], *((1,) * (grad_tensor.ndim - 2)))
                new_grads_dict[param_name] = grad_tensor * mod_unsqueezed
            grads = TensorDict(new_grads_dict)

        surprises = grads.mul(-1)
        
        current_past_weights_batch_start, current_past_momentum_chunk_start = default(
            past_state,
            (weights.apply(lambda t: torch.zeros_like(t) if t.numel() > 0 else t),
             self.init_momentum(batch))
        )
        if not isinstance(current_past_weights_batch_start, TensorDict): current_past_weights_batch_start = TensorDict(current_past_weights_batch_start)
        if not isinstance(current_past_momentum_chunk_start, TensorDict): current_past_momentum_chunk_start = TensorDict(current_past_momentum_chunk_start)

        updates_M_values_over_chunks = TensorDict()
        next_last_M_values_for_batch = TensorDict()
        next_last_S_values_for_batch = TensorDict()

        for (param_name, surprise_param_chunks) in surprises.items():
            effective_update_for_decay_scan = surprise_param_chunks

            if has_momentum:
                momentum_deltas_accum = []
                last_S_param_for_orders = current_past_momentum_chunk_start[param_name]
                next_S_orders_final_state_for_param = []

                for order_idx in range(self.momentum_order):
                    one_adaptive_momentum_for_order = adaptive_momentum[order_idx]
                    one_last_S_state_for_order = last_S_param_for_orders[order_idx]
                    alpha_momentum_scan = rearrange(one_adaptive_momentum_for_order, 'bh n 1 -> bh n')
                    
                    momentum_scan_out_for_order = self.assoc_scan(
                        alpha_momentum_scan, surprise_param_chunks, prev=one_last_S_state_for_order
                    )
                    momentum_deltas_accum.append(momentum_scan_out_for_order)
                    if momentum_scan_out_for_order.numel() > 0 and momentum_scan_out_for_order.shape[1] > 0:
                        next_S_orders_final_state_for_param.append(momentum_scan_out_for_order[:, -1])
                    else:
                        next_S_orders_final_state_for_param.append(one_last_S_state_for_order)
                
                momentum_deltas_accum_stacked = stack(momentum_deltas_accum)
                next_last_S_values_for_batch[param_name] = stack(next_S_orders_final_state_for_param)

                if learned_combine:
                    current_momentums_for_combine = momentum_deltas_accum_stacked
                    if self.learned_combine_include_zeroth:
                        surprise_param_chunks_expanded = rearrange(surprise_param_chunks, 'bh n ... -> 1 bh n ...')
                        current_momentums_for_combine = cat((surprise_param_chunks_expanded, momentum_deltas_accum_stacked), dim=0)
                    effective_update_for_decay_scan = einsum(combine_momentums, current_momentums_for_combine, 'o_comb bh n, o_comb bh n ...P -> bh n ...P')
                else:
                    effective_update_for_decay_scan = momentum_deltas_accum_stacked[-1]
            else:
                 next_last_S_values_for_batch[param_name] = self.init_momentum(batch)[param_name]

            decay_alpha_scan_for_param = rearrange(1. - decay_factor, 'bh n 1 -> bh n')
            
            accumulated_M_values_scan_out = self.assoc_scan(
                decay_alpha_scan_for_param, effective_update_for_decay_scan, prev=current_past_weights_batch_start[param_name]
            )

            updates_M_values_over_chunks[param_name] = accumulated_M_values_scan_out
            if accumulated_M_values_scan_out.numel() > 0 and accumulated_M_values_scan_out.shape[1] > 0:
                next_last_M_values_for_batch[param_name] = accumulated_M_values_scan_out[:, -1]
            else:
                next_last_M_values_for_batch[param_name] = current_past_weights_batch_start[param_name]

        current_call_final_state_for_M_and_S = (next_last_M_values_for_batch, next_last_S_values_for_batch)

        next_store_state = NeuralMemState(
            seq_index=next_seq_len_index,
            weights=weights, 
            cache_store_segment=remainder,
            states=current_call_final_state_for_M_and_S,
            updates=updates_M_values_over_chunks,
            memory_weights_updated_in_call=False # Will be set in NeuralMemory.forward
        )

        if not return_surprises:
            return updates_M_values_over_chunks, next_store_state
        return updates_M_values_over_chunks, next_store_state, (unweighted_mem_model_loss_reshaped, adaptive_lr_reshaped_for_surprise)

    def retrieve_memories(
        self,
        seq,
        weights: TensorDict,
    ):
        original_seq_for_gate = seq
        local_retrieve_chunk_size = self.retrieve_chunk_size
        
        batch, seq_len = seq.shape[:2]
        
        is_one_token = seq_len == 1
        if is_one_token: local_retrieve_chunk_size = 1

        need_pad = local_retrieve_chunk_size > 1 and seq_len > 0 and (seq_len % local_retrieve_chunk_size != 0)
        original_retrieve_seq_len = seq_len

        seq_for_queries = seq
        if need_pad:
            padding_for_chunking = local_retrieve_chunk_size - (seq_len % local_retrieve_chunk_size)
            seq_for_queries = pad_at_dim(seq, (0, padding_for_chunking), dim=1)
        
        if seq_for_queries.shape[1] == 0:
             return torch.empty(batch, original_retrieve_seq_len, self.dim, device=seq.device, dtype=seq.dtype)

        normed_seq_for_queries = self.retrieve_norm(seq_for_queries)
        queries = self.to_queries(normed_seq_for_queries)
        queries = self.split_heads(queries)

        if self.qk_l2norm:
            queries = l2norm(queries)
        elif self.qk_rmsnorm:
            queries = self.q_norm_module(queries)
        
        queries_rearranged_for_chunking = rearrange(queries, 'b h (nq c) d -> (b h nq) c d', c=local_retrieve_chunk_size)
        num_query_chunks_per_bh = queries.shape[2] // local_retrieve_chunk_size

        if num_query_chunks_per_bh == 0 and original_retrieve_seq_len > 0 :
             return torch.empty(batch, original_retrieve_seq_len, self.dim, device=seq.device, dtype=seq.dtype)
        if num_query_chunks_per_bh == 0 and original_retrieve_seq_len == 0:
             return torch.empty(batch, 0, self.dim, device=seq.device, dtype=seq.dtype)

        weights_for_functional_call = weights.apply(
            lambda t: repeat(t, 'bh ... -> (bh nq) ...', nq=num_query_chunks_per_bh) if t.numel() > 0 else t.expand(num_query_chunks_per_bh * t.shape[0], *t.shape[1:])
        )
        
        def single_model_call_retrieve(params_single_set: Dict[str, Tensor], single_query_chunk_input: Tensor):
            return functional_call(self.memory_model, params_single_set, (single_query_chunk_input,))

        vmapped_model_call_retrieve = vmap(single_model_call_retrieve, in_dims=(0, 0))

        values_from_mem_model_chunked = vmapped_model_call_retrieve(
            dict(weights_for_functional_call),
            queries_rearranged_for_chunking
        )

        values_from_mem_model = rearrange(values_from_mem_model_chunked,
                                          '(b h nq) c d -> b h (nq c) d',
                                          b=batch, h=self.heads, c=local_retrieve_chunk_size)

        values_from_mem_model = self.multihead_rmsnorm(values_from_mem_model)

        if exists(self.retrieve_gate):
            gate_input_seq = original_seq_for_gate
            if need_pad:
                gate_input_seq = pad_at_dim(original_seq_for_gate, (0, padding_for_chunking), dim=1)
            if gate_input_seq.shape[1] > 0:
                gate_values = self.retrieve_gate(gate_input_seq)
                values_from_mem_model = values_from_mem_model * gate_values

        values_merged = self.merge_heads(values_from_mem_model)
        final_values = self.combine_heads(values_merged)

        return final_values[:, :original_retrieve_seq_len]

    def forward(
        self,
        seq,
        store_seq_arg_external = None,
        state: NeuralMemState | None = None,
        detach_mem_state = False,
        prev_weights = None,
        store_mask: Tensor | None = None,
        return_surprises = False,
        ttt_batch_size: int | None = None,
        disable_ttl: bool = False
    ):
        assert seq.ndim == 4, f"NeuralMemory expects a 4D input (views, b, n, d), got {seq.ndim}D"
        num_views = seq.shape[0]
        batch_dim_size = seq.shape[1]

        primary_seq_3d_for_nm = seq[0]
        keys_values_views_for_store_memories = None
        if self.qkv_receives_diff_views and num_views > 1:
            keys_values_views_for_store_memories = seq[1:]
            if keys_values_views_for_store_memories.shape[0] == 0:
                keys_values_views_for_store_memories = None
        
        retrieve_seq_3d = primary_seq_3d_for_nm
        store_seq_base_3d = default(store_seq_arg_external, primary_seq_3d_for_nm)

        if isinstance(store_seq_base_3d, Tensor) and store_seq_base_3d.ndim == 2:
            store_seq_base_3d = rearrange(store_seq_base_3d, 'b d -> b 1 d')
        
        initial_seq_index, initial_weights_td, cache_store_seq_tensor, initial_past_M_and_S_state_tuple, _, _ = \
            default(state, NeuralMemState(0, None, None, None, None, False)) # Added default for new field

        current_weights_td = initial_weights_td
        if not exists(current_weights_td):
            current_weights_td = self.init_weights(batch_dim_size)
        current_weights_td = TensorDict(current_weights_td)

        current_past_M_and_S_state_tuple = initial_past_M_and_S_state_tuple
        if not exists(current_past_M_and_S_state_tuple):
            current_past_M_and_S_state_tuple = (current_weights_td.clone(), self.init_momentum(batch_dim_size))
        
        current_M_for_next_batch_start, current_S_for_next_chunk_start = current_past_M_and_S_state_tuple
        if not isinstance(current_M_for_next_batch_start, TensorDict): current_M_for_next_batch_start = TensorDict(current_M_for_next_batch_start)
        if not isinstance(current_S_for_next_chunk_start, TensorDict): current_S_for_next_chunk_start = TensorDict(current_S_for_next_chunk_start)
        current_past_M_and_S_state_tuple = (current_M_for_next_batch_start, current_S_for_next_chunk_start)

        store_seq_input_3d = store_seq_base_3d
        if exists(cache_store_seq_tensor) and cache_store_seq_tensor.numel() > 0 :
            assert cache_store_seq_tensor.ndim == 3, "cache_store_seq must be 3D"
            store_seq_input_3d = safe_cat((cache_store_seq_tensor, store_seq_input_3d), dim=1)
        
        assert store_seq_input_3d.ndim == 3, "store_seq_input_3d for splitting must be 3D"
        current_call_store_seq_len = store_seq_input_3d.shape[1]
        
        batch_size_for_ttl_update = default(ttt_batch_size, self.batch_size)
        
        accumulated_M_values_over_call_chunks = None # Initialize to None
        final_cache_store_segment_for_next_state = store_seq_input_3d
        
        final_M_after_call = current_M_for_next_batch_start
        final_S_after_call = current_S_for_next_chunk_start
        final_seq_index_after_call = initial_seq_index + current_call_store_seq_len
        
        all_chunk_surprises_list = []
        memory_weights_actually_updated_this_call = False

        if current_call_store_seq_len > 0 and not disable_ttl and exists(batch_size_for_ttl_update):
            offset_in_store_seq = 0
            temp_seq_idx_tracker = initial_seq_index
            
            weights_at_start_of_current_ttl_batch = current_M_for_next_batch_start.clone() 
            momentum_at_start_of_current_ttl_batch_first_chunk = current_S_for_next_chunk_start

            while offset_in_store_seq < current_call_store_seq_len:
                current_pos_in_ttl_batch = temp_seq_idx_tracker % batch_size_for_ttl_update
                len_to_fill_ttl_batch = batch_size_for_ttl_update - current_pos_in_ttl_batch
                len_this_chunk = min(len_to_fill_ttl_batch, current_call_store_seq_len - offset_in_store_seq)

                store_chunk_iter = store_seq_input_3d[:, offset_in_store_seq : offset_in_store_seq + len_this_chunk, :]
                
                kv_views_chunk_iter = None
                if exists(keys_values_views_for_store_memories) and keys_values_views_for_store_memories.shape[1] == current_call_store_seq_len:
                    kv_views_chunk_iter = keys_values_views_for_store_memories[:, :, offset_in_store_seq : offset_in_store_seq + len_this_chunk, :]

                mask_chunk_iter = None
                if exists(store_mask) and store_mask.shape[-1] == current_call_store_seq_len:
                    mask_chunk_iter = store_mask[:, offset_in_store_seq : offset_in_store_seq + len_this_chunk]
                
                chunk_M_values, internal_next_state_from_chunk, chunk_surprises = self.store_memories(
                    store_chunk_iter,
                    keys_values_views_seq=kv_views_chunk_iter,
                    weights=weights_at_start_of_current_ttl_batch,
                    seq_index=temp_seq_idx_tracker,
                    past_state=(weights_at_start_of_current_ttl_batch, momentum_at_start_of_current_ttl_batch_first_chunk),
                    prev_weights=prev_weights,
                    mask=mask_chunk_iter,
                    return_surprises=True,
                    disable_ttl=False 
                )
                
                if chunk_M_values.keys(): # If store_memories returned updates
                    accumulated_M_values_over_call_chunks = chunk_M_values # Only keep the latest
                
                all_chunk_surprises_list.append(chunk_surprises)
                final_cache_store_segment_for_next_state = internal_next_state_from_chunk.cache_store_segment
                
                temp_seq_idx_tracker = internal_next_state_from_chunk.seq_index
                final_M_after_chunk, final_S_after_chunk = internal_next_state_from_chunk.states
                
                if divisible_by(temp_seq_idx_tracker, batch_size_for_ttl_update):
                    gate_val = self.transition_gate.sigmoid() if exists(self.transition_gate) else 1.0
                    
                    weights_at_start_of_current_ttl_batch = weights_at_start_of_current_ttl_batch.apply(
                        lambda w_old, m_new: w_old.lerp(m_new, gate_val) if exists(gate_val) and gate_val != 1.0 else m_new,
                        final_M_after_chunk
                    )
                    memory_weights_actually_updated_this_call = True 
                    
                    momentum_at_start_of_current_ttl_batch_first_chunk = final_S_after_chunk
                else:
                    momentum_at_start_of_current_ttl_batch_first_chunk = final_S_after_chunk

                offset_in_store_seq += len_this_chunk
            
            final_M_after_call = weights_at_start_of_current_ttl_batch
            final_S_after_call = momentum_at_start_of_current_ttl_batch_first_chunk
            final_seq_index_after_call = temp_seq_idx_tracker

        elif current_call_store_seq_len > 0: 
            chunk_M_values, internal_next_state_from_chunk, chunk_surprises = self.store_memories(
                store_seq_input_3d, keys_values_views_seq=keys_values_views_for_store_memories,
                weights=current_M_for_next_batch_start, seq_index=initial_seq_index,
                past_state=(current_M_for_next_batch_start, current_S_for_next_chunk_start),
                prev_weights=prev_weights, mask=store_mask, return_surprises=True,
                disable_ttl=True 
            )
            if chunk_M_values.keys():
                accumulated_M_values_over_call_chunks = chunk_M_values # Only keep the latest
            all_chunk_surprises_list.append(chunk_surprises)
            final_cache_store_segment_for_next_state = internal_next_state_from_chunk.cache_store_segment
            _, final_S_after_call = internal_next_state_from_chunk.states
            final_seq_index_after_call = internal_next_state_from_chunk.seq_index

        final_output_state = NeuralMemState(
             seq_index = final_seq_index_after_call,
             weights = final_M_after_call,
             cache_store_segment=final_cache_store_segment_for_next_state,
             states = (final_M_after_call, final_S_after_call),
             updates = accumulated_M_values_over_call_chunks,
             memory_weights_updated_in_call = memory_weights_actually_updated_this_call
        )

        retrieved = self.retrieve_memories(
            retrieve_seq_3d,
            final_output_state.weights
        )

        if detach_mem_state:
            final_output_state = mem_state_detach(final_output_state)

        if not return_surprises:
            return retrieved, final_output_state

        final_surprises_tuple = (None, None)
        if all_chunk_surprises_list:
             all_losses = [s[0] for s in all_chunk_surprises_list if exists(s) and exists(s[0]) and s[0].numel() > 0]
             all_lrs = [s[1] for s in all_chunk_surprises_list if exists(s) and exists(s[1]) and s[1].numel() > 0]
             
             final_loss_tensor = cat(all_losses, dim=-1) if all_losses else torch.empty(batch_dim_size, self.heads, 0, device=retrieve_seq_3d.device, dtype=retrieve_seq_3d.dtype)
             final_lr_tensor = cat(all_lrs, dim=-1) if all_lrs else torch.empty(batch_dim_size, self.heads, 0, device=retrieve_seq_3d.device, dtype=retrieve_seq_3d.dtype)

             if final_loss_tensor.numel() > 0 or final_lr_tensor.numel() > 0 :
                 final_surprises_tuple = (final_loss_tensor, final_lr_tensor)
        
        return retrieved, final_output_state, final_surprises_tuple
