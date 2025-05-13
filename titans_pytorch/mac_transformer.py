

# FILE: mac_transformer.py

from __future__ import annotations
from typing import Callable, Tuple, Optional, Union # Added Union

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm

import torch
from torch import nn, stack, cat, Tensor # Added Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

def create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding = False):

    def create_mac_mask(_, __, q_idx, kv_idx):
        is_persist_mem = kv_idx < persist_mem_len
        kv_without_mem = kv_idx - persist_mem_len
        causal_mask = q_idx >= kv_without_mem

        if not sliding:
            block_diagonal = (q_idx // window_size) == (kv_without_mem // window_size)
            causal_mask = causal_mask & block_diagonal
        else:
            sliding_mask = (q_idx - kv_without_mem) <= window_size
            causal_mask = causal_mask & sliding_mask

        return is_persist_mem | (~is_persist_mem & causal_mask)

    block_mask = create_block_mask(create_mac_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len + persist_mem_len, _compile = True)
    return block_mask

# einstein notation related

from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange

# b - batch
# n - sequence
# h - heads
# d - feature dimension

# absolute and relative positions

from titans_pytorch.axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding # Import RoPE

# hyper connections / attend from x-transformers, which handles different queries and key lengths better

from x_transformers.attend import Attend

# Assuming hyper_connections is accessible
from hyper_connections import get_init_and_expand_reduce_stream_functions

# proposed neural memory

from titans_pytorch.neural_memory import NeuralMemory

# constants

LinearNoBias = partial(Linear, bias = False) # Define LinearNoBias

AttnIntermediates = namedtuple('AttnIntermediates', ('value_residual', 'cached_key_values'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def round_down_multiple(seq, mult):
    return seq // mult * mult

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        return unpack(out, packed_shape, default(inv_pattern, pattern))

    return packed, inverse

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    batch, seq_len_orig_input = seq.shape[:2] # Use a different name to avoid conflict
    next_seq_len_mult = round_up_multiple(seq_len_orig_input, segment_len)

    padding = next_seq_len_mult - seq_len_orig_input
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding)) # Pad sequence dimension

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n = segment_len)

    def inverse(out):
        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :seq_len_orig_input, :] # Slice to original input length

        return out

    return seq, inverse

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    if not exists(logits): return logits
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype), logits)


# --- FeedForward / MLP ---

# New MLP class mimicking Qwen's structure
class QwenMimicMLP(nn.Module):
    def __init__(self, dim, hidden_dim): # hidden_dim is intermediate_size
        super().__init__()
        # Use LinearNoBias as Qwen MLP layers are typically bias-free
        self.gate_proj = LinearNoBias(dim, hidden_dim)
        self.up_proj = LinearNoBias(dim, hidden_dim)
        self.down_proj = LinearNoBias(hidden_dim, dim)
        self.act_fn = nn.SiLU() # Qwen uses SiLU

    def forward(self, x):
        # Qwen computes: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        return self.down_proj(gate_output * up_output)

# Modified FeedForward function to use the new MLP structure
def FeedForward(dim, mult = 4, norm_eps=1e-6): # Add norm_eps
    intermediate_size = int(dim * mult)
    return nn.Sequential(
        nn.RMSNorm(dim, eps=norm_eps), # Use passed eps
        QwenMimicMLP(dim=dim, hidden_dim=intermediate_size)
    )

# --- Attention ---

class SegmentedAttention(Module):
    def __init__(
        self,
        dim,
        segment_len,
        num_persist_mem_tokens = 0,
        num_longterm_mem_tokens = 0,
        dim_head = 64,
        heads = 8,
        sliding = False,
        accept_value_residual = False,
        attend_kwargs: dict = dict(),
        use_flex_attn = False,
        rope_theta = 10000, # Add rope_theta parameter
        norm_eps = 1e-6 # Add norm_eps parameter
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=norm_eps) # Use passed eps
        # Initialize RoPE with the provided theta
        self.rotary_emb = RotaryEmbedding(dim=dim_head, theta=rope_theta)

        self.attend = Attend(causal = True, **attend_kwargs)

        # Use LinearNoBias for Qwen mimicry of QKV projections
        self.to_qkv = LinearNoBias(dim, (dim_head * heads) * 3)

        # Qwen's o_proj has bias, so use standard Linear
        self.to_out = Linear(dim_head * heads, dim)

        self.to_learned_v_mix = nn.Sequential(
            nn.Linear(dim, heads), # Keep bias=True here for flexibility, init carefully
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if accept_value_residual else None

        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens

        total_segment_len = segment_len + num_longterm_mem_tokens
        self.total_segment_len = total_segment_len

        self.sliding = sliding

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head)) if num_persist_mem_tokens > 0 else None
        assert not (use_flex_attn and not exists(flex_attention)), 'FlexAttention requested but not available. Ensure PyTorch version supports it and CUDA is available.'
        self.use_flex_attn = use_flex_attn
        self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward_inference(
        self,
        token,
        cache,
        value_residual = None,
        output_gating = None,
    ):
        batch = token.shape[0]
        token = self.norm(token)
        qkv = self.to_qkv(token)
        q, k, v = qkv.chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))
        orig_v = v
        if exists(self.to_learned_v_mix) and exists(value_residual):
            mix = self.to_learned_v_mix(token)
            v = v.lerp(value_residual, mix)

        ck, cv = (None, None) if cache is None else cache

        if ck is not None:
            k = cat((ck, k), dim = -2)
        if cv is not None:
            v = cat((cv, v), dim = -2)

        next_cache = (k, v)
        # Apply RoPE using the initialized self.rotary_emb
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        if exists(self.persistent_memory) and self.num_persist_mem_tokens > 0:
            # Ensure persistent_memory is correctly indexed (it has shape [2, h, n_persist, d])
            pmk = repeat(self.persistent_memory[0], 'h n d -> b h n d', b = k.shape[0])
            pmv = repeat(self.persistent_memory[1], 'h n d -> b h n d', b = v.shape[0])
            k = cat((pmk, k), dim = -2)
            v = cat((pmv, v), dim = -2)


        out, _ = self.attend(q, k, v)
        out = self.merge_heads(out)
        out = self.to_out(out)
        if exists(output_gating):
            out = out * output_gating
        return out, AttnIntermediates(orig_v, next_cache)

    def forward_flex(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        output_gating = None,
        cache = None # Cache is not typically used with flex_attention full sequence processing
    ):
        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))
        batch, seq_len = seq.shape[:2]
        seq = self.norm(seq)
        qkv = self.to_qkv(seq)
        q, k, v = qkv.chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))
        orig_v = v
        if exists(self.to_learned_v_mix) and exists(value_residual):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)

        # For flex attention on full sequence, next_cache is typically the current k,v for potential future use if mixed mode
        next_cache = (k, v)

        # Apply RoPE using the initialized self.rotary_emb
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k) # Should handle non-cached k correctly

        if exists(self.persistent_memory) and self.num_persist_mem_tokens > 0:
            pmk = repeat(self.persistent_memory[0], 'h n d -> b h n d', b = batch)
            pmv = repeat(self.persistent_memory[1], 'h n d -> b h n d', b = batch)
            k = cat((pmk, k), dim = -2)
            v = cat((pmv, v), dim = -2)


        if not exists(flex_attn_fn):
            # Ensure seq_len for block_mask is the actual query length for this flex attention call
            block_mask = create_mac_block_mask(q.shape[-2], self.total_segment_len, self.num_persist_mem_tokens, self.sliding)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        out = flex_attn_fn(q, k, v)
        out = self.merge_heads(out)
        out = self.to_out(out)
        if exists(output_gating):
            out = out * output_gating
        return out, AttnIntermediates(orig_v, next_cache)

    def forward(
        self,
        seq,
        value_residual = None,
        flex_attn_fn: Callable | None = None,
        disable_flex_attn = False,
        output_gating = None,
        cache = None
    ):
        is_inferencing = exists(cache)
        if is_inferencing:
            assert seq.shape[-2] == 1, "For inference with cache, input sequence length must be 1"
            return self.forward_inference(seq, cache, value_residual, output_gating = output_gating)

        if seq.is_cuda and self.use_flex_attn and not disable_flex_attn:
            return self.forward_flex(seq, value_residual, flex_attn_fn, output_gating = output_gating)

        # Fallback to segmented attention without flex_attention
        assert not (exists(value_residual) ^ exists(self.to_learned_v_mix))

        batch, seq_len_orig = seq.shape[:2] # Original input sequence length

        # Pad the sequence to be a multiple of total_segment_len for block processing
        seq_proc, inverse_segment = pad_and_segment_with_inverse(seq, self.total_segment_len, fold_into_batch = False, inverse_remove_pad=True)

        # After inverse_segment, seq_proc has length multiple of total_segment_len
        # inverse_remove_pad=True in inverse_segment will handle trimming back to seq_len_orig

        seq_proc_normed = self.norm(seq_proc)
        qkv = self.to_qkv(seq_proc_normed)
        q, k, v = qkv.chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v)) # q, k, v are (b, h, padded_len, d_h)

        orig_v = v
        if exists(self.to_learned_v_mix) and exists(value_residual):
            # value_residual should also correspond to seq_proc length if used here
            # This might require value_residual to be padded/segmented if it comes from previous layer raw output
            mix_seq_input = seq_proc # Use the processed sequence for mix
            if mix_seq_input.shape[1] != v.shape[2]: # If seq_proc was padded differently than v
                 mix_seq_input = F.pad(mix_seq_input, (0,0,0, v.shape[2] - mix_seq_input.shape[1]))

            mix = self.to_learned_v_mix(mix_seq_input) # mix is (b,h,padded_len,1)
            v = v.lerp(value_residual, mix)

        # For non-flex, non-inference, cache is effectively (k,v) of the current full processed sequence
        next_cache_for_attn_intermediates = (k,v)

        # Apply RoPE. rotate_queries_with_cached_keys should handle non-cached k by rotating both q and k.
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # Rearrange into segments/windows for attention
        # q_seg, k_seg, v_seg are (b*w, h, total_segment_len, d_h)
        q_seg, k_seg, v_seg = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = self.total_segment_len) for t in (q, k, v))

        attend_kwargs = dict() # causal=True is default in self.attend

        if exists(self.persistent_memory) and self.num_persist_mem_tokens > 0:
            pmk = repeat(self.persistent_memory[0], 'h n d -> b_w h n d', b_w = q_seg.shape[0])
            pmv = repeat(self.persistent_memory[1], 'h n d -> b_w h n d', b_w = v_seg.shape[0])
            k_seg = cat((pmk, k_seg), dim = -2) # Prepend persistent keys
            v_seg = cat((pmv, v_seg), dim = -2) # Prepend persistent values

        out_seg, _ = self.attend(q_seg, k_seg, v_seg, **attend_kwargs)

        out_remerged = self.merge_heads(out_seg) # (b*w, total_segment_len, h*d_h)
        out_remerged = self.to_out(out_remerged) # (b*w, total_segment_len, dim)

        # Rearrange back to (b, padded_len, dim)
        out_full = rearrange(out_remerged, '(b w) n d -> b (w n) d', b = batch)

        # Apply inverse segmentation (trims padding to original seq_len_orig)
        out_final = inverse_segment(out_full)

        if exists(output_gating):
            # output_gating should correspond to original sequence length
            if output_gating.shape[1] != out_final.shape[1]:
                 # This needs careful handling if gating comes from a source with different padding
                 # For simplicity, assume output_gating matches out_final's seq dim
                 pass
            out_final = out_final * output_gating

        return out_final, AttnIntermediates(orig_v, next_cache_for_attn_intermediates)


# MAC transformer

class MemoryAsContextTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        segment_len,
        neural_memory_segment_len = None,
        neural_mem_gate_attn_output = False,
        neural_memory_add_value_residual = False,
        num_longterm_mem_tokens = 0,
        num_persist_mem_tokens = 0,
        neural_memory_batch_size = None,
        neural_mem_qkv_receives_diff_views = False, # Added this parameter
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        num_residual_streams = 4,
        neural_memory_model: Module | None = None,
        neural_memory_kwargs: dict = dict(),
        neural_memory_layers: tuple[int, ...] | None = None,
        use_flex_attn = False,
        sliding_window_attn = False,
        neural_mem_weight_residual = False,
        token_emb: Module | None = None,
        max_seq_len_for_axial_dims: Optional[int] = None,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6
    ):
        super().__init__()

        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)

        self.token_emb = token_emb
        self.axial_pos_emb = ContinuousAxialPositionalEmbedding(
            dim=dim,
            num_axial_dims=2,
            max_seq_len_derive_axial_from=max_seq_len_for_axial_dims
        )
        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        has_longterm_mems = num_longterm_mem_tokens > 0
        if has_longterm_mems:
            self.longterm_mems = nn.Parameter(torch.randn(num_longterm_mem_tokens, dim) * 0.02)
        else:
            self.longterm_mems = None

        self.sliding_window_attn_mac = sliding_window_attn
        self.attn_window_size = segment_len + num_longterm_mem_tokens

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, dim = dim, add_stream_embed = True, disable = num_residual_streams == 1)

        self.layers = ModuleList([])
        self.neural_memory_segment_len = default(neural_memory_segment_len, self.attn_window_size)

        layers_map = tuple(range(1, depth + 1))
        neural_memory_layers_set = set(default(neural_memory_layers, layers_map))
        self.neural_memory_layers_config = neural_memory_layers

        self.neural_mem_weight_residual = neural_mem_weight_residual
        self.neural_memory_qkv_receives_diff_views = neural_mem_qkv_receives_diff_views # Store the parameter
        is_first_neural_mem_layer_encountered = True

        for layer_idx in layers_map:
            is_first_layer_overall = layer_idx == 1

            attn = SegmentedAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                segment_len = segment_len,
                use_flex_attn = use_flex_attn,
                accept_value_residual = not is_first_layer_overall and neural_memory_add_value_residual,
                num_longterm_mem_tokens = num_longterm_mem_tokens,
                num_persist_mem_tokens = num_persist_mem_tokens,
                sliding = sliding_window_attn,
                rope_theta = rope_theta,
                norm_eps = norm_eps
            )

            mem = None
            mem_qkv_layer_selector = None
            mem_hyper_conn = None

            if layer_idx in neural_memory_layers_set:
                mem_hyper_conn = init_hyper_conn(add_branch_out_to_residual = not neural_mem_gate_attn_output)

                if not is_first_layer_overall and self.neural_memory_qkv_receives_diff_views: # Use stored parameter
                    num_prev_value_sources = (layer_idx - 1) * 4 + 1
                    mem_qkv_layer_selector = nn.Sequential(
                        nn.RMSNorm(dim, eps=norm_eps),
                        nn.Linear(dim, 3 * num_prev_value_sources),
                        Rearrange('... (views layers) -> views ... layers', views = 3),
                        nn.Softmax(dim = -1)
                    )

                current_neural_memory_kwargs = deepcopy(neural_memory_kwargs)
                if 'dim_head' not in current_neural_memory_kwargs:
                    current_neural_memory_kwargs['dim_head'] = dim_head
                if 'heads' not in current_neural_memory_kwargs:
                    current_neural_memory_kwargs['heads'] = heads

                current_neural_memory_kwargs['norm_eps'] = norm_eps


                mem = NeuralMemory(
                    dim = dim,
                    chunk_size = self.neural_memory_segment_len,
                    batch_size = neural_memory_batch_size,
                    model = deepcopy(neural_memory_model),
                    accept_weight_residual = neural_mem_weight_residual and not is_first_neural_mem_layer_encountered,
                    **current_neural_memory_kwargs
                )
                is_first_neural_mem_layer_encountered = False

            ff = FeedForward(dim = dim, mult = ff_mult, norm_eps=norm_eps)

            self.layers.append(ModuleList([
                mem_hyper_conn,
                init_hyper_conn(),
                init_hyper_conn(),
                mem_qkv_layer_selector,
                mem,
                attn,
                ff,
            ]))

        self.final_norm = nn.RMSNorm(dim, eps=norm_eps) # Added final_norm
        self.to_logits = Linear(dim, num_tokens)
        self.gate_attn_output = neural_mem_gate_attn_output

        assert not (use_flex_attn and not exists(flex_attention)), 'FlexAttention requested but not available.'
        self.use_flex_attn = use_flex_attn
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.neural_memory_add_value_residual = neural_memory_add_value_residual


    def seq_index_is_longterm(
        self,
        seq_index
    ):
        if self.num_longterm_mem_tokens == 0:
            return False

        block_len_for_pattern = self.segment_len + self.num_longterm_mem_tokens
        pos_in_block = seq_index % block_len_for_pattern

        return pos_in_block >= self.segment_len


    def seq_len_with_longterm_mem(
        self,
        seq_len
    ):
        if self.num_longterm_mem_tokens == 0 or seq_len == 0:
            return seq_len

        num_full_segments = (seq_len -1) // self.segment_len
        len_with_mem = num_full_segments * (self.segment_len + self.num_longterm_mem_tokens)

        remaining_data_tokens = seq_len - (num_full_segments * self.segment_len)
        len_with_mem += remaining_data_tokens

        return len_with_mem


    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.0,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(min_p = 0.01),
        show_progress = True,
        use_cache = True,
        disable_ttl: bool = False
    ):
        was_training = self.training
        self.eval()

        prompt_seq_len = prompt.shape[-1]
        out = prompt.clone()

        sample_num_times = max(0, seq_len - prompt_seq_len)

        current_inference_idx = 0

        max_len_for_pos_emb_calc = self.seq_len_with_longterm_mem(seq_len)

        factorized_pos_emb = None
        if use_cache and hasattr(self, 'axial_pos_emb') and self.axial_pos_emb is not None:
            axial_dims_for_sample = self.axial_pos_emb.get_axial_dims(max_len_for_pos_emb_calc)
            factorized_pos_emb = self.axial_pos_emb(axial_dims_for_sample, return_factorized=True)

        loop_cache = (current_inference_idx, None, None)

        if prompt_seq_len > 0:
            _, loop_cache = self.forward(
                out,
                return_loss=False,
                disable_flex_attn=not (self.use_flex_attn and out.is_cuda),
                cache=loop_cache,
                return_cache=True,
                factorized_pos_emb=factorized_pos_emb,
                disable_ttl=disable_ttl
            )
            current_inference_idx = loop_cache[0]


        pbar = tqdm.tqdm(total = sample_num_times, disable = not show_progress, desc="Generating (Titans)")

        for _ in range(sample_num_times):
            current_token_input = out[:, -1:]

            logits, next_loop_cache = self.forward(
                current_token_input,
                return_loss=False,
                disable_flex_attn=True,
                cache=loop_cache,
                return_cache=True,
                factorized_pos_emb=factorized_pos_emb,
                disable_ttl=disable_ttl
            )

            loop_cache = next_loop_cache
            current_inference_idx = loop_cache[0]

            if not exists(logits):
                if self.num_longterm_mem_tokens > 0 and self.seq_index_is_longterm(current_inference_idx -1):
                     pbar.update(0)
                if current_inference_idx >= max_len_for_pos_emb_calc :
                    break
                continue


            next_token_logits = logits[:, -1, :]

            next_token_logits = filter_fn(next_token_logits, **filter_kwargs)
            sample = gumbel_sample(next_token_logits, temperature = temperature)

            out = torch.cat((out, sample), dim = -1)
            pbar.update(1)

            if current_inference_idx >= max_len_for_pos_emb_calc or out.shape[-1] >= seq_len:
                break

        pbar.close()
        self.train(was_training)

        return out[..., prompt_seq_len:]


    def forward(
        self,
        x: Tensor,
        return_loss = False,
        return_loss_breakdown = False,
        disable_flex_attn = False,
        cache = None,
        return_cache = False,
        factorized_pos_emb: Optional[Tuple[Tensor, ...]] = None,
        disable_ttl: bool = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch_size, input_data_seq_len = x.shape[:2]

        current_token_idx_for_state = 0
        initial_kv_caches_per_layer = None
        initial_nm_states_per_layer = None

        is_processing_single_data_token = (input_data_seq_len == 1)

        if exists(cache):
            current_token_idx_for_state, initial_kv_caches_per_layer, initial_nm_states_per_layer = cache

        x_emb = self.token_emb(x)

        final_seq_for_attn_ff_list = []

        current_conceptual_idx = current_token_idx_for_state

        for i in range(input_data_seq_len):
            if self.num_longterm_mem_tokens > 0:
                while self.seq_index_is_longterm(current_conceptual_idx):
                    if self.longterm_mems is not None:
                        block_len_for_pattern = self.segment_len + self.num_longterm_mem_tokens
                        pos_in_block = current_conceptual_idx % block_len_for_pattern
                        longterm_mem_idx_in_block = pos_in_block - self.segment_len

                        if 0 <= longterm_mem_idx_in_block < self.num_longterm_mem_tokens:
                            mem_token_emb = self.longterm_mems[longterm_mem_idx_in_block].unsqueeze(0).repeat(batch_size, 1)
                            final_seq_for_attn_ff_list.append(mem_token_emb.unsqueeze(1))
                    current_conceptual_idx += 1

            final_seq_for_attn_ff_list.append(x_emb[:, i:i+1, :])
            current_conceptual_idx += 1

        if self.num_longterm_mem_tokens > 0 and not is_processing_single_data_token :
             while self.seq_index_is_longterm(current_conceptual_idx) and \
                   (current_conceptual_idx < current_token_idx_for_state + self.seq_len_with_longterm_mem(input_data_seq_len)):
                if self.longterm_mems is not None:
                    block_len_for_pattern = self.segment_len + self.num_longterm_mem_tokens
                    pos_in_block = current_conceptual_idx % block_len_for_pattern
                    longterm_mem_idx_in_block = pos_in_block - self.segment_len
                    if 0 <= longterm_mem_idx_in_block < self.num_longterm_mem_tokens:
                        mem_token_emb = self.longterm_mems[longterm_mem_idx_in_block].unsqueeze(0).repeat(batch_size, 1)
                        final_seq_for_attn_ff_list.append(mem_token_emb.unsqueeze(1))
                current_conceptual_idx += 1


        if not final_seq_for_attn_ff_list:
             final_seq_for_attn_ff = torch.empty((batch_size, 0, self.token_emb.embedding_dim), device=x.device, dtype=x_emb.dtype)
        else:
            final_seq_for_attn_ff = cat(final_seq_for_attn_ff_list, dim=1)

        actual_processed_len = final_seq_for_attn_ff.shape[1]

        if hasattr(self, 'axial_pos_emb') and self.axial_pos_emb is not None and actual_processed_len > 0:
            indices_for_pos_emb = torch.arange(
                current_token_idx_for_state,
                current_token_idx_for_state + actual_processed_len,
                device=x.device
            ).unsqueeze(0).repeat(batch_size, 1)

            pos_emb = self.axial_pos_emb.forward_tokens_at_indices(
                indices=indices_for_pos_emb,
                factorized_cache=factorized_pos_emb
            )
            final_seq_for_attn_ff = final_seq_for_attn_ff + pos_emb


        use_flex_attn_for_this_pass = final_seq_for_attn_ff.is_cuda and self.use_flex_attn and \
                                      not disable_flex_attn and not is_processing_single_data_token

        flex_attn_fn = None
        if use_flex_attn_for_this_pass and actual_processed_len > 0:
            q_len_for_mask = actual_processed_len
            block_mask = create_mac_block_mask(q_len_for_mask, self.attn_window_size, self.num_persist_mem_tokens, self.sliding_window_attn_mac)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        kv_caches_iter = iter(default(initial_kv_caches_per_layer, [None]*len(self.layers)))
        nm_states_iter = iter(default(initial_nm_states_per_layer, [None]*len(self.layers)))

        next_kv_caches_list = []
        next_nm_states_list = []

        value_residual_from_prev_attn = None
        mem_input_layers_for_nm_selector = []

        x_stream = self.expand_streams(final_seq_for_attn_ff)

        for layer_idx, (mem_hyper_conn, attn_hyper_conn, ff_hyper_conn, mem_qkv_layer_selector, mem, attn, ff) in enumerate(self.layers):
            retrieved_memory_output = None
            attn_output_gates_from_mem = None

            current_layer_nm_state_input = next(nm_states_iter, None)
            if exists(mem):
                mem_input_for_nm, add_residual_nm = mem_hyper_conn(x_stream)
                qkv_sources_for_nm_input_4d = None
                if not exists(mem_qkv_layer_selector):
                    qkv_sources_for_nm_input_4d = stack((mem_input_for_nm, mem_input_for_nm, mem_input_for_nm), dim=0)
                else:
                    choices = [mem_input_for_nm] + mem_input_layers_for_nm_selector if mem_input_layers_for_nm_selector else [mem_input_for_nm]
                    layers_to_choose_from = stack(choices, dim=0)
                    selected_weights = mem_qkv_layer_selector(mem_input_for_nm)
                    if layers_to_choose_from.shape[0] != selected_weights.shape[-1]:
                         pass
                    qkv_sources_for_nm_input_4d = einsum(layers_to_choose_from, selected_weights, 'l b n d, v b n l -> v b n d')

                if current_layer_nm_state_input is not None:
                    current_layer_nm_state_input = current_layer_nm_state_input._replace(seq_index=current_token_idx_for_state)

                retrieved_memory_output, next_nm_state = mem.forward(
                    qkv_sources_for_nm_input_4d,
                    state = current_layer_nm_state_input,
                    disable_ttl = disable_ttl,
                )
                next_nm_states_list.append(next_nm_state)

                if self.gate_attn_output and exists(retrieved_memory_output):
                    attn_output_gates_from_mem = retrieved_memory_output.sigmoid()
                elif exists(retrieved_memory_output):
                    x_stream = add_residual_nm(retrieved_memory_output)
            else:
                next_nm_states_list.append(None)

            attn_input, add_residual_attn = attn_hyper_conn(x_stream)
            mem_input_layers_for_nm_selector.append(attn_input)

            attn_output, attn_intermediates = attn(
                attn_input,
                value_residual = value_residual_from_prev_attn if self.neural_memory_add_value_residual else None,
                disable_flex_attn = disable_flex_attn,
                flex_attn_fn = flex_attn_fn,
                output_gating = attn_output_gates_from_mem,
                cache = next(kv_caches_iter, None)
            )
            mem_input_layers_for_nm_selector.append(attn_output)

            if self.neural_memory_add_value_residual:
                 value_residual_from_prev_attn = attn_intermediates.value_residual
            next_kv_caches_list.append(attn_intermediates.cached_key_values)

            x_stream = add_residual_attn(attn_output)
            ff_input, add_residual_ff = ff_hyper_conn(x_stream)
            mem_input_layers_for_nm_selector.append(ff_input)

            ff_output = ff(ff_input)
            mem_input_layers_for_nm_selector.append(ff_output)

            x_stream = add_residual_ff(ff_output)

        final_output_reduced = self.reduce_streams(x_stream)

        data_token_outputs_list = []
        temp_conceptual_idx = current_token_idx_for_state
        output_idx_in_final_reduced = 0
        for _ in range(input_data_seq_len):
            if self.num_longterm_mem_tokens > 0:
                while self.seq_index_is_longterm(temp_conceptual_idx):
                    temp_conceptual_idx += 1
                    output_idx_in_final_reduced +=1

            if output_idx_in_final_reduced < actual_processed_len:
                 data_token_outputs_list.append(final_output_reduced[:, output_idx_in_final_reduced:output_idx_in_final_reduced+1, :])
            temp_conceptual_idx += 1
            output_idx_in_final_reduced +=1

        final_logits = None
        if data_token_outputs_list:
            output_for_norm_and_logits = cat(data_token_outputs_list, dim=1)
            if output_for_norm_and_logits.shape[1] > 0:
                normed_output = self.final_norm(output_for_norm_and_logits) # Use self.final_norm
                final_logits = self.to_logits(normed_output)

        next_token_idx_for_state_calc = current_token_idx_for_state + actual_processed_len


        if return_cache:
            processed_kv_caches_for_next_step = []
            if next_kv_caches_list:
                for layer_kv_cache in next_kv_caches_list:
                    if layer_kv_cache is None or layer_kv_cache[0] is None or layer_kv_cache[1] is None:
                        processed_kv_caches_for_next_step.append(None)
                        continue
                    k_c, v_c = layer_kv_cache
                    max_cache_len = self.attn_window_size
                    if self.num_persist_mem_tokens > 0 :
                         max_cache_len -= self.num_persist_mem_tokens


                    if k_c is not None: k_c = k_c[..., -max_cache_len:, :]
                    if v_c is not None: v_c = v_c[..., -max_cache_len:, :]
                    processed_kv_caches_for_next_step.append((k_c, v_c) if k_c is not None and v_c is not None else None)

            cache_to_return = (
                next_token_idx_for_state_calc,
                processed_kv_caches_for_next_step if processed_kv_caches_for_next_step else None,
                next_nm_states_list if next_nm_states_list else None
            )
            return final_logits, cache_to_return


        if return_loss:
            if not exists(final_logits) or final_logits.shape[1] == 0:
                return torch.tensor(0.0, device=x.device, requires_grad=True)

            if final_logits.shape[1] != labels.shape[1]:
                 print(f"Warning: Mismatch between logit outputs ({final_logits.shape[1]}) and labels ({labels.shape[1]}) for loss. This is unexpected.")
                 min_len = min(final_logits.shape[1], labels.shape[1])
                 if min_len == 0: return torch.tensor(0.0, device=x.device, requires_grad=True)
                 final_logits_for_loss = final_logits[:, :min_len, :]
                 labels_for_loss = labels[:, :min_len]
            else:
                final_logits_for_loss = final_logits
                labels_for_loss = labels

            return F.cross_entropy(rearrange(final_logits_for_loss, 'b n l -> b l n'), labels_for_loss)

        return final_logits
