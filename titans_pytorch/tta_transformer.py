# FILE: titans_pytorch/tta_transformer.py

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
from tqdm import tqdm

from einops import rearrange, repeat, pack, unpack, einsum
from einops.layers.torch import Rearrange

from titans_pytorch.axial_positional_embedding import ContinuousAxialPositionalEmbedding
from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState, mem_state_detach
from titans_pytorch.mac_transformer import gumbel_sample, min_p_filter # Keep utility funcs
from titans_pytorch.memory_models import MemoryMLP

from titans_pytorch.qwen_direct_copy_transformer import (
    Qwen3CopiedAttention, Qwen3CopiedMLP, Qwen3Config,
    Qwen3CopiedRotaryEmbedding,
    _make_causal_mask as qwen_make_causal_mask,
    _expand_mask as qwen_expand_mask
)

# helpers
def exists(v): return v is not None
def default(v, d): return v if exists(v) else d
def identity(t): return t

# Cache structure for TTA
TTALayerCache = namedtuple('TTALayerCache', ['nm_state', 'attn_kv_cache']) # attn_kv_cache from Qwen3CopiedAttention
TTACache = namedtuple('TTACache', ['current_token_idx', 'layer_caches'])


class TTALayer(Module):
    def __init__(
        self,
        dim: int,
        attention_module: Qwen3CopiedAttention, # Type hint updated
        ff_module: Qwen3CopiedMLP,               # Type hint updated
        nm_module: Optional[NeuralMemory],
        norm_eps: float = 1e-6,
        nm_gate_initial_bias: float = -10.0
    ):
        super().__init__()
        self.dim = dim
        self.attention_module = attention_module
        self.ff_module = ff_module
        self.nm_module = nm_module # Can be None if mimic mode

        self.norm_input = nn.RMSNorm(dim, eps=norm_eps)
        self.norm_post_attn = nn.RMSNorm(dim, eps=norm_eps)

        if self.nm_module is not None:
            self.nm_gate_scale_logit = Parameter(torch.full((dim,), nm_gate_initial_bias))
        else:
            self.register_parameter('nm_gate_scale_logit', None)

    def forward(
        self,
        x: Tensor,
        nm_state_in: Optional[NeuralMemState] = None,
        # Args for Qwen3CopiedAttention (which is now always the attention_module)
        attention_mask_for_attn: Optional[Tensor] = None,
        position_embeddings_for_attn: Optional[Tuple[Tensor, Tensor]] = None,
        past_key_value_for_attn: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions_for_attn: bool = False,
        disable_ttl: bool = False
    ):
        residual = x
        x_normed1 = self.norm_input(x)

        attn_out, _attn_weights, new_kv_cache_attn = self.attention_module(
            hidden_states=x_normed1,
            attention_mask=attention_mask_for_attn,
            position_embeddings=position_embeddings_for_attn,
            past_key_value=past_key_value_for_attn,
            output_attentions=output_attentions_for_attn
        )
        # new_kv_cache_attn is the attn_kv_cache_out for this layer

        x_after_attn = residual + attn_out # First Residual

        residual_ff = x_after_attn
        x_normed2 = self.norm_post_attn(x_after_attn)
        ff_out = self.ff_module(x_normed2)

        gated_nm_contribution = torch.zeros_like(ff_out)
        nm_state_out = nm_state_in # Initialize with input state

        if self.nm_module is not None and self.nm_gate_scale_logit is not None:
            nm_input_for_branch = x_normed2 # Use the already normed input for FF as input to NM branch
            
            # Pass nm_state_in directly without modifying its seq_index here.
            # NeuralMemory module will manage its own seq_index.
            nm_out_raw, current_nm_state_out, surprises_from_nm = self.nm_module(
                nm_input_for_branch.unsqueeze(0), # NM expects (views, b, n, d)
                state=nm_state_in, 
                return_surprises=True,
                disable_ttl=disable_ttl
            )
            nm_state_out = current_nm_state_out # Update nm_state_out with the returned state
            gate_val = torch.sigmoid(self.nm_gate_scale_logit)
            gated_nm_contribution = nm_out_raw.squeeze(0) * gate_val # Squeeze views dim
            
            if not disable_ttl:
                # ADD DEBUG PRINT
                # print(f"DEBUG TTALayer: NM raw out norm: {torch.norm(nm_out_raw).item()}, Gate val mean: {torch.mean(gate_val).item()}, Gated NM contrib norm: {torch.norm(gated_nm_contribution).item()}")
                if surprises_from_nm and surprises_from_nm[0] is not None: # Check if surprises were returned
                    pass
                    # print(f"DEBUG TTALayer: NM Surprises (loss mean): {torch.mean(surprises_from_nm[0]).item() if surprises_from_nm[0].numel() > 0 else 'N/A'}")

        combined_ff_nm = ff_out + gated_nm_contribution
        output = residual_ff + combined_ff_nm # Second Residual

        layer_cache_out = TTALayerCache(nm_state=nm_state_out, attn_kv_cache=new_kv_cache_attn)
        return output, layer_cache_out

class TTATransformer(Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        qwen_config_for_mimic: Qwen3Config, # Now mandatory, not optional
        dim_head: int = 64, # Retained for NM default model, though QwenConfig has head_dim
        heads: int = 8,     # Retained for NM default model, though QwenConfig has num_attention_heads
        num_persist_mem_tokens: int = 0,
        ff_mult: int = 4, # Retained for NM default model, QwenConfig has intermediate_size
        neural_memory_model_proto: Optional[Module] = None,
        neural_memory_kwargs: dict = dict(),
        nm_chunk_size: int = 256, # Explicit NM chunk size
        token_emb: Optional[Module] = None,
        max_seq_len_for_axial_dims: Optional[int] = 2048,
        norm_eps: float = 1e-6, # RoPE theta will come from qwen_config
        nm_gate_initial_bias: float = -10.0,
        vanilla_qwen_mimic: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.vanilla_qwen_mimic = vanilla_qwen_mimic
        self.qwen_config = qwen_config_for_mimic

        self.token_emb = default(token_emb, nn.Embedding(num_tokens, dim))

        if not self.vanilla_qwen_mimic:
            self.axial_pos_emb = ContinuousAxialPositionalEmbedding(
                dim=dim, num_axial_dims=2, max_seq_len_derive_axial_from=max_seq_len_for_axial_dims
            )
            if num_persist_mem_tokens > 0:
                self.persistent_mems_for_attention = Parameter(torch.randn(num_persist_mem_tokens, dim))
            else:
                self.persistent_mems_for_attention = None
        else:
            self.axial_pos_emb = None
            self.persistent_mems_for_attention = None

        self.rotary_emb_for_layers = Qwen3CopiedRotaryEmbedding(config=self.qwen_config)


        self.layers = ModuleList([])
        for layer_idx in range(depth):
            attention_module_instance = Qwen3CopiedAttention(
                config=self.qwen_config,
                layer_idx=layer_idx
            )
            ff_module_instance = Qwen3CopiedMLP(config=self.qwen_config)

            nm_module_instance = None
            if not self.vanilla_qwen_mimic:
                nm_dim_head_default = self.qwen_config.head_dim if hasattr(self.qwen_config, 'head_dim') else dim_head
                nm_heads_default = self.qwen_config.num_attention_heads if hasattr(self.qwen_config, 'num_attention_heads') else heads
                nm_model_for_layer = default(deepcopy(neural_memory_model_proto), MemoryMLP(dim=nm_dim_head_default, depth=2))
                current_nm_kwargs_for_layer = deepcopy(neural_memory_kwargs)
                current_nm_kwargs_for_layer.setdefault('dim', dim)
                current_nm_kwargs_for_layer.setdefault('dim_head', nm_dim_head_default)
                current_nm_kwargs_for_layer.setdefault('heads', nm_heads_default)
                current_nm_kwargs_for_layer.setdefault('norm_eps', norm_eps)
                current_nm_kwargs_for_layer.setdefault('chunk_size', nm_chunk_size)
                nm_module_instance = NeuralMemory(model=nm_model_for_layer, **current_nm_kwargs_for_layer)

            self.layers.append(TTALayer(
                dim=dim,
                attention_module=attention_module_instance,
                ff_module=ff_module_instance,
                nm_module=nm_module_instance,
                norm_eps=norm_eps,
                nm_gate_initial_bias=nm_gate_initial_bias
            ))

        self.final_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.to_logits = Linear(dim, num_tokens)


    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        bsz, seq_len = input_shape
        causal_mask = qwen_make_causal_mask(
            (bsz, seq_len), inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length
        )
        if attention_mask is not None:
            if attention_mask.ndim == 2: # (bsz, seq_len_kv)
                expanded_attn_mask = qwen_expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_len)
                # expanded_attn_mask is (bsz, 1, tgt_len, src_len_from_mask)
                # causal_mask is (bsz, 1, tgt_len, tgt_len + past_kv_len)
                # We need to align the kv_len dimension.
                # If user mask covers full effective kv_len (past + current), it should align.
                # If user mask only covers current input (src_len_from_mask == tgt_len), pad for past.
                if expanded_attn_mask.shape[-1] == seq_len and past_key_values_length > 0: # This condition means user mask is for current input only
                    expanded_attn_mask = F.pad(expanded_attn_mask, (past_key_values_length, 0), value=0) # Pad for past keys, allowing attention

            elif attention_mask.ndim == 4: # Already expanded
                expanded_attn_mask = attention_mask
            else:
                raise ValueError(f"Unsupported attention_mask ndim: {attention_mask.ndim}")

            if expanded_attn_mask.shape[-1] == causal_mask.shape[-1]:
                # Qwen uses maximum, not minimum like Llama for combining masks
                causal_mask = torch.maximum(causal_mask, expanded_attn_mask)
            else:
                # This case should ideally be handled by padding in the qwen_expand_mask or here.
                # If shapes still don't match, it's an issue. For safety, log and use causal.
                # print(f"Warning: Attention mask shape mismatch after potential padding. User: {expanded_attn_mask.shape}, Causal: {causal_mask.shape}. Using only causal mask.")
                pass # Rely on causal mask if alignment failed
        return causal_mask

    def forward(
        self,
        x: Tensor,
        return_loss: bool = False,
        cache: Optional[TTACache] = None,
        return_cache: bool = False,
        factorized_pos_emb: Optional[Tuple[Tensor, ...]] = None, # Not used by TTA directly, RoPE is Qwen's
        disable_ttl: bool = False,
        attention_mask_external: Optional[Tensor] = None # User-provided mask (bsz, seq_len_kv)
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        batch_size, data_seq_len = x.shape # data_seq_len is the length of the new tokens in x

        current_token_idx = 0 # This is the starting conceptual index for RoPE for the current x
        layer_caches_in_list = [None] * len(self.layers)
        past_kv_len_from_cache = 0

        if exists(cache):
            current_token_idx, layer_caches_in_list = cache.current_token_idx, cache.layer_caches
            if layer_caches_in_list is None: layer_caches_in_list = [None] * len(self.layers)
            # Correctly access past_kv_len from the first layer's cache if available
            if layer_caches_in_list and len(layer_caches_in_list) > 0 and \
               layer_caches_in_list[0] is not None and \
               layer_caches_in_list[0].attn_kv_cache is not None and \
               layer_caches_in_list[0].attn_kv_cache[0] is not None: # Check key tensor
                past_kv_len_from_cache = layer_caches_in_list[0].attn_kv_cache[0].shape[2]


        x_emb = self.token_emb(x)
        current_input_for_layers = x_emb
        num_prepended_persist_effective = 0

        if not self.vanilla_qwen_mimic:
            if exists(self.persistent_mems_for_attention):
                if current_token_idx == 0: # Only prepend persist mems if at the very start of a sequence context
                    pm = repeat(self.persistent_mems_for_attention, 'n d -> b n d', b=batch_size)
                    current_input_for_layers = torch.cat((pm, current_input_for_layers), dim=1)
                    num_prepended_persist_effective = self.persistent_mems_for_attention.shape[0]

        effective_input_seq_len = current_input_for_layers.shape[1]

        position_ids_for_rope = torch.arange(
            past_kv_len_from_cache, # Start RoPE from where the cache left off
            past_kv_len_from_cache + effective_input_seq_len,
            dtype=torch.long, device=x.device
        ).unsqueeze(0)


        position_embeddings_for_attn_layers = self.rotary_emb_for_layers(current_input_for_layers, position_ids_for_rope)

        current_attention_mask_for_qwen_layers = None
        if attention_mask_external is not None:
            # Ensure attention_mask_external covers the effective_input_seq_len (data + persist)
            if num_prepended_persist_effective > 0:
                if attention_mask_external.shape[1] == data_seq_len: # User mask only for data tokens
                    persist_mask_part = torch.ones(batch_size, num_prepended_persist_effective, device=x.device, dtype=attention_mask_external.dtype)
                    current_attention_mask_for_qwen_layers = torch.cat([persist_mask_part, attention_mask_external], dim=1)
                elif attention_mask_external.shape[1] == effective_input_seq_len: # User mask for data + persist
                    current_attention_mask_for_qwen_layers = attention_mask_external
                else: # Fallback if mismatch
                    current_attention_mask_for_qwen_layers = torch.ones(batch_size, effective_input_seq_len, device=x.device, dtype=torch.long)
            else: # No persist tokens, user mask is for data tokens
                current_attention_mask_for_qwen_layers = attention_mask_external
        elif effective_input_seq_len > 0 : # If no external mask, create a default one allowing all attention
             current_attention_mask_for_qwen_layers = torch.ones(batch_size, effective_input_seq_len, device=x.device, dtype=torch.long)


        effective_attention_mask_for_attn_layers = self._prepare_decoder_attention_mask(
            current_attention_mask_for_qwen_layers,
            (batch_size, effective_input_seq_len),
            current_input_for_layers,
            past_kv_len_from_cache
        )

        next_layer_caches_list = []
        for i, layer_instance in enumerate(self.layers):
            layer_cache_in = layer_caches_in_list[i] if layer_caches_in_list and i < len(layer_caches_in_list) else None
            nm_state_in_this_layer = layer_cache_in.nm_state if exists(layer_cache_in) else None
            past_kv_for_this_layer_attn = layer_cache_in.attn_kv_cache if exists(layer_cache_in) else None

            # NM state's seq_index is managed internally by NeuralMemory.
            # Do not reset it here using current_token_idx from TTACache.

            current_input_for_layers, layer_cache_out = layer_instance(
                current_input_for_layers,
                nm_state_in=nm_state_in_this_layer, # Pass the NM state as is
                attention_mask_for_attn=effective_attention_mask_for_attn_layers,
                position_embeddings_for_attn=position_embeddings_for_attn_layers,
                past_key_value_for_attn=past_kv_for_this_layer_attn,
                output_attentions_for_attn=False,
                disable_ttl=disable_ttl
            )
            next_layer_caches_list.append(layer_cache_out)

        output_from_layers = current_input_for_layers
        if not self.vanilla_qwen_mimic and num_prepended_persist_effective > 0:
            output_from_layers = output_from_layers[:, num_prepended_persist_effective:, :]

        final_normed_output = self.final_norm(output_from_layers)
        logits = self.to_logits(final_normed_output)

        if return_cache:
            # TTACache.current_token_idx should reflect the total number of tokens processed
            # that contributed to the K/V cache.
            # If persist mems were prepended AND this is the first call (current_token_idx from input cache was 0),
            # they contribute to past_kv_len_from_cache for the *next* call.
            # The RoPE position IDs already account for past_kv_len_from_cache.
            # The new current_token_idx for the *output cache* should be the sum of
            # input cache's current_token_idx + length of new data tokens processed.
            # If persist tokens were added, they are part of the K/V cache now,
            # so the effective sequence length for K/V cache has increased by effective_input_seq_len.
            # The `current_token_idx` in TTACache tracks the conceptual end position of the K/V cache.
            next_total_processed_tokens_for_kv_cache = past_kv_len_from_cache + effective_input_seq_len
            cache_out = TTACache(current_token_idx=next_total_processed_tokens_for_kv_cache, layer_caches=next_layer_caches_list)
            return logits, cache_out

        if return_loss:
            if logits.shape[1] == 0: # Check if logits sequence is empty
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
        attention_mask: Optional[Tensor] = None, # User-provided mask for the prompt
        disable_ttl: bool = False, # Global TTL disable for this sample call (affects prompt processing)
        cache: Optional[TTACache] = None,
        return_cache_override: bool = False,
        analysis_callback_for_generation_fn: Optional[Callable] = None,
        analyze_tokens_target_ids_for_callback: Optional[List[int]] = None,
        analyze_tokens_start_tag_id_for_callback: Optional[int] = None,
        analyze_tokens_end_tag_id_for_callback: Optional[int] = None
    ):
        was_training = self.training
        self.eval()

        batch_size, prompt_len_orig = prompt.shape
        out = prompt.clone()

        current_model_cache = cache
        if current_model_cache is None:
             current_model_cache = TTACache(current_token_idx=0, layer_caches=[None] * len(self.layers))

        # Process the prompt part if it hasn't been processed into the cache yet.
        # current_model_cache.current_token_idx tracks the length of sequence already in K/V cache.
        if prompt_len_orig > current_model_cache.current_token_idx:
            # Only process the part of the prompt not yet in cache
            prompt_to_process_now = out[:, current_model_cache.current_token_idx:]
            attention_mask_for_prompt_processing = None
            if attention_mask is not None:
                # Adjust user-provided attention mask to align with prompt_to_process_now
                if attention_mask.shape[1] == prompt_len_orig: # Mask covers full original prompt
                    attention_mask_for_prompt_processing = attention_mask[:, current_model_cache.current_token_idx:]
                elif attention_mask.shape[1] == prompt_to_process_now.shape[1]: # Mask already for the part to process
                    attention_mask_for_prompt_processing = attention_mask
                # else: use None, _prepare_decoder_attention_mask will handle causal

            if prompt_to_process_now.numel() > 0: # Check if there's anything to process
                _, current_model_cache = self.forward(
                    prompt_to_process_now,
                    return_loss=False,
                    cache=current_model_cache, # Pass the current cache state
                    return_cache=True,
                    disable_ttl=disable_ttl, # TTL state for prompt processing
                    attention_mask_external=attention_mask_for_prompt_processing
                )
        
        # out contains the full sequence processed so far (original prompt + any newly generated)
        # current_model_cache.current_token_idx is now updated to reflect prompt_len_orig

        num_tokens_to_generate = max(0, seq_len - out.shape[1])

        pbar_desc = f"Generating (Titans TTA {'' if self.vanilla_qwen_mimic else ('TTL OFF' if disable_ttl else 'TTL ON') if not self.vanilla_qwen_mimic else 'Vanilla Mimic'})"
        pbar = tqdm(total=num_tokens_to_generate, disable=not show_progress, desc=pbar_desc)

        gen_step_counter = 0
        for _ in range(num_tokens_to_generate):
            if out.shape[1] >= seq_len: break

            current_token_input = out[:, -1:] # Next token to feed is the last generated token

            # For single token generation, attention_mask_external should allow attending to all previous tokens in `out`
            # The K/V cache in current_model_cache handles the past.
            # The _prepare_decoder_attention_mask inside self.forward will create the correct causal mask.
            # We can pass a simple mask indicating the valid length of `out` if needed, or rely on causal.
            # For single token generation, an attention_mask allowing it to see all previous cached tokens is implicit.
            # We only need to ensure the RoPE positions are correct, handled by current_model_cache.current_token_idx.
            step_attention_mask = torch.ones((batch_size, current_token_input.shape[1]), device=out.device, dtype=torch.long)


            logits, current_model_cache = self.forward(
                current_token_input,
                return_loss=False,
                cache=current_model_cache, # Pass updated cache
                return_cache=True,
                disable_ttl=False, #lets test instead of True, # TTL is OFF for generating subsequent tokens within a single sample call
                attention_mask_external=step_attention_mask # Mask for the single current token
            )

            next_token_logits = logits[:, -1, :]

            if analysis_callback_for_generation_fn is not None:
                analysis_callback_for_generation_fn(
                    next_token_logits.detach().clone(),
                    out.detach().clone(), # Pass the sequence generated so far
                    gen_step_counter
                )

            filtered_next_token_logits = filter_fn(next_token_logits, **filter_kwargs)
            sample_token = gumbel_sample(filtered_next_token_logits, temperature=temperature)
            out = torch.cat((out, sample_token), dim=-1)
            pbar.update(1)
            gen_step_counter += 1

            qwen_eos_ids_config = self.qwen_config.eos_token_id
            current_eos_set = set()
            if isinstance(qwen_eos_ids_config, int): current_eos_set.add(qwen_eos_ids_config)
            elif isinstance(qwen_eos_ids_config, list): current_eos_set.update(qwen_eos_ids_config)

            if sample_token.item() in current_eos_set:
                break

        pbar.close()
        self.train(was_training)

        generated_part = out[..., prompt_len_orig:]

        if return_cache_override:
            return generated_part, current_model_cache
        else:
            return generated_part
