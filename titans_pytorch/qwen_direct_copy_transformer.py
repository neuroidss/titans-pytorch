# FILE: titans_pytorch/qwen_direct_copy_transformer.py

# coding=utf-utf-8
# Copyright 2025 The Qwen team, Alibaba Group.
# Adapted for standalone use.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Tuple, Union, NamedTuple, List
import math
import logging # Standard Python logging

import torch
from torch import nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

# --- Locally Defined Configuration ---
class Qwen3Config:
    def __init__(self, vocab_size=151936, hidden_size=1024, intermediate_size=2816,
                 num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=8,
                 hidden_act="silu", max_position_embeddings=32768, initializer_range=0.02,
                 rms_norm_eps=1e-6, use_cache=True, pad_token_id=151643, eos_token_id=151643,
                 attention_bias=False, attention_dropout=0.0, rope_theta=10000.0,
                 rope_scaling=None, sliding_window=None, max_window_layers=float('inf'),
                 use_sliding_window=False, _attn_implementation="eager", head_dim=None,
                 classifier_dropout=None, # For TokenClassification
                 output_attentions=False, # Added
                 output_hidden_states=False, # Added
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id # Can be int or list
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.use_sliding_window = use_sliding_window
        self._attn_implementation = _attn_implementation
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.classifier_dropout = classifier_dropout
        self.output_attentions = output_attentions # Added
        self.output_hidden_states = output_hidden_states # Added


        if isinstance(eos_token_id, list) and len(eos_token_id) > 0:
            self.eos_token_id_single = eos_token_id[0] # For simple cases
        elif isinstance(eos_token_id, int):
            self.eos_token_id_single = eos_token_id
        else:
            self.eos_token_id_single = None


        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err
    
    def get_text_config(self): 
        return self


# --- Locally Defined Output Structures ---
class BaseModelOutputWithPast(NamedTuple):
    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class CausalLMOutputWithPast(NamedTuple):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

KVCacheType = Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]

# --- Sampling Helper ---
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

# --- Locally Defined Qwen3 components ---

class Qwen3CopiedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3CopiedMLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        elif config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            self.act_fn = lambda x: x 
            logger.warning(f"Unsupported activation function: {config.hidden_act}. Using identity.")


    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        slice_len = key_states.shape[-2]
        if slice_len > 0 :
            causal_mask = attention_mask[:, :, :, :slice_len]
            attn_weights = attn_weights + causal_mask
        elif attention_mask.shape[-1] == 0 and key_states.shape[-2] == 0 :
             pass 
    
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3CopiedRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        self.dim = config.head_dim
        self.max_seq_len_cached = config.max_position_embeddings
        self.base = config.rope_theta

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.attention_scaling = 1.0
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            scaling_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            if scaling_type == "linear":
                scaling_factor = config.rope_scaling.get("factor", 1.0)
                self.attention_scaling = scaling_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3CopiedAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.q_norm = Qwen3CopiedRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3CopiedRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.sliding_window = config.sliding_window
        if not (config.use_sliding_window and getattr(config, "sliding_window", None) is not None and layer_idx >= config.max_window_layers):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs: dict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        query_states = self.q_norm(query_states).transpose(1,2)
        key_states = self.k_norm(key_states).transpose(1,2)
        value_states = value_states.transpose(1,2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        current_key_value = (key_states, value_states)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, key_states], dim=-2) # dim is 2 for (bsz, num_heads, seq_len, head_dim)
            value_states = torch.cat([past_v, value_states], dim=-2)
            current_key_value = (key_states, value_states)
        
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, current_key_value


class Qwen3CopiedDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3CopiedAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3CopiedMLP(config)
        self.input_layernorm = Qwen3CopiedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3CopiedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False # Not implemented here

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # Passed for RoPE
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None, # Not directly used by layer, but by model
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: dict,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)

        # self_attn always returns three values: attn_output, attn_weights, new_key_value
        # attn_weights can be None if output_attentions is False.
        attn_outputs, self_attn_weights, new_key_value = self.self_attn(
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (new_key_value,)
        # No 'else' branch to append None if use_cache is False.
        # The tuple length will vary based on flags.

        return outputs


class Qwen3CopiedModuleBase(nn.Module):
    def __init__(self, config: Qwen3Config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, Qwen3Config):
            raise ValueError(f"Parameter config should be an instance of Qwen3Config. Got {config.__class__}")
        self.config = config
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3CopiedRMSNorm):
            module.weight.data.fill_(1.0)
    
    def post_init(self):
        pass

    def _gradient_checkpointing_func(self, func, *args):
        return func(*args)


class Qwen3CopiedModel(Qwen3CopiedModuleBase):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3CopiedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3CopiedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3CopiedRotaryEmbedding(config=config)
        self.post_init()


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        bsz, seq_len = input_shape
        
        causal_mask = _make_causal_mask(
            (bsz, seq_len), inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length
        )
        
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=seq_len).to(inputs_embeds.device)
            
            # Ensure expanded_attn_mask covers the full key length (past + current)
            # The key dimension of causal_mask is (tgt_len + past_key_values_length)
            # The key dimension of expanded_attn_mask is src_len (from original attention_mask)
            # We need to align them. expanded_attn_mask is for the *current* input tokens.
            # If past_key_values_length > 0, we need to pad expanded_attn_mask for the past keys.
            
            current_mask_key_len = expanded_attn_mask.shape[-1] # This is src_len from original mask
            expected_key_len_for_causal = causal_mask.shape[-1] # This is tgt_len + past_key_values_length
            
            if past_key_values_length > 0:
                # expanded_attn_mask is (bsz, 1, tgt_len, src_len)
                # We assume src_len from attention_mask corresponds to the current input's seq_len (tgt_len)
                # So, we need to pad for the past_key_values_length part.
                # The mask from user (attention_mask) applies to the *current* keys.
                # The causal part handles causality.
                # The padding part of the mask should apply to the *current* keys.
                # So, if expanded_attn_mask is (bsz, 1, tgt_len, tgt_len), and causal is (bsz, 1, tgt_len, tgt_len + past_len)
                # We need to combine them carefully.
                
                # Let's assume expanded_attn_mask is (bsz, 1, q_len, kv_len_current_input)
                # And causal_mask is (bsz, 1, q_len, kv_len_total)
                # We want to apply the user's padding mask to the current keys part of the total keys.
                
                # If user provides attention_mask (bsz, total_kv_len_including_past_if_any_from_user_perspective)
                # _expand_mask makes it (bsz, 1, q_len, total_kv_len_from_user_mask)
                # This needs to align with causal_mask's (bsz, 1, q_len, q_len + past_kv_len_from_cache)
                
                # Simpler: causal_mask already handles causality.
                # User's attention_mask handles padding for the *entire* sequence (past + current).
                # So, if attention_mask is (bsz, total_kv_sequence_length),
                # _expand_mask should make it (bsz, 1, q_len_current, total_kv_sequence_length)
                # This expanded_attn_mask can then be combined with causal_mask.
                
                # Re-evaluate _expand_mask: it takes mask (bsz, src_len) and tgt_len.
                # It creates (bsz, 1, tgt_len, src_len).
                # Here, src_len is the length of the attention_mask provided by user.
                # tgt_len is current input_ids.shape[1].
                
                # The causal_mask is (bsz, 1, current_q_len, current_q_len + past_kv_len)
                # The expanded_attn_mask from user is (bsz, 1, current_q_len, user_mask_len)
                # We need user_mask_len to be current_q_len + past_kv_len for direct combination.
                # This means the user's attention_mask should cover the full effective sequence.
                
                # If attention_mask is (bsz, current_q_len + past_kv_len)
                # then expanded_attn_mask will be (bsz, 1, current_q_len, current_q_len + past_kv_len)
                # This matches causal_mask shape.
                
                # If attention_mask is just (bsz, current_q_len) for the current input (e.g. during prompt processing with no past)
                # then expanded_attn_mask is (bsz, 1, current_q_len, current_q_len)
                # And causal_mask is (bsz, 1, current_q_len, current_q_len + past_kv_len)
                # We need to pad expanded_attn_mask for the past part.
                
                if expanded_attn_mask.shape[-1] < causal_mask.shape[-1]:
                    padding_for_past_keys = causal_mask.shape[-1] - expanded_attn_mask.shape[-1]
                    # Pad with 0 (allow attention) for past keys, assuming user mask only covers current input.
                    expanded_attn_mask = F.pad(expanded_attn_mask, (padding_for_past_keys, 0), value=0)


            causal_mask = torch.minimum(causal_mask, expanded_attn_mask)

        return causal_mask


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: KVCacheType = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, 
        **kwargs: dict,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = [None] * self.config.num_hidden_layers
        
        current_seq_len = inputs_embeds.shape[1]
        past_kv_len = 0
        if past_key_values is not None and past_key_values[0] is not None and past_key_values[0][0] is not None:
            past_kv_len = past_key_values[0][0].shape[2] 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_kv_len, past_kv_len + current_seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        input_shape = inputs_embeds.shape[:-1]
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_kv_len
        )
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # If output_attentions is True, attentions are at index 1
                if len(layer_outputs) > 1:
                    all_self_attns += (layer_outputs[1],)
            
            if use_cache:
                # If use_cache is True, the cache is the last element of layer_outputs
                if layer_outputs[-1] is not None: # Ensure cache is actually returned
                    next_decoder_cache.append(layer_outputs[-1])
                else: # Should not happen if use_cache=True and layer returns cache
                    next_decoder_cache.append(None)


        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen3CopiedForCausalLM(Qwen3CopiedModuleBase):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3CopiedModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.generation_config = config 
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: KVCacheType = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: dict,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        attention_mask: Optional[torch.Tensor] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs,
    ) -> torch.LongTensor:
        self.eval()
        
        if eos_token_id is None:
            eos_token_id = self.generation_config.eos_token_id_single 
        if isinstance(eos_token_id, int):
            eos_token_id_list = [eos_token_id]
        else: 
            eos_token_id_list = eos_token_id if eos_token_id is not None else []


        batch_size = input_ids.shape[0]
        current_input_ids = input_ids
        past_key_values: KVCacheType = None
        
        generated_sequences = input_ids.clone()

        for _ in range(max_new_tokens):
            seq_len = current_input_ids.shape[1] # This is 1 after the first token
            
            current_attention_mask = None
            if attention_mask is not None:
                # If this is the first token (prompt processing), use the provided mask.
                # For subsequent tokens, the mask needs to cover the whole sequence generated so far.
                if generated_sequences.shape[1] == input_ids.shape[1]: # Prompt phase
                    current_attention_mask = attention_mask
                else: # Generation phase
                    # Create a mask that attends to all previous tokens + current token
                    # This assumes left padding if attention_mask was provided for prompt.
                    # For simplicity, if generating token by token, the mask for the new token is 1.
                    # And we assume previous tokens in generated_sequences are valid.
                    current_attention_mask = torch.ones(batch_size, generated_sequences.shape[1], dtype=torch.long, device=input_ids.device)


            model_inputs = {
                "input_ids": current_input_ids, 
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": current_attention_mask 
            }
            
            outputs: CausalLMOutputWithPast = self.forward(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :] 

            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_sequences = torch.cat([generated_sequences, next_tokens], dim=-1)
            
            current_input_ids = next_tokens 
            past_key_values = outputs.past_key_values
            
            if eos_token_id_list and (next_tokens.item() in eos_token_id_list):
                break
        
        return generated_sequences


# Helper functions for mask creation
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    # Mask is 0 for tokens to attend to, 1 for tokens to ignore (like HF standard)
    # We need to convert it to additive mask: 0 for attend, large negative for ignore
    inverted_mask = 1.0 - expanded_mask # Now 1 for attend, 0 for ignore
    return inverted_mask.masked_fill(inverted_mask == 0, torch.finfo(dtype).min) # 0 for attend, min_dtype for ignore
