# FILE: titans_pytorch/run_titans_with_qwen_init.py

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import argparse
import sys
from copy import deepcopy
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
from typing import Callable, Tuple, Optional, List, Union, NamedTuple, Dict, Any
import re


from titans_pytorch import (
    MemoryAsContextTransformer,
    MAGTransformer, MAGCache, MAGLayerCache, MAGLayer,
    MALTransformer, MALCache, MALLayerCache, MALLayer,
    TTATransformer, TTACache, TTALayerCache, TTALayer,
    NeuralMemory, NeuralMemState,
    MemoryMLP,
    ContinuousAxialPositionalEmbedding,
    SegmentedAttention, FeedForward,
    Qwen3CopiedForCausalLM, Qwen3Config, Qwen3CopiedAttention, Qwen3CopiedMLP,
    QwenMimicMLP
)


QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EFFECTIVE_TITANS_VOCAB_SIZE = None
TITANS_DIM = None
TITANS_DEPTH = None
TITANS_FF_MULT = None

TITANS_MAC_SEGMENT_LEN = 256
TITANS_MAC_NEURAL_MEMORY_SEGMENT_LEN = TITANS_MAC_SEGMENT_LEN
TITANS_MAC_NUM_LONGTERM_MEM_TOKENS = 0
TITANS_MAC_NEURAL_MEM_QKV_DIFF_VIEWS = False
TITANS_MAC_NEURAL_MEMORY_ADD_VALUE_RESIDUAL = False
TITANS_MAC_NEURAL_MEM_GATE_ATTN_OUTPUT = False

TITANS_NM_CHUNK_SIZE_DEFAULT = 256
TITANS_NEURAL_MEMORY_BATCH_SIZE_FOR_TTL = 16

TITANS_NUM_RESIDUAL_STREAMS = 1
TITANS_NUM_PERSIST_MEM_TOKENS = 0
TITANS_NEURAL_MEM_QKV_DIFF_VIEWS_INTERNAL_NM = False

NM_INIT_ADAPTIVE_STEP_BIAS = 3.0/10
NM_INIT_MOMENTUM_BIAS = 1.0/10
NM_INIT_DECAY_BIAS = -2.0/10
INITIAL_WEIGHT_STD = 0.02 # 0.02
NM_GATE_INITIAL_BIAS_TTA = 0.0 # Default for TTALayer's NM gate
NM_TRANSITION_GATE_BIAS_FOR_TTL = 5.0/4 # 5.0 # New: For NeuralMemory's TTL update strength (sigmoid(5.0) approx 0.993)

MAX_SEQ_LEN_FOR_AXIAL_DIMS = 2048
NM_DEFAULT_STEP_TRANSFORM_MAX_LR = 1
NM_QK_L2NORM_ENABLED = True
NM_GATED_TRANSITION_ENABLED_FOR_TTL = True


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_code_from_response(response_text: str) -> Optional[str]:
    match = re.search(r"<code>(.*?)</code>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def load_qwen_components():
    logger.info(f"Loading Qwen tokenizer and model: {QWEN_MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        hf_model.eval().to(DEVICE)
    except Exception as e:
        logger.error(f"Error loading Qwen model: {e}. Ensure you have internet and model access.")
        sys.exit(1)
    logger.info("Qwen model loaded.")

    if hasattr(hf_model.model, 'layers') and hf_model.model.layers:
        first_qwen_layer = hf_model.model.layers[0]
        q_proj_example = first_qwen_layer.self_attn.q_proj
        k_proj_example = first_qwen_layer.self_attn.k_proj
        v_proj_example = first_qwen_layer.self_attn.v_proj
        o_proj_example = first_qwen_layer.self_attn.o_proj

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        else:
            new_pad_token = '<|pad|>'
            tokenizer.add_special_tokens({'pad_token': new_pad_token})
            hf_model.resize_token_embeddings(len(tokenizer))
            if hasattr(hf_model, 'config'):
                hf_model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"Added new pad_token: {new_pad_token}, ID: {tokenizer.pad_token_id}")

    if isinstance(hf_model.config.eos_token_id, list):
        tokenizer.eos_token_id_set = set(hf_model.config.eos_token_id)
    elif hf_model.config.eos_token_id is not None:
        tokenizer.eos_token_id_set = {hf_model.config.eos_token_id}
    elif tokenizer.eos_token_id is not None:
        tokenizer.eos_token_id_set = {tokenizer.eos_token_id}
    else:
        eos_tokens_in_vocab = [v for k, v in tokenizer.vocab.items() if "eos" in k.lower() or "im_end" in k.lower() or k == "<|endoftext|>"]
        if eos_tokens_in_vocab:
            tokenizer.eos_token_id_set = set(eos_tokens_in_vocab)
            logger.warning(f"EOS token ID not explicitly set in config or tokenizer. Derived from vocab: {tokenizer.eos_token_id_set}")
        else:
            tokenizer.eos_token_id_set = set()
            logger.critical("CRITICAL Warning: No EOS token ID could be determined. Generation might not stop correctly.")

    return tokenizer, hf_model

def initialize_titans_model(qwen_tokenizer, qwen_hf_model, model_type_str: str, vanilla_qwen_mimic=False, nm_gate_bias_override_for_tta_analysis=None):
    logger.info(f"Initializing Titans model (type: {model_type_str}) from Qwen model...")
    if vanilla_qwen_mimic and model_type_str == "tta":
        logger.info("TTA Mode: Vanilla Qwen Mimic. Titans-specific Neural Memory and features will be disabled.")
    elif vanilla_qwen_mimic:
        logger.info(f"Mode: Vanilla Qwen Mimic for {model_type_str}. Titans-specific features will be architecturally disabled or neutralized.")

    qwen_config = qwen_hf_model.config

    global EFFECTIVE_TITANS_VOCAB_SIZE, TITANS_DIM, TITANS_DEPTH, TITANS_FF_MULT

    EFFECTIVE_TITANS_VOCAB_SIZE = qwen_config.vocab_size
    TITANS_DIM = qwen_config.hidden_size
    TITANS_DEPTH = qwen_config.num_hidden_layers
    TITANS_FF_MULT = qwen_config.intermediate_size / qwen_config.hidden_size

    actual_q_dim_out = 0
    actual_kv_dim_out = 0
    actual_o_dim_in = 0
    actual_dim_head = 0
    actual_num_q_heads = 0
    actual_num_kv_heads = 0

    if hasattr(qwen_hf_model.model, 'layers') and qwen_hf_model.model.layers:
        first_qwen_layer = qwen_hf_model.model.layers[0]
        q_weight = first_qwen_layer.self_attn.q_proj.weight
        k_weight = first_qwen_layer.self_attn.k_proj.weight
        v_weight = first_qwen_layer.self_attn.v_proj.weight
        o_weight = first_qwen_layer.self_attn.o_proj.weight

        actual_q_dim_out = q_weight.shape[0]
        actual_kv_dim_out = k_weight.shape[0]
        actual_o_dim_in = o_weight.shape[1]

        config_num_q_heads = qwen_config.num_attention_heads
        config_num_kv_heads = qwen_config.num_key_value_heads

        if actual_q_dim_out % config_num_q_heads == 0:
            derived_head_dim_from_q = actual_q_dim_out // config_num_q_heads
            if actual_kv_dim_out % config_num_kv_heads == 0 and (actual_kv_dim_out // config_num_kv_heads) == derived_head_dim_from_q:
                if actual_o_dim_in == config_num_q_heads * derived_head_dim_from_q :
                    actual_dim_head = derived_head_dim_from_q
                    actual_num_q_heads = config_num_q_heads
                    actual_num_kv_heads = config_num_kv_heads

        if actual_dim_head == 0 and hasattr(qwen_config, 'head_dim') and qwen_config.head_dim is not None and qwen_config.head_dim > 0:
             config_dim_head_val = qwen_config.head_dim
             if actual_q_dim_out == config_num_q_heads * config_dim_head_val and \
                actual_kv_dim_out == config_num_kv_heads * config_dim_head_val and \
                actual_o_dim_in == config_num_q_heads * config_dim_head_val:
                 actual_dim_head = config_dim_head_val
                 actual_num_q_heads = config_num_q_heads
                 actual_num_kv_heads = config_num_kv_heads

        if actual_dim_head == 0 or actual_num_q_heads == 0 or actual_num_kv_heads == 0:
             logger.error(f"Error: Could not reliably derive actual attention parameters from loaded Qwen model.")
             logger.error(f"Derived: Q_dim_out={actual_q_dim_out}, KV_dim_out={actual_kv_dim_out}, O_dim_in={actual_o_dim_in}")
             logger.error(f"Config: Q_heads={config_num_q_heads}, KV_heads={config_num_kv_heads}, Head_dim_config={getattr(qwen_config, 'head_dim', 'N/A')}")
             sys.exit("Exiting due to attention dimension mismatch during Titans model initialization.")
    else:
        logger.error("Error: Cannot inspect Qwen layers to derive actual attention dimensions.")
        sys.exit(1)

    logger.info(f"Titans Config (derived from Qwen weights): Vocab={EFFECTIVE_TITANS_VOCAB_SIZE}, Dim={TITANS_DIM}, Depth={TITANS_DEPTH}")
    logger.info(f"Attention (derived from Qwen): Q_Heads={actual_num_q_heads}, KV_Heads={actual_num_kv_heads}, Dim_Head={actual_dim_head}")
    logger.info(f"FeedForward Multiplier={TITANS_FF_MULT}")

    rope_theta = getattr(qwen_config, 'rope_theta', 10000.0)
    norm_eps = getattr(qwen_config, 'rms_norm_eps', 1e-6)
    logger.info(f"Using RoPE Theta: {rope_theta}")
    logger.info(f"Using RMSNorm Epsilon: {norm_eps}")

    hf_config_for_qwen_parts = qwen_hf_model.config
    qwen_config_for_titans_internal = Qwen3Config(
        vocab_size=hf_config_for_qwen_parts.vocab_size, hidden_size=TITANS_DIM,
        intermediate_size=int(TITANS_DIM * TITANS_FF_MULT), num_hidden_layers=TITANS_DEPTH,
        num_attention_heads=actual_num_q_heads,
        num_key_value_heads=actual_num_kv_heads,
        head_dim=actual_dim_head,
        hidden_act=hf_config_for_qwen_parts.hidden_act, max_position_embeddings=hf_config_for_qwen_parts.max_position_embeddings,
        initializer_range=hf_config_for_qwen_parts.initializer_range, rms_norm_eps=norm_eps,
        use_cache=True,
        pad_token_id=qwen_tokenizer.pad_token_id,
        eos_token_id=list(qwen_tokenizer.eos_token_id_set),
        attention_bias=getattr(hf_config_for_qwen_parts, 'attention_bias', False),
        attention_dropout=hf_config_for_qwen_parts.attention_dropout,
        rope_theta=rope_theta,
        rope_scaling=getattr(hf_config_for_qwen_parts, 'rope_scaling', None),
        sliding_window=getattr(hf_config_for_qwen_parts, 'sliding_window', None),
        max_window_layers=getattr(hf_config_for_qwen_parts, 'max_window_layers', float('inf')),
        use_sliding_window=getattr(hf_config_for_qwen_parts, 'use_sliding_window', False),
        _attn_implementation = getattr(hf_config_for_qwen_parts, '_attn_implementation', "eager")
    )

    default_mem_model_for_nm = MemoryMLP(
        dim=actual_dim_head,
        depth=2,
        expansion_factor=4
    )
    if hasattr(default_mem_model_for_nm, 'model') and isinstance(default_mem_model_for_nm.model, nn.Sequential):
        for mlp_layer_module in default_mem_model_for_nm.model:
            if isinstance(mlp_layer_module, nn.Linear):
                if mlp_layer_module is default_mem_model_for_nm.model[-1]:
                    nn.init.normal_(mlp_layer_module.weight, 0, INITIAL_WEIGHT_STD / (TITANS_DEPTH if TITANS_DEPTH > 0 else 1))
                else:
                    nn.init.kaiming_uniform_(mlp_layer_module.weight, nonlinearity='relu')
                if mlp_layer_module.bias is not None:
                    nn.init.zeros_(mlp_layer_module.bias)
            elif isinstance(mlp_layer_module, nn.Sequential):
                for hidden_layer_submodule in mlp_layer_module:
                     if isinstance(hidden_layer_submodule, nn.Linear):
                        nn.init.kaiming_uniform_(hidden_layer_submodule.weight, nonlinearity='relu')
                        if hidden_layer_submodule.bias is not None:
                            nn.init.zeros_(hidden_layer_submodule.bias)
    else:
        logger.warning("Could not iterate over default_mem_model_for_nm.model for weight initialization. Structure might have changed.")

    logger.info("Using MemoryMLP(depth=2) for NeuralMemory's internal model.")

    neural_mem_kwargs_common_for_tta = dict(
        heads=actual_num_q_heads,
        dim_head=actual_dim_head,
        init_adaptive_step_bias=NM_INIT_ADAPTIVE_STEP_BIAS,
        init_momentum_bias=NM_INIT_MOMENTUM_BIAS,
        init_decay_bias=NM_INIT_DECAY_BIAS,
        qkv_receives_diff_views=TITANS_NEURAL_MEM_QKV_DIFF_VIEWS_INTERNAL_NM,
        activation = torch.nn.SiLU(),
        qk_l2norm = NM_QK_L2NORM_ENABLED,
        norm_eps = norm_eps,
        batch_size=TITANS_NEURAL_MEMORY_BATCH_SIZE_FOR_TTL,
        default_step_transform_max_lr=NM_DEFAULT_STEP_TRANSFORM_MAX_LR,
        gated_transition=NM_GATED_TRANSITION_ENABLED_FOR_TTL,
        init_transition_gate_bias=NM_TRANSITION_GATE_BIAS_FOR_TTL, # Use the new high bias for TTL
        mem_model_norm_add_residual=True
    )

    nm_batch_size_for_others = getattr(sys.modules[__name__], 'NEURAL_MEM_BATCH_SIZE', None)
    neural_mem_kwargs_common_others = deepcopy(neural_mem_kwargs_common_for_tta)
    neural_mem_kwargs_common_others['batch_size'] = nm_batch_size_for_others
    neural_mem_kwargs_common_others['gated_transition'] = False
    neural_mem_kwargs_common_others['init_transition_gate_bias'] = -5.0 # Keep default for non-TTA NM
    neural_mem_kwargs_common_others['mem_model_norm_add_residual'] = True

    titans_model_instance = None

    # Determine the NM gate bias for TTA layers
    effective_nm_gate_initial_bias_tta = NM_GATE_INITIAL_BIAS_TTA
    if model_type_str == "tta" and nm_gate_bias_override_for_tta_analysis is not None:
        effective_nm_gate_initial_bias_tta = nm_gate_bias_override_for_tta_analysis
        logger.info(f"Overriding TTA Layer NM Gate Bias for analysis: {effective_nm_gate_initial_bias_tta}")


    if model_type_str == "mac":
        effective_segment_len_for_mac = TITANS_MAC_SEGMENT_LEN
        current_use_flex_attn_for_mac = False
        current_num_longterm_mem_tokens_mac = TITANS_MAC_NUM_LONGTERM_MEM_TOKENS
        current_neural_memory_layers_mac = tuple(range(1, TITANS_DEPTH + 1))

        if vanilla_qwen_mimic:
            current_neural_memory_layers_mac = None
            current_num_longterm_mem_tokens_mac = 0
            qwen_max_pos_emb = getattr(qwen_config, 'max_position_embeddings', 32768)
            effective_segment_len_for_mac = qwen_max_pos_emb
            try:
                from titans_pytorch.mac_transformer import flex_attention as mac_flex_attn
                if mac_flex_attn is not None and torch.cuda.is_available(): current_use_flex_attn_for_mac = True
            except ImportError: pass

        titans_model_instance = MemoryAsContextTransformer(
            num_tokens=EFFECTIVE_TITANS_VOCAB_SIZE, dim=TITANS_DIM, depth=TITANS_DEPTH,
            heads=actual_num_q_heads, dim_head=actual_dim_head,
            segment_len=effective_segment_len_for_mac,
            use_flex_attn=current_use_flex_attn_for_mac,
            ff_mult=TITANS_FF_MULT,
            neural_memory_segment_len=TITANS_MAC_NEURAL_MEMORY_SEGMENT_LEN,
            neural_mem_qkv_receives_diff_views=TITANS_MAC_NEURAL_MEM_QKV_DIFF_VIEWS if not vanilla_qwen_mimic else False,
            num_residual_streams=TITANS_NUM_RESIDUAL_STREAMS if not vanilla_qwen_mimic else 1,
            num_persist_mem_tokens=TITANS_NUM_PERSIST_MEM_TOKENS if not vanilla_qwen_mimic else 0,
            num_longterm_mem_tokens=current_num_longterm_mem_tokens_mac,
            neural_memory_model=deepcopy(default_mem_model_for_nm) if not vanilla_qwen_mimic else None,
            neural_memory_kwargs=neural_mem_kwargs_common_others if not vanilla_qwen_mimic else {},
            neural_memory_layers=current_neural_memory_layers_mac,
            token_emb = torch.nn.Embedding(EFFECTIVE_TITANS_VOCAB_SIZE, TITANS_DIM),
            max_seq_len_for_axial_dims=MAX_SEQ_LEN_FOR_AXIAL_DIMS,
            neural_memory_add_value_residual=TITANS_MAC_NEURAL_MEMORY_ADD_VALUE_RESIDUAL if not vanilla_qwen_mimic else False,
            neural_mem_gate_attn_output=TITANS_MAC_NEURAL_MEM_GATE_ATTN_OUTPUT if not vanilla_qwen_mimic else False,
            rope_theta=rope_theta, norm_eps=norm_eps
        ).to(DEVICE)

    elif model_type_str == "mag":
        effective_swa_window_size = getattr(sys.modules[__name__], 'TITANS_ATTENTION_WINDOW_SIZE', 256)
        current_use_flex_attn_for_swa = False
        if vanilla_qwen_mimic:
            effective_swa_window_size = getattr(qwen_config, 'max_position_embeddings', 32768)
        titans_model_instance = MAGTransformer(
            num_tokens=EFFECTIVE_TITANS_VOCAB_SIZE, dim=TITANS_DIM, depth=TITANS_DEPTH,
            dim_head=actual_dim_head, heads=actual_num_q_heads,
            swa_window_size=effective_swa_window_size,
            num_persist_mem_tokens=TITANS_NUM_PERSIST_MEM_TOKENS if not vanilla_qwen_mimic else 0,
            ff_mult=TITANS_FF_MULT,
            neural_memory_model=deepcopy(default_mem_model_for_nm) if not vanilla_qwen_mimic else None,
            neural_memory_kwargs=neural_mem_kwargs_common_others if not vanilla_qwen_mimic else {},
            token_emb=torch.nn.Embedding(EFFECTIVE_TITANS_VOCAB_SIZE, TITANS_DIM),
            max_seq_len_for_axial_dims=MAX_SEQ_LEN_FOR_AXIAL_DIMS,
            rope_theta=rope_theta, norm_eps=norm_eps,
            use_flex_attn_for_swa=current_use_flex_attn_for_swa
        ).to(DEVICE)
        if vanilla_qwen_mimic:
            for layer in titans_model_instance.layers:
                if isinstance(layer, MAGLayer):
                    if hasattr(layer, 'nm_gate_scale') and layer.nm_gate_scale is not None:
                        layer.nm_gate_scale.data.fill_(-float('inf'))
                    if hasattr(layer, 'swa_gate_scale') and layer.swa_gate_scale is not None:
                        layer.swa_gate_scale.data.fill_(0.0)

    elif model_type_str == "mal":
             effective_swa_window_size = getattr(sys.modules[__name__], 'TITANS_ATTENTION_WINDOW_SIZE', 256)
             current_use_flex_attn_for_swa = False
             if vanilla_qwen_mimic:
                 effective_swa_window_size = getattr(qwen_config, 'max_position_embeddings', 32768)

             titans_model_instance = MALTransformer(
                 num_tokens=EFFECTIVE_TITANS_VOCAB_SIZE, dim=TITANS_DIM, depth=TITANS_DEPTH,
                 dim_head=actual_dim_head, heads=actual_num_q_heads,
                 swa_window_size=effective_swa_window_size,
                 num_persist_mem_tokens=TITANS_NUM_PERSIST_MEM_TOKENS if not vanilla_qwen_mimic else 0,
                 ff_mult=TITANS_FF_MULT,
                 neural_memory_model=deepcopy(default_mem_model_for_nm) if not vanilla_qwen_mimic else None,
                 neural_memory_kwargs=neural_mem_kwargs_common_others if not vanilla_qwen_mimic else {},
                 token_emb=torch.nn.Embedding(EFFECTIVE_TITANS_VOCAB_SIZE, TITANS_DIM),
                 max_seq_len_for_axial_dims=MAX_SEQ_LEN_FOR_AXIAL_DIMS,
                 rope_theta=rope_theta, norm_eps=norm_eps,
                 use_flex_attn_for_swa=current_use_flex_attn_for_swa
             ).to(DEVICE)

    elif model_type_str == "tta":
        if TITANS_NEURAL_MEMORY_BATCH_SIZE_FOR_TTL is not None:
            effective_nm_chunk_size_for_tta = TITANS_NEURAL_MEMORY_BATCH_SIZE_FOR_TTL
            logger.info(f"Setting NeuralMemory's internal chunk_size for TTA to match its batch_size (TITANS_NEURAL_MEMORY_BATCH_SIZE_FOR_TTL): {effective_nm_chunk_size_for_tta}")
        else:
            effective_nm_chunk_size_for_tta = TITANS_NM_CHUNK_SIZE_DEFAULT
            logger.warning(f"TITANS_NEURAL_MEMORY_BATCH_SIZE_FOR_TTL is None. Using default NM chunk size: {effective_nm_chunk_size_for_tta}. This might cause issues.")

        titans_model_instance = TTATransformer(
            num_tokens=EFFECTIVE_TITANS_VOCAB_SIZE, dim=TITANS_DIM, depth=TITANS_DEPTH,
            qwen_config_for_mimic=qwen_config_for_titans_internal,
            dim_head=actual_dim_head,
            heads=actual_num_q_heads,
            num_persist_mem_tokens=TITANS_NUM_PERSIST_MEM_TOKENS,
            ff_mult=TITANS_FF_MULT,
            neural_memory_model_proto=deepcopy(default_mem_model_for_nm),
            neural_memory_kwargs=neural_mem_kwargs_common_for_tta,
            nm_chunk_size=effective_nm_chunk_size_for_tta,
            token_emb=torch.nn.Embedding(EFFECTIVE_TITANS_VOCAB_SIZE, TITANS_DIM),
            max_seq_len_for_axial_dims=MAX_SEQ_LEN_FOR_AXIAL_DIMS,
            norm_eps=norm_eps,
            nm_gate_initial_bias=effective_nm_gate_initial_bias_tta, # Use the potentially overridden value
            vanilla_qwen_mimic=vanilla_qwen_mimic
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type_str for initialization: {model_type_str}")

    if hasattr(titans_model_instance, 'axial_pos_emb') and titans_model_instance.axial_pos_emb is not None:
        if not (model_type_str == "tta" and vanilla_qwen_mimic):
            logger.info("Initializing Axial Positional Embeddings to zero.")
            for param in titans_model_instance.axial_pos_emb.parameters():
                param.data.fill_(0.0)

    if model_type_str == "mac":
        if hasattr(titans_model_instance, 'longterm_mems') and titans_model_instance.longterm_mems is not None:
            logger.info(f"Initializing MAC longterm_mems to zero.")
            titans_model_instance.longterm_mems.data.fill_(0.0)

    if model_type_str != "tta":
        if hasattr(titans_model_instance, 'persistent_mems') and titans_model_instance.persistent_mems is not None:
            logger.info(f"Initializing common persistent_mems (for main attention) to zero for {model_type_str}.")
            titans_model_instance.persistent_mems.data.fill_(0.0)
    elif model_type_str == "tta" and not vanilla_qwen_mimic:
        if hasattr(titans_model_instance, 'persistent_mems_for_attention') and titans_model_instance.persistent_mems_for_attention is not None:
            logger.info(f"Initializing TTA persistent_mems_for_attention to zero.")
            titans_model_instance.persistent_mems_for_attention.data.fill_(0.0)

    if model_type_str == "mac" and hasattr(titans_model_instance, 'layers'):
        for layer_module_list in titans_model_instance.layers:
            attn_module = layer_module_list[5] if len(layer_module_list) > 5 else None
            if attn_module and isinstance(attn_module, SegmentedAttention) and \
               hasattr(attn_module, 'persistent_memory') and attn_module.persistent_memory is not None:
                 logger.info(f"Initializing MAC SegmentedAttention internal persistent_memory to zero.")
                 attn_module.persistent_memory.data.fill_(0.0)
    elif model_type_str in ["mag", "mal"] and hasattr(titans_model_instance, 'layers'):
        for layer_instance in titans_model_instance.layers:
            if hasattr(layer_instance, 'swa') and isinstance(layer_instance.swa, SegmentedAttention) and \
               hasattr(layer_instance.swa, 'persistent_memory') and layer_instance.swa.persistent_memory is not None:
                logger.info(f"Initializing {model_type_str.upper()} SWA internal persistent_memory to zero.")
                layer_instance.swa.persistent_memory.data.fill_(0.0)

    if not vanilla_qwen_mimic:
        if model_type_str == "mac":
            logger.info("Initializing MAC Titans-specific components for initial neutrality.")
            for layer_module_list in titans_model_instance.layers:
                mem_qkv_selector = layer_module_list[3]
                neural_mem_module = layer_module_list[4]
                attn_module = layer_module_list[5]

                if neural_mem_module is not None:
                    for proj_seq_attr_name in ["to_queries", "to_keys", "to_values"]:
                        proj_seq = getattr(neural_mem_module, proj_seq_attr_name, None)
                        if proj_seq is not None:
                            target_linear_layer = proj_seq[0] if isinstance(proj_seq, torch.nn.Sequential) and len(proj_seq) > 0 and isinstance(proj_seq[0], torch.nn.Linear) else None
                            if target_linear_layer:
                                target_linear_layer.weight.data.normal_(0, INITIAL_WEIGHT_STD)
                                if hasattr(target_linear_layer, 'bias') and target_linear_layer.bias is not None: target_linear_layer.bias.data.fill_(0.0)
                    if hasattr(neural_mem_module, 'combine_heads') and isinstance(neural_mem_module.combine_heads, torch.nn.Linear):
                        ch_linear = neural_mem_module.combine_heads
                        torch.nn.init.normal_(ch_linear.weight, 0, INITIAL_WEIGHT_STD / (actual_num_q_heads if actual_num_q_heads > 0 else 1))
                        if ch_linear.bias is not None: ch_linear.bias.data.fill_(0.0)
                    if hasattr(neural_mem_module, 'retrieve_gate') and neural_mem_module.retrieve_gate:
                         if len(neural_mem_module.retrieve_gate) > 0 and isinstance(neural_mem_module.retrieve_gate[0], torch.nn.Linear):
                            rg_linear = neural_mem_module.retrieve_gate[0]
                            rg_linear.weight.data.normal_(0, INITIAL_WEIGHT_STD)
                            if hasattr(rg_linear, 'bias') and rg_linear.bias is not None: rg_linear.bias.data.fill_(effective_nm_gate_initial_bias_tta) # Use effective bias

                if mem_qkv_selector and isinstance(mem_qkv_selector, torch.nn.Sequential) and \
                   len(mem_qkv_selector) > 1 and isinstance(mem_qkv_selector[1], torch.nn.Linear):
                    sel_linear = mem_qkv_selector[1]
                    sel_linear.weight.data.normal_(0, INITIAL_WEIGHT_STD)
                    if sel_linear.bias is not None: sel_linear.bias.data.fill_(0.0)

                if attn_module and hasattr(attn_module, 'to_learned_v_mix') and attn_module.to_learned_v_mix:
                    if len(attn_module.to_learned_v_mix) > 0 and isinstance(attn_module.to_learned_v_mix[0], torch.nn.Linear):
                        vmix_linear = attn_module.to_learned_v_mix[0]
                        vmix_linear.weight.data.normal_(0, INITIAL_WEIGHT_STD)
                        if hasattr(vmix_linear, 'bias') and vmix_linear.bias is not None: vmix_linear.bias.data.fill_(effective_nm_gate_initial_bias_tta - 5.0) # Use effective bias

        elif model_type_str == "mag":
            logger.info("Initializing MAG Titans-specific components for initial neutrality.")
            for layer in titans_model_instance.layers:
                if isinstance(layer, MAGLayer):
                    if layer.nm is not None:
                        for proj_seq_attr_name in ["to_queries", "to_keys", "to_values"]:
                            proj_seq = getattr(layer.nm, proj_seq_attr_name, None)
                            if proj_seq is not None:
                                target_linear_layer = proj_seq[0] if isinstance(proj_seq, torch.nn.Sequential) and len(proj_seq) > 0 and isinstance(proj_seq[0], torch.nn.Linear) else None
                                if target_linear_layer:
                                    target_linear_layer.weight.data.normal_(0, INITIAL_WEIGHT_STD)
                                    if hasattr(target_linear_layer, 'bias') and target_linear_layer.bias is not None: target_linear_layer.bias.data.fill_(0.0)
                        if hasattr(layer.nm, 'combine_heads') and isinstance(layer.nm.combine_heads, torch.nn.Linear):
                            ch_linear = layer.nm.combine_heads
                            torch.nn.init.normal_(ch_linear.weight, 0, INITIAL_WEIGHT_STD / (actual_num_q_heads if actual_num_q_heads > 0 else 1))
                            if ch_linear.bias is not None: ch_linear.bias.data.fill_(0.0)
                    layer.swa_gate_scale.data.normal_(0.0, 0.02)
                    layer.nm_gate_scale.data.normal_(effective_nm_gate_initial_bias_tta, 0.02) # Use effective bias

        elif model_type_str == "mal":
            logger.info("Initializing MAL Titans-specific components for initial neutrality.")
            for layer in titans_model_instance.layers:
                if isinstance(layer, MALLayer):
                    if layer.nm is not None:
                        for proj_seq_attr_name in ["to_queries", "to_keys", "to_values"]:
                            proj_seq = getattr(layer.nm, proj_seq_attr_name, None)
                            if proj_seq is not None:
                                target_linear_layer = proj_seq[0] if isinstance(proj_seq, torch.nn.Sequential) and len(proj_seq) > 0 and isinstance(proj_seq[0], torch.nn.Linear) else None
                                if target_linear_layer:
                                    target_linear_layer.weight.data.normal_(0, INITIAL_WEIGHT_STD)
                                    if hasattr(target_linear_layer, 'bias') and target_linear_layer.bias is not None: target_linear_layer.bias.data.fill_(0.0)
                        if hasattr(layer.nm, 'combine_heads') and isinstance(layer.nm.combine_heads, torch.nn.Linear):
                            ch_linear = layer.nm.combine_heads
                            torch.nn.init.normal_(ch_linear.weight, 0, INITIAL_WEIGHT_STD / (actual_num_q_heads if actual_num_q_heads > 0 else 1))
                            if ch_linear.bias is not None: ch_linear.bias.data.fill_(0.0)

        elif model_type_str == "tta":
            logger.info("Initializing TTA Titans-specific (Neural Memory) components.")
            for layer in titans_model_instance.layers:
                if isinstance(layer, TTALayer) and layer.nm_module is not None:
                     for proj_seq_attr_name in ["to_queries", "to_keys", "to_values"]:
                        proj_seq = getattr(layer.nm_module, proj_seq_attr_name, None)
                        if proj_seq is not None:
                            target_linear_layer = proj_seq[0] if isinstance(proj_seq, torch.nn.Sequential) and len(proj_seq) > 0 and isinstance(proj_seq[0], torch.nn.Linear) else None
                            if target_linear_layer:
                                target_linear_layer.weight.data.normal_(0, INITIAL_WEIGHT_STD)
                                if hasattr(target_linear_layer, 'bias') and target_linear_layer.bias is not None: target_linear_layer.bias.data.fill_(0.0)
                     if hasattr(layer.nm_module, 'combine_heads') and isinstance(layer.nm_module.combine_heads, torch.nn.Linear):
                        ch_linear = layer.nm_module.combine_heads
                        torch.nn.init.normal_(ch_linear.weight, 0, INITIAL_WEIGHT_STD / (actual_num_q_heads if actual_num_q_heads > 0 else 1))
                        if ch_linear.bias is not None: ch_linear.bias.data.fill_(0.0)
                     # TTALayer's nm_gate_scale_logit is already initialized in its constructor via nm_gate_initial_bias

    qwen_embed_tokens_weight = qwen_hf_model.model.embed_tokens.weight.detach().clone()
    num_embeddings_to_copy = min(EFFECTIVE_TITANS_VOCAB_SIZE, qwen_embed_tokens_weight.shape[0])
    if titans_model_instance.token_emb.weight.shape[1] == qwen_embed_tokens_weight.shape[1]:
        titans_model_instance.token_emb.weight.data[:num_embeddings_to_copy, :] = qwen_embed_tokens_weight.data[:num_embeddings_to_copy, :]
    else:
        logger.error(f"Error: Embedding dimension mismatch for token_emb. Qwen: {qwen_embed_tokens_weight.shape[1]}, Titans: {titans_model_instance.token_emb.weight.shape[1]}")
        sys.exit(1)

    qwen_lm_head_weight = qwen_hf_model.lm_head.weight.detach().clone()
    num_lm_head_outputs_to_copy = min(EFFECTIVE_TITANS_VOCAB_SIZE, qwen_lm_head_weight.shape[0])
    if titans_model_instance.to_logits.weight.shape[1] == qwen_lm_head_weight.shape[1]:
        titans_model_instance.to_logits.weight.data[:num_lm_head_outputs_to_copy, :] = qwen_lm_head_weight.data[:num_lm_head_outputs_to_copy, :]
        if hasattr(titans_model_instance.to_logits, 'bias') and titans_model_instance.to_logits.bias is not None:
            if hasattr(qwen_hf_model.lm_head, 'bias') and qwen_hf_model.lm_head.bias is not None:
                qwen_lm_head_bias = qwen_hf_model.lm_head.bias.detach().clone()
                num_lm_head_bias_elements_to_copy = min(EFFECTIVE_TITANS_VOCAB_SIZE, qwen_lm_head_bias.shape[0])
                titans_model_instance.to_logits.bias.data[:num_lm_head_bias_elements_to_copy] = qwen_lm_head_bias.data[:num_lm_head_bias_elements_to_copy]
            else:
                titans_model_instance.to_logits.bias.data.fill_(0.0)
    else:
        logger.error(f"Error: Dimension mismatch for lm_head weights. Qwen input dim: {qwen_lm_head_weight.shape[1]}, Titans input dim: {titans_model_instance.to_logits.weight.shape[1]}")
        sys.exit(1)

    if hasattr(qwen_hf_model.model, 'norm') and qwen_hf_model.model.norm is not None:
         if hasattr(titans_model_instance, 'final_norm') and titans_model_instance.final_norm is not None:
            if titans_model_instance.final_norm.weight.shape == qwen_hf_model.model.norm.weight.shape:
                titans_model_instance.final_norm.weight.data.copy_(qwen_hf_model.model.norm.weight.detach().clone())
            else:
                logger.warning(f"Warning: Shape mismatch for final_norm weights. Qwen: {qwen_hf_model.model.norm.weight.shape}, Titans ({model_type_str}): {titans_model_instance.final_norm.weight.shape}. Skipping final norm weight copy.")
         else:
            logger.warning(f"Warning: Titans model ({model_type_str}) does not have a `final_norm` attribute. Skipping final norm weight copy.")
    else:
        logger.warning("Warning: Qwen model does not have a final norm at `model.norm`. Skipping final norm weight copy.")

    num_layers_to_copy = min(len(qwen_hf_model.model.layers), len(titans_model_instance.layers))
    logger.info(f"Copying {num_layers_to_copy} transformer layers...")

    for i in tqdm(range(num_layers_to_copy), desc="Copying Layers"):
        qwen_layer = qwen_hf_model.model.layers[i]
        titans_layer_generic = titans_model_instance.layers[i]

        titans_attn_module = None
        titans_ff_module_actual_mlp = None
        titans_norm_input_module = None
        titans_norm_post_attn_module = None

        if model_type_str == "mac":
            titans_attn_module = titans_layer_generic[5]
            titans_ff_module_container = titans_layer_generic[6]
            if hasattr(titans_attn_module, 'norm'): titans_norm_input_module = titans_attn_module.norm
            if isinstance(titans_ff_module_container, torch.nn.Sequential) and len(titans_ff_module_container) > 0:
                 titans_norm_post_attn_module = titans_ff_module_container[0]
                 titans_ff_module_actual_mlp = titans_ff_module_container[1]

        elif model_type_str == "mag":
            if isinstance(titans_layer_generic, MAGLayer):
                titans_attn_module = titans_layer_generic.swa
                titans_ff_module_container = titans_layer_generic.ff
                titans_norm_input_module = titans_layer_generic.norm_input
                titans_norm_post_attn_module = titans_layer_generic.norm_after_gate
                if isinstance(titans_ff_module_container, torch.nn.Sequential) and len(titans_ff_module_container) > 1:
                    titans_ff_module_actual_mlp = titans_ff_module_container[1]

        elif model_type_str == "mal":
             if isinstance(titans_layer_generic, MALLayer):
                titans_attn_module = titans_layer_generic.swa
                titans_ff_module_container = titans_layer_generic.ff
                titans_norm_input_module = titans_layer_generic.norm_before_nm
                titans_norm_post_attn_module = titans_layer_generic.norm_before_ff
                if isinstance(titans_ff_module_container, torch.nn.Sequential) and len(titans_ff_module_container) > 1:
                    titans_ff_module_actual_mlp = titans_ff_module_container[1]

        elif model_type_str == "tta":
             if isinstance(titans_layer_generic, TTALayer):
                titans_attn_module = titans_layer_generic.attention_module
                titans_ff_module_actual_mlp = titans_layer_generic.ff_module
                titans_norm_input_module = titans_layer_generic.norm_input
                titans_norm_post_attn_module = titans_layer_generic.norm_post_attn

        if titans_attn_module is None or titans_ff_module_actual_mlp is None or \
           titans_norm_input_module is None or titans_norm_post_attn_module is None:
            logger.error(f"Error: Could not find all required attention/FF/norm modules for layer {i} in Titans model type {model_type_str}")
            sys.exit(1)

        if isinstance(titans_attn_module, Qwen3CopiedAttention):
            titans_attn_module.q_proj.weight.data.copy_(qwen_layer.self_attn.q_proj.weight.detach().clone())
            titans_attn_module.k_proj.weight.data.copy_(qwen_layer.self_attn.k_proj.weight.detach().clone())
            titans_attn_module.v_proj.weight.data.copy_(qwen_layer.self_attn.v_proj.weight.detach().clone())
            titans_attn_module.o_proj.weight.data.copy_(qwen_layer.self_attn.o_proj.weight.detach().clone())
            if qwen_config_for_titans_internal.attention_bias:
                if hasattr(titans_attn_module.q_proj, 'bias') and titans_attn_module.q_proj.bias is not None and qwen_layer.self_attn.q_proj.bias is not None: titans_attn_module.q_proj.bias.data.copy_(qwen_layer.self_attn.q_proj.bias.detach().clone())
                if hasattr(titans_attn_module.k_proj, 'bias') and titans_attn_module.k_proj.bias is not None and qwen_layer.self_attn.k_proj.bias is not None: titans_attn_module.k_proj.bias.data.copy_(qwen_layer.self_attn.k_proj.bias.detach().clone())
                if hasattr(titans_attn_module.v_proj, 'bias') and titans_attn_module.v_proj.bias is not None and qwen_layer.self_attn.v_proj.bias is not None: titans_attn_module.v_proj.bias.data.copy_(qwen_layer.self_attn.v_proj.bias.detach().clone())
                if hasattr(titans_attn_module.o_proj, 'bias') and titans_attn_module.o_proj.bias is not None and qwen_layer.self_attn.o_proj.bias is not None: titans_attn_module.o_proj.bias.data.copy_(qwen_layer.self_attn.o_proj.bias.detach().clone())

            if hasattr(qwen_layer.self_attn, 'q_norm') and qwen_layer.self_attn.q_norm is not None and \
               hasattr(titans_attn_module, 'q_norm') and titans_attn_module.q_norm is not None:
                titans_attn_module.q_norm.weight.data.copy_(qwen_layer.self_attn.q_norm.weight.data)
            if hasattr(qwen_layer.self_attn, 'k_norm') and qwen_layer.self_attn.k_norm is not None and \
               hasattr(titans_attn_module, 'k_norm') and titans_attn_module.k_norm is not None:
                titans_attn_module.k_norm.weight.data.copy_(qwen_layer.self_attn.k_norm.weight.data)

        elif isinstance(titans_attn_module, SegmentedAttention):
            q_w = qwen_layer.self_attn.q_proj.weight.detach().clone()
            k_w = qwen_layer.self_attn.k_proj.weight.detach().clone()
            v_w = qwen_layer.self_attn.v_proj.weight.detach().clone()

            effective_k_w = k_w
            effective_v_w = v_w
            if actual_num_kv_heads < actual_num_q_heads:
                repeat_factor = actual_num_q_heads // actual_num_kv_heads
                k_w_reshaped = k_w.view(actual_num_kv_heads, actual_dim_head, TITANS_DIM)
                effective_k_w = k_w_reshaped.repeat_interleave(repeat_factor, dim=0).view(-1, TITANS_DIM)
                v_w_reshaped = v_w.view(actual_num_kv_heads, actual_dim_head, TITANS_DIM)
                effective_v_w = v_w_reshaped.repeat_interleave(repeat_factor, dim=0).view(-1, TITANS_DIM)

            combined_qkv_weight = torch.cat([q_w, effective_k_w, effective_v_w], dim=0)
            titans_attn_module.to_qkv.weight.data.copy_(combined_qkv_weight)

            titans_attn_module.to_out.weight.data.copy_(qwen_layer.self_attn.o_proj.weight.detach().clone())
            if hasattr(titans_attn_module.to_out, 'bias') and titans_attn_module.to_out.bias is not None:
                if hasattr(qwen_layer.self_attn.o_proj, 'bias') and qwen_layer.self_attn.o_proj.bias is not None:
                     titans_attn_module.to_out.bias.data.copy_(qwen_layer.self_attn.o_proj.bias.detach().clone())
                else:
                     titans_attn_module.to_out.bias.data.fill_(0.0)

        if isinstance(titans_ff_module_actual_mlp, (QwenMimicMLP, Qwen3CopiedMLP)):
            titans_ff_module_actual_mlp.gate_proj.weight.data.copy_(qwen_layer.mlp.gate_proj.weight.detach().clone())
            titans_ff_module_actual_mlp.up_proj.weight.data.copy_(qwen_layer.mlp.up_proj.weight.detach().clone())
            titans_ff_module_actual_mlp.down_proj.weight.data.copy_(qwen_layer.mlp.down_proj.weight.detach().clone())
            if isinstance(titans_ff_module_actual_mlp, Qwen3CopiedMLP):
                if hasattr(titans_ff_module_actual_mlp.down_proj, 'bias') and titans_ff_module_actual_mlp.down_proj.bias is not None and \
                   hasattr(qwen_layer.mlp.down_proj, 'bias') and qwen_layer.mlp.down_proj.bias is not None:
                    titans_ff_module_actual_mlp.down_proj.bias.data.copy_(qwen_layer.mlp.down_proj.bias.detach().clone())

        if hasattr(titans_norm_input_module, 'weight'):
            titans_norm_input_module.weight.data.copy_(qwen_layer.input_layernorm.weight.detach().clone())

        if hasattr(titans_norm_post_attn_module, 'weight'):
            titans_norm_post_attn_module.weight.data.copy_(qwen_layer.post_attention_layernorm.weight.detach().clone())

    logger.info(f"Finished copying {num_layers_to_copy} layers.")
    titans_model_instance.eval()
    logger.info(f"Titans model (type: {model_type_str}) initialized and weights copied.")
    return titans_model_instance

def generate_response_titans(
    model, tokenizer, prompt_text, max_new_tokens=50, temperature=0.6, min_p=0.01,
    disable_ttl=False, model_type_str="mac", cache_for_generation=None,
    return_cache_after_generation=False,
    analysis_callback_for_generation_fn: Optional[Callable] = None,
    analyze_tokens_target_ids_for_callback: Optional[List[int]] = None,
    analyze_tokens_start_tag_id_for_callback: Optional[int] = None,
    analyze_tokens_end_tag_id_for_callback: Optional[int] = None,
    is_pre_templated_text: bool = False
):
    is_nm_active_structurally = True
    if hasattr(model, 'vanilla_qwen_mimic') and model.vanilla_qwen_mimic:
        mode_str = f"Vanilla Qwen Mimic ({model_type_str.upper()} - NM Disabled)"
        is_nm_active_structurally = False
    else:
        mode_str = "TTL Disabled" if disable_ttl else "TTL Enabled"

    log_prompt_display = prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text
    logger.info(f"\n--- Generating with TITANS ({model_type_str.upper()}) for Prompt ({mode_str}) ---\n{log_prompt_display}\n")

    templated_prompt_text_for_model = ""
    if is_pre_templated_text:
        templated_prompt_text_for_model = prompt_text
    else:
        messages_for_template = [{"role": "user", "content": prompt_text}]
        try:
            if hasattr(tokenizer, 'apply_chat_template'):
                templated_prompt_text_for_model = tokenizer.apply_chat_template(
                    messages_for_template, tokenize=False, add_generation_prompt=True
                )
            else:
                templated_prompt_text_for_model = "user: " + prompt_text + "\nassistant:"
                logger.warning("Warning: tokenizer.apply_chat_template not found. Using basic prompt formatting.")

        except Exception as e:
            logger.warning(f"Warning: Failed to apply chat template locally: {e}. Using basic prompt text formatting.")
            templated_prompt_text_for_model = "user: " + prompt_text + "\nassistant:"


    input_ids = tokenizer.encode(templated_prompt_text_for_model, return_tensors="pt").to(DEVICE)
    attention_mask_for_generation = torch.ones_like(input_ids)


    sample_kwargs = {
        "prompt": input_ids,
        "seq_len": input_ids.shape[1] + max_new_tokens,
        "temperature": temperature,
        "filter_kwargs": dict(min_p=min_p),
        "show_progress": False,
        "attention_mask": attention_mask_for_generation
    }

    effective_disable_ttl_val = disable_ttl
    if not is_nm_active_structurally :
        effective_disable_ttl_val = True

    sample_kwargs["disable_ttl"] = effective_disable_ttl_val

    if model_type_str == "tta":
        sample_kwargs["cache"] = cache_for_generation
        sample_kwargs["return_cache_override"] = return_cache_after_generation
        if analysis_callback_for_generation_fn:
            sample_kwargs["analysis_callback_for_generation_fn"] = analysis_callback_for_generation_fn
            # These were specific to the old callback, might not be needed directly by TTATransformer.sample
            # but the callback itself will use them.
            # sample_kwargs["analyze_tokens_target_ids_for_callback"] = analyze_tokens_target_ids_for_callback
            # sample_kwargs["analyze_tokens_start_tag_id_for_callback"] = analyze_tokens_start_tag_id_for_callback
            # sample_kwargs["analyze_tokens_end_tag_id_for_callback"] = analyze_tokens_end_tag_id_for_callback


    elif model_type_str == "mac":
        sample_kwargs["use_cache"] = True
        if cache_for_generation is not None: sample_kwargs["cache"] = cache_for_generation

    elif model_type_str in ["mag", "mal"]:
        sample_kwargs["cache"] = cache_for_generation
        if hasattr(model, 'sample') and 'return_cache_override' in model.sample.__code__.co_varnames:
            sample_kwargs["return_cache_override"] = return_cache_after_generation


    returned_from_sample = model.sample(**sample_kwargs)

    generated_suffix_ids_tensor = None
    updated_cache_after_generation = None

    if return_cache_after_generation and isinstance(returned_from_sample, tuple) and len(returned_from_sample) == 2:
        generated_suffix_ids_tensor, updated_cache_after_generation = returned_from_sample
    else:
        generated_suffix_ids_tensor = returned_from_sample

    generated_suffix_ids = generated_suffix_ids_tensor[0].tolist()

    assistant_response_cleaned = tokenizer.decode(generated_suffix_ids, skip_special_tokens=True).strip()

    if return_cache_after_generation:
        return assistant_response_cleaned, updated_cache_after_generation
    else:
        return assistant_response_cleaned

def initialize_qwen_direct_copy_model(qwen_tokenizer, qwen_hf_model):
    logger.info("Initializing Qwen Direct Architectural Copy model...")
    qwen_config_orig = qwen_hf_model.config
    copied_config = Qwen3Config(
        vocab_size=qwen_config_orig.vocab_size, hidden_size=qwen_config_orig.hidden_size,
        intermediate_size=qwen_config_orig.intermediate_size, num_hidden_layers=qwen_config_orig.num_hidden_layers,
        num_attention_heads=qwen_config_orig.num_attention_heads,
        num_key_value_heads=qwen_config_orig.num_key_value_heads,
        hidden_act=qwen_config_orig.hidden_act, max_position_embeddings=qwen_config_orig.max_position_embeddings,
        initializer_range=qwen_config_orig.initializer_range, rms_norm_eps=qwen_config_orig.rms_norm_eps,
        use_cache=True,
        pad_token_id=qwen_tokenizer.pad_token_id,
        eos_token_id=list(qwen_tokenizer.eos_token_id_set if hasattr(qwen_tokenizer, 'eos_token_id_set') else [qwen_tokenizer.eos_token_id]),
        attention_bias=getattr(qwen_config_orig, 'attention_bias', False),
        attention_dropout=qwen_config_orig.attention_dropout,
        rope_theta=getattr(qwen_config_orig, 'rope_theta', 10000.0),
        rope_scaling=getattr(qwen_config_orig, 'rope_scaling', None),
        sliding_window=getattr(qwen_config_orig, 'sliding_window', None),
        max_window_layers=getattr(qwen_config_orig, 'max_window_layers', float('inf')),
        use_sliding_window=getattr(qwen_config_orig, 'use_sliding_window', False),
        _attn_implementation=getattr(qwen_config_orig, '_attn_implementation', "eager"),
        head_dim = getattr(qwen_config_orig, 'head_dim', qwen_config_orig.hidden_size // qwen_config_orig.num_attention_heads)
    )
    direct_copy_model = Qwen3CopiedForCausalLM(copied_config).to(DEVICE)
    direct_copy_model.model.embed_tokens.weight.data.copy_(qwen_hf_model.model.embed_tokens.weight.data)
    direct_copy_model.lm_head.weight.data.copy_(qwen_hf_model.lm_head.weight.data)
    if hasattr(qwen_hf_model.model, 'norm') and qwen_hf_model.model.norm is not None:
         if hasattr(direct_copy_model.model, 'norm') and direct_copy_model.model.norm is not None:
            direct_copy_model.model.norm.weight.data.copy_(qwen_hf_model.model.norm.weight.data)

    for i in range(copied_config.num_hidden_layers):
        direct_copy_model.model.layers[i].load_state_dict(qwen_hf_model.model.layers[i].state_dict(), strict=True)

    direct_copy_model.eval()
    logger.info("Qwen Direct Architectural Copy model initialized and weights copied.")
    return direct_copy_model

def generate_response_qwen_hf_style_copied(model, tokenizer, prompt_text, max_new_tokens=50, disable_ttl=None, cache_for_generation=None, return_cache_after_generation=False):
    logger.info(f"\n--- Generating with Qwen Direct Copy for Prompt ---\n{prompt_text}\n")

    messages = [{"role": "user", "content": prompt_text}]
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            text_prompt_for_model = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text_prompt_for_model = "user: " + prompt_text + "\nassistant:"
            logger.warning("Warning: tokenizer.apply_chat_template not found for Qwen Direct Copy. Using basic prompt formatting.")

    except Exception as e:
        logger.warning(f"Warning: Could not apply chat template for Qwen Direct Copy: {e}. Using basic prompt.")
        text_prompt_for_model = "user: " + prompt_text + "\nassistant:"

    model_inputs = tokenizer([text_prompt_for_model], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        eos_token_ids_list = list(tokenizer.eos_token_id_set)
        if not eos_token_ids_list and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            eos_token_ids_list = [tokenizer.eos_token_id]
        elif not eos_token_ids_list: eos_token_ids_list = None

        gen_config = model.generation_config if hasattr(model, 'generation_config') else qwen_hf_model_global.generation_config

        current_pad_token_id = tokenizer.pad_token_id
        if current_pad_token_id is None:
            current_pad_token_id = gen_config.pad_token_id if hasattr(gen_config, 'pad_token_id') and gen_config.pad_token_id is not None else \
                                 (gen_config.eos_token_id if hasattr(gen_config, 'eos_token_id') and gen_config.eos_token_id is not None and not isinstance(gen_config.eos_token_id, list) else \
                                 (gen_config.eos_token_id[0] if hasattr(gen_config, 'eos_token_id') and isinstance(gen_config.eos_token_id, list) and gen_config.eos_token_id else \
                                 (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else \
                                 (next(iter(tokenizer.eos_token_id_set)) if tokenizer.eos_token_id_set else 0))))

        generated_ids_direct_copy = model.generate(
            input_ids=model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=current_pad_token_id,
            eos_token_id=eos_token_ids_list if eos_token_ids_list else gen_config.eos_token_id,
            do_sample=getattr(gen_config, "do_sample", True),
            temperature=getattr(gen_config, "temperature", 0.6),
            top_p=getattr(gen_config, "top_p", 0.95),
            top_k=getattr(gen_config, "top_k", 20),
        )

    output_ids_full = generated_ids_direct_copy[0].tolist()
    prompt_len = len(model_inputs.input_ids[0])
    output_ids_new = output_ids_full[prompt_len:]

    assistant_response = tokenizer.decode(output_ids_new, skip_special_tokens=True).strip()

    if return_cache_after_generation:
        return assistant_response, None
    else:
        return assistant_response


def generate_response_qwen(model, tokenizer, prompt_text, max_new_tokens=50, disable_ttl=None, cache_for_generation=None, return_cache_after_generation=False):
    logger.info(f"\n--- Generating with Qwen for Prompt ---\n{prompt_text}\n")
    messages = [{"role": "user", "content": prompt_text}]
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text_prompt = "user: " + prompt_text + "\nassistant:"
            logger.warning("Warning: tokenizer.apply_chat_template not found for Qwen. Using basic prompt formatting.")
    except Exception as e:
        logger.warning(f"Warning: Could not apply chat template for Qwen: {e}. Using basic prompt.")
        text_prompt = "user: " + prompt_text + "\nassistant:"

    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(DEVICE)
    attention_mask_for_gen = torch.ones_like(model_inputs.input_ids)

    with torch.no_grad():
        eos_token_ids_list = list(tokenizer.eos_token_id_set if hasattr(tokenizer, 'eos_token_id_set') and tokenizer.eos_token_id_set else [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])

        gen_config_source = model if hasattr(model, 'generation_config') and model.generation_config is not None else qwen_hf_model_global
        gen_config = gen_config_source.generation_config

        current_pad_token_id = tokenizer.pad_token_id
        if current_pad_token_id is None: current_pad_token_id = getattr(gen_config, 'pad_token_id', getattr(gen_config, 'eos_token_id', 0))
        if isinstance(current_pad_token_id, list): current_pad_token_id = current_pad_token_id[0]

        effective_eos_token_id_for_gen = eos_token_ids_list if eos_token_ids_list else getattr(gen_config, 'eos_token_id', None)
        if isinstance(effective_eos_token_id_for_gen, list) and not effective_eos_token_id_for_gen: effective_eos_token_id_for_gen = None

        generated_ids_hf = model.generate(
            model_inputs.input_ids, max_new_tokens=max_new_tokens,
            attention_mask=attention_mask_for_gen,
            pad_token_id=current_pad_token_id,
            eos_token_id=effective_eos_token_id_for_gen,
            do_sample=getattr(gen_config, "do_sample", True), temperature=getattr(gen_config, "temperature", 0.6),
            top_p=getattr(gen_config, "top_p", 0.95), top_k=getattr(gen_config, "top_k", 50),
        )
    output_ids_full = generated_ids_hf[0].tolist()
    prompt_len = len(model_inputs.input_ids[0])
    output_ids_new = output_ids_full[prompt_len:]
    assistant_response = tokenizer.decode(output_ids_new, skip_special_tokens=True).strip()

    if return_cache_after_generation:
        return assistant_response, None
    else:
        return assistant_response

active_model = None


def ttl_analysis_callback_for_generation_manager(
    logits_for_next_token: torch.Tensor,
    full_generated_sequence_tensor: torch.Tensor, # Full `out` tensor from TTATransformer.sample
    gen_step_idx: int,
    # --- Args passed via partial ---
    secret_code_target_token_ids: List[int],
    analysis_capture_list: List[Dict], # Mutable list to append results
    callback_state: Dict, # Mutable dict for state like {"inside_code_tag": False, "current_target_idx": 0}
    tokenizer_for_analysis: Any,
    model_vocab_size_for_analysis: int,
    current_learning_iteration_for_analysis: int # For logging
    ):

    if callback_state.get("stop_analysis", False):
        return

    generated_ids_list_so_far = full_generated_sequence_tensor[0].tolist()
    current_generated_text = tokenizer_for_analysis.decode(generated_ids_list_so_far, skip_special_tokens=False) # Use skip_special_tokens=False to detect tags

    code_tag_prefix = callback_state.get("code_tag_prefix", "<code>")
    code_tag_suffix = callback_state.get("code_tag_suffix", "</code>")

    # Check for start of code tag
    if not callback_state["inside_code_tag"] and current_generated_text.endswith(code_tag_prefix):
        callback_state["inside_code_tag"] = True
        callback_state["current_target_idx"] = 0
        # logger.info(f"    [Iter {current_learning_iteration_for_analysis} Gen Step {gen_step_idx}] Callback: Detected '{code_tag_prefix}'. Starting token-level analysis.")
        # This log is now too verbose for every step, moved to main loop.

    # If inside code tag, analyze prediction for the next expected token of the secret code
    if callback_state["inside_code_tag"]:
        current_target_idx = callback_state["current_target_idx"]

        if current_target_idx < len(secret_code_target_token_ids):
            expected_next_token_id = secret_code_target_token_ids[current_target_idx]
            expected_next_token_str = tokenizer_for_analysis.decode([expected_next_token_id])

            # Analyze the logits for this expected_next_token_id
            probs_tensor = F.softmax(logits_for_next_token, dim=-1)
            rank_of_expected, prob_of_expected, _ = analyze_probabilities_for_ttl_tracking(
                probs_tensor,
                target_ids=[[expected_next_token_id]], # analyze_probabilities expects list of lists
                target_strs_legend=[expected_next_token_str],
                tokenizer=tokenizer_for_analysis,
                k_top=0, # Not needed for this specific token's rank/prob
                model_vocab_size=model_vocab_size_for_analysis
            )

            analysis_capture_list.append({
                "target_idx_in_secret": current_target_idx,
                "expected_token_id": expected_next_token_id,
                "expected_token_str": expected_next_token_str,
                "rank": rank_of_expected.get(expected_next_token_str, model_vocab_size_for_analysis + 1),
                "prob": prob_of_expected.get(expected_next_token_str, 0.0)
            })
            # logger.info(f"    [Iter {current_learning_iteration_for_analysis} Gen Step {gen_step_idx}] Callback: Analyzing for target_idx {current_target_idx} ('{expected_next_token_str}'). Rank: {rank_of_expected.get(expected_next_token_str)}, Prob: {prob_of_expected.get(expected_next_token_str):.4e}")

            # IMPORTANT: The callback doesn't know what token WILL be generated next.
            # It only records the prediction for the *expected* token.
            # The main loop will advance current_target_idx based on actual generation.
            # For simplicity in this callback, we'll let the main loop handle the "actual generated" part.
            # The callback just provides the prediction for the *next token in the target sequence*.
            # The `current_target_idx` will be incremented in the main loop if the model generates the correct token.
            # However, for this callback to provide analysis for *subsequent* target tokens,
            # it needs to know that the *previous* target token was indeed generated.
            # This makes the callback logic complex if it tries to perfectly align.

            # Simpler: The callback always logs the prediction for the *current* `callback_state["current_target_idx"]`.
            # The main loop, after generation, will see how many of these predictions were "used"
            # by checking the actual generated code.
            # If the model generates the expected token, the main loop will increment `callback_state["current_target_idx"]`
            # *before the next generation step's callback*. This is not possible with current `model.sample` structure.

            # Alternative: The callback logs for its current_target_idx.
            # The main loop, after the *entire* query generation, iterates through `analysis_capture_list`
            # and aligns it with the `parsed_code`.

            # For now, the callback will log the prediction for `secret_code_target_token_ids[callback_state["current_target_idx"]]`.
            # The `callback_state["current_target_idx"]` will be incremented by the callback itself,
            # assuming it's predicting the sequence. This might lead to analysis for tokens
            # that are never actually reached if the model deviates.
            callback_state["current_target_idx"] += 1

        else: # We've analyzed all tokens in the secret code
            callback_state["inside_code_tag"] = False # Stop further analysis within this tag
            callback_state["stop_analysis"] = True    # Stop for this generation call
            # logger.info(f"    [Iter {current_learning_iteration_for_analysis} Gen Step {gen_step_idx}] Callback: Reached end of secret code analysis.")

    # Check for end of code tag to stop analysis for this call
    if callback_state["inside_code_tag"] and current_generated_text.endswith(code_tag_suffix):
        callback_state["inside_code_tag"] = False
        callback_state["stop_analysis"] = True # Stop for this generation call
        # logger.info(f"    [Iter {current_learning_iteration_for_analysis} Gen Step {gen_step_idx}] Callback: Detected '{code_tag_suffix}'. Ending token-level analysis for this call.")


def analyze_probabilities_for_ttl_tracking(probs_tensor, target_ids, target_strs_legend, tokenizer, k_top=5, model_vocab_size=None):
    ranks, probabilities = {}, {}
    if probs_tensor.ndim == 2 and probs_tensor.shape[0] == 1:
        probs_np = probs_tensor[0].cpu().numpy()
    elif probs_tensor.ndim == 1:
        probs_np = probs_tensor.cpu().numpy()
    else:
        logger.error(f"probs_tensor has unexpected shape: {probs_tensor.shape}")
        for token_legend_str in target_strs_legend:
            ranks[token_legend_str], probabilities[token_legend_str] = (model_vocab_size or 0) + 1, 0.0
        return ranks, probabilities, []

    actual_vocab_size = model_vocab_size if model_vocab_size is not None else len(probs_np)

    for token_id_list, token_legend_str in zip(target_ids, target_strs_legend):
        # For this function, when called by the callback for a *specific* target token,
        # token_id_list will contain just that one token ID.
        token_id = token_id_list[0] if token_id_list else None

        if token_id is None or not (0 <= token_id < actual_vocab_size):
            prob, rank = 0.0, actual_vocab_size + 1
        else:
            prob = probs_np[token_id] if token_id < len(probs_np) else 0.0
            valid_probs_for_rank = probs_np[:actual_vocab_size]
            rank = np.sum(valid_probs_for_rank > prob) + 1
        ranks[token_legend_str], probabilities[token_legend_str] = rank, float(prob)

    valid_probs_for_sort = probs_np[:actual_vocab_size]
    top_k_indices_full = np.argsort(valid_probs_for_sort)[::-1]
    top_k_tokens = []

    count_added = 0
    for idx_item in top_k_indices_full:
        if k_top > 0 and count_added >= k_top:
            break
        idx_int = int(idx_item)
        if not (0 <= idx_int < actual_vocab_size): continue

        try:
            decoded_tok = tokenizer.decode([idx_int], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            decoded_tok = f"[DecodeErr ID:{idx_int}]"

        top_k_tokens.append((decoded_tok, float(probs_np[idx_int])))
        count_added +=1
    return ranks, probabilities, top_k_tokens

def print_ttl_iteration_results(iteration, results_data, secret_code_to_learn_token_ids, tokenizer, top_k_to_display_count):
    logger.info(f"--- TTL Analysis Results for Iteration {iteration} ---")
    secret_code_str = tokenizer.decode(secret_code_to_learn_token_ids, skip_special_tokens=True)
    logger.info(f"  Target Secret Code: '{secret_code_str}' (Tokens: {len(secret_code_to_learn_token_ids)})")

    if not results_data["code_tag_found"]:
        logger.info(f"  Model did NOT generate '<code>' tag. Generated response: '{results_data['raw_response_text']}'")
        logger.info("  No token-by-token analysis available for plotting for this iteration.")
        return

    generated_code_str = results_data["generated_code_str"]
    logger.info(f"  Generated Code in Tag: '{generated_code_str}'")

    token_predictions_analysis = results_data["token_predictions_analysis"]

    if not token_predictions_analysis:
        logger.info("  No detailed token predictions were captured (e.g., model generated <code> then immediately </code> or something unexpected).")
        return

    max_tokens_to_compare = len(secret_code_to_learn_token_ids)
    generated_code_tokens = tokenizer.encode(generated_code_str if generated_code_str else "", add_special_tokens=False)

    for i in range(max_tokens_to_compare):
        target_token_id = secret_code_to_learn_token_ids[i]
        target_token_char = tokenizer.decode([target_token_id])
        
        generated_char_at_pos = tokenizer.decode([generated_code_tokens[i]]) if i < len(generated_code_tokens) else "[N/A]"
        
        analysis_for_this_pos = None
        # The token_predictions_analysis list is built sequentially by the callback
        # as it *expects* the target tokens.
        if i < len(token_predictions_analysis):
            analysis_for_this_pos = token_predictions_analysis[i]
            # Sanity check if the analysis entry corresponds to the current target token index
            if analysis_for_this_pos["target_idx_in_secret"] != i or analysis_for_this_pos["expected_token_id"] != target_token_id:
                # This might happen if the callback's state got misaligned or if the model deviated early
                # and the callback continued to log predictions for the *original* sequence.
                logger.warning(f"    Mismatch in analysis data for target position {i}. Expected token ID {target_token_id}, analysis for {analysis_for_this_pos['expected_token_id']}. Using this analysis entry anyway.")
                # analysis_for_this_pos = None # Or decide to invalidate it

        if analysis_for_this_pos:
            rank = analysis_for_this_pos['rank']
            prob = analysis_for_this_pos['prob']
            logger.info(f"  Pos {i+1}: Target='{target_token_char}' (ID:{target_token_id}) | Generated='{generated_char_at_pos}' | Predicted Rank for Target: {rank}, Prob: {prob:.4e}")
        else:
            # This case means the model stopped generating inside <code> before reaching this target position,
            # or the callback stopped.
            logger.info(f"  Pos {i+1}: Target='{target_token_char}' (ID:{target_token_id}) | Generated='{generated_char_at_pos}' | No prediction analysis captured for this position (model may have deviated or stopped early).")

    # Display top-k for the *first* token prediction if available (as a summary)
    if token_predictions_analysis:
        first_token_analysis_entry = token_predictions_analysis[0]
        # To show top-k, we'd need to have stored them in the callback or re-run analyze_probabilities
        # For now, this part is simplified as the main focus was rank/prob of target.
        # The old top_k_display_list was from a single capture point.
        # We can log the top-k for the prediction of the *first token of the secret code*.
        # This requires the callback to also store top_k if we want to display it here.
        # Let's assume `analyze_probabilities_for_ttl_tracking` can be called again if needed,
        # or that the callback is enhanced to store this.
        # For now, we'll skip detailed top-k printout here to keep it focused on the per-token rank/prob.
        pass
    else:
        logger.info("  No top-k predictions to display as no token analysis was captured.")


def plot_ttl_learning_live(fig_rank, ax_rank, fig_prob, ax_prob, results_history, secret_code_token_ids, tokenizer, model_vocab_size_for_plot):
    iterations_all = [r["iteration"] for r in results_history]
    vocab_size = model_vocab_size_for_plot

    ax_rank.clear()
    ax_prob.clear()

    num_target_tokens = len(secret_code_token_ids)

    for token_idx_in_secret in range(num_target_tokens):
        target_token_id = secret_code_token_ids[token_idx_in_secret]
        target_token_char = tokenizer.decode([target_token_id])
        legend_label_rank = f"Rank TgtTok[{token_idx_in_secret}] ('{target_token_char}')"
        legend_label_prob = f"Prob TgtTok[{token_idx_in_secret}] ('{target_token_char}')"

        ranks_for_this_token_pos = []
        probs_for_this_token_pos = []
        iterations_with_data_for_this_pos = []

        for res_item in results_history:
            iteration_val = res_item["iteration"]
            current_rank = np.nan
            current_prob = np.nan

            if res_item.get("code_tag_found", False) and \
               res_item.get("token_predictions_analysis") and \
               len(res_item["token_predictions_analysis"]) > token_idx_in_secret:
                
                analysis_entry = res_item["token_predictions_analysis"][token_idx_in_secret]
                # Ensure this analysis entry is indeed for the current token_idx_in_secret
                if analysis_entry.get("target_idx_in_secret") == token_idx_in_secret and \
                   analysis_entry.get("expected_token_id") == target_token_id:
                    current_rank = analysis_entry.get("rank", vocab_size + 1)
                    current_prob = analysis_entry.get("prob", 0.0)
            
            # Only add if we have a valid point for this iteration, even if it's NaN
            # This ensures x-axis (iterations_all) aligns with y-data that might have NaNs
            if iteration_val in iterations_all: # Should always be true
                 ranks_for_this_token_pos.append(current_rank)
                 probs_for_this_token_pos.append(current_prob)
        
        # Ensure ranks_for_this_token_pos and probs_for_this_token_pos have same length as iterations_all
        # by filling with NaN if an iteration is missing (though current loop structure should prevent this)
        
        if len(ranks_for_this_token_pos) == len(iterations_all):
            ax_rank.plot(iterations_all, ranks_for_this_token_pos, marker='o', linestyle='-', label=legend_label_rank)
        if len(probs_for_this_token_pos) == len(iterations_all):
            ax_prob.plot(iterations_all, probs_for_this_token_pos, marker='x', linestyle='--', label=legend_label_prob)

    secret_code_str_for_title = tokenizer.decode(secret_code_token_ids, skip_special_tokens=True)
    ax_rank.set_xlabel("Learning Iteration")
    ax_rank.set_ylabel("Rank (Lower is Better)")
    ax_rank.set_title(f"TTL Ranks (Learned Code: '{secret_code_str_for_title[:30].replace(chr(10), ' ')}...')")
    ax_rank.invert_yaxis()
    ax_rank.legend(loc='best', fontsize='small')
    ax_rank.grid(True, which='both', linestyle='--')

    all_ranks_flat = [analysis["rank"] for res_item in results_history for analysis in res_item.get("token_predictions_analysis", []) if "rank" in analysis and not np.isnan(analysis["rank"])]
    if all_ranks_flat:
        min_r, max_r = min(all_ranks_flat) if all_ranks_flat else 1, max(all_ranks_flat) if all_ranks_flat else vocab_size
        ax_rank.set_ylim(min(vocab_size +10, max_r + 0.1 * max_r +10 ) , max(0,min_r - 0.1*min_r -1) )
    else:
        ax_rank.set_ylim(vocab_size + 10, 0)


    ax_prob.set_xlabel("Learning Iteration")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_title(f"TTL Probabilities (Learned Code: '{secret_code_str_for_title[:30].replace(chr(10), ' ')}...')")
    ax_prob.legend(loc='best', fontsize='small')
    ax_prob.grid(True, which='both', linestyle='--')
    all_probs_flat = [analysis["prob"] for res_item in results_history for analysis in res_item.get("token_predictions_analysis", []) if "prob" in analysis and not np.isnan(analysis["prob"])]
    if all_probs_flat:
        max_p_overall = max(all_probs_flat) if all_probs_flat else 0.0
        ax_prob.set_ylim(0, max(0.0001, max_p_overall * 1.2) if max_p_overall > 0 else 0.01 )
    else:
        ax_prob.set_ylim(0, 0.01)


    fig_rank.tight_layout()
    fig_rank.canvas.draw_idle()
    fig_prob.tight_layout()
    fig_prob.canvas.draw_idle()
    plt.pause(0.1)


def run_ttl_learning_analysis_loop(model, tokenizer, args, secret_code_to_learn_override: str):
    logger.info("\n=== Starting TTL Learning Analysis for Code in Tag ===")

    import random
    # Ensure secret_code_to_learn_override is used if provided, otherwise generate
    if secret_code_to_learn_override:
        secret_code_to_learn = secret_code_to_learn_override.strip()
    else:
        secret_code_to_learn = format(random.randint(0,999999), "06d")
    
    logger.info(f"Using Secret Code for this run: '{secret_code_to_learn}'")

    expected_code_tag_prefix = "<code>"
    expected_code_tag_suffix = "</code>"

    prompt_learn_user_template = f"<|im_start|>user\nPlease learn and remember this code {secret_code_to_learn} for stardust powder, secret ingredient for the space muffin. Acknowledge by saying 'Understood, {secret_code_to_learn} remembered as code'.{args.no_think}<|im_end|>\n<|im_start|>assistant\n"
    prompt_query_user_for_analysis = f"<|im_start|>user\nWhat was the specific 6-digit code I told you to remember? Please respond *only* with the code enclosed in <code></code> tags. If not correctly remember then just guess, but answer with code any way.{args.no_think}<|im_end|>\n<|im_start|>assistant\n"

    n_repetitions = args.analysis_repetitions
    results_history = []
    current_model_cache_for_learning = None
    model_depth_for_cache = model.qwen_config.num_hidden_layers if hasattr(model, 'qwen_config') else TITANS_DEPTH
    if args.model_type == "tta":
        current_model_cache_for_learning = TTACache(current_token_idx=0, layer_caches=[TTALayerCache(nm_state=None, attn_kv_cache=None) for _ in range(model_depth_for_cache)])

    secret_code_token_ids = tokenizer.encode(secret_code_to_learn, add_special_tokens=False)
    if not secret_code_token_ids:
        logger.error(f"Secret code '{secret_code_to_learn}' tokenized to an empty list. Aborting analysis.")
        return
    logger.info(f"Tokenized Secret Code '{secret_code_to_learn}': {secret_code_token_ids} (Decoded check: '{tokenizer.decode(secret_code_token_ids)}')")


    plt.ion()
    fig_rank_ttl, ax_rank_ttl = plt.subplots(figsize=(12, 7))
    fig_prob_ttl, ax_prob_ttl = plt.subplots(figsize=(12, 7))

    # --- Iteration 0: Baseline Query ---
    logger.info("\n--- TTL Iteration 0 (Code in Tag): Baseline Query (Full Generation Analysis) ---")
    
    analysis_capture_list_iter0 = []
    callback_state_iter0 = {"inside_code_tag": False, "current_target_idx": 0, "stop_analysis": False, "code_tag_prefix": expected_code_tag_prefix, "code_tag_suffix": expected_code_tag_suffix}
    baseline_analysis_callback = partial(
        ttl_analysis_callback_for_generation_manager,
        secret_code_target_token_ids=secret_code_token_ids,
        analysis_capture_list=analysis_capture_list_iter0,
        callback_state=callback_state_iter0,
        tokenizer_for_analysis=tokenizer,
        model_vocab_size_for_analysis=model.qwen_config.vocab_size,
        current_learning_iteration_for_analysis=0
    )
    baseline_query_cache_for_tta = TTACache(current_token_idx=0, layer_caches=[TTALayerCache(nm_state=None, attn_kv_cache=None) for _ in range(model_depth_for_cache)]) if args.model_type == "tta" else None

    generated_response_0, _ = generate_response_titans(
        model, tokenizer, prompt_text=prompt_query_user_for_analysis, is_pre_templated_text=True,
        max_new_tokens=len(secret_code_token_ids) + 30, # Ample space for code and tags
        temperature=0.01,
        disable_ttl=True,
        model_type_str=args.model_type, cache_for_generation=baseline_query_cache_for_tta,
        return_cache_after_generation=True,
        analysis_callback_for_generation_fn=baseline_analysis_callback
    )
    parsed_code_0 = parse_code_from_response(generated_response_0)
    code_tag_found_0 = parsed_code_0 is not None
    
    results_history.append({
        "iteration": 0, "code_tag_found": code_tag_found_0,
        "generated_code_str": parsed_code_0,
        "token_predictions_analysis": list(analysis_capture_list_iter0), # Store copy
        "raw_response_text": generated_response_0
    })
    print_ttl_iteration_results(0, results_history[-1], secret_code_token_ids, tokenizer, args.analysis_top_k_display)
    plot_ttl_learning_live(fig_rank_ttl, ax_rank_ttl, fig_prob_ttl, ax_prob_ttl, results_history, secret_code_token_ids, tokenizer, model.qwen_config.vocab_size)

    # --- Learning Iterations ---
    for i in range(1, n_repetitions + 1):
        logger.info(f"\n--- TTL Iteration {i} (Code in Tag): Learning Confirmation & TTL Update & Query Recall Analysis ---")
        
        learn_user_prompt_text_i = prompt_learn_user_template
        logger.info(f"Iter {i}: User teaches (TTL {'Enabled' if not args.disable_ttl else 'Disabled'}): '{learn_user_prompt_text_i[:200].replace(chr(10), ' ')}...'")

        if args.model_type == "tta" and current_model_cache_for_learning is not None:
            current_model_cache_for_learning = reset_attention_cache_for_tta(current_model_cache_for_learning, model_depth_for_cache)

        assistant_confirmation_text, updated_cache_after_learn_gen = generate_response_titans(
            model, tokenizer, learn_user_prompt_text_i, is_pre_templated_text=True,
            max_new_tokens=len(secret_code_to_learn.split()) + 30,
            temperature=0.01,
            disable_ttl=False, # TTL is active during learning phase
            model_type_str=args.model_type,
            cache_for_generation=current_model_cache_for_learning,
            return_cache_after_generation=True
        )
        current_model_cache_for_learning = updated_cache_after_learn_gen
        logger.info(f"Iter {i}: Assistant confirmed learning: '{assistant_confirmation_text[:200].replace(chr(10), ' ')}...'")

        # Query Phase
        query_cache_for_iter_i = None
        if args.model_type == "tta" and current_model_cache_for_learning:
            query_cache_for_iter_i = reset_attention_cache_for_tta(current_model_cache_for_learning, model_depth_for_cache)
        elif args.model_type == "mac": # Basic cache reset for MAC
            nm_state_mac = current_model_cache_for_learning[2] if current_model_cache_for_learning and len(current_model_cache_for_learning) == 3 else None
            query_cache_for_iter_i = (0, None, nm_state_mac)
        # MAG/MAL would need similar cache reset logic if stateful across teach/query cycles

        analysis_capture_list_iter_i = []
        callback_state_iter_i = {"inside_code_tag": False, "current_target_idx": 0, "stop_analysis": False, "code_tag_prefix": expected_code_tag_prefix, "code_tag_suffix": expected_code_tag_suffix}
        query_phase_analysis_callback = partial(
            ttl_analysis_callback_for_generation_manager,
            secret_code_target_token_ids=secret_code_token_ids,
            analysis_capture_list=analysis_capture_list_iter_i,
            callback_state=callback_state_iter_i,
            tokenizer_for_analysis=tokenizer,
            model_vocab_size_for_analysis=model.qwen_config.vocab_size,
            current_learning_iteration_for_analysis=i
        )
        
        generated_response_i, _ = generate_response_titans(
            model, tokenizer, prompt_text=prompt_query_user_for_analysis, is_pre_templated_text=True,
            max_new_tokens=len(secret_code_token_ids) + 30,
            temperature=0.01,
            disable_ttl=True, # TTL is off for query generation itself to see what was learned
            model_type_str=args.model_type,
            cache_for_generation=query_cache_for_iter_i,
            return_cache_after_generation=True, # Cache might be needed if subsequent queries build on this
            analysis_callback_for_generation_fn=query_phase_analysis_callback
        )
        parsed_code_i = parse_code_from_response(generated_response_i)
        code_tag_found_i = parsed_code_i is not None

        results_history.append({
            "iteration": i, "code_tag_found": code_tag_found_i,
            "generated_code_str": parsed_code_i,
            "token_predictions_analysis": list(analysis_capture_list_iter_i),
            "raw_response_text": generated_response_i
        })
        print_ttl_iteration_results(i, results_history[-1], secret_code_token_ids, tokenizer, args.analysis_top_k_display)
        plot_ttl_learning_live(fig_rank_ttl, ax_rank_ttl, fig_prob_ttl, ax_prob_ttl, results_history, secret_code_token_ids, tokenizer, model.qwen_config.vocab_size)

    if args.analysis_plot_file_prefix:
        try:
            fig_rank_ttl.savefig(f"{args.analysis_plot_file_prefix}_code_tag_rank_evolution.png")
            fig_prob_ttl.savefig(f"{args.analysis_plot_file_prefix}_code_tag_probability_evolution.png")
            logger.info(f"\nSaved Code in Tag TTL analysis plots to '{args.analysis_plot_file_prefix}_code_tag_*.png'")
        except Exception as e:
            logger.error(f"Error saving Code in Tag TTL analysis plots: {e}")

    plt.ioff()
    if secret_code_token_ids and results_history:
        try:
            logger.info("Displaying Code in Tag TTL analysis plots. Close plot windows to exit.")
            plt.show(block=True)
        except Exception as e:
            logger.error(f"Error showing Code in Tag TTL analysis plots: {e}")
    logger.info("\n=== Code in Tag TTL Learning Analysis Finished ===")


qwen_tokenizer_global, qwen_hf_model_global = None, None

def reset_attention_cache_for_tta(tta_cache: Optional[TTACache], num_layers: int) -> Optional[TTACache]:
    if tta_cache is None:
        return TTACache(current_token_idx=0, layer_caches=[TTALayerCache(nm_state=None, attn_kv_cache=None) for _ in range(num_layers)])

    new_layer_caches = []
    if tta_cache.layer_caches:
        for lc in tta_cache.layer_caches:
            if lc is not None:
                new_layer_caches.append(TTALayerCache(nm_state=lc.nm_state, attn_kv_cache=None)) # Reset only attn_kv_cache
            else:
                new_layer_caches.append(TTALayerCache(nm_state=None, attn_kv_cache=None))
    else:
        new_layer_caches = [TTALayerCache(nm_state=None, attn_kv_cache=None) for _ in range(num_layers)]

    return TTACache(current_token_idx=0, layer_caches=new_layer_caches) # Reset current_token_idx for new attention context


def main():
    parser = argparse.ArgumentParser(description="Run demo with Titans or original Qwen model, with optional TTL learning analysis.")
    parser.add_argument(
        "--model_type", type=str,
        choices=["mac", "mag", "mal", "tta", "qwen", "qwen_direct_copy"],
        required=True,
        help="Specify Titans variant, original 'qwen', or 'qwen_direct_copy'."
    )
    parser.add_argument(
        "--disable_ttl", action="store_true",
        help="For Titans models (non-mimic), disable Test-Time Learning for Neural Memory during generation and analysis learning steps."
    )
    parser.add_argument(
        "--vanilla_qwen_mimic", action="store_true",
        help="For Titans MAC/MAG/MAL/TTA models, configure to architecturally mimic Qwen, disabling Titans-specific modules."
    )
    parser.add_argument(
        "--test_prompts", type=str,
        default="hello,llm,fact,instruction",
        help="Comma-separated list of prompt types to test for generation (e.g., hello,llm,fact,instruction). Set to 'none' to skip generation tests."
    )
    parser.add_argument(
        "--analyze_fact_ttl_learning", action="store_true",
        help="Enable TTL learning analysis for a factual statement. Requires --model_type tta."
    )
    parser.add_argument(
        "--test_code_tag_learning_scenario", action="store_true",
        help="Run the new scenario: baseline, contextual learning, and TTL analysis for learning a code within <code> tags. Designed for TTA."
    )
    parser.add_argument(
        "--test_code_tag_learning_scenario_mode", type=str,
        choices=["char", "digit"],
        default="digit",
        help="Run the new scenario: baseline, contextual learning, and TTL analysis for learning a code within <code> tags. Designed for TTA."
    )
    parser.add_argument(
        "--analysis_fact", type=str,
        default="The secret code is Starlight-Omega-Seven.",
        help="The fact/string to be learned during general TTL fact analysis."
    )
    parser.add_argument(
        "--analysis_query_prompt", type=str,
        default="What was the secret code I told you to remember?",
        help="The user's part of the query prompt used to query the model during general TTL fact analysis."
    )
    parser.add_argument(
        "--analysis_target_tokens", type=str,
        default="Starlight,Omega,Seven,secret,code",
        help="Comma-separated list of tokens to track during TTL analysis (typically the first token of the fact/code or key parts)."
    )
    parser.add_argument(
        "--analysis_repetitions", type=int, default=30,
        help="Number of learning repetitions for TTL analysis."
    )
    parser.add_argument(
        "--analysis_plot_file_prefix", type=str, default="ttl_analysis",
        help="Prefix for saving TTL analysis plot files."
    )
    parser.add_argument(
        "--analysis_nm_gate_bias", type=float, default=5.0, # Changed default to 5.0 for "max on"
        help="Initial bias for NM gate logit for TTA model during analysis (e.g., 0.0, 1.0, 5.0 for max)."
    )
    parser.add_argument(
        "--analysis_top_k_display", type=int, default=5,
        help="Number of top-k tokens to display during TTL analysis printout."
    )
    parser.add_argument(
        "--no_think", type=str, default="/no_think",
        help="empty thinking tag for qwen3"
    )


    args = parser.parse_args()

    global qwen_tokenizer_global, qwen_hf_model_global
    qwen_tokenizer_global, qwen_hf_model_global = load_qwen_components()

    global active_model
    generate_fn = None
    model_name_str = ""
    current_model_cache = None

    is_effective_mimic = args.vanilla_qwen_mimic
    current_model_type_for_init = args.model_type
    current_disable_ttl_for_run = args.disable_ttl

    if args.vanilla_qwen_mimic:
        is_effective_mimic = True
        current_disable_ttl_for_run = True

    if current_model_type_for_init == "qwen_direct_copy":
        active_model = initialize_qwen_direct_copy_model(
            qwen_tokenizer_global, qwen_hf_model_global
        )
        generate_fn = partial(generate_response_qwen_hf_style_copied, model=active_model, tokenizer=qwen_tokenizer_global)
        model_name_str = f"Qwen Direct Architectural Copy ({QWEN_MODEL_NAME} core)"
        if args.vanilla_qwen_mimic: logger.warning("Warning: --vanilla_qwen_mimic ignored for 'qwen_direct_copy'.")
        if args.disable_ttl: logger.warning("Warning: --disable_ttl ignored for 'qwen_direct_copy'.")

    elif current_model_type_for_init in ["mac", "mag", "mal", "tta"]:
        nm_gate_bias_for_init_override = None
        if (args.analyze_fact_ttl_learning or args.test_code_tag_learning_scenario) and current_model_type_for_init == 'tta':
            nm_gate_bias_for_init_override = args.analysis_nm_gate_bias
            logger.info(f"Using NM gate bias for TTA model (analysis) from command line: {nm_gate_bias_for_init_override}")

        active_model = initialize_titans_model(
            qwen_tokenizer_global, qwen_hf_model_global,
            model_type_str=current_model_type_for_init,
            vanilla_qwen_mimic=is_effective_mimic,
            nm_gate_bias_override_for_tta_analysis=nm_gate_bias_for_init_override
        )

        generate_fn = partial(generate_response_titans, model=active_model, tokenizer=qwen_tokenizer_global, model_type_str=current_model_type_for_init)
        model_name_str = f"Titans ({current_model_type_for_init.upper()}) ({QWEN_MODEL_NAME} core)"
        if is_effective_mimic:
            model_name_str += " [Vanilla Qwen Mimic]"
            model_name_str += " (NM disabled by mimic, TTL effectively OFF)"
        else:
            model_name_str += " [Titans Features Active]"
            model_name_str += " [TTL DISABLED]" if current_disable_ttl_for_run else " [TTL ENABLED]"

        model_depth_for_cache = active_model.qwen_config.num_hidden_layers if hasattr(active_model, 'qwen_config') else TITANS_DEPTH
        if current_model_type_for_init == "tta":
            current_model_cache = TTACache(current_token_idx=0, layer_caches=[TTALayerCache(nm_state=None, attn_kv_cache=None) for _ in range(model_depth_for_cache)])
        elif current_model_type_for_init == "mac":
             current_model_cache = (0, None, None)
        elif current_model_type_for_init == "mag":
            current_model_cache = MAGCache(current_token_idx=0, layer_caches=[MAGLayerCache(nm_state=None, swa_kv_cache=None) for _ in range(model_depth_for_cache)])
        elif current_model_type_for_init == "mal":
            current_model_cache = MALCache(current_token_idx=0, layer_caches=[MALLayerCache(nm_state=None, swa_kv_cache=None) for _ in range(model_depth_for_cache)])


    elif current_model_type_for_init == "qwen":
        active_model = qwen_hf_model_global
        if not hasattr(active_model, 'generation_config') and hasattr(qwen_hf_model_global, 'generation_config'):
            active_model.generation_config = qwen_hf_model_global.generation_config
        elif not hasattr(active_model, 'generation_config'):
             active_model.generation_config = qwen_hf_model_global.config

        generate_fn = partial(generate_response_qwen, model=active_model, tokenizer=qwen_tokenizer_global)
        model_name_str = f"Original {QWEN_MODEL_NAME}"
        if args.vanilla_qwen_mimic: logger.warning("Warning: --vanilla_qwen_mimic ignored for 'qwen' model_type.")
        if args.disable_ttl: logger.warning("Warning: --disable_ttl ignored for 'qwen' model_type.")
    else:
        logger.error(f"Invalid model type specified or unhandled combination: {args.model_type}")
        return

    logger.info(f"\n>>> Running with: {model_name_str} on {DEVICE} <<<")

    if args.test_code_tag_learning_scenario:
      if current_model_type_for_init != "tta" or is_effective_mimic:
            logger.error("--test_code_tag_learning_scenario is designed for TTA models with TTL active (not in mimic mode).")
            return
      if 'char' in args.test_code_tag_learning_scenario_mode:
          import uuid
          secret_code_for_scenario = str(uuid.uuid4())[:6] # Generate a random 6-char code
      elif 'digit' in args.test_code_tag_learning_scenario_mode:
          import random
          secret_code_for_scenario = format(random.randint(0,999999), "06d") # Generate a random 6-digit code
      else:
          secret_code_for_scenario = args.test_code_tag_learning_scenario_mode
      logger.info(f"\n\n=== SCENARIO: Testing Code Learning ('{secret_code_for_scenario}') and Tag Formatting ===\n")

      run_ttl_learning_analysis_loop(active_model, qwen_tokenizer_global, args, secret_code_to_learn_override=secret_code_for_scenario)

    elif args.analyze_fact_ttl_learning:
        if current_model_type_for_init != "tta" or is_effective_mimic:
            logger.error("--analyze_fact_ttl_learning is designed for TTA models with TTL active (not in mimic mode).")
        else:
            logger.info("Running general fact TTL analysis. This will use args.analysis_fact as the 'code' and expect <code> tags due to current loop structure.")
            run_ttl_learning_analysis_loop(active_model, qwen_tokenizer_global, args, secret_code_to_learn_override=args.analysis_fact)

    else:
        logger.info("TTL Learning Analysis not enabled. Running generation prompts if specified.")
        tests_to_run = args.test_prompts.split(',')

        if "none" in tests_to_run:
            logger.info("Skipping generation tests as per --test_prompts none.")
        else:
            model_depth_for_cache_reset = active_model.qwen_config.num_hidden_layers if hasattr(active_model, 'qwen_config') and active_model.qwen_config else TITANS_DEPTH

            if "hello" in tests_to_run:
                prompt1_text = "Hello, who are you?"
                response1, current_model_cache = generate_fn(prompt_text=prompt1_text + args.no_think, max_new_tokens=500, temperature=0.01, disable_ttl=current_disable_ttl_for_run, cache_for_generation=current_model_cache, return_cache_after_generation=True, is_pre_templated_text=False)
                logger.info(f"\nFinal Assistant Response (Prompt 1): '{response1}'")

            if "llm" in tests_to_run:
                if current_model_type_for_init == "tta":
                    logger.info(f"Resetting TTA attention cache, preserving NM state before 'LLM' task.")
                    current_model_cache = reset_attention_cache_for_tta(current_model_cache, model_depth_for_cache_reset)
                elif current_model_type_for_init == "mac":
                    current_model_cache = (0, None, current_model_cache[2] if current_model_cache and len(current_model_cache) == 3 else None)


                prompt2_text = "Can you tell me about large language models?"
                response2, current_model_cache = generate_fn(prompt_text=prompt2_text + args.no_think, max_new_tokens=1000, temperature=0.01, disable_ttl=current_disable_ttl_for_run, cache_for_generation=current_model_cache, return_cache_after_generation=True, is_pre_templated_text=False)
                logger.info(f"\nFinal Assistant Response (Prompt 2): '{response2}'")

            if "fact" in tests_to_run:
                if current_model_type_for_init == "tta":
                    logger.info(f"Resetting TTA attention cache, preserving NM state before 'fact' task.")
                    current_model_cache = reset_attention_cache_for_tta(current_model_cache, model_depth_for_cache_reset)
                elif current_model_type_for_init == "mac":
                     current_model_cache = (0, None, current_model_cache[2] if current_model_cache and len(current_model_cache) == 3 else None)

                import uuid
                unique_code_fact = str(uuid.uuid4())[:6]
                fact_to_remember_orig = f"The secret ingredient for the space muffin is stardust powder code {unique_code_fact}."

                logger.info("\n--- Stating a fact (User teaches model) ---")
                prompt_text_fact_user_teach = f"<|im_start|>user\nPlease learn and remember this: {fact_to_remember_orig}. When you acknowledge, please repeat the fact exactly as I stated it.{args.no_think}<|im_end|>\n<|im_start|>assistant\n"

                effective_disable_ttl_for_teach = current_disable_ttl_for_run
                if is_effective_mimic: effective_disable_ttl_for_teach = True


                ack_response, current_model_cache = generate_fn(
                    prompt_text=prompt_text_fact_user_teach, is_pre_templated_text=True,
                    max_new_tokens=len(fact_to_remember_orig.split()) + 300,
                    temperature=0.01,
                    disable_ttl=effective_disable_ttl_for_teach,
                    cache_for_generation=current_model_cache,
                    return_cache_after_generation=True
                )
                logger.info(f"Model's acknowledgement: {ack_response}")

                prompt_text_recall_user = f"<|im_start|>user\nA few moments ago, I told you a secret ingredient for space muffins. What was that ingredient?{args.no_think}<|im_end|>\n<|im_start|>assistant\n"
                logger.info("\n--- Recalling the fact ---")

                cache_for_recall_fact = current_model_cache
                if current_model_type_for_init == "tta":
                    logger.info(f"Resetting TTA attention cache for recall, preserving NM state.")
                    cache_for_recall_fact = reset_attention_cache_for_tta(current_model_cache, model_depth_for_cache_reset)
                elif current_model_type_for_init == "mac":
                     cache_for_recall_fact = (0, None, current_model_cache[2] if current_model_cache and len(current_model_cache) == 3 else None)

                response_fact_recall, current_model_cache = generate_fn(
                    prompt_text=prompt_text_recall_user, is_pre_templated_text=True,
                    max_new_tokens=len(fact_to_remember_orig.split()) + 200,
                    temperature=0.01,
                    disable_ttl=True,
                    cache_for_generation=cache_for_recall_fact,
                    return_cache_after_generation=True
                )

                if response_fact_recall:
                    logger.info(f"Model's recall response: {response_fact_recall}")
                    if unique_code_fact.lower() in response_fact_recall.lower():
                        logger.info(f"Stateful TTL Test (Recall Across Calls for fact with code {unique_code_fact}): PASSED - Model recalled the specific fact.")
                    else:
                        logger.info(f"Stateful TTL Test (Recall Across Calls for fact with code {unique_code_fact}): FAILED - Model did not recall the specific fact. Expected part of: '{fact_to_remember_orig}'")
                logger.info("\n---------------------------------------------\n")


            if "instruction" in tests_to_run:
                if current_model_type_for_init == "tta":
                    logger.info(f"Resetting TTA attention cache, preserving NM state before 'instruction' task.")
                    current_model_cache = reset_attention_cache_for_tta(current_model_cache, model_depth_for_cache_reset)
                elif current_model_type_for_init == "mac":
                     current_model_cache = (0, None, current_model_cache[2] if current_model_cache and len(current_model_cache) == 3 else None)

                instruction = "From now on, whenever I say 'Alpha', you say 'Beta'."
                prompt_text_instruction_setup_user = f"<|im_start|>user\n{instruction} Please confirm you understood by saying 'Understood, Alpha means Beta'.{args.no_think}<|im_end|>\n<|im_start|>assistant\n"

                logger.info("\n--- Setting up an instruction (User teaches model) ---")
                effective_disable_ttl_for_teach = current_disable_ttl_for_run
                if is_effective_mimic: effective_disable_ttl_for_teach = True

                instruction_ack_response, current_model_cache = generate_fn(
                    prompt_text=prompt_text_instruction_setup_user, is_pre_templated_text=True,
                    max_new_tokens=500,
                    temperature=0.01,
                    disable_ttl=effective_disable_ttl_for_teach,
                    cache_for_generation=current_model_cache,
                    return_cache_after_generation=True
                )
                logger.info(f"Model's instruction acknowledgement: {instruction_ack_response}")

                prompt_text_instruction_test = f"<|im_start|>user\nAlpha{args.no_think}<|im_end|>\n<|im_start|>assistant\n"
                logger.info("\n--- Testing the instruction ---")

                cache_for_instruction_test = current_model_cache
                if current_model_type_for_init == "tta":
                    logger.info(f"Resetting TTA attention cache for instruction test, preserving NM state.")
                    cache_for_instruction_test = reset_attention_cache_for_tta(current_model_cache, model_depth_for_cache_reset)
                elif current_model_type_for_init == "mac":
                     cache_for_instruction_test = (0, None, current_model_cache[2] if current_model_cache and len(current_model_cache) == 3 else None)

                response_instruction_test, current_model_cache = generate_fn(
                    prompt_text=prompt_text_instruction_test, is_pre_templated_text=True,
                    max_new_tokens=500,
                    temperature=0.01,
                    disable_ttl=True,
                    cache_for_generation=cache_for_instruction_test,
                    return_cache_after_generation=True
                )
                if response_instruction_test:
                    logger.info(f"Model's response to 'Alpha': {response_instruction_test}")
                    if "beta" in response_instruction_test.lower():
                        logger.info("Stateful TTL Test (Instruction Across Calls): PASSED")
                    else:
                        logger.info("Stateful TTL Test (Instruction Across Calls): FAILED")

    if current_model_type_for_init == "qwen_direct_copy":
        logger.info(f"\nNote for Qwen Direct Architectural Copy: Model definition is a direct copy of Qwen3 style, with weights transferred from {QWEN_MODEL_NAME}.")
    elif current_model_type_for_init in ["mac", "mag", "mal", "tta"]:
        logger.info(f"\nNote for Titans ({current_model_type_for_init.upper()}): Core weights are copied from {QWEN_MODEL_NAME}.")
        if is_effective_mimic:
            logger.info(f"Titans ({current_model_type_for_init.upper()}) was configured to architecturally mimic Qwen. Titans-specific Neural Memory modules were disabled/neutralized.")
            logger.info("Test-Time Learning was effectively DISABLED due to mimic mode.")
        else:
            logger.info(f"Titans ({current_model_type_for_init.upper()}) specific components were initialized and active.")
            if args.test_code_tag_learning_scenario or args.analyze_fact_ttl_learning:
                 logger.info("NeuralMemory Test-Time Learning for prompt processing was ENABLED for the analysis loop's teaching phase.")
            else:
                logger.info("NeuralMemory Test-Time Learning for prompt processing was DISABLED." if current_disable_ttl_for_run else "NeuralMemory Test-Time Learning for prompt processing was ENABLED.")
            logger.info("NeuralMemory Test-Time Learning for model's own token generation is always DISABLED within a single sample call's generation loop (after prompt processing).")


    else:
        logger.info("\nNote for Qwen: This is the original pre-trained model behavior.")

    logger.info("\nDemo finished.")

if __name__ == "__main__":
    main()

