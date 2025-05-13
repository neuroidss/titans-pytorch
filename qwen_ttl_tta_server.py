# FILE: qwen_ttl_tta_server.py

import torch
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import uvicorn
import argparse
import asyncio
import json
import os 
import time
import sys
import uuid # For API key generation

# Assuming run_titans_with_qwen_init.py is in the same directory or PYTHONPATH
from run_titans_with_qwen_init import (
    load_qwen_components,
    initialize_titans_model,
    DEVICE,
)
from titans_pytorch import NeuralMemory, MemoryMLP, TTATransformer, Qwen3Config, TTACache, TTALayerCache, NeuralMemState

# --- FastAPI App Setup ---
app = FastAPI(title="Qwen TTA TTL Stateful Server", version="0.2.1") # Incremented version

# --- Global Model and Tokenizer ---
model_global: Optional[TTATransformer] = None
tokenizer_global = None
loaded_base_model_name_global = None 
effective_tta_model_name_global = None 
num_model_layers_global = 0 # Will be set at startup

# This will be updated in startup_event to reflect the actual running model
POTENTIAL_TTA_MODELS: List[str] = []


# --- Per-User Neural Memory State Store (In-Memory) ---
# Dict[api_key, List[Optional[NeuralMemState]]]
# The list stores the NM state for each layer of the TTA model.
user_specific_nm_states_store: Dict[str, List[Optional[NeuralMemState]]] = {}

# --- API Key Management (Simplified In-Memory) ---
generated_api_keys: set[str] = set()

auth_scheme = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials not in generated_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# --- Pydantic Models (largely the same, ensure ChatMessage handles tool roles) ---
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None 
    tool_calls: Optional[List[Dict]] = None 
    tool_call_id: Optional[str] = None 

class ChatCompletionRequest(BaseModel):
    model: str 
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "system"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage 
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None 

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str 
    object: str = "chat.completion"
    created: int 
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage
    system_fingerprint: Optional[str] = "fp_TTA_TTL_Qwen_Stateful"

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    system_fingerprint: Optional[str] = "fp_TTA_TTL_Qwen_Stateful"
    choices: List[ChatCompletionStreamResponseChoice]

# --- Helper Functions ---
def format_prompt(messages: List[ChatMessage]) -> str:
    if tokenizer_global and hasattr(tokenizer_global, 'apply_chat_template'):
        try:
            chat_to_template = []
            for m in messages:
                entry = {"role": m.role, "content": m.content or ""} 
                if m.role == "assistant" and m.tool_calls: entry["tool_calls"] = m.tool_calls
                if m.role == "tool" and m.tool_call_id: entry["tool_call_id"] = m.tool_call_id
                chat_to_template.append(entry)
            return tokenizer_global.apply_chat_template(
                chat_to_template, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"Failed to apply chat template: {e}. Using basic concatenation.")
    prompt_str = ""
    for msg in messages:
        prompt_str += f"{msg.role}: {msg.content or ''}\n"
        if msg.role == "assistant" and msg.tool_calls: prompt_str += f"Tool Calls: {json.dumps(msg.tool_calls)}\n"
    if not prompt_str.strip().endswith("assistant:") and not prompt_str.strip().endswith("<|im_start|>assistant"):
        prompt_str += "assistant:" # Qwen specific end token for assistant generation
    return prompt_str

# --- Admin Endpoint for API Key Generation ---
@app.post("/admin/create_api_key", status_code=status.HTTP_201_CREATED)
async def create_api_key(key_description: Optional[str] = "default_key"):
    new_key = str(uuid.uuid4())
    generated_api_keys.add(new_key)
    if num_model_layers_global > 0:
        user_specific_nm_states_store[new_key] = [None] * num_model_layers_global
    else: 
        user_specific_nm_states_store[new_key] = []
    print(f"Created API key: {new_key} (Description: {key_description})")
    return {"api_key": new_key, "description": key_description, "detail": "Key created successfully. Store it securely."}

# --- API Endpoints ---
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    # POTENTIAL_TTA_MODELS is now updated at startup
    model_cards = [ModelCard(id=model_id) for model_id in POTENTIAL_TTA_MODELS]
    return ModelList(data=model_cards)

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest, 
    api_key: str = Depends(verify_api_key) 
):
    global effective_tta_model_name_global, num_model_layers_global
    if model_global is None or tokenizer_global is None or effective_tta_model_name_global is None:
        raise HTTPException(status_code=503, detail="Model not loaded or not ready.")
    if request.model != effective_tta_model_name_global:
        raise HTTPException(status_code=400, detail=f"Invalid model requested. This server is running '{effective_tta_model_name_global}'.")

    prompt_text = format_prompt(request.messages)
    
    # Max prompt length calculation (Qwen specific, needs to be robust)
    # model_global.qwen_config.max_position_embeddings might be the theoretical max.
    # Effective max prompt len also depends on tokenizer behavior for control tokens.
    # For now, a simpler check based on a reasonable limit minus max_tokens.
    # Max sequence length for Qwen models is often 32768 or similar.
    # Actual context window for attention could be model_global.qwen_config.sliding_window
    # or model_global.qwen_config.max_position_embeddings
    
    # A more conservative max_prompt_len
    # This needs to be based on the actual model's context window capacity.
    # For Qwen3-0.6B, max_position_embeddings is typically 32768.
    # Sliding window might also play a role if enabled.
    model_context_limit = getattr(model_global.qwen_config, 'max_position_embeddings', 2048) # Default to 2048 if not found
    max_prompt_len = model_context_limit - request.max_tokens - 20 # -20 for safety margin & control tokens

    if max_prompt_len <= 0: raise HTTPException(status_code=400, detail="max_tokens too large for model context window.")
    
    input_ids = tokenizer_global.encode(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(DEVICE)
    
    if input_ids.shape[1] == 0: 
        print(f"Warning: Prompt resulted in empty input_ids after tokenization/truncation. Prompt text: '{prompt_text[:200]}...'")
        raise HTTPException(status_code=400, detail="Prompt too long or resulted in empty input after tokenization.")
    
    prompt_attention_mask = torch.ones_like(input_ids) # Attention mask for the prompt itself
    prompt_tokens = input_ids.shape[1]
    request_id = f"chatcmpl-TTA-S-{int(time.time_ns() / 1000)}"
    created_time = int(time.time())

    persisted_user_nm_states = user_specific_nm_states_store.get(api_key)
    if persisted_user_nm_states is None or len(persisted_user_nm_states) != num_model_layers_global:
        persisted_user_nm_states = [None] * num_model_layers_global
        user_specific_nm_states_store[api_key] = persisted_user_nm_states
        print(f"Initialized new NM state list for API key {api_key[:8]}... ({num_model_layers_global} layers)")

    # TTACache for the call: current_token_idx is 0 for new call's attention context.
    # NM state is loaded from persisted store. Attention K/V cache starts empty for the call.
    initial_layer_caches_for_call: List[Optional[TTALayerCache]] = []
    for i in range(num_model_layers_global):
        nm_state_for_layer = persisted_user_nm_states[i]
        initial_layer_caches_for_call.append(TTALayerCache(nm_state=nm_state_for_layer, attn_kv_cache=None))
    
    # current_call_tta_cache contains the *initial* state for this API call.
    # current_token_idx for TTACache is the starting position for RoPE and K/V cache for *this specific call*.
    # If processing a long prompt in one go, it starts at 0.
    # The NM state (nm_state_for_layer) carries its own independent seq_index for TTL.
    current_call_tta_cache = TTACache(current_token_idx=0, layer_caches=initial_layer_caches_for_call)

    if request.stream:
        async def stream_generator():
            nonlocal current_call_tta_cache # Ensure we can update this cache
            returned_tta_cache_from_sample_stream = None
            generated_ids_stream = []
            try:
                with torch.no_grad():
                    generated_suffix_ids_tensor, returned_tta_cache_from_sample_stream = model_global.sample(
                        prompt=input_ids,
                        seq_len=input_ids.shape[1] + request.max_tokens,
                        temperature=request.temperature,
                        attention_mask=prompt_attention_mask, # Mask for the prompt
                        disable_ttl=False,
                        cache=current_call_tta_cache, 
                        return_cache_override=True 
                    )
                generated_ids_stream = generated_suffix_ids_tensor[0].tolist()
            except Exception as e:
                print(f"Error during model.sample (stream): {e}")
                error_choice = ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(content=f" Error during generation: {str(e)}"), finish_reason="error")
                yield f"data: {ChatCompletionStreamResponse(id=request_id, model=effective_tta_model_name_global, choices=[error_choice], created=created_time).model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            finish_reason_stream = "stop" 
            assistant_role_sent = False
            for i, token_id in enumerate(generated_ids_stream):
                if token_id in tokenizer_global.eos_token_id_set:
                    finish_reason_stream = "stop"; break
                if i >= request.max_tokens : # Double check max_tokens limit
                    finish_reason_stream = "length"; break

                decoded_token = tokenizer_global.decode([token_id]) 
                delta_msg = DeltaMessage(content=decoded_token)
                if not assistant_role_sent: 
                    delta_msg.role = "assistant"; assistant_role_sent = True
                
                stream_choice = ChatCompletionStreamResponseChoice(index=0, delta=delta_msg, finish_reason=None)
                stream_chunk = ChatCompletionStreamResponse(id=request_id, model=effective_tta_model_name_global, choices=[stream_choice], created=created_time)
                yield f"data: {stream_chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.001) # Small delay for streaming effect
            
            if len(generated_ids_stream) >= request.max_tokens and finish_reason_stream == "stop": # If max_tokens reached before EOS
                 if not any(tid in tokenizer_global.eos_token_id_set for tid in generated_ids_stream[-1:]):
                    finish_reason_stream = "length"

            final_stream_choice = ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(), finish_reason=finish_reason_stream)
            final_stream_chunk = ChatCompletionStreamResponse(id=request_id, model=effective_tta_model_name_global, choices=[final_stream_choice], created=created_time)
            yield f"data: {final_stream_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

            if returned_tta_cache_from_sample_stream and returned_tta_cache_from_sample_stream.layer_caches:
                new_nm_states_for_user = [lc.nm_state for lc in returned_tta_cache_from_sample_stream.layer_caches if lc]
                if len(new_nm_states_for_user) == num_model_layers_global:
                    user_specific_nm_states_store[api_key] = new_nm_states_for_user
                    print(f"Updated NM states for API key {api_key[:8]}... after stream.")
                else:
                    print(f"Warning: NM state update mismatch after stream. Expected {num_model_layers_global}, got {len(new_nm_states_for_user)}.")
            else:
                print(f"Warning: No cache returned from model.sample (stream) or cache empty. NM state not updated for {api_key[:8]}.")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else: # Non-streaming
        returned_tta_cache_from_sample_non_stream = None
        generated_ids_non_stream = []
        try:
            with torch.no_grad():
                generated_suffix_ids_tensor, returned_tta_cache_from_sample_non_stream = model_global.sample(
                    prompt=input_ids,
                    seq_len=input_ids.shape[1] + request.max_tokens,
                    temperature=request.temperature,
                    attention_mask=prompt_attention_mask, 
                    disable_ttl=False,
                    cache=current_call_tta_cache, 
                    return_cache_override=True 
                )
            generated_ids_non_stream = generated_suffix_ids_tensor[0].tolist()
        except Exception as e:
            print(f"Error during model.sample (non-stream): {e}")
            raise HTTPException(status_code=500, detail=f"Error during model generation: {str(e)}")
        
        processed_generated_ids = []
        for token_id in generated_ids_non_stream:
            if token_id in tokenizer_global.eos_token_id_set: break
            processed_generated_ids.append(token_id)
            if len(processed_generated_ids) >= request.max_tokens: break # Respect max_tokens
        
        completion_tokens = len(processed_generated_ids)
        assistant_response_content = tokenizer_global.decode(processed_generated_ids, skip_special_tokens=True).strip()
        finish_reason = "stop"
        if completion_tokens >= request.max_tokens:
            if not any(tid in tokenizer_global.eos_token_id_set for tid in generated_ids_non_stream[-1:]):
                 finish_reason = "length"
        
        assistant_message = ChatMessage(role="assistant", content=assistant_response_content)
        response_choice = ChatCompletionResponseChoice(index=0, message=assistant_message, finish_reason=finish_reason)
        usage_stats = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)
        
        if returned_tta_cache_from_sample_non_stream and returned_tta_cache_from_sample_non_stream.layer_caches:
            new_nm_states_for_user = [lc.nm_state for lc in returned_tta_cache_from_sample_non_stream.layer_caches if lc]
            if len(new_nm_states_for_user) == num_model_layers_global:
                user_specific_nm_states_store[api_key] = new_nm_states_for_user
                print(f"Updated NM states for API key {api_key[:8]}... after non-stream.")
            else:
                print(f"Warning: NM state update mismatch after non-stream. Expected {num_model_layers_global}, got {len(new_nm_states_for_user)}.")
        else:
            print(f"Warning: No cache returned from model.sample (non-stream) or cache empty. NM state not updated for {api_key[:8]}.")


        return ChatCompletionResponse(
            id=request_id, model=effective_tta_model_name_global, choices=[response_choice], created=created_time, usage=usage_stats
        )

def startup_event():
    global model_global, tokenizer_global, loaded_base_model_name_global, effective_tta_model_name_global, num_model_layers_global, POTENTIAL_TTA_MODELS
    
    base_model_name_from_env = os.environ.get("QWEN_BASE_MODEL_FOR_TTA_SERVER", "Qwen/Qwen3-0.6B")
    loaded_base_model_name_global = base_model_name_from_env
    simple_model_name_part = loaded_base_model_name_global.split('/')[-1]
    effective_tta_model_name_global = f"{simple_model_name_part}-tta-ttl"
    
    # Update POTENTIAL_TTA_MODELS to only list the currently effective model
    POTENTIAL_TTA_MODELS = [effective_tta_model_name_global]
    print(f"Effective TTA Model Name for server: {effective_tta_model_name_global}")


    print(f"Attempting to load base Qwen model: {loaded_base_model_name_global} for TTA initialization...")
    
    original_qwen_model_name_in_module = None
    run_titans_module = sys.modules.get('run_titans_with_qwen_init')
    if run_titans_module and hasattr(run_titans_module, 'QWEN_MODEL_NAME'):
        original_qwen_model_name_in_module = run_titans_module.QWEN_MODEL_NAME
    if run_titans_module: run_titans_module.QWEN_MODEL_NAME = loaded_base_model_name_global
    
    tokenizer, hf_model = load_qwen_components()
    tokenizer_global = tokenizer
    
    if run_titans_module:
        if original_qwen_model_name_in_module is not None: run_titans_module.QWEN_MODEL_NAME = original_qwen_model_name_in_module
        elif hasattr(run_titans_module, 'QWEN_MODEL_NAME'): delattr(run_titans_module, 'QWEN_MODEL_NAME')

    print(f"Initializing Titans TTA model (TTL Enabled) from {loaded_base_model_name_global}...")
    model_global = initialize_titans_model(
        qwen_tokenizer=tokenizer_global, qwen_hf_model=hf_model, model_type_str="tta", vanilla_qwen_mimic=False 
    )
    num_model_layers_global = len(model_global.layers) if model_global and hasattr(model_global, 'layers') else 0
    print(f"Titans TTA model ('{effective_tta_model_name_global}') loaded with {num_model_layers_global} layers. Ready.")
    print(f"Model running on device: {DEVICE}")

app.add_event_handler("startup", startup_event)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_qwen_name", type=str, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    os.environ["QWEN_BASE_MODEL_FOR_TTA_SERVER"] = args.base_model_qwen_name
    uvicorn.run(app, host="0.0.0.0", port=8000)

